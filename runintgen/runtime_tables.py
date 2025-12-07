"""Runtime table mapping for code generation.

This module provides utilities to map between the analysis results
and the runtime table structure used in generated kernels.

Key concepts:
- basix.tabulate returns shape [nderivs, nq, ndofs, ncomps] - all derivatives together
- At runtime, we pass ONE table per unique element, containing all needed derivatives
- Each table has shape [nderivs, nq, ndofs] (component flattened or separate tables)

Structure:
1. UniqueElement: A distinct (element, max_derivative) that needs tabulation
2. ElementUsage: How a unique element is used (test, trial, coefficient, coordinate)
3. RuntimeElementMapping: Complete mapping for all elements in an integral

The runtime kernel signature is:
    void kernel(
        double* restrict A,           // Output tensor
        const double* restrict w,     // Coefficients
        const double* restrict c,     // Constants
        const double* restrict coords,// Coordinate dofs
        const runintgen_data* data    // Runtime quadrature and tables
    );

where runintgen_data contains:
    - nq, weights, points
    - nelements, elements[] (per-element: ndofs, nderivs, pointer to table data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analysis import ArgumentRole, RuntimeIntegralInfo


@dataclass
class ElementUsage:
    """How a unique element is used in the form.

    Attributes:
        role: ArgumentRole (TEST, TRIAL, COEFFICIENT, GEOMETRY).
        terminal_index: Argument number (0=test, 1=trial) or coefficient index.
        component: Flat component index for blocked elements.
        derivatives_needed: Set of derivative tuples needed for this usage.
    """

    role: ArgumentRole
    terminal_index: int
    component: int = 0
    derivatives_needed: set[tuple[int, ...]] = field(default_factory=set)


@dataclass
class UniqueElementInfo:
    """Information about a unique element in the form.

    At runtime, we tabulate each unique element once with all needed derivatives.
    basix.tabulate returns shape [nderivs, nq, ndofs, ncomps].

    Attributes:
        index: Index into the runtime elements array.
        element: The basix/UFL element object.
        element_id: String identifier for this element.
        ndofs: Number of DOFs for this element.
        ncomps: Number of components (1 for scalar, >1 for vector/tensor).
        max_derivative_order: Maximum derivative order needed across all usages.
        derivative_tuples: Set of all derivative tuples needed.
        usages: List of how this element is used (test, trial, coefficients, etc.).
    """

    index: int
    element: Any  # basix element
    element_id: str
    ndofs: int
    ncomps: int
    max_derivative_order: int
    derivative_tuples: set[tuple[int, ...]] = field(default_factory=set)
    usages: list[ElementUsage] = field(default_factory=list)

    @property
    def is_argument(self) -> bool:
        """Check if this element is used for any argument (test/trial)."""
        return any(
            u.role in (ArgumentRole.TEST, ArgumentRole.TRIAL) for u in self.usages
        )

    @property
    def is_test(self) -> bool:
        """Check if this element is used for test function."""
        return any(u.role == ArgumentRole.TEST for u in self.usages)

    @property
    def is_trial(self) -> bool:
        """Check if this element is used for trial function."""
        return any(u.role == ArgumentRole.TRIAL for u in self.usages)

    @property
    def is_coefficient(self) -> bool:
        """Check if this element is used for any coefficient."""
        return any(u.role == ArgumentRole.COEFFICIENT for u in self.usages)

    @property
    def is_coordinate(self) -> bool:
        """Check if this element is used for coordinate/jacobian."""
        return any(u.role == ArgumentRole.GEOMETRY for u in self.usages)

    def roles_str(self) -> str:
        """String representation of all roles."""
        parts = []
        for u in self.usages:
            parts.append(f"{u.role.name}[{u.terminal_index}]")
        return ", ".join(sorted(set(parts)))


class DerivativeMapping:
    """Maps a derivative tuple to an index in the basix tabulation array.

    basix.tabulate(n, pts) returns shape [num_derivs, nq, ndofs, ncomps]
    where num_derivs = (n+1)*(n+2)/2 for 2D, (n+1)*(n+2)*(n+3)/6 for 3D.

    Uses basix.index() which follows the basix derivative ordering convention.
    """

    @staticmethod
    def derivative_to_index(deriv: tuple[int, ...]) -> int:
        """Convert derivative tuple to basix index.

        Uses basix.index() which handles both 2D and 3D cases.

        Args:
            deriv: Derivative tuple, e.g., (1, 0) for d/dx in 2D,
                   (0, 1, 0) for d/dy in 3D.

        Returns:
            Index into the basix tabulation array's derivative dimension.
        """
        import basix

        return basix.index(*deriv)

    @staticmethod
    def derivative_to_index_2d(deriv: tuple[int, ...]) -> int:
        """Convert derivative tuple to basix index for 2D.

        Convenience method that ensures the tuple is 2D.
        """
        import basix

        if len(deriv) != 2:
            d0 = deriv[0] if len(deriv) > 0 else 0
            d1 = deriv[1] if len(deriv) > 1 else 0
            deriv = (d0, d1)

        return basix.index(*deriv)

    @staticmethod
    def derivative_to_index_3d(deriv: tuple[int, ...]) -> int:
        """Convert derivative tuple to basix index for 3D.

        Convenience method that ensures the tuple is 3D.
        """
        import basix

        if len(deriv) != 3:
            d = list(deriv) + [0] * (3 - len(deriv))
            deriv = (d[0], d[1], d[2])

        return basix.index(*deriv)


@dataclass
class RuntimeElementMapping:
    """Mapping from unique elements to runtime table structure.

    This is the main interface for code generation. It tracks:
    - Unique elements (each tabulated once at runtime)
    - How each element is used (test, trial, coefficients, jacobian)
    - Derivative requirements per element

    Attributes:
        elements: List of UniqueElementInfo objects (one per unique element).
        id_to_index: Map from element_id to element index.
        tdim: Topological dimension of the cell.
    """

    elements: list[UniqueElementInfo] = field(default_factory=list)
    id_to_index: dict[str, int] = field(default_factory=dict)
    tdim: int = 2

    def get_or_create_element(
        self,
        element: Any,
        element_id: str,
        ndofs: int,
        ncomps: int = 1,
    ) -> UniqueElementInfo:
        """Get existing element or create new one."""
        if element_id in self.id_to_index:
            return self.elements[self.id_to_index[element_id]]

        idx = len(self.elements)
        info = UniqueElementInfo(
            index=idx,
            element=element,
            element_id=element_id,
            ndofs=ndofs,
            ncomps=ncomps,
            max_derivative_order=0,
        )
        self.elements.append(info)
        self.id_to_index[element_id] = idx
        return info

    def add_usage(
        self,
        element_id: str,
        role: ArgumentRole,
        terminal_index: int,
        derivatives: set[tuple[int, ...]],
    ) -> None:
        """Add a usage to an element."""
        if element_id not in self.id_to_index:
            raise ValueError(f"Element id {element_id} not found")

        info = self.elements[self.id_to_index[element_id]]

        # Update max derivative order and derivatives
        for deriv in derivatives:
            deriv_order = sum(deriv)
            if deriv_order > info.max_derivative_order:
                info.max_derivative_order = deriv_order
            info.derivative_tuples.add(deriv)

        # Find or create usage for this role/terminal
        usage = None
        for u in info.usages:
            if u.role == role and u.terminal_index == terminal_index:
                usage = u
                break
        if usage is None:
            usage = ElementUsage(
                role=role,
                terminal_index=terminal_index,
                derivatives_needed=set(),
            )
            info.usages.append(usage)

        usage.derivatives_needed.update(derivatives)

    def get_table_access(
        self,
        element_id: str,
        deriv: tuple[int, ...],
        q_idx: str,
        dof_idx: str,
    ) -> str:
        """Generate C code for accessing a table value.

        Args:
            element_id: Element identifier.
            deriv: Derivative tuple.
            q_idx: C expression for quadrature point index.
            dof_idx: C expression for DOF index.

        Returns:
            C expression to access the table value.
        """
        if element_id not in self.id_to_index:
            raise ValueError(f"Unknown element: {element_id}")

        elem_idx = self.id_to_index[element_id]
        info = self.elements[elem_idx]

        # Compute derivative index in basix format
        if self.tdim == 2:
            deriv_idx = DerivativeMapping.derivative_to_index_2d(deriv)
        else:
            deriv_idx = DerivativeMapping.derivative_to_index_3d(deriv)

        ndofs = info.ndofs

        # Access pattern: data->elements[elem_idx].table[deriv_idx * nq * ndofs + q * ndofs + dof]
        return (
            f"data->elements[{elem_idx}].table["
            f"{deriv_idx} * data->nq * {ndofs} + "
            f"({q_idx}) * {ndofs} + ({dof_idx})]"
        )


def build_runtime_element_mapping(
    integral_info: RuntimeIntegralInfo,
) -> RuntimeElementMapping:
    """Build a mapping from unique elements to runtime table structure.

    This function takes the analysis result (RuntimeIntegralInfo) and builds
    the RuntimeElementMapping needed for code generation.

    Args:
        integral_info: The RuntimeIntegralInfo from the analysis phase.

    Returns:
        RuntimeElementMapping object with unique elements and derivative info.
    """
    mapping = RuntimeElementMapping()

    # Process all elements from the analysis
    for element_id, elem_info in integral_info.elements.items():
        element = elem_info.element

        # Get ncomps from element (for blocked elements)
        ncomps = 1
        if hasattr(element, "block_size"):
            ncomps = element.block_size

        # Get ndofs from element
        # For blocked elements, use sub-element dimension (per-component ndofs)
        ndofs = 1
        if ncomps > 1 and hasattr(element, "sub_elements") and element.sub_elements:
            # Blocked element: use sub-element's dimension
            sub_elem = element.sub_elements[0]
            if hasattr(sub_elem, "dim"):
                ndofs = sub_elem.dim
            elif hasattr(sub_elem, "space_dimension"):
                ndofs = sub_elem.space_dimension()
        elif hasattr(element, "dim"):
            ndofs = element.dim
        elif hasattr(element, "space_dimension"):
            ndofs = element.space_dimension()

        mapping.get_or_create_element(
            element=element,
            element_id=element_id,
            ndofs=ndofs,
            ncomps=ncomps,
        )

    # Process all arguments and their usages
    for (role, index), arg_info in integral_info.arguments.items():
        if arg_info.element_id in mapping.id_to_index:
            mapping.add_usage(
                element_id=arg_info.element_id,
                role=role,
                terminal_index=index,
                derivatives=arg_info.derivative_tuples,
            )

    return mapping


def build_runtime_element_mapping_from_ir(
    integral_ir: Any,
) -> RuntimeElementMapping:
    """Build element mapping directly from FFCX IR.

    This is an alternative entry point that works directly with the FFCX IR
    without going through RuntimeIntegralInfo. Useful for backward compatibility.

    Args:
        integral_ir: The FFCX integral IR.

    Returns:
        RuntimeElementMapping object with unique elements and derivative info.
    """
    from ffcx.ir.elementtables import get_modified_terminal_element
    from ufl import Argument, Coefficient
    from ufl.classes import Jacobian, SpatialCoordinate

    mapping = RuntimeElementMapping()
    expr_ir = integral_ir.expression

    # Get topological dimension from cell type
    for (cell_type_key, qrule), integrand_data in expr_ir.integrand.items():
        ct_name = (
            cell_type_key.name if hasattr(cell_type_key, "name") else str(cell_type_key)
        )
        if "triangle" in ct_name.lower() or "quadrilateral" in ct_name.lower():
            mapping.tdim = 2
        elif "tetrahedron" in ct_name.lower() or "hexahedron" in ct_name.lower():
            mapping.tdim = 3
        else:
            mapping.tdim = 2

        factorization = integrand_data.get("factorization")
        if not factorization:
            continue

        for node_id, node_data in factorization.nodes.items():
            tr = node_data.get("tr")
            mt = node_data.get("mt")

            if tr is None or mt is None:
                continue

            mte = get_modified_terminal_element(mt)
            if mte is None:
                continue

            element, averaged, local_derivatives, flat_component = mte
            terminal = mt.terminal

            # Determine role and index
            if isinstance(terminal, Argument):
                if terminal.number() == 0:
                    role = ArgumentRole.TEST
                else:
                    role = ArgumentRole.TRIAL
                terminal_index = terminal.number()
            elif isinstance(terminal, Coefficient):
                role = ArgumentRole.COEFFICIENT
                terminal_index = expr_ir.coefficient_numbering.get(terminal, -1)
            elif isinstance(terminal, (SpatialCoordinate, Jacobian)):
                role = ArgumentRole.GEOMETRY
                terminal_index = 0
            else:
                continue

            element_id = str(hash(element))
            ndofs = tr.values.shape[-1]
            ncomps = 1
            if hasattr(element, "block_size"):
                ncomps = element.block_size

            # Get or create element info
            mapping.get_or_create_element(
                element=element,
                element_id=element_id,
                ndofs=ndofs,
                ncomps=ncomps,
            )

            # Add this usage
            mapping.add_usage(
                element_id=element_id,
                role=role,
                terminal_index=terminal_index,
                derivatives={tuple(local_derivatives)},
            )

    return mapping
