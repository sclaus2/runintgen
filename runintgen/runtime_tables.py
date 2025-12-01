"""Runtime table mapping for code generation.

This module provides utilities to map between FFCX's compile-time table
structure and the runtime table structure we use.

Key concepts:
- FFCX uses named tables like "FE0_C0_D10_Q083" with shape [perm][entity][point][dof]
- basix.tabulate returns shape [nderivs, nq, ndofs, ncomps] - all derivatives together
- At runtime, we pass ONE table per unique element, containing all needed derivatives
- Each table has shape [nderivs, nq, ndofs] (component flattened or separate tables)

Structure:
1. UniqueElement: A distinct (element, max_derivative) that needs tabulation
2. ElementUsage: How a unique element is used (test, trial, coefficient, coordinate)
3. RuntimeElementInfo: Complete info for each unique element

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

from .fe_tables import IntegralRuntimeMeta


@dataclass
class ElementUsage:
    """How a unique element is used in the form.

    Attributes:
        role: "argument", "coefficient", "jacobian", "coordinate".
        terminal_index: Argument number (0=test, 1=trial) or coefficient index.
        component: Flat component index for blocked elements.
        derivatives_needed: Set of derivative tuples needed for this usage.
    """

    role: str
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
        element_hash: Hash for element identification.
        ndofs: Number of DOFs for this element.
        ncomps: Number of components (1 for scalar, >1 for vector/tensor).
        max_derivative_order: Maximum derivative order needed across all usages.
        usages: List of how this element is used (test, trial, coefficients, etc.).
        ffcx_table_names: Original FFCX table names that map to this element.
    """

    index: int
    element: Any  # basix element
    element_hash: int
    ndofs: int
    ncomps: int
    max_derivative_order: int
    usages: list[ElementUsage] = field(default_factory=list)
    ffcx_table_names: list[str] = field(default_factory=list)

    @property
    def is_argument(self) -> bool:
        """Check if this element is used for any argument (test/trial)."""
        return any(u.role == "argument" for u in self.usages)

    @property
    def is_coefficient(self) -> bool:
        """Check if this element is used for any coefficient."""
        return any(u.role == "coefficient" for u in self.usages)

    @property
    def is_coordinate(self) -> bool:
        """Check if this element is used for coordinate/jacobian."""
        return any(u.role in ("jacobian", "coordinate") for u in self.usages)

    def roles_str(self) -> str:
        """String representation of all roles."""
        parts = []
        for u in self.usages:
            parts.append(f"{u.role}[{u.terminal_index}]")
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
            # Extend or truncate to 2D
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
            # Extend or truncate to 3D
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
        hash_to_index: Map from element hash to element index.
        ffcx_table_to_element: Map from FFCX table name to (element_idx, deriv, comp).
        tdim: Topological dimension of the cell.
    """

    elements: list[UniqueElementInfo] = field(default_factory=list)
    hash_to_index: dict[int, int] = field(default_factory=dict)
    ffcx_table_to_element: dict[str, tuple[int, tuple[int, ...], int]] = field(
        default_factory=dict
    )
    tdim: int = 2  # Will be set during construction

    def get_or_create_element(
        self,
        element: Any,
        element_hash: int,
        ndofs: int,
        ncomps: int = 1,
    ) -> UniqueElementInfo:
        """Get existing element or create new one."""
        if element_hash in self.hash_to_index:
            return self.elements[self.hash_to_index[element_hash]]

        idx = len(self.elements)
        info = UniqueElementInfo(
            index=idx,
            element=element,
            element_hash=element_hash,
            ndofs=ndofs,
            ncomps=ncomps,
            max_derivative_order=0,
        )
        self.elements.append(info)
        self.hash_to_index[element_hash] = idx
        return info

    def add_usage(
        self,
        element_hash: int,
        role: str,
        terminal_index: int,
        component: int,
        derivative: tuple[int, ...],
        ffcx_table_name: str,
    ) -> None:
        """Add a usage to an element."""
        if element_hash not in self.hash_to_index:
            raise ValueError(f"Element hash {element_hash} not found")

        info = self.elements[self.hash_to_index[element_hash]]
        deriv_order = sum(derivative)

        # Update max derivative order
        if deriv_order > info.max_derivative_order:
            info.max_derivative_order = deriv_order

        # Find or create usage for this role/terminal
        usage = None
        for u in info.usages:
            if (
                u.role == role
                and u.terminal_index == terminal_index
                and u.component == component
            ):
                usage = u
                break
        if usage is None:
            usage = ElementUsage(
                role=role,
                terminal_index=terminal_index,
                component=component,
                derivatives_needed=set(),
            )
            info.usages.append(usage)

        usage.derivatives_needed.add(derivative)

        # Map FFCX table to this element
        info.ffcx_table_names.append(ffcx_table_name)
        self.ffcx_table_to_element[ffcx_table_name] = (
            info.index,
            derivative,
            component,
        )

    def get_table_access(
        self,
        ffcx_table_name: str,
        q_idx: str,
        dof_idx: str,
    ) -> str:
        """Generate C code for accessing a table value.

        Args:
            ffcx_table_name: Original FFCX table name.
            q_idx: C expression for quadrature point index.
            dof_idx: C expression for DOF index.

        Returns:
            C expression to access the table value.
        """
        if ffcx_table_name not in self.ffcx_table_to_element:
            raise ValueError(f"Unknown table: {ffcx_table_name}")

        elem_idx, derivative, component = self.ffcx_table_to_element[ffcx_table_name]
        info = self.elements[elem_idx]

        # Compute derivative index in basix format
        if self.tdim == 2:
            deriv_idx = DerivativeMapping.derivative_to_index_2d(derivative)
        else:
            deriv_idx = DerivativeMapping.derivative_to_index_3d(derivative)

        ndofs = info.ndofs
        ncomps = info.ncomps

        # Access pattern: data->elements[elem_idx].table[deriv_idx][q * ndofs * ncomps + dof * ncomps + comp]
        if ncomps == 1:
            # Scalar element: data->elements[i].table[d_idx * nq * ndofs + q * ndofs + dof]
            return (
                f"data->elements[{elem_idx}].table["
                f"{deriv_idx} * data->nq * {ndofs} + "
                f"({q_idx}) * {ndofs} + ({dof_idx})]"
            )
        else:
            # Vector/blocked element
            return (
                f"data->elements[{elem_idx}].table["
                f"{deriv_idx} * data->nq * {ndofs} * {ncomps} + "
                f"({q_idx}) * {ndofs} * {ncomps} + "
                f"({dof_idx}) * {ncomps} + {component}]"
            )


def build_runtime_element_mapping(
    integral_ir: Any,
    meta: IntegralRuntimeMeta | None = None,
) -> RuntimeElementMapping:
    """Build a mapping from unique elements to runtime table structure.

    Args:
        integral_ir: The FFCX integral IR.
        meta: The IntegralRuntimeMeta from our analysis (optional, not used yet).

    Returns:
        RuntimeElementMapping object with unique elements and derivative info.
    """
    from ffcx.ir.elementtables import get_modified_terminal_element
    from ufl import Argument, Coefficient
    from ufl.classes import Jacobian, SpatialCoordinate

    mapping = RuntimeElementMapping()
    expr_ir = integral_ir.expression

    # Get topological dimension from cell type
    # Collect all table references from the factorization graph
    for (cell_type_key, qrule), integrand_data in expr_ir.integrand.items():
        # Set tdim based on cell type (basix.CellType is an enum)
        ct_name = (
            cell_type_key.name if hasattr(cell_type_key, "name") else str(cell_type_key)
        )
        if "triangle" in ct_name.lower() or "quadrilateral" in ct_name.lower():
            mapping.tdim = 2
        elif "tetrahedron" in ct_name.lower() or "hexahedron" in ct_name.lower():
            mapping.tdim = 3
        else:
            mapping.tdim = 2  # Default

        factorization = integrand_data.get("factorization")
        if not factorization:
            continue

        for node_id, node_data in factorization.nodes.items():
            tr = node_data.get("tr")  # Table reference
            mt = node_data.get("mt")  # Modified terminal

            if tr is None or mt is None:
                continue

            mte = get_modified_terminal_element(mt)
            if mte is None:
                continue

            element, averaged, local_derivatives, flat_component = mte
            terminal = mt.terminal

            # Determine role and index
            if isinstance(terminal, Argument):
                role = "argument"
                terminal_index = terminal.number()
            elif isinstance(terminal, Coefficient):
                role = "coefficient"
                terminal_index = expr_ir.coefficient_numbering.get(terminal, -1)
            elif isinstance(terminal, SpatialCoordinate):
                role = "coordinate"
                terminal_index = 0
            elif isinstance(terminal, Jacobian):
                role = "jacobian"
                terminal_index = 0
            else:
                continue

            # Get element hash and properties
            element_hash = (
                hash(element) if hasattr(element, "__hash__") else id(element)
            )
            ndofs = tr.values.shape[-1]
            # Get number of components
            ncomps = 1
            if hasattr(element, "block_size"):
                ncomps = element.block_size

            # Get or create element info
            mapping.get_or_create_element(
                element=element,
                element_hash=element_hash,
                ndofs=ndofs,
                ncomps=ncomps,
            )

            # Add this usage
            mapping.add_usage(
                element_hash=element_hash,
                role=role,
                terminal_index=terminal_index,
                component=flat_component,
                derivative=local_derivatives,
                ffcx_table_name=tr.name,
            )

    return mapping


# Keep old interface for backward compatibility
@dataclass
class TableUsage:
    """A single usage of a table by a terminal (backward compat)."""

    role: str
    terminal_index: int


@dataclass
class RuntimeTableInfo:
    """Information about a runtime FE table (backward compat)."""

    index: int
    name: str
    element_hash: int
    component: int
    derivative: tuple[int, ...]
    ndofs: int
    usages: list[TableUsage] = field(default_factory=list)

    @property
    def role(self) -> str:
        return self.usages[0].role if self.usages else "unknown"

    @property
    def terminal_index(self) -> int:
        return self.usages[0].terminal_index if self.usages else -1


@dataclass
class RuntimeTableMapping:
    """Mapping from FFCX table names to runtime table indices (backward compat)."""

    tables: list[RuntimeTableInfo] = field(default_factory=list)
    name_to_index: dict[str, int] = field(default_factory=dict)
    request_to_index: dict[tuple, int] = field(default_factory=dict)

    def add_table(
        self,
        name: str,
        element_hash: int,
        role: str,
        terminal_index: int,
        component: int,
        derivative: tuple[int, ...],
        ndofs: int,
    ) -> int:
        usage = TableUsage(role=role, terminal_index=terminal_index)

        if name in self.name_to_index:
            index = self.name_to_index[name]
            info = self.tables[index]
            if usage not in info.usages:
                info.usages.append(usage)
            key = (element_hash, role, terminal_index, component, derivative)
            self.request_to_index[key] = index
            return index

        index = len(self.tables)
        info = RuntimeTableInfo(
            index=index,
            name=name,
            element_hash=element_hash,
            component=component,
            derivative=derivative,
            ndofs=ndofs,
            usages=[usage],
        )
        self.tables.append(info)
        self.name_to_index[name] = index
        key = (element_hash, role, terminal_index, component, derivative)
        self.request_to_index[key] = index
        return index

    def get_index(self, name: str) -> int | None:
        return self.name_to_index.get(name)


def build_runtime_table_mapping(
    integral_ir: Any,
    meta: IntegralRuntimeMeta,
) -> RuntimeTableMapping:
    """Build a mapping from FFCX tables to runtime table indices (backward compat)."""
    from ffcx.ir.elementtables import get_modified_terminal_element
    from ufl import Argument, Coefficient
    from ufl.classes import Jacobian, SpatialCoordinate

    mapping = RuntimeTableMapping()
    expr_ir = integral_ir.expression

    for (cell_type, qrule), integrand_data in expr_ir.integrand.items():
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

            if isinstance(terminal, Argument):
                role = "argument"
                terminal_index = terminal.number()
            elif isinstance(terminal, Coefficient):
                role = "coefficient"
                terminal_index = expr_ir.coefficient_numbering.get(terminal, -1)
            elif isinstance(terminal, SpatialCoordinate):
                role = "coordinate"
                terminal_index = 0
            elif isinstance(terminal, Jacobian):
                role = "jacobian"
                terminal_index = 0
            else:
                continue

            element_hash = (
                hash(element) if hasattr(element, "__hash__") else id(element)
            )
            ndofs = tr.values.shape[-1]

            mapping.add_table(
                name=tr.name,
                element_hash=element_hash,
                role=role,
                terminal_index=terminal_index,
                component=flat_component,
                derivative=local_derivatives,
                ndofs=ndofs,
            )

    return mapping


def generate_runtime_table_access(
    table_name: str,
    mapping: RuntimeTableMapping,
    quadrature_index: str,
    dof_index: str,
) -> str:
    """Generate C code to access a runtime table (old interface)."""
    idx = mapping.get_index(table_name)
    if idx is None:
        raise ValueError(f"Unknown table: {table_name}")

    info = mapping.tables[idx]
    ndofs = info.ndofs

    return f"rtables[{idx}][({quadrature_index}) * {ndofs} + ({dof_index})]"
