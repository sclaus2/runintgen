"""Form-level metadata for runtime integration (Plan v2).

This module provides the form-level and integral-level metadata structures
needed to support runtime quadrature with efficient element tabulation.

Key concepts (Plan v2):
1. Form-level: unique elements across all runtime integrals
2. Integral-level: which form elements are used, with what max derivative
3. Per-terminal: table_slot for generated kernel to access correct table

The metadata allows C++ to:
- Build a vector of basix::FiniteElement* from DOLFINx function spaces
- Tabulate each element once per (integral, cell) with correct max derivative
- Pass tables to kernels in a consistent order

C Struct Layout (passed from C++ to kernel):
    typedef struct {
        int nq;
        const double* w;              // quadrature weights
        const double* x_ref;          // reference points
        int n_tables;
        struct {
            int form_elem_index;      // index into form-level element vector
            int max_derivative;
            const double* table;
        } tables[];
    } runintgen_integral_data_t;
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# -----------------------------------------------------------------------------
# ElementKey - Efficient element identification using basix properties
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementKey:
    """Unique identifier for a basix finite element.

    This replaces string-based element hashing with a structured key
    based on the actual basix element properties. This is:
    - More efficient (no string operations)
    - Directly compatible with basix C++ API
    - Unambiguous (uses actual element properties)

    A basix::FiniteElement is uniquely identified by:
    (family, cell_type, degree, value_shape, discontinuous)

    Attributes:
        family: Element family (basix enum as int).
        cell_type: Cell type (basix enum as int).
        degree: Polynomial degree.
        value_shape: Shape of values (tuple of ints).
        discontinuous: Whether element is discontinuous.
    """

    family: int
    cell_type: int
    degree: int
    value_shape: tuple[int, ...] = ()
    discontinuous: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "family": self.family,
            "cell_type": self.cell_type,
            "degree": self.degree,
            "value_shape": list(self.value_shape),
            "discontinuous": self.discontinuous,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ElementKey":
        """Create from dictionary."""
        return cls(
            family=d["family"],
            cell_type=d["cell_type"],
            degree=d["degree"],
            value_shape=tuple(d.get("value_shape", ())),
            discontinuous=d.get("discontinuous", False),
        )

    def to_int64(self) -> int:
        """Pack into a 64-bit integer for fast hashing/comparison.

        Layout (bits):
        - [0:8]   family (8 bits, 256 values)
        - [8:16]  cell_type (8 bits, 256 values)
        - [16:24] degree (8 bits, 256 values)
        - [24:32] discontinuous + reserved (8 bits)
        - [32:64] value_shape hash (32 bits)
        """
        packed = self.family & 0xFF
        packed |= (self.cell_type & 0xFF) << 8
        packed |= (self.degree & 0xFF) << 16
        packed |= (int(self.discontinuous) & 0xFF) << 24
        # Use Python hash for value_shape, masked to 32 bits
        shape_hash = hash(self.value_shape) & 0xFFFFFFFF
        packed |= shape_hash << 32
        return packed

    @classmethod
    def from_int64(cls, packed: int) -> "ElementKey":
        """Reconstruct from packed integer (partial - value_shape is lost)."""
        # Note: This cannot fully reconstruct value_shape from hash
        # Use only when you have the original value_shape available
        raise NotImplementedError(
            "Cannot reconstruct value_shape from hash. "
            "Use from_dict() for full reconstruction."
        )


def element_key_from_basix(element: Any) -> ElementKey:
    """Create ElementKey from a basix element.

    Args:
        element: A basix FiniteElement (or wrapped UFL element).

    Returns:
        ElementKey with the element's identifying properties.
    """
    # Handle both basix elements and basix.ufl wrapped elements
    if hasattr(element, "_element"):
        # basix.ufl wrapped element
        basix_elem = element._element
    elif hasattr(element, "basix_element"):
        # Another wrapper type
        basix_elem = element.basix_element
    else:
        basix_elem = element

    # Get properties from basix element
    family = int(basix_elem.family)
    cell_type = int(basix_elem.cell_type)
    degree = basix_elem.degree

    # Get value shape
    vs = basix_elem.value_shape
    if hasattr(vs, "__iter__"):
        value_shape = tuple(int(x) for x in vs)
    else:
        value_shape = ()

    # Get discontinuous flag
    discontinuous = getattr(basix_elem, "discontinuous", False)

    return ElementKey(
        family=family,
        cell_type=cell_type,
        degree=degree,
        value_shape=value_shape,
        discontinuous=discontinuous,
    )


def element_key_from_ufl(element: Any) -> ElementKey:
    """Create ElementKey from a UFL element.

    This handles various UFL element types and extracts the underlying
    basix element to get the canonical identification.

    Args:
        element: A UFL element (FiniteElement, VectorElement, etc.).

    Returns:
        ElementKey identifying the element.
    """
    # Try to get basix element from UFL element
    if hasattr(element, "_element"):
        return element_key_from_basix(element._element)
    elif hasattr(element, "basix_element"):
        return element_key_from_basix(element.basix_element)
    elif hasattr(element, "family"):
        # Direct basix element
        return element_key_from_basix(element)
    else:
        # Fallback: use hash (should not happen with proper basix elements)
        raise TypeError(
            f"Cannot extract ElementKey from {type(element)}. "
            "Expected a basix element or basix.ufl wrapped element."
        )


# Type alias for element identification
ElementID = ElementKey  # Use structured key, not string


class Role(Enum):
    """Role of a terminal in the form."""

    TEST = auto()
    TRIAL = auto()
    COEFFICIENT = auto()
    GEOMETRY = auto()


@dataclass
class FormElementInfo:
    """Information about a unique element at the form level.

    This represents one distinct element (by ElementKey) across all
    runtime integrals in the form. The role and index provide one
    representative location where this element can be found in DOLFINx.

    C++ can use (role, index) to obtain the basix::FiniteElement*:
        switch (role):
            case Trial/Test:
                basix_elem = &form.function_spaces()[index]
                    ->element().basix_element();
            case Coefficient:
                basix_elem = &form.coefficient_spaces()[index]
                    ->element().basix_element();
            case Geometry:
                basix_elem = &form.mesh()->geometry().cmap().basix_element();

    Attributes:
        form_elem_index: Index of this element in the form's unique_elements list.
        element_key: ElementKey identifying this basix element uniquely.
        element: The actual basix/UFL element object (for Python-side tabulation).
        role: Representative role (where to find this element in DOLFINx).
        index: Representative index (arg index for test/trial, coeff index for coeff).
        ndofs: Number of DOFs for this element (per component).
        ncomps: Number of components (1 for scalar, >1 for vector/tensor).
    """

    form_elem_index: int
    element_key: ElementKey
    element: Any  # basix element
    role: Role
    index: int
    ndofs: int = 1
    ncomps: int = 1


@dataclass
class IntegralElementUsage:
    """Usage of a form-level element in a specific integral.

    For each runtime integral, we track which form-level elements are used
    and with what maximum derivative order. This is needed to call tabulate()
    with the correct derivative order at runtime.

    Attributes:
        form_elem_index: Index into the form's unique_elements list.
        max_derivative: Maximum derivative order needed for this integral.
        table_slot: Index into the tables[] array passed to the kernel.
    """

    form_elem_index: int
    max_derivative: int
    table_slot: int


@dataclass
class IntegralRuntimeLayout:
    """Complete runtime layout for one integral.

    This captures everything needed for C++ to prepare runtime data
    for a specific integral and for the generated kernel to access
    the correct table slot for each terminal.

    Attributes:
        integral_type: Type of integral ("cell", "exterior_facet", etc.).
        ir_index: Index in FFCx IR for this integral type.
        subdomain_id: Subdomain identifier.
        element_usages: List of IntegralElementUsage for elements in this integral.
        terminal_to_table_slot: Map from (role, terminal_index) to table slot index.
    """

    integral_type: str
    ir_index: int
    subdomain_id: int
    element_usages: list[IntegralElementUsage] = field(default_factory=list)
    terminal_to_table_slot: dict[tuple[Role, int], int] = field(default_factory=dict)

    def get_table_slot(self, role: Role, terminal_index: int) -> int:
        """Get the table slot for a specific terminal.

        Args:
            role: The role of the terminal (TEST, TRIAL, COEFFICIENT, GEOMETRY).
            terminal_index: Argument number or coefficient index.

        Returns:
            Index into the tables[] array.

        Raises:
            KeyError: If the terminal is not found.
        """
        return self.terminal_to_table_slot[(role, terminal_index)]


@dataclass
class FormRuntimeMetadata:
    """Complete runtime metadata for a form (Plan v2).

    This is the main container that holds:
    1. Form-level unique elements (for building basix::FiniteElement* vector)
    2. Per-integral layouts (for tabulation and kernel table slot mapping)

    Attributes:
        form_name: Name/identifier for this form.
        unique_elements: Form-level list of unique elements (by ElementKey).
        key_to_form_index: Map from ElementKey to form_elem_index.
        integral_layouts: Map from (integral_type, ir_index) to layout.
    """

    form_name: str = ""
    unique_elements: list[FormElementInfo] = field(default_factory=list)
    key_to_form_index: dict[ElementKey, int] = field(default_factory=dict)
    integral_layouts: dict[tuple[str, int], IntegralRuntimeLayout] = field(
        default_factory=dict
    )

    def get_form_element(self, element_key: ElementKey) -> FormElementInfo | None:
        """Get FormElementInfo by ElementKey."""
        idx = self.key_to_form_index.get(element_key)
        if idx is not None:
            return self.unique_elements[idx]
        return None

    def get_form_element_index(self, element_key: ElementKey) -> int | None:
        """Get form element index by ElementKey."""
        return self.key_to_form_index.get(element_key)

    def add_unique_element(
        self,
        element_key: ElementKey,
        element: Any,
        role: Role,
        index: int,
        ndofs: int = 1,
        ncomps: int = 1,
    ) -> FormElementInfo:
        """Add a unique element to the form-level list.

        If element_key already exists, returns existing FormElementInfo.
        Otherwise creates and adds a new one.

        Args:
            element_key: ElementKey identifying this element.
            element: The basix/UFL element object.
            role: Representative role for DOLFINx lookup.
            index: Representative index for DOLFINx lookup.
            ndofs: Number of DOFs.
            ncomps: Number of components.

        Returns:
            The FormElementInfo (existing or newly created).
        """
        if element_key in self.key_to_form_index:
            return self.unique_elements[self.key_to_form_index[element_key]]

        form_idx = len(self.unique_elements)
        info = FormElementInfo(
            form_elem_index=form_idx,
            element_key=element_key,
            element=element,
            role=role,
            index=index,
            ndofs=ndofs,
            ncomps=ncomps,
        )
        self.unique_elements.append(info)
        self.key_to_form_index[element_key] = form_idx
        return info

    def get_integral_layout(
        self, integral_type: str, ir_index: int
    ) -> IntegralRuntimeLayout | None:
        """Get the IntegralRuntimeLayout for a specific integral."""
        return self.integral_layouts.get((integral_type, ir_index))


# -----------------------------------------------------------------------------
# Builder Functions
# -----------------------------------------------------------------------------


def _role_from_analysis_role(analysis_role: Any) -> Role:
    """Convert analysis.ArgumentRole to form_metadata.Role."""
    # Import here to avoid circular imports
    from .analysis import ArgumentRole

    if analysis_role == ArgumentRole.TEST:
        return Role.TEST
    elif analysis_role == ArgumentRole.TRIAL:
        return Role.TRIAL
    elif analysis_role == ArgumentRole.COEFFICIENT:
        return Role.COEFFICIENT
    elif analysis_role == ArgumentRole.GEOMETRY:
        return Role.GEOMETRY
    else:
        raise ValueError(f"Unknown role: {analysis_role}")


def build_form_runtime_metadata(
    analysis: Any,  # RuntimeAnalysisInfo
) -> FormRuntimeMetadata:
    """Build FormRuntimeMetadata from analysis results.

    This is the main entry point for building Plan v2 metadata from
    the analysis phase results.

    Args:
        analysis: RuntimeAnalysisInfo from build_runtime_analysis().

    Returns:
        FormRuntimeMetadata with form-level unique elements and per-integral layouts.
    """
    from .analysis import RuntimeAnalysisInfo

    if not isinstance(analysis, RuntimeAnalysisInfo):
        raise TypeError(f"Expected RuntimeAnalysisInfo, got {type(analysis)}")

    metadata = FormRuntimeMetadata()

    # Step 1: Collect all unique elements across all runtime integrals
    # and record one representative usage for each
    # We use ElementKey for proper identification
    element_first_usage: dict[ElementKey, tuple[Any, Role, int, int, int]] = {}

    for (itype, ir_index), integral_info in analysis.integral_infos.items():
        for (role, idx), arg_info in integral_info.arguments.items():
            element = arg_info.element
            elem_key = element_key_from_basix(element)
            if elem_key not in element_first_usage:
                # Get ndofs and ncomps from element
                ndofs, ncomps = _get_element_dims(element)
                element_first_usage[elem_key] = (
                    element,
                    _role_from_analysis_role(role),
                    idx,
                    ndofs,
                    ncomps,
                )

    # Add all unique elements to metadata
    for elem_key, (element, role, idx, ndofs, ncomps) in element_first_usage.items():
        metadata.add_unique_element(
            element_key=elem_key,
            element=element,
            role=role,
            index=idx,
            ndofs=ndofs,
            ncomps=ncomps,
        )

    # Step 2: Build IntegralRuntimeLayout for each runtime integral
    for (itype, ir_index), integral_info in analysis.integral_infos.items():
        subdomain_id = integral_info.subdomain_id

        layout = IntegralRuntimeLayout(
            integral_type=itype,
            ir_index=ir_index,
            subdomain_id=subdomain_id,
        )

        # Collect elements used in this integral with their max derivatives
        # ElementKey -> max_derivative (for this integral only)
        integral_elem_maxderiv: dict[ElementKey, int] = {}

        for elem_id, elem_info in integral_info.elements.items():
            elem_key = element_key_from_basix(elem_info.element)
            integral_elem_maxderiv[elem_key] = elem_info.max_derivative_order

        # Also collect from arguments (may have different derivatives)
        for (role, idx), arg_info in integral_info.arguments.items():
            elem_key = element_key_from_basix(arg_info.element)
            if elem_key in integral_elem_maxderiv:
                integral_elem_maxderiv[elem_key] = max(
                    integral_elem_maxderiv[elem_key], arg_info.max_derivative_order
                )
            else:
                integral_elem_maxderiv[elem_key] = arg_info.max_derivative_order

        # Create element usages (assign table slots in order)
        key_to_table_slot: dict[ElementKey, int] = {}

        for elem_key, max_deriv in integral_elem_maxderiv.items():
            form_idx = metadata.key_to_form_index[elem_key]
            table_slot = len(layout.element_usages)
            key_to_table_slot[elem_key] = table_slot

            layout.element_usages.append(
                IntegralElementUsage(
                    form_elem_index=form_idx,
                    max_derivative=max_deriv,
                    table_slot=table_slot,
                )
            )

        # Build terminal_to_table_slot mapping
        for (role, idx), arg_info in integral_info.arguments.items():
            elem_key = element_key_from_basix(arg_info.element)
            table_slot = key_to_table_slot[elem_key]
            layout.terminal_to_table_slot[(_role_from_analysis_role(role), idx)] = (
                table_slot
            )

        metadata.integral_layouts[(itype, ir_index)] = layout

    return metadata


def _get_element_dims(element: Any) -> tuple[int, int]:
    """Get (ndofs, ncomps) from an element.

    Args:
        element: A basix/UFL element.

    Returns:
        Tuple of (ndofs, ncomps).
    """
    ncomps = 1
    if hasattr(element, "block_size"):
        ncomps = element.block_size

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

    return ndofs, ncomps


def _basix_hash(element: Any) -> int:
    """Return the Basix element hash, or zero when unavailable."""
    if hasattr(element, "_element"):
        basix_elem = element._element
    elif hasattr(element, "basix_element"):
        basix_elem = element.basix_element
    else:
        basix_elem = element

    if hasattr(basix_elem, "basix_hash"):
        return int(basix_elem.basix_hash())
    if hasattr(basix_elem, "hash"):
        return int(basix_elem.hash())
    return 0


# -----------------------------------------------------------------------------
# Export utilities for C++
# -----------------------------------------------------------------------------


def export_metadata_for_cpp(metadata: FormRuntimeMetadata) -> dict[str, Any]:
    """Export FormRuntimeMetadata in a format suitable for C++ consumption.

    This returns a dictionary that can be serialized to JSON or passed
    via pybind11 to C++ code.

    Returns:
        Dictionary with:
        - unique_elements: List of dicts with form element info (including ElementKey)
        - integral_layouts: Dict from "integral_type_ir_index" to layout info
    """
    unique_elems = []
    for fe in metadata.unique_elements:
        unique_elems.append(
            {
                "form_elem_index": fe.form_elem_index,
                "element_key": fe.element_key.to_dict(),
                "basix_hash": _basix_hash(fe.element),
                "role": fe.role.name.lower(),
                "index": fe.index,
                "ndofs": fe.ndofs,
                "ncomps": fe.ncomps,
            }
        )

    layouts = {}
    for (itype, ir_index), layout in metadata.integral_layouts.items():
        key = f"{itype}_{ir_index}"
        element_usages = [
            {
                "form_elem_index": eu.form_elem_index,
                "max_derivative": eu.max_derivative,
                "table_slot": eu.table_slot,
            }
            for eu in layout.element_usages
        ]
        terminal_map = {
            f"{role.name.lower()}_{tidx}": slot
            for (role, tidx), slot in layout.terminal_to_table_slot.items()
        }
        layouts[key] = {
            "integral_type": layout.integral_type,
            "ir_index": layout.ir_index,
            "subdomain_id": layout.subdomain_id,
            "element_usages": element_usages,
            "terminal_to_table_slot": terminal_map,
        }

    return {
        "unique_elements": unique_elems,
        "integral_layouts": layouts,
    }
