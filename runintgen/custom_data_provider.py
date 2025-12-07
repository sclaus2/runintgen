"""Custom data provider for DOLFINx per-entity quadrature.

This module provides the runtime data structures and lookup functions
compatible with the proposed DOLFINx CustomDataFunction pattern:

    using CustomDataFunction = std::function<void*(std::span<const int32_t> entity_info)>;

Where entity_info contains:
    - Cell integrals: {loop_index, cell}
    - Exterior facet: {loop_index, cell, local_facet}
    - Interior facet: {loop_index, cell0, local_facet0, cell1, local_facet1}

The user can access:
    - entity_info[0]: loop_index (for contiguous array storage)
    - entity_info[1]: cell index (for map-based storage)

Design goals:
1. Efficient contiguous storage indexed by loop position
2. Support for tabulated FE values at runtime quadrature points
3. Memory-safe lifetime management
4. Simple API for creating the lookup function
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass


# =============================================================================
# C Structure Layout (matching what the kernel expects)
# =============================================================================
#
# The kernel receives a void* that points to runintgen_quadrature_config:
#
# struct runintgen_element {
#     int32_t ndofs;
#     int32_t nderivs;
#     const double* table;      // [nderivs, nq, ndofs] flattened
# };
#
# struct runintgen_quadrature_config {
#     int32_t nq;
#     int32_t tdim;
#     const double* points;     // [nq, tdim] flattened
#     const double* weights;    // [nq]
#     int32_t nelements;
#     const runintgen_element* elements;
# };
#
# =============================================================================


@dataclass
class ElementTable:
    """Tabulated finite element values at quadrature points.

    Attributes:
        ndofs: Number of degrees of freedom.
        nderivs: Number of derivative components.
        table: Values with shape [nderivs, nq, ndofs], C-contiguous.
    """

    ndofs: int
    nderivs: int
    table: npt.NDArray[np.float64]

    def __post_init__(self):
        self.table = np.ascontiguousarray(self.table, dtype=np.float64)


@dataclass
class QuadratureRule:
    """A quadrature rule with optional tabulated element values.

    Attributes:
        points: Quadrature points, shape [nq, tdim], C-contiguous.
        weights: Quadrature weights, shape [nq], C-contiguous.
        elements: Optional list of tabulated element values.
    """

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    elements: list[ElementTable] = field(default_factory=list)

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)

    @property
    def nq(self) -> int:
        """Number of quadrature points."""
        return len(self.weights)

    @property
    def tdim(self) -> int:
        """Topological dimension."""
        return self.points.shape[1] if self.points.ndim > 1 else 1


@dataclass
class PackedQuadratureData:
    """Pre-packed quadrature data ready to pass to kernels.

    This is a single contiguous buffer containing all the data
    the kernel needs, laid out to match the C struct.

    The buffer layout:
    [header: nq(i32), tdim(i32), nelements(i32), pad(i32)]
    [points: nq*tdim doubles]
    [weights: nq doubles]
    [element_headers: nelements * (ndofs(i32), nderivs(i32), table_offset(i32), pad(i32))]
    [element_tables: all tables concatenated]
    """

    buffer: npt.NDArray[np.uint8]
    _source_rule: QuadratureRule  # Keep reference for debugging

    @property
    def data_ptr(self) -> int:
        """Get the pointer to pass to kernels."""
        return self.buffer.ctypes.data

    @classmethod
    def from_rule(cls, rule: QuadratureRule) -> "PackedQuadratureData":
        """Pack a QuadratureRule into a contiguous buffer."""
        return _pack_quadrature_rule(rule)


def _pack_quadrature_rule(rule: QuadratureRule) -> PackedQuadratureData:
    """Pack a quadrature rule into a single contiguous buffer.

    The layout matches what the C kernel expects to unpack.
    """
    nq = rule.nq
    tdim = rule.tdim
    nelements = len(rule.elements)

    # Calculate sizes
    header_size = 4 * 4  # nq, tdim, nelements, pad (all int32)
    points_size = nq * tdim * 8  # doubles
    weights_size = nq * 8  # doubles
    elem_header_size = (
        nelements * 4 * 4
    )  # ndofs, nderivs, table_offset, pad per element

    # Calculate table sizes and offsets
    table_sizes = []
    table_offset = header_size + points_size + weights_size + elem_header_size
    for elem in rule.elements:
        size = elem.nderivs * nq * elem.ndofs * 8
        table_sizes.append(size)

    total_tables_size = sum(table_sizes)
    total_size = table_offset + total_tables_size

    # Allocate buffer
    buffer = np.zeros(total_size, dtype=np.uint8)

    # Write header
    header = np.array([nq, tdim, nelements, 0], dtype=np.int32)
    buffer[0:16] = header.view(np.uint8)

    # Write points
    offset = header_size
    points_flat = rule.points.flatten()
    buffer[offset : offset + points_size] = points_flat.view(np.uint8)
    offset += points_size

    # Write weights
    buffer[offset : offset + weights_size] = rule.weights.view(np.uint8)
    offset += weights_size

    # Write element headers and tables
    current_table_offset = table_offset
    for i, elem in enumerate(rule.elements):
        # Element header: ndofs, nderivs, table_offset, pad
        elem_header = np.array(
            [elem.ndofs, elem.nderivs, current_table_offset, 0], dtype=np.int32
        )
        buffer[offset : offset + 16] = elem_header.view(np.uint8)
        offset += 16

        # Write table at its offset
        table_flat = elem.table.flatten()
        table_bytes = table_flat.view(np.uint8)
        buffer[current_table_offset : current_table_offset + len(table_bytes)] = (
            table_bytes
        )
        current_table_offset += table_sizes[i]

    return PackedQuadratureData(buffer=buffer, _source_rule=rule)


class PerEntityQuadratureProvider:
    """Provider for per-entity quadrature data compatible with DOLFINx CustomDataFunction.

    This class manages quadrature rules for a set of entities (cells or facets)
    and provides a lookup function that can be passed to DOLFINx.

    There are two storage strategies:

    1. **By loop index** (default, most efficient):
       Quadrature rules are stored in a list indexed by loop position.
       The entity list order determines which rule goes with which entity.

    2. **By cell index** (when needed):
       Quadrature rules are stored in a dict keyed by cell index.
       Useful when the cell-to-rule mapping is sparse or dynamic.

    Example usage:
        # Create provider with rules for specific cells
        provider = PerEntityQuadratureProvider()

        # Add rules indexed by loop position (cells will be [1, 3, 7, 10])
        provider.add_rule_by_index(0, rule_for_cell_1)
        provider.add_rule_by_index(1, rule_for_cell_3)
        provider.add_rule_by_index(2, rule_for_cell_7)
        provider.add_rule_by_index(3, rule_for_cell_10)

        # OR add rules indexed by cell index
        provider.add_rule_by_cell(1, rule_for_cell_1)
        provider.add_rule_by_cell(3, rule_for_cell_3)
        # ...

        # Get the lookup function for DOLFINx
        lookup_fn = provider.get_lookup_function()
    """

    def __init__(self, use_cell_index: bool = False):
        """Initialize the provider.

        Args:
            use_cell_index: If True, use cell index for lookup (entity_info[1]).
                           If False (default), use loop index (entity_info[0]).
        """
        self.use_cell_index = use_cell_index
        self._rules_by_index: list[PackedQuadratureData | None] = []
        self._rules_by_cell: dict[int, PackedQuadratureData] = {}

    def add_rule_by_index(
        self, loop_index: int, rule: QuadratureRule | PackedQuadratureData
    ) -> None:
        """Add a quadrature rule at a specific loop index.

        Args:
            loop_index: The index in the assembly loop (0, 1, 2, ...).
            rule: The quadrature rule to use for this entity.
        """
        # Ensure list is long enough
        while len(self._rules_by_index) <= loop_index:
            self._rules_by_index.append(None)

        if isinstance(rule, QuadratureRule):
            rule = PackedQuadratureData.from_rule(rule)

        self._rules_by_index[loop_index] = rule

    def add_rule_by_cell(
        self, cell_index: int, rule: QuadratureRule | PackedQuadratureData
    ) -> None:
        """Add a quadrature rule for a specific cell.

        Args:
            cell_index: The mesh cell index.
            rule: The quadrature rule to use for this cell.
        """
        if isinstance(rule, QuadratureRule):
            rule = PackedQuadratureData.from_rule(rule)

        self._rules_by_cell[cell_index] = rule

    def add_rules_for_entities(
        self, rules: list[QuadratureRule | PackedQuadratureData]
    ) -> None:
        """Add quadrature rules for all entities in loop order.

        Args:
            rules: List of rules, one per entity in loop order.
        """
        for i, rule in enumerate(rules):
            self.add_rule_by_index(i, rule)

    def get_lookup_function(self) -> Callable[[list[int]], int | None]:
        """Get the lookup function for DOLFINx CustomDataFunction.

        Returns a function that takes entity_info and returns a pointer.

        The returned function signature matches:
            void*(std::span<const int32_t> entity_info)

        Where entity_info[0] = loop_index, entity_info[1] = cell_index.
        """
        if self.use_cell_index:
            return self._lookup_by_cell
        else:
            return self._lookup_by_index

    def _lookup_by_index(self, entity_info: list[int]) -> int | None:
        """Lookup by loop index (entity_info[0])."""
        if len(entity_info) < 1:
            return None

        loop_index = entity_info[0]
        if 0 <= loop_index < len(self._rules_by_index):
            packed = self._rules_by_index[loop_index]
            if packed is not None:
                return packed.data_ptr
        return None

    def _lookup_by_cell(self, entity_info: list[int]) -> int | None:
        """Lookup by cell index (entity_info[1])."""
        if len(entity_info) < 2:
            return None

        cell_index = entity_info[1]
        if cell_index in self._rules_by_cell:
            return self._rules_by_cell[cell_index].data_ptr
        return None


def tabulate_element_at_points(
    element,  # basix finite element
    points: npt.NDArray[np.float64],
    max_deriv_order: int = 1,
) -> ElementTable:
    """Tabulate a basix element at given quadrature points.

    Args:
        element: A basix finite element.
        points: Quadrature points, shape [nq, tdim].
        max_deriv_order: Maximum derivative order to compute.

    Returns:
        ElementTable with tabulated values.
    """
    # basix.tabulate returns [nderivs, nq, ndofs, ncomps]
    tables = element.tabulate(max_deriv_order, points)

    # For scalar elements, squeeze the last dimension
    if tables.shape[-1] == 1:
        tables = tables[..., 0]  # [nderivs, nq, ndofs]

    return ElementTable(
        ndofs=element.dim,
        nderivs=tables.shape[0],
        table=tables,
    )


def create_quadrature_rule(
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    elements: list | None = None,
    max_deriv_order: int = 1,
) -> QuadratureRule:
    """Create a QuadratureRule, optionally tabulating elements.

    Args:
        points: Quadrature points, shape [nq, tdim].
        weights: Quadrature weights, shape [nq].
        elements: Optional list of basix elements to tabulate.
        max_deriv_order: Maximum derivative order for tabulation.

    Returns:
        QuadratureRule with tabulated element values.
    """
    rule = QuadratureRule(
        points=np.asarray(points),
        weights=np.asarray(weights),
    )

    if elements:
        for elem in elements:
            rule.elements.append(
                tabulate_element_at_points(elem, rule.points, max_deriv_order)
            )

    return rule
