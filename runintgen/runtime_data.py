"""Runtime data structures for runintgen kernels.

This module provides Python classes and helpers to create and manage
the runtime data structures (runintgen_data) that are passed
to generated kernels via the custom_data pointer.

Usage:
    # Create runtime data
    builder = RuntimeDataBuilder(ffi)
    builder.set_quadrature(quadrature_points, quadrature_weights)
    builder.add_element_table(ElementTableInfo(ndofs=3, nderivs=3, table=...))
    data_ptr = builder.build()

    # Use with DOLFINx Form via custom_data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cffi


@dataclass
class ElementTableInfo:
    """Information about a finite element's tabulated values.

    Attributes:
        ndofs: Number of degrees of freedom.
        nderivs: Number of derivative components in the table.
        table: Tabulated values with shape [nderivs, nq, ndofs] or flattened.
    """

    ndofs: int
    nderivs: int
    table: npt.NDArray[np.float64]


class RuntimeDataBuilder:
    """Builder for runintgen_data structures.

    This class helps construct the C structures needed by runtime kernels,
    handling memory management and layout for CFFI.

    Example:
        builder = RuntimeDataBuilder(ffi)
        builder.set_quadrature(points, weights)
        builder.add_element_table(ElementTableInfo(ndofs=3, nderivs=3, table=table))
        data = builder.build()
        # data can be passed to kernel via custom_data
    """

    def __init__(self, ffi: "cffi.FFI") -> None:
        """Initialize the builder.

        Args:
            ffi: CFFI FFI instance with runintgen struct definitions.
        """
        self.ffi = ffi
        self._points: npt.NDArray[np.float64] | None = None
        self._weights: npt.NDArray[np.float64] | None = None
        self._elements: list[ElementTableInfo] = []
        # Keep references to prevent garbage collection
        self._refs: list = []

    def set_quadrature(
        self,
        points: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> None:
        """Set the quadrature rule.

        Args:
            points: Quadrature points, shape [nq, tdim].
            weights: Quadrature weights, shape [nq].
        """
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if points.shape[0] != weights.shape[0]:
            raise ValueError(
                f"Number of points ({points.shape[0]}) must match "
                f"number of weights ({weights.shape[0]})"
            )
        self._points = points
        self._weights = weights

    def add_element_table(self, table_info: ElementTableInfo) -> int:
        """Add an element table.

        Args:
            table_info: ElementTableInfo with tabulated values.

        Returns:
            The index of the added element in data->elements.
        """
        self._elements.append(table_info)
        return len(self._elements) - 1

    def build(self) -> "cffi.CData":
        """Build the runintgen_data structure.

        Returns:
            A CFFI pointer to the runintgen_data structure.
        """
        if self._points is None or self._weights is None:
            raise ValueError("Quadrature not set. Call set_quadrature() first.")
        if not self._elements:
            raise ValueError("No element tables added. Call add_element_table() first.")

        ffi = self.ffi
        nq = self._points.shape[0]

        # Ensure arrays are contiguous
        points = np.ascontiguousarray(self._points.flatten(), dtype=np.float64)
        weights = np.ascontiguousarray(self._weights, dtype=np.float64)
        self._refs.extend([points, weights])

        # Allocate element array
        nelements = len(self._elements)
        c_elements = ffi.new(f"runintgen_element[{nelements}]")
        self._refs.append(c_elements)

        for j, elem in enumerate(self._elements):
            table = np.ascontiguousarray(elem.table.flatten(), dtype=np.float64)
            expected_size = elem.nderivs * nq * elem.ndofs
            if table.size != expected_size:
                raise ValueError(
                    f"Element {j} table size mismatch: got {table.size}, "
                    f"expected {expected_size} (nderivs={elem.nderivs}, nq={nq}, ndofs={elem.ndofs})"
                )
            self._refs.append(table)

            c_elements[j].ndofs = elem.ndofs
            c_elements[j].nderivs = elem.nderivs
            c_elements[j].table = ffi.cast("const double*", table.ctypes.data)

        # Allocate main structure
        c_data = ffi.new("runintgen_data*")
        self._refs.append(c_data)

        c_data.nq = nq
        c_data.points = ffi.cast("const double*", points.ctypes.data)
        c_data.weights = ffi.cast("const double*", weights.ctypes.data)
        c_data.nelements = nelements
        c_data.elements = c_elements

        return c_data


def to_intptr(ffi: "cffi.FFI", data: "cffi.CData") -> int:
    """Return an intptr_t representation of a runintgen_data*.

    This is useful for passing the pointer to DOLFINx.

    Args:
        ffi: CFFI FFI instance.
        data: A runintgen_data* pointer.

    Returns:
        Integer representation of the pointer.
    """
    return int(ffi.cast("intptr_t", data))


def tabulate_element(
    element,  # basix.finite_element
    points: npt.NDArray[np.float64],
    max_deriv_order: int = 1,
) -> ElementTableInfo:
    """Tabulate a basix element at given points.

    Args:
        element: A basix finite element.
        points: Quadrature points, shape [nq, tdim].
        max_deriv_order: Maximum derivative order to tabulate.

    Returns:
        ElementTableInfo with tabulated values.
    """
    # basix.tabulate returns [nderivs, nq, ndofs, ncomps]
    tables = element.tabulate(max_deriv_order, points)

    # For scalar elements, ncomps=1, so squeeze that dimension
    if tables.shape[-1] == 1:
        tables = tables[..., 0]  # [nderivs, nq, ndofs]

    return ElementTableInfo(
        ndofs=element.dim,
        nderivs=tables.shape[0],
        table=tables,
    )


def create_quadrature_config(
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    elements: list,  # List of basix elements
    max_deriv_order: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[ElementTableInfo]]:
    """Create quadrature and element tables for a runtime integral.

    Args:
        points: Quadrature points, shape [nq, tdim].
        weights: Quadrature weights, shape [nq].
        elements: List of basix finite elements to tabulate.
        max_deriv_order: Maximum derivative order.

    Returns:
        Tuple of (points, weights, element_tables).
    """
    element_tables = []
    for elem in elements:
        element_tables.append(tabulate_element(elem, points, max_deriv_order))

    return np.asarray(points), np.asarray(weights), element_tables


# CFFI definition string for the runtime structures
# This should be passed to ffi.cdef() before using RuntimeDataBuilder
CFFI_DEF = """
typedef struct {
  int ndofs;
  int nderivs;
  const double* table;
} runintgen_element;

typedef struct {
  int nq;
  const double* points;
  const double* weights;
  int nelements;
  const runintgen_element* elements;
} runintgen_data;
"""
