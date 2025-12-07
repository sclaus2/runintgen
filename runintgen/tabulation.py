"""Generic tabulation utilities for runtime kernels.

This module provides utilities to pre-compute and fill finite element
tables from the table_info metadata returned by compile_runtime_integrals.

The main entry point is `prepare_runtime_data()` which:
1. Takes a RuntimeKernelInfo (with table_info)
2. Takes quadrature points and weights
3. Creates all required element tables by looking up basix elements
4. Returns a RuntimeDataBuilder ready for use

Example:
    from runintgen import compile_runtime_integrals
    from runintgen.tabulation import prepare_runtime_data

    # Compile the form
    module = compile_runtime_integrals(form)
    kernel_info = module.kernels[0]

    # Get quadrature rule (e.g., from basix or custom)
    points, weights = basix.make_quadrature(basix.CellType.triangle, degree)

    # Scale weights by |detJ| for the cell
    weights_scaled = weights * abs(detJ)

    # Prepare runtime data
    builder, ffi = prepare_runtime_data(kernel_info, points, weights_scaled)
    data = builder.build()

    # Now call the kernel with data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import basix
import cffi

from .runtime_data import CFFI_DEF, ElementTableInfo, RuntimeDataBuilder

if TYPE_CHECKING:
    from .runtime_api import RuntimeKernelInfo


@dataclass
class PreparedTables:
    """Container for prepared runtime tables and CFFI objects.

    Attributes:
        builder: The RuntimeDataBuilder with quadrature and tables set.
        ffi: The CFFI FFI instance.
        table_arrays: List of numpy arrays for each element table (for reference).
    """

    builder: RuntimeDataBuilder
    ffi: cffi.FFI
    table_arrays: list[npt.NDArray[np.float64]]


def _get_basix_element_from_table_info(
    table_info: dict[str, Any],
) -> basix.finite_element.FiniteElement:
    """Reconstruct a basix element from table_info metadata.

    This function determines the basix element type from the table_info
    dictionary returned by compile_runtime_integrals.

    Args:
        table_info: Dictionary with element metadata including:
            - ndofs: Number of degrees of freedom
            - ncomps: Number of components
            - is_coordinate: Whether this is a coordinate element
            - Additional metadata that helps identify the element

    Returns:
        A basix finite element.

    Note:
        This is a heuristic approach. For full generality, the table_info
        should include the actual basix element family and degree.
    """
    # For now, we use heuristics based on ndofs and geometry
    # In a complete implementation, table_info should store the element family/degree
    ndofs = table_info["ndofs"]
    ncomps = table_info.get("ncomps", 1)
    is_coordinate = table_info.get("is_coordinate", False)

    # Common triangle elements
    # P1: 3 dofs, P2: 6 dofs, P3: 10 dofs
    # Coordinate elements are typically P1 (3 dofs in 2D)
    if is_coordinate:
        # Coordinate element is typically P1 Lagrange
        return basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

    # Heuristics for common Lagrange elements on triangles
    degree_map = {3: 1, 6: 2, 10: 3, 15: 4}

    if ncomps == 1 and ndofs in degree_map:
        degree = degree_map[ndofs]
        return basix.create_element(
            basix.ElementFamily.P, basix.CellType.triangle, degree
        )

    # For vector elements, ndofs = scalar_ndofs * ncomps
    if ncomps > 1:
        scalar_ndofs = ndofs // ncomps
        if scalar_ndofs in degree_map:
            degree = degree_map[scalar_ndofs]
            return basix.create_element(
                basix.ElementFamily.P, basix.CellType.triangle, degree
            )

    raise ValueError(
        f"Cannot determine basix element from table_info: ndofs={ndofs}, ncomps={ncomps}"
    )


def tabulate_from_table_info(
    table_info: dict[str, Any],
    points: npt.NDArray[np.float64],
    basix_element: basix.finite_element.FiniteElement | None = None,
) -> ElementTableInfo:
    """Tabulate a finite element at quadrature points based on table_info.

    Args:
        table_info: Element metadata from RuntimeKernelInfo.table_info.
        points: Quadrature points, shape [nq, tdim].
        basix_element: Optional pre-created basix element. If None, attempts
            to reconstruct from table_info.

    Returns:
        ElementTableInfo with tabulated values.
    """
    if basix_element is None:
        basix_element = _get_basix_element_from_table_info(table_info)

    max_deriv = table_info.get("max_derivative_order", 1)

    # basix.tabulate returns [nderivs, nq, ndofs, ncomps]
    tables = basix_element.tabulate(max_deriv, points)

    nderivs = tables.shape[0]
    ndofs = table_info["ndofs"]
    ncomps = table_info.get("ncomps", 1)

    if ncomps == 1:
        # Scalar element: squeeze component dimension
        # Result shape: [nderivs, nq, ndofs]
        table = tables[..., 0]
    else:
        # Vector/tensor element: keep all components
        # For blocked elements, we may need to reshape
        # basix gives [nderivs, nq, scalar_ndofs, ncomps]
        # We want [nderivs, nq, ndofs] where ndofs = scalar_ndofs * ncomps
        table = tables.reshape(nderivs, -1, ndofs)

    return ElementTableInfo(
        ndofs=ndofs,
        nderivs=nderivs,
        table=table.astype(np.float64),
    )


def prepare_runtime_data(
    kernel_info: "RuntimeKernelInfo",
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    basix_elements: list[basix.finite_element.FiniteElement] | None = None,
) -> PreparedTables:
    """Prepare all runtime data for a kernel from its table_info.

    This is the main utility function for setting up runtime data.
    It creates all element tables and packages them with the quadrature rule.

    Args:
        kernel_info: RuntimeKernelInfo from compile_runtime_integrals.
        points: Quadrature points, shape [nq, tdim].
        weights: Quadrature weights, shape [nq]. Should already include |detJ|.
        basix_elements: Optional list of basix elements corresponding to each
            entry in kernel_info.table_info. If None, elements are reconstructed
            from metadata (works for simple cases).

    Returns:
        PreparedTables with builder, ffi, and table arrays.

    Example:
        >>> module = compile_runtime_integrals(form)
        >>> kernel_info = module.kernels[0]
        >>> points, weights = basix.make_quadrature(basix.CellType.triangle, 4)
        >>> weights_scaled = weights * abs(detJ)
        >>> prepared = prepare_runtime_data(kernel_info, points, weights_scaled)
        >>> data = prepared.builder.build()
    """
    if kernel_info.table_info is None:
        raise ValueError("kernel_info.table_info is None")

    # Create FFI instance
    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)

    # Create builder
    builder = RuntimeDataBuilder(ffi)
    builder.set_quadrature(points, weights)

    # Tabulate each element
    table_arrays = []
    for i, tinfo in enumerate(kernel_info.table_info):
        elem = basix_elements[i] if basix_elements else None
        elem_table = tabulate_from_table_info(tinfo, points, elem)
        builder.add_element_table(elem_table)
        table_arrays.append(elem_table.table)

    return PreparedTables(
        builder=builder,
        ffi=ffi,
        table_arrays=table_arrays,
    )


def compute_detJ_triangle(coords: npt.NDArray[np.float64]) -> float:
    """Compute the Jacobian determinant for a P1 triangle.

    For a linear (P1) triangle, the Jacobian is constant over the cell.

    Args:
        coords: Vertex coordinates, shape [3, 2] or [3, 3].

    Returns:
        The Jacobian determinant (can be negative for inverted elements).
    """
    x0, y0 = coords[0, 0], coords[0, 1]
    x1, y1 = coords[1, 0], coords[1, 1]
    x2, y2 = coords[2, 0], coords[2, 1]

    # Jacobian: J = [[x1-x0, x2-x0], [y1-y0, y2-y0]]
    detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    return detJ


def prepare_runtime_data_for_cell(
    kernel_info: "RuntimeKernelInfo",
    coords: npt.NDArray[np.float64],
    quadrature_degree: int,
    cell_type: basix.CellType = basix.CellType.triangle,
    basix_elements: list[basix.finite_element.FiniteElement] | None = None,
) -> PreparedTables:
    """Prepare runtime data for a specific cell with automatic |detJ| scaling.

    Convenience function that:
    1. Creates a quadrature rule of the given degree
    2. Computes |detJ| for the cell
    3. Scales the weights by |detJ|
    4. Tabulates all elements

    Args:
        kernel_info: RuntimeKernelInfo from compile_runtime_integrals.
        coords: Cell vertex coordinates, shape [nverts, gdim].
        quadrature_degree: Degree of the quadrature rule.
        cell_type: basix cell type (default: triangle).
        basix_elements: Optional list of basix elements.

    Returns:
        PreparedTables ready for kernel invocation.

    Example:
        >>> module = compile_runtime_integrals(form)
        >>> kernel_info = module.kernels[0]
        >>> coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        >>> prepared = prepare_runtime_data_for_cell(kernel_info, coords, degree=4)
        >>> data = prepared.builder.build()
    """
    # Get quadrature rule
    points, weights = basix.make_quadrature(cell_type, quadrature_degree)

    # Compute |detJ| and scale weights
    if cell_type == basix.CellType.triangle:
        detJ = compute_detJ_triangle(coords)
    else:
        raise NotImplementedError(f"Cell type {cell_type} not yet supported")

    weights_scaled = weights * np.abs(detJ)

    return prepare_runtime_data(kernel_info, points, weights_scaled, basix_elements)
