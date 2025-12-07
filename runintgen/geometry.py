"""Geometry and quadrature mapping utilities using basix.

This module provides implementations for mapping quadrature rules between
reference cells, physical subcells, and parent reference cells.
It uses basix for proper geometry handling of all cell types:
  - interval (1D)
  - triangle, quadrilateral (2D)
  - tetrahedron, hexahedron, prism, pyramid (3D)
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import basix
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


class CellType(IntEnum):
    """Cell types matching basix.CellType."""

    point = 0
    interval = 1
    triangle = 2
    tetrahedron = 3
    quadrilateral = 4
    hexahedron = 5
    prism = 6
    pyramid = 7


def _basix_cell_type(cell_type: CellType | int) -> basix.CellType:
    """Convert to basix CellType."""
    cell_map = {
        CellType.point: basix.CellType.point,
        CellType.interval: basix.CellType.interval,
        CellType.triangle: basix.CellType.triangle,
        CellType.tetrahedron: basix.CellType.tetrahedron,
        CellType.quadrilateral: basix.CellType.quadrilateral,
        CellType.hexahedron: basix.CellType.hexahedron,
        CellType.prism: basix.CellType.prism,
        CellType.pyramid: basix.CellType.pyramid,
    }
    return cell_map[CellType(cell_type)]


def cell_tdim(cell_type: CellType | int) -> int:
    """Get topological dimension of a cell type."""
    tdim_map = {
        CellType.point: 0,
        CellType.interval: 1,
        CellType.triangle: 2,
        CellType.quadrilateral: 2,
        CellType.tetrahedron: 3,
        CellType.hexahedron: 3,
        CellType.prism: 3,
        CellType.pyramid: 3,
    }
    return tdim_map[CellType(cell_type)]


def cell_num_vertices(cell_type: CellType | int) -> int:
    """Get number of vertices of a cell type."""
    nv_map = {
        CellType.point: 1,
        CellType.interval: 2,
        CellType.triangle: 3,
        CellType.quadrilateral: 4,
        CellType.tetrahedron: 4,
        CellType.hexahedron: 8,
        CellType.prism: 6,
        CellType.pyramid: 5,
    }
    return nv_map[CellType(cell_type)]


def reference_cell_vertices(
    cell_type: CellType | int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.floating]:
    """Get reference cell vertex coordinates from basix.

    Args:
        cell_type: The cell type
        dtype: Data type for coordinates

    Returns:
        Array of vertex coordinates, shape (num_vertices, tdim)
    """
    bcell = _basix_cell_type(cell_type)
    geom = basix.geometry(bcell)
    return np.asarray(geom, dtype=dtype)


def create_coordinate_element(
    cell_type: CellType | int, degree: int = 1, dtype: npt.DTypeLike = np.float64
) -> basix.finite_element.FiniteElement:
    """Create a basix Lagrange coordinate element for geometry mapping.

    Args:
        cell_type: The cell type
        degree: Polynomial degree (1 for linear/affine elements)
        dtype: Scalar type

    Returns:
        Basix finite element for coordinate mapping
    """
    bcell = _basix_cell_type(cell_type)
    return basix.create_element(
        basix.ElementFamily.P,
        bcell,
        degree,
        basix.LagrangeVariant.equispaced,
        dtype=dtype,
    )


def compute_jacobian(
    dphi: npt.NDArray[np.floating],
    coord_dofs: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute Jacobian matrix from basis function derivatives and coordinates.

    J[i,j] = sum_k coord_dofs[k,i] * dphi[j,k]

    This computes J = coord_dofs.T @ dphi.T = (dphi @ coord_dofs).T

    Args:
        dphi: Derivatives of coordinate basis functions at a point,
              shape (tdim, num_dofs)
        coord_dofs: Physical coordinates of cell vertices/dofs,
                    shape (num_dofs, gdim)

    Returns:
        Jacobian matrix, shape (gdim, tdim)
    """
    # J = coord_dofs.T @ dphi.T
    return coord_dofs.T @ dphi.T


def compute_jacobian_determinant(
    J: npt.NDArray[np.floating],
) -> np.floating:
    """Compute determinant of Jacobian (or pseudo-determinant for non-square).

    For square Jacobians: returns det(J)
    For non-square (gdim > tdim): returns sqrt(det(J.T @ J))

    Args:
        J: Jacobian matrix, shape (gdim, tdim)

    Returns:
        Determinant or pseudo-determinant
    """
    gdim, tdim = J.shape
    if gdim == tdim:
        return np.linalg.det(J)
    else:
        # Non-square: det = sqrt(det(J^T J))
        JTJ = J.T @ J
        return np.sqrt(np.linalg.det(JTJ))


def compute_jacobian_inverse(
    J: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute inverse (or pseudo-inverse) of Jacobian.

    For square Jacobians: returns J^{-1}
    For non-square (gdim > tdim): returns (J^T J)^{-1} J^T (left pseudo-inverse)

    Args:
        J: Jacobian matrix, shape (gdim, tdim)

    Returns:
        Inverse Jacobian, shape (tdim, gdim)
    """
    gdim, tdim = J.shape
    if gdim == tdim:
        return np.linalg.inv(J)
    else:
        # Pseudo-inverse: K = (J^T J)^{-1} J^T
        return np.linalg.pinv(J)


def push_forward(
    X: npt.NDArray[np.floating],
    coord_dofs: npt.NDArray[np.floating],
    cell_type: CellType | int,
) -> npt.NDArray[np.floating]:
    """Map points from reference cell to physical coordinates.

    x = sum_i coord_dofs[i] * phi_i(X)

    Args:
        X: Reference coordinates, shape (num_points, tdim)
        coord_dofs: Physical coordinates of cell dofs, shape (num_dofs, gdim)
        cell_type: The cell type

    Returns:
        Physical coordinates, shape (num_points, gdim)
    """
    # Create coordinate element and tabulate basis
    celem = create_coordinate_element(cell_type, degree=1, dtype=X.dtype)

    # Tabulate basis functions (no derivatives)
    tab = celem.tabulate(0, X.reshape(-1))  # shape: (1, num_points, num_dofs, 1)
    phi = tab[0, :, :, 0]  # shape: (num_points, num_dofs)

    # x = phi @ coord_dofs
    return phi @ coord_dofs


def pull_back_affine(
    x: npt.NDArray[np.floating],
    coord_dofs: npt.NDArray[np.floating],
    cell_type: CellType | int,
) -> npt.NDArray[np.floating]:
    """Map points from physical coordinates to reference cell (affine map).

    For affine cells: X = K @ (x - x0) where K = J^{-1}

    Args:
        x: Physical coordinates, shape (num_points, gdim)
        coord_dofs: Physical coordinates of cell dofs, shape (num_dofs, gdim)
        cell_type: The cell type

    Returns:
        Reference coordinates, shape (num_points, tdim)
    """
    # Get coordinate element and tabulate derivatives at origin
    celem = create_coordinate_element(cell_type, degree=1, dtype=x.dtype)
    tdim = cell_tdim(cell_type)

    # Tabulate at reference origin to get Jacobian
    origin = np.zeros((1, tdim), dtype=x.dtype)
    tab = celem.tabulate(1, origin.reshape(-1))

    # Extract derivatives: dphi[d] has shape (1, num_dofs, 1)
    # We want dphi shape (tdim, num_dofs)
    dphi = np.zeros((tdim, tab.shape[2]), dtype=x.dtype)
    for d in range(tdim):
        dphi[d, :] = tab[1 + d, 0, :, 0]

    # Compute Jacobian and its inverse
    J = compute_jacobian(dphi, coord_dofs)
    K = compute_jacobian_inverse(J)

    # x0 = physical coordinate of reference origin
    phi0 = tab[0, 0, :, 0]  # basis at origin
    x0 = phi0 @ coord_dofs

    # X = K @ (x - x0)
    return (x - x0) @ K.T


def map_quadrature_to_physical(
    ref_points: npt.NDArray[np.floating],
    ref_weights: npt.NDArray[np.floating],
    coord_dofs: npt.NDArray[np.floating],
    cell_type: CellType | int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Map quadrature rule from reference to physical cell.

    Args:
        ref_points: Reference quadrature points, shape (num_points, tdim)
        ref_weights: Reference quadrature weights, shape (num_points,)
        coord_dofs: Physical coordinates of cell dofs, shape (num_dofs, gdim)
        cell_type: The cell type

    Returns:
        Tuple of (physical_points, scaled_weights)
    """
    # Create coordinate element
    celem = create_coordinate_element(cell_type, degree=1, dtype=ref_points.dtype)
    tdim = cell_tdim(cell_type)
    num_points = ref_points.shape[0]

    # Tabulate with first derivatives
    tab = celem.tabulate(1, ref_points.reshape(-1))

    # Compute physical points: x = phi @ coord_dofs
    phi = tab[0, :, :, 0]  # (num_points, num_dofs)
    phys_points = phi @ coord_dofs

    # Compute Jacobian determinant at each point for weight scaling
    scaled_weights = np.zeros(num_points, dtype=ref_points.dtype)
    for q in range(num_points):
        dphi = np.zeros((tdim, tab.shape[2]), dtype=ref_points.dtype)
        for d in range(tdim):
            dphi[d, :] = tab[1 + d, q, :, 0]
        J = compute_jacobian(dphi, coord_dofs)
        detJ = compute_jacobian_determinant(J)
        scaled_weights[q] = ref_weights[q] * np.abs(detJ)

    return phys_points, scaled_weights


def map_quadrature_to_subcell(
    points: npt.NDArray[np.floating],
    subcell_vertices: npt.NDArray[np.floating],
    cell_type: CellType | int | None = None,
) -> npt.NDArray[np.floating]:
    """Map quadrature points from reference cell to physical subcell coordinates.

    Args:
        points: Reference quadrature points, shape (num_points, tdim)
        subcell_vertices: Vertices of the subcell in physical coordinates,
                          shape (num_vertices, gdim)
        cell_type: The cell type (inferred from num_vertices if None)

    Returns:
        Physical quadrature points, shape (num_points, gdim)
    """
    if cell_type is None:
        # Infer cell type from number of vertices
        nv = subcell_vertices.shape[0]
        tdim = points.shape[1]
        cell_type = _infer_cell_type(nv, tdim)

    return push_forward(points, subcell_vertices, cell_type)


def map_subcell_to_parent_reference(
    points: npt.NDArray[np.floating],
    parent_vertices: npt.NDArray[np.floating],
    cell_type: CellType | int | None = None,
) -> npt.NDArray[np.floating]:
    """Map physical subcell points to parent reference cell coordinates.

    Args:
        points: Physical quadrature points, shape (num_points, gdim)
        parent_vertices: Vertices of the parent cell in physical coordinates,
                         shape (num_vertices, gdim)
        cell_type: The parent cell type (inferred from num_vertices if None)

    Returns:
        Reference quadrature points in parent cell, shape (num_points, tdim)
    """
    if cell_type is None:
        nv = parent_vertices.shape[0]
        gdim = parent_vertices.shape[1]
        cell_type = _infer_cell_type(nv, gdim)

    return pull_back_affine(points, parent_vertices, cell_type)


def scale_weights_by_jacobian(
    weights: npt.NDArray[np.floating],
    subcell_vertices: npt.NDArray[np.floating],
    cell_type: CellType | int | None = None,
) -> npt.NDArray[np.floating]:
    """Scale quadrature weights by the Jacobian determinant of the subcell.

    Args:
        weights: Reference quadrature weights, shape (num_points,)
        subcell_vertices: Vertices of the subcell in physical coordinates,
                          shape (num_vertices, gdim)
        cell_type: The cell type (inferred if None)

    Returns:
        Scaled quadrature weights, shape (num_points,)
    """
    if cell_type is None:
        nv = subcell_vertices.shape[0]
        gdim = subcell_vertices.shape[1]
        cell_type = _infer_cell_type(nv, gdim)

    # Get Jacobian determinant
    celem = create_coordinate_element(cell_type, degree=1, dtype=weights.dtype)
    tdim = cell_tdim(cell_type)

    # Tabulate derivatives at origin (constant for affine elements)
    origin = np.zeros((1, tdim), dtype=weights.dtype)
    tab = celem.tabulate(1, origin.reshape(-1))

    dphi = np.zeros((tdim, tab.shape[2]), dtype=weights.dtype)
    for d in range(tdim):
        dphi[d, :] = tab[1 + d, 0, :, 0]

    J = compute_jacobian(dphi, subcell_vertices)
    detJ = compute_jacobian_determinant(J)

    return weights * np.abs(detJ)


def generate_subcell_quadrature(
    ref_points: npt.NDArray[np.floating],
    ref_weights: npt.NDArray[np.floating],
    subcell_vertices_list: Sequence[npt.NDArray[np.floating]],
    parent_vertices: npt.NDArray[np.floating],
    subcell_type: CellType | int | None = None,
    parent_type: CellType | int | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate mapped quadrature rules for multiple subcells.

    This function takes a reference quadrature rule and a list of subcell
    vertex arrays, and produces a combined quadrature rule in the parent
    reference cell coordinates.

    Args:
        ref_points: Reference quadrature points, shape (num_points, tdim)
        ref_weights: Reference quadrature weights, shape (num_points,)
        subcell_vertices_list: List of subcell vertex arrays, each shape
                               (num_vertices, gdim)
        parent_vertices: Vertices of the parent cell in physical coordinates,
                         shape (num_vertices, gdim)
        subcell_type: Cell type of subcells (inferred if None)
        parent_type: Cell type of parent (inferred if None)

    Returns:
        Tuple of (mapped_points, scaled_weights) where:
        - mapped_points: shape (total_points, tdim) in parent reference coords
        - scaled_weights: shape (total_points,) scaled by subcell Jacobians
    """
    all_points = []
    all_weights = []

    # Infer types if needed
    if subcell_type is None and len(subcell_vertices_list) > 0:
        nv = subcell_vertices_list[0].shape[0]
        tdim = ref_points.shape[1]
        subcell_type = _infer_cell_type(nv, tdim)

    if parent_type is None:
        nv = parent_vertices.shape[0]
        gdim = parent_vertices.shape[1]
        parent_type = _infer_cell_type(nv, gdim)

    for subcell_vertices in subcell_vertices_list:
        # Scale weights by subcell Jacobian
        scaled_wts = scale_weights_by_jacobian(
            ref_weights, subcell_vertices, subcell_type
        )

        # Map reference -> physical subcell -> parent reference
        phys_points = map_quadrature_to_subcell(
            ref_points, subcell_vertices, subcell_type
        )
        parent_ref_points = map_subcell_to_parent_reference(
            phys_points, parent_vertices, parent_type
        )

        all_points.append(parent_ref_points)
        all_weights.append(scaled_wts)

    if len(all_points) == 0:
        tdim = cell_tdim(parent_type) if parent_type else ref_points.shape[1]
        return (
            np.zeros((0, tdim), dtype=ref_points.dtype),
            np.zeros((0,), dtype=ref_weights.dtype),
        )

    return np.vstack(all_points), np.concatenate(all_weights)


def _infer_cell_type(num_vertices: int, dim: int) -> CellType:
    """Infer cell type from number of vertices and dimension.

    Args:
        num_vertices: Number of vertices
        dim: Dimension (topological or geometric)

    Returns:
        Inferred cell type
    """
    if dim == 1:
        if num_vertices == 2:
            return CellType.interval
    elif dim == 2:
        if num_vertices == 3:
            return CellType.triangle
        elif num_vertices == 4:
            return CellType.quadrilateral
    elif dim == 3:
        if num_vertices == 4:
            return CellType.tetrahedron
        elif num_vertices == 5:
            return CellType.pyramid
        elif num_vertices == 6:
            return CellType.prism
        elif num_vertices == 8:
            return CellType.hexahedron
    raise ValueError(
        f"Cannot infer cell type from {num_vertices} vertices in dim {dim}"
    )


__all__ = [
    "CellType",
    "cell_num_vertices",
    "cell_tdim",
    "compute_jacobian",
    "compute_jacobian_determinant",
    "compute_jacobian_inverse",
    "create_coordinate_element",
    "generate_subcell_quadrature",
    "map_quadrature_to_physical",
    "map_quadrature_to_subcell",
    "map_subcell_to_parent_reference",
    "pull_back_affine",
    "push_forward",
    "reference_cell_vertices",
    "scale_weights_by_jacobian",
]
