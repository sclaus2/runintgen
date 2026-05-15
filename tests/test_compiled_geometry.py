"""Tests for compiled geometry mapping helpers."""

from __future__ import annotations

import numpy as np
import pytest
from basix import CellType, LagrangeVariant

_basix_runtime = pytest.importorskip("runintgen._basix_runtime")


def test_compiled_per_entity_geometry_maps_and_scales_weights() -> None:
    """The C++ helper should map points and optionally scale weights."""
    geometry_x = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    geometry_dofmap = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    points = np.array(
        [[0.25, 0.25], [0.5, 0.25], [0.25, 0.5]],
        dtype=np.float64,
    )
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    offsets = np.array([0, 2, 3], dtype=np.int64)
    parent_map = np.array([0, 1], dtype=np.int32)

    physical_points, scaled_weights = _basix_runtime.map_per_entity_geometry(
        int(CellType.triangle),
        1,
        int(LagrangeVariant.equispaced),
        2,
        2,
        points,
        weights,
        offsets,
        parent_map,
        geometry_x,
        geometry_dofmap,
        True,
    )

    np.testing.assert_allclose(
        physical_points,
        [[0.5, 1.0, 1.5], [0.25, 0.25, 1.5]],
    )
    np.testing.assert_allclose(scaled_weights, [0.2, 0.4, 0.6])


def test_compiled_shared_geometry_repeats_reference_rule() -> None:
    """The C++ helper should repeat shared reference points per entity."""
    geometry_x = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    geometry_dofmap = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    points = np.array([[0.25, 0.25], [0.5, 0.25]], dtype=np.float64)
    weights = np.array([0.1, 0.2], dtype=np.float64)
    parent_map = np.array([0, 1], dtype=np.int32)

    physical_points, scaled_weights = _basix_runtime.map_shared_geometry(
        int(CellType.triangle),
        1,
        int(LagrangeVariant.equispaced),
        2,
        2,
        points,
        weights,
        parent_map,
        geometry_x,
        geometry_dofmap,
        True,
    )

    np.testing.assert_allclose(
        physical_points,
        [[0.5, 1.0, 1.5, 2.0], [0.25, 0.25, 1.25, 1.25]],
    )
    np.testing.assert_allclose(scaled_weights, [0.2, 0.4, 0.2, 0.4])
