"""Tests for DOLFINx quadrature physical-point helpers."""

from __future__ import annotations

import numpy as np
import pytest

from runintgen.dolfinx import compute_physical_points
from runintgen.dolfinx import utils as dolfinx_utils
from runintgen.runtime_data import QuadratureRules


class _FakeCoordinateMap:
    """Affine triangle coordinate map with a DOLFINx-like API."""

    def push_forward(
        self,
        X: np.ndarray,
        cell_geometry: np.ndarray,
    ) -> np.ndarray:
        """Map reference triangle points to physical points."""
        return (
            cell_geometry[0]
            + X[:, [0]] * (cell_geometry[1] - cell_geometry[0])
            + X[:, [1]] * (cell_geometry[2] - cell_geometry[0])
        )


class _FakeGeometry:
    """Small DOLFINx-like geometry object."""

    dim = 2

    def __init__(self) -> None:
        """Initialise two affine triangle cells."""
        self.x = np.array(
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
        self.dofmap = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    def cmap(self) -> _FakeCoordinateMap:
        """Return the coordinate map."""
        return _FakeCoordinateMap()


class _FakeMesh:
    """Small DOLFINx-like mesh object."""

    def __init__(self) -> None:
        """Initialise mesh geometry."""
        self.geometry = _FakeGeometry()


def test_compute_physical_points_for_per_entity_rules() -> None:
    """Per-entity rules should map ragged slices and preserve rule identity."""
    points = np.array(
        [[0.25, 0.25], [0.5, 0.25], [0.25, 0.5]],
        dtype=np.float64,
    )
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    offsets = np.array([0, 2, 3], dtype=np.int64)
    parent_map = np.array([0, 1], dtype=np.int32)
    rules = QuadratureRules(
        kind="per_entity",
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
        rule_id="per-entity-rule",
    )

    mapped = compute_physical_points(_FakeMesh(), rules)

    assert mapped is not rules
    assert mapped.rule_id == rules.rule_id
    assert mapped.points is points
    assert mapped.weights is weights
    assert mapped.offsets is offsets
    assert mapped.parent_map is parent_map
    assert mapped.gdim == 2
    np.testing.assert_allclose(
        mapped.physical_points,
        [[0.5, 1.0, 1.5], [0.25, 0.25, 1.5]],
    )


def test_compute_physical_points_for_shared_rules() -> None:
    """Shared rules should reuse one reference rule for every parent cell."""
    points = np.array([[0.25, 0.25], [0.5, 0.25]], dtype=np.float64)
    weights = np.array([0.1, 0.2], dtype=np.float64)
    parent_map = np.array([0, 1], dtype=np.int32)
    rules = QuadratureRules(
        kind="shared",
        tdim=2,
        points=points,
        weights=weights,
        parent_map=parent_map,
        rule_id="shared-rule",
    )

    mapped = compute_physical_points(_FakeMesh(), rules)

    assert mapped.rule_id == rules.rule_id
    assert mapped.kind == "shared"
    assert mapped.total_points == 4
    np.testing.assert_allclose(
        mapped.physical_points,
        [[0.5, 1.0, 1.5, 2.0], [0.25, 0.25, 1.25, 1.25]],
    )


def test_compute_physical_points_requires_parent_map() -> None:
    """The DOLFINx helper needs parent cells for every rule/entity."""
    rules = QuadratureRules(
        kind="per_entity",
        tdim=2,
        points=np.array([[0.25, 0.25]], dtype=np.float64),
        weights=np.array([0.1], dtype=np.float64),
        offsets=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="parent_map"):
        compute_physical_points(_FakeMesh(), rules)


def test_compute_physical_points_allows_empty_local_rules() -> None:
    """Parallel ranks may have no local entities for a rule set."""
    rules = QuadratureRules(
        kind="per_entity",
        tdim=2,
        points=np.empty((0, 2), dtype=np.float64),
        weights=np.empty(0, dtype=np.float64),
        offsets=np.array([0], dtype=np.int64),
        parent_map=np.empty(0, dtype=np.int32),
    )

    mapped = compute_physical_points(_FakeMesh(), rules)

    assert mapped.physical_points.shape == (2, 0)


def test_dolfinx_quadrature_function_factory_tags_function(monkeypatch) -> None:
    """The DOLFINx factory should attach runintgen metadata and set_values."""

    class FakeSpace:
        mesh = object()

        def ufl_domain(self):
            return None

        def ufl_element(self):
            return None

    class FakeFunction:
        def __init__(self, V, *, name=None, dtype=None) -> None:
            self.function_space = V
            self.name = "f" if name is None else name
            self.dtype = dtype
            self.ufl_shape = ()

    class FakeFem:
        Function = FakeFunction

    monkeypatch.setattr(
        dolfinx_utils,
        "_require_dolfinx",
        lambda: (FakeFem, object()),
    )

    alpha = dolfinx_utils.QuadratureFunction(FakeSpace(), name="alpha")
    rules = QuadratureRules(
        kind="per_entity",
        tdim=2,
        points=np.array([[0.25, 0.25]], dtype=np.float64),
        weights=np.array([0.1], dtype=np.float64),
        offsets=np.array([0, 1], dtype=np.int64),
        rule_id="rule",
    )
    values = np.array([2.0], dtype=np.float64)

    alpha.set_values(rules, values)

    assert alpha._runintgen_quadrature_function.name == "alpha"
    assert alpha._runintgen_values["rule"] is values
    assert alpha.is_cellwise_constant() is False


def test_background_dolfinx_evaluator_uses_parent_cells() -> None:
    """Background DOLFINx Function evaluation should receive one cell per point."""

    class FakeSpace:
        mesh = object()

    class FakeFunction:
        function_space = FakeSpace()

        def eval(self, x, cells):
            np.testing.assert_allclose(x, [[0.2, 0.4, 0.0], [0.6, 0.8, 0.0]])
            np.testing.assert_array_equal(cells, [7, 8])
            return np.array([[3.0], [4.0]], dtype=np.float64)

    class FakeInfo:
        terminal = FakeFunction()
        label = "alpha"
        value_size = 1

    rules = QuadratureRules(
        kind="per_entity",
        tdim=2,
        gdim=2,
        points=np.array([[0.25, 0.25], [0.5, 0.25]], dtype=np.float64),
        weights=np.array([0.1, 0.2], dtype=np.float64),
        offsets=np.array([0, 1, 2], dtype=np.int64),
        parent_map=np.array([7, 8], dtype=np.int32),
        physical_points=np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float64),
    )

    values = dolfinx_utils._evaluate_background_quadrature_function(
        FakeInfo(),
        rules,
    )

    np.testing.assert_allclose(values, [[3.0], [4.0]])
