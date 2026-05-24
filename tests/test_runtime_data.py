"""Tests for runtime quadrature custom-data containers."""

from __future__ import annotations

import cffi
import numpy as np
import pytest

from runintgen.runtime_data import (
    CFFI_DEF,
    QuadratureRules,
    RuntimeContextBuilder,
    RuntimeEntityMap,
    RuntimeQuadraturePayload,
    as_runtime_quadrature_payload,
    facet_runtime_quadrature_payload,
)


def _ptr(ffi: cffi.FFI, value) -> int:
    """Return an integer pointer value."""
    return int(ffi.cast("intptr_t", value))


def test_quadrature_rules_borrows_flat_arrays() -> None:
    """QuadratureRules should not copy flat quadrature storage."""
    points = np.arange(10, dtype=np.float64)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    offsets = np.array([0, 2, 3, 5], dtype=np.int32)
    parent_map = np.array([8, 10, 11], dtype=np.int32)

    rules = QuadratureRules(
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
    )

    assert rules.points is points
    assert rules.weights is weights
    assert rules.offsets is offsets
    assert rules.parent_map is parent_map


def test_runtime_payload_collects_mixed_form_entities() -> None:
    """Payload builder should derive the full loop map from form data."""
    points = np.arange(10, dtype=np.float64)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    offsets = np.array([0, 2, 3, 5], dtype=np.int32)
    parent_map = np.array([8, 10, 11], dtype=np.int32)
    standard_entities = np.array([1, 4, 5, 6], dtype=np.int32)
    rules = QuadratureRules(
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
    )
    payload = as_runtime_quadrature_payload([standard_entities, rules])

    assert payload.rules is rules
    np.testing.assert_array_equal(
        payload.entity_indices, [1, 4, 5, 6, 8, 10, 11]
    )
    np.testing.assert_array_equal(payload.is_cut, [0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(payload.rule_indices, [-1, -1, -1, -1, 0, 1, 2])


def test_runtime_entity_map_accepts_facet_rows() -> None:
    """Entity maps should preserve DOLFINx-shaped facet integration rows."""
    rows = np.array([[3, 0], [5, 2]], dtype=np.int32)
    is_cut = np.array([1, 1], dtype=np.uint8)
    rule_indices = np.array([0, 1], dtype=np.int32)

    entities = RuntimeEntityMap(
        entity_indices=rows,
        is_cut=is_cut,
        rule_indices=rule_indices,
    )

    assert entities.num_entities == 2
    assert entities.entity_indices is rows
    np.testing.assert_array_equal(entities.entity_indices, rows)


def test_runtime_payload_preserves_facet_rows() -> None:
    """RuntimeQuadraturePayload should allow row-shaped entity domains."""
    points = np.array([[0.2], [0.7]], dtype=np.float64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    offsets = np.array([0, 1, 2], dtype=np.int32)
    parent_map = np.array([4, 8], dtype=np.int32)
    rules = QuadratureRules(
        tdim=1,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
    )
    entities = RuntimeEntityMap(
        entity_indices=np.array([[3, 0], [5, 2]], dtype=np.int32),
        is_cut=np.ones(2, dtype=np.uint8),
        rule_indices=np.array([0, 1], dtype=np.int32),
    )

    payload = RuntimeQuadraturePayload(rules=rules, entities=entities)

    assert payload.num_entities == 2
    assert payload.entity_indices.shape == (2, 2)


def test_quadrature_rules_rejects_implicit_quadrature_copy() -> None:
    """The zero-copy constructor should reject list-backed storage."""
    weights = np.array([0.5], dtype=np.float64)
    offsets = np.array([0, 1], dtype=np.int32)

    with pytest.raises(TypeError, match="points must be a NumPy ndarray"):
        QuadratureRules(
            tdim=2,
            points=[[1.0 / 3.0, 1.0 / 3.0]],
            weights=weights,
            offsets=offsets,
        )


def test_runtime_context_builder_borrows_quadrature_pointers() -> None:
    """CFFI context builder should expose the provider's array pointers."""
    points = np.array([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    offsets = np.array([0, 2], dtype=np.int32)
    parent_map = np.array([8], dtype=np.int32)
    standard_entities = np.array([1, 4], dtype=np.int32)
    rules = QuadratureRules(
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
    )

    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)
    builder = RuntimeContextBuilder(ffi)
    ctx = builder.build_context([standard_entities, rules])

    assert ctx.quadrature.num_rules == 1
    assert ctx.quadrature.tdim == 2
    assert ctx.entities.num_entities == 3
    assert _ptr(ffi, ctx.quadrature.points) == points.ctypes.data
    assert _ptr(ffi, ctx.quadrature.weights) == weights.ctypes.data
    assert _ptr(ffi, ctx.quadrature.offsets) == offsets.ctypes.data
    assert _ptr(ffi, ctx.quadrature.parent_map) == parent_map.ctypes.data
    assert ctx.entities.is_cut[0] == 0
    assert ctx.entities.rule_indices[0] == -1
    assert ctx.entities.is_cut[2] == 1
    assert ctx.entities.rule_indices[2] == 0


def test_runtime_context_builder_counts_facet_rows() -> None:
    """CFFI context builder should count row-shaped entities by first axis."""
    points = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    offsets = np.array([0, 1, 2], dtype=np.int32)
    rules = QuadratureRules(
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=np.array([3, 5], dtype=np.int32),
    )
    entities = RuntimeEntityMap(
        entity_indices=np.array([[3, 0], [5, 2]], dtype=np.int32),
        is_cut=np.ones(2, dtype=np.uint8),
        rule_indices=np.array([0, 1], dtype=np.int32),
    )
    payload = RuntimeQuadraturePayload(rules=rules, entities=entities)

    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)
    builder = RuntimeContextBuilder(ffi)
    ctx = builder.build_context(payload)

    assert ctx.entities.num_entities == 2
    assert _ptr(ffi, ctx.entities.entity_indices) == entities.entity_indices.ctypes.data


def test_facet_runtime_payload_maps_triangle_facets_to_parent_cell() -> None:
    """The Basix-facing facet payload should carry parent-cell reference points."""
    basix = pytest.importorskip("basix")
    points = np.array([[0.25], [0.75], [0.5]], dtype=np.float64)
    weights = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    offsets = np.array([0, 2, 3], dtype=np.int32)
    parent_map = np.array([4, 8], dtype=np.int32)
    rows = np.array([[10, 0], [12, 1]], dtype=np.int32)
    rules = QuadratureRules(
        tdim=1,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
    )

    payload = facet_runtime_quadrature_payload(
        parent_cell_type=basix.CellType.triangle,
        quadrature=rules,
        entity_indices=rows,
    )

    assert payload.rules.tdim == 2
    assert payload.entity_indices.shape == (2, 2)
    np.testing.assert_array_equal(payload.rules.parent_map, rows[:, 0])
    np.testing.assert_array_equal(payload.rules.weights, weights)

    geometry = np.asarray(basix.geometry(basix.CellType.triangle), dtype=np.float64)
    topology = basix.topology(basix.CellType.triangle)
    facet0 = geometry[np.asarray(topology[1][0], dtype=np.int32)]
    facet1 = geometry[np.asarray(topology[1][1], dtype=np.int32)]
    expected = np.vstack(
        [
            (1.0 - 0.25) * facet0[0] + 0.25 * facet0[1],
            (1.0 - 0.75) * facet0[0] + 0.75 * facet0[1],
            (1.0 - 0.5) * facet1[0] + 0.5 * facet1[1],
        ]
    )
    np.testing.assert_allclose(payload.rules.points, expected)


def test_facet_runtime_payload_handles_interval_boundary_points() -> None:
    """Point facets of an interval map to parent interval reference points."""
    basix = pytest.importorskip("basix")
    rules = QuadratureRules(
        tdim=0,
        points=np.empty(0, dtype=np.float64),
        weights=np.array([1.0, 1.0], dtype=np.float64),
        offsets=np.array([0, 1, 2], dtype=np.int32),
        parent_map=np.array([3, 4], dtype=np.int32),
    )
    rows = np.array([[3, 0], [4, 1]], dtype=np.int32)

    payload = facet_runtime_quadrature_payload(
        parent_cell_type=basix.CellType.interval,
        quadrature=rules,
        entity_indices=rows,
    )

    assert payload.rules.tdim == 1
    np.testing.assert_allclose(payload.rules.points, [[0.0], [1.0]])


def test_facet_runtime_payload_does_not_force_lazy_physical_points() -> None:
    """Facet point remapping must not evaluate provider-specific lazy caches."""
    basix = pytest.importorskip("basix")

    class LazyPhysicalRules(QuadratureRules):
        def __getattribute__(self, name: str):
            if name == "physical_points" and object.__getattribute__(
                self, "__dict__"
            ).get("_raise_on_physical_points", False):
                raise RuntimeError("physical_points should stay lazy")
            return super().__getattribute__(name)

    rules = LazyPhysicalRules(
        tdim=1,
        points=np.array([[0.25], [0.75]], dtype=np.float64),
        weights=np.array([0.5, 0.5], dtype=np.float64),
        offsets=np.array([0, 1, 2], dtype=np.int32),
        parent_map=np.array([5, 6], dtype=np.int32),
    )
    rules._raise_on_physical_points = True

    payload = facet_runtime_quadrature_payload(
        parent_cell_type=basix.CellType.triangle,
        quadrature=rules,
        entity_indices=np.array([[3, 0], [4, 1]], dtype=np.int32),
    )

    assert payload.rules.physical_points is None


def test_basix_custom_data_accepts_facet_payload_rows() -> None:
    """The Basix-only CustomData backend should accept row-shaped entities."""
    basix = pytest.importorskip("basix")
    pytest.importorskip("runintgen._basix_runtime")
    from runintgen.basix_runtime import CustomData
    from runintgen.form_metadata import FormRuntimeMetadata

    rules = QuadratureRules(
        tdim=1,
        points=np.array([[0.25], [0.75]], dtype=np.float64),
        weights=np.array([0.5, 0.5], dtype=np.float64),
        offsets=np.array([0, 1, 2], dtype=np.int32),
        parent_map=np.array([5, 6], dtype=np.int32),
    )
    payload = facet_runtime_quadrature_payload(
        parent_cell_type=basix.CellType.triangle,
        quadrature=rules,
        entity_indices=np.array([[3, 0], [4, 1]], dtype=np.int32),
    )

    custom_data = CustomData(FormRuntimeMetadata(), payload)

    assert custom_data.num_rules == 2
    assert custom_data.num_entities == 2
