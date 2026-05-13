"""Tests for runtime quadrature custom-data containers."""

from __future__ import annotations

import cffi
import numpy as np
import pytest

from runintgen.runtime_data import (
    CFFI_DEF,
    RuntimeContextBuilder,
    RuntimeQuadratureRules,
    as_runtime_quadrature_payload,
)


def _ptr(ffi: cffi.FFI, value) -> int:
    """Return an integer pointer value."""
    return int(ffi.cast("intptr_t", value))


def test_runtime_quadrature_rules_borrows_flat_arrays() -> None:
    """RuntimeQuadratureRules should not copy flat quadrature storage."""
    points = np.arange(10, dtype=np.float64)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    offsets = np.array([0, 2, 3, 5], dtype=np.int64)
    parent_map = np.array([8, 10, 11], dtype=np.int32)

    rules = RuntimeQuadratureRules(
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
    offsets = np.array([0, 2, 3, 5], dtype=np.int64)
    parent_map = np.array([8, 10, 11], dtype=np.int32)
    standard_entities = np.array([1, 4, 5, 6], dtype=np.int32)
    rules = RuntimeQuadratureRules(
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


def test_runtime_quadrature_rules_rejects_implicit_quadrature_copy() -> None:
    """The zero-copy constructor should reject list-backed storage."""
    weights = np.array([0.5], dtype=np.float64)
    offsets = np.array([0, 1], dtype=np.int64)

    with pytest.raises(TypeError, match="points must be a NumPy ndarray"):
        RuntimeQuadratureRules(
            tdim=2,
            points=[[1.0 / 3.0, 1.0 / 3.0]],
            weights=weights,
            offsets=offsets,
        )


def test_runtime_context_builder_borrows_quadrature_pointers() -> None:
    """CFFI context builder should expose the provider's array pointers."""
    points = np.array([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    offsets = np.array([0, 2], dtype=np.int64)
    parent_map = np.array([8], dtype=np.int32)
    standard_entities = np.array([1, 4], dtype=np.int32)
    rules = RuntimeQuadratureRules(
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
