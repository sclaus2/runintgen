"""Basix-only runtime custom-data owner.

This module provides the Python-facing API for building the ``void*`` context
used by generated runtime kernels without depending on DOLFINx. The compiled
``_basix_runtime`` extension owns the C++ Basix elements, quadrature storage,
and table scratch used by the generated C kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .form_metadata import FormRuntimeMetadata, export_metadata_for_cpp
from .runtime_data import RuntimeQuadratureRule


def _load_extension() -> Any:
    """Import the compiled Basix runtime extension with a useful error."""
    try:
        from . import _basix_runtime
    except ImportError as exc:  # pragma: no cover - depends on build mode
        raise ImportError(
            "runintgen._basix_runtime is not available. Install runintgen with "
            "the Basix runtime extension, for example `pip install -e .` in an "
            "environment with fenics-basix and a C++20 compiler."
        ) from exc
    return _basix_runtime


def _form_metadata(metadata_or_module: Any) -> FormRuntimeMetadata:
    """Return form metadata from a metadata object or RunintModule-like object."""
    metadata = getattr(metadata_or_module, "form_metadata", metadata_or_module)
    if metadata is None:
        raise ValueError("Runtime form metadata is required to build custom_data.")
    if not isinstance(metadata, FormRuntimeMetadata):
        raise TypeError(
            "Expected FormRuntimeMetadata or an object with a form_metadata "
            f"attribute, got {type(metadata_or_module)!r}."
        )
    return metadata


def element_specs_from_metadata(metadata_or_module: Any) -> list[dict[str, Any]]:
    """Return Basix element constructor specs for a runtime module/metadata."""
    metadata = _form_metadata(metadata_or_module)
    exported = export_metadata_for_cpp(metadata)
    return [dict(item["element_spec"]) for item in exported["unique_elements"]]


def _normalise_rule(rule: Any) -> dict[str, npt.NDArray[np.float64]]:
    """Return contiguous quadrature arrays in the shape expected by C++."""
    if isinstance(rule, RuntimeQuadratureRule):
        points = rule.points
        weights = rule.weights
    elif isinstance(rule, dict):
        points = rule["points"]
        weights = rule["weights"]
    else:
        points = getattr(rule, "points")
        weights = getattr(rule, "weights")

    points_array = np.ascontiguousarray(points, dtype=np.float64)
    weights_array = np.ascontiguousarray(weights, dtype=np.float64)
    if points_array.ndim != 2:
        raise ValueError("Runtime quadrature points must have shape (nq, tdim).")
    if weights_array.ndim != 1:
        weights_array = np.ravel(weights_array)
    if points_array.shape[0] != weights_array.shape[0]:
        raise ValueError("Runtime quadrature points and weights disagree on nq.")
    return {"points": points_array, "weights": weights_array}


def _normalise_rules(quadrature: Any) -> list[dict[str, npt.NDArray[np.float64]]]:
    """Normalise one or more quadrature rules."""
    if isinstance(quadrature, RuntimeQuadratureRule):
        return [_normalise_rule(quadrature)]
    if isinstance(quadrature, (list, tuple)):
        return [_normalise_rule(rule) for rule in quadrature]
    return [_normalise_rule(quadrature)]


def _normalise_is_cut(
    is_cut: Any | None,
    num_rules: int,
) -> npt.NDArray[np.uint8] | None:
    """Normalise optional runtime/standard branch flags."""
    if is_cut is None:
        return None
    flags = np.ascontiguousarray(is_cut, dtype=np.uint8)
    if flags.ndim != 1:
        flags = np.ravel(flags)
    if flags.size != num_rules:
        raise ValueError("is_cut must have one entry per quadrature rule.")
    return flags


class CustomData:
    """Basix-only owner for generated-kernel ``custom_data``.

    Keep this object alive for as long as any generated kernel may use the
    pointer returned by :attr:`ptr`.
    """

    def __init__(
        self,
        metadata_or_module: Any,
        quadrature: Any,
        *,
        is_cut: Any | None = None,
    ) -> None:
        """Build a Basix-backed runtime context.

        Args:
            metadata_or_module: ``FormRuntimeMetadata`` or a ``RunintModule``.
            quadrature: One ``RuntimeQuadratureRule``-like object, or a list of
                them. Weights must already include measure scaling.
            is_cut: Optional mixed-integral branch flags, one per rule.
        """
        extension = _load_extension()
        self._element_specs = element_specs_from_metadata(metadata_or_module)
        self._rules = _normalise_rules(quadrature)
        self._is_cut = _normalise_is_cut(is_cut, len(self._rules))
        self._owner = extension.CustomData(
            self._element_specs,
            self._rules,
            self._is_cut,
        )

    @property
    def ptr(self) -> int:
        """Integer representation of the ``runintgen_context*`` pointer."""
        return int(self._owner.ptr)

    @property
    def num_elements(self) -> int:
        """Number of registered form elements."""
        return int(self._owner.num_elements)

    @property
    def num_rules(self) -> int:
        """Number of runtime quadrature rules."""
        return int(self._owner.num_rules)

    def __int__(self) -> int:
        """Return :attr:`ptr` for APIs that accept integer pointers."""
        return self.ptr


def create_custom_data(
    metadata_or_module: Any,
    quadrature: Any,
    *,
    is_cut: Any | None = None,
) -> CustomData:
    """Build a :class:`CustomData` owner."""
    return CustomData(metadata_or_module, quadrature=quadrature, is_cut=is_cut)


__all__ = [
    "CustomData",
    "create_custom_data",
    "element_specs_from_metadata",
]
