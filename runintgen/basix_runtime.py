"""Basix-only runtime custom-data owner.

This module provides the Python-facing API for building the ``void*`` context
used by generated runtime kernels without depending on DOLFINx. The compiled
``_basix_runtime`` extension owns the C++ Basix elements and table scratch used
by generated C kernels. Quadrature buffers are borrowed from Python/provider
objects and must outlive kernel calls.
"""

from __future__ import annotations

from typing import Any

from .form_metadata import FormRuntimeMetadata, export_metadata_for_cpp
from .runtime_data import (
    RuntimeQuadraturePayload,
    as_runtime_quadrature_payload,
    build_quadrature_function_value_set,
)


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


def _quadrature_function_infos(metadata_or_module: Any) -> list[Any]:
    """Return generated quadrature-function infos from a module-like object."""
    return list(getattr(metadata_or_module, "quadrature_functions", []) or [])


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
        quadrature_function_evaluator: Any | None = None,
    ) -> None:
        """Build a Basix-backed runtime context.

        Args:
            metadata_or_module: ``FormRuntimeMetadata`` or a ``RunintModule``.
            quadrature: A ``QuadratureRules`` flat-buffer object, one
                ``RuntimeQuadratureRule``-like object, a list of such rules, or
                mixed form subdomain data such as
                ``[standard_entities, per_entity_rules]``. Weights must already
                include measure scaling. Flat per-entity rules are borrowed.
            is_cut: Optional legacy branch flags, one per packed rule.
            quadrature_function_evaluator: Optional backend callback used when
                a quadrature function has neither explicit values nor a
                callable source. The core runtime never imports backend modules.
        """
        extension = _load_extension()
        self._element_specs = element_specs_from_metadata(metadata_or_module)
        self._payload: RuntimeQuadraturePayload = as_runtime_quadrature_payload(
            quadrature, is_cut=is_cut
        )
        q_values = build_quadrature_function_value_set(
            _quadrature_function_infos(metadata_or_module),
            self._payload.rules,
            fallback_evaluator=quadrature_function_evaluator,
        )
        if q_values is not None:
            self._payload = RuntimeQuadraturePayload(
                rules=self._payload.rules,
                entities=self._payload.entities,
                quadrature_functions=q_values,
            )
        self._owner = extension.CustomData(
            self._element_specs,
            self._payload,
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

    @property
    def num_entities(self) -> int:
        """Number of runtime integration-loop entities."""
        return int(self._owner.num_entities)

    def __int__(self) -> int:
        """Return :attr:`ptr` for APIs that accept integer pointers."""
        return self.ptr


def create_custom_data(
    metadata_or_module: Any,
    quadrature: Any,
    *,
    is_cut: Any | None = None,
    quadrature_function_evaluator: Any | None = None,
) -> CustomData:
    """Build a :class:`CustomData` owner."""
    return CustomData(
        metadata_or_module,
        quadrature=quadrature,
        is_cut=is_cut,
        quadrature_function_evaluator=quadrature_function_evaluator,
    )


__all__ = [
    "CustomData",
    "create_custom_data",
    "element_specs_from_metadata",
]
