"""Runtime measures and quadrature helpers for runintgen.

This module provides utilities to detect UFL integrals that carry runtime
quadrature data. New code should put quadrature-rule objects directly in UFL
``subdomain_data``; legacy ``quadrature_rule="runtime"`` metadata is still
accepted.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import ufl

# Metadata value that marks an integral as runtime.
RUNTIME_QUADRATURE_RULE = "runtime"


class RuntimeIntegralMode(Enum):
    """Code-generation mode requested by an integral's measure data."""

    STANDARD = "standard"
    RUNTIME = "runtime"
    MIXED = "mixed"


def is_runtime_quadrature_rule(value: Any) -> bool:
    """Return whether a value looks like a runtime quadrature rule.

    The check intentionally uses a small structural protocol so callers can use
    either runintgen's Python containers or their own quadrature-rule objects.
    """
    if value is None:
        return False
    return hasattr(value, "points") and hasattr(value, "weights")


def _iter_subdomain_data_values(subdomain_data: Any) -> list[Any]:
    """Return payload values from common UFL subdomain-data containers."""
    if isinstance(subdomain_data, dict):
        return list(subdomain_data.values())

    if isinstance(subdomain_data, (list, tuple)):
        values = []
        for item in subdomain_data:
            if isinstance(item, tuple) and len(item) == 2:
                values.append(item[1])
        return values

    return []


def has_runtime_quadrature(subdomain_data: Any) -> bool:
    """Return whether subdomain data contains runtime quadrature rules."""
    if is_runtime_quadrature_rule(subdomain_data):
        return True
    return any(
        is_runtime_quadrature_rule(value)
        for value in _iter_subdomain_data_values(subdomain_data)
    )


def has_standard_subdomain_data(subdomain_data: Any) -> bool:
    """Return whether subdomain data contains non-runtime entity payloads."""
    if subdomain_data is None or is_runtime_quadrature_rule(subdomain_data):
        return False

    values = _iter_subdomain_data_values(subdomain_data)
    if not values:
        return False
    return any(not is_runtime_quadrature_rule(value) for value in values)


def runtime_integral_mode(integral: ufl.classes.Integral) -> RuntimeIntegralMode:
    """Return the code-generation mode requested by an integral."""
    subdomain_data = integral.subdomain_data()
    metadata = dict(integral.metadata())

    if has_runtime_quadrature(subdomain_data):
        if has_standard_subdomain_data(subdomain_data):
            return RuntimeIntegralMode.MIXED
        return RuntimeIntegralMode.RUNTIME

    if metadata.get("quadrature_rule") == RUNTIME_QUADRATURE_RULE:
        return RuntimeIntegralMode.RUNTIME

    return RuntimeIntegralMode.STANDARD


class RuntimeMeasure(ufl.Measure):
    """UFL measure that preserves the runtime quadrature metadata flag."""

    __slots__ = ()

    def reconstruct(
        self,
        integral_type: str | None = None,
        subdomain_id: Any | None = None,
        domain: Any | None = None,
        metadata: dict[str, Any] | None = None,
        subdomain_data: Any | None = None,
        intersect_measures: tuple[ufl.Measure, ...] | None = None,
    ) -> RuntimeMeasure:
        """Construct a new runtime measure with selected properties replaced."""
        if integral_type is None:
            integral_type = self.integral_type()
        if subdomain_id is None:
            subdomain_id = self.subdomain_id()
        if domain is None:
            domain = self.ufl_domain()
        if intersect_measures is None:
            intersect_measures = self.intersect_measures()

        if metadata is None:
            new_metadata = dict(self.metadata())
        else:
            new_metadata = dict(self.metadata())
            new_metadata.update(metadata)
        new_metadata["quadrature_rule"] = RUNTIME_QUADRATURE_RULE

        if subdomain_data is None:
            new_subdomain_data = self.subdomain_data()
        else:
            new_subdomain_data = subdomain_data

        return RuntimeMeasure(
            integral_type,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=new_metadata,
            subdomain_data=new_subdomain_data,
            intersect_measures=intersect_measures,
        )


def _runtime_metadata(
    metadata: dict[str, Any] | None,
    metadata_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge explicit metadata sources for runtime measure helpers."""
    runtime_metadata = dict(metadata or {})
    runtime_metadata.update(metadata_kwargs)
    runtime_metadata["quadrature_rule"] = RUNTIME_QUADRATURE_RULE
    return runtime_metadata


def runtime_measure(
    integral_type: str,
    *,
    domain: Any | None = None,
    subdomain_id: Any = "everywhere",
    quadrature_provider: Any | None = None,
    subdomain_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    intersect_measures: tuple[ufl.Measure, ...] | None = None,
    **metadata_kwargs: Any,
) -> RuntimeMeasure:
    """Create a UFL measure marked for runtime quadrature.

    Args:
        integral_type: UFL measure type, e.g. ``"dx"``, ``"ds"``, or ``"dS"``.
        domain: UFL integration domain.
        subdomain_id: UFL subdomain id or tuple of ids.
        quadrature_provider: Optional caller-owned runtime quadrature provider.
            This is stored directly in UFL ``subdomain_data``.
        subdomain_data: UFL ``subdomain_data`` value. This is an alias for
            ``quadrature_provider`` and is also stored directly.
        metadata: Optional UFL compiler metadata.
        intersect_measures: Optional UFL intersect measures.
        **metadata_kwargs: Additional metadata entries.

    Returns:
        A runtime-marked UFL measure.
    """
    if quadrature_provider is not None and subdomain_data is not None:
        raise ValueError("Pass either quadrature_provider or subdomain_data, not both.")
    runtime_subdomain_data = (
        subdomain_data if subdomain_data is not None else quadrature_provider
    )
    return RuntimeMeasure(
        integral_type,
        domain=domain,
        subdomain_id=subdomain_id,
        metadata=_runtime_metadata(metadata, metadata_kwargs),
        subdomain_data=runtime_subdomain_data,
        intersect_measures=intersect_measures,
    )


def dxq(
    subdomain_id: Any = "everywhere",
    domain: Any | None = None,
    *,
    quadrature_provider: Any | None = None,
    subdomain_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    **metadata_kwargs: Any,
) -> RuntimeMeasure:
    """Create a runtime-quadrature cell measure."""
    return runtime_measure(
        "dx",
        domain=domain,
        subdomain_id=subdomain_id,
        quadrature_provider=quadrature_provider,
        subdomain_data=subdomain_data,
        metadata=metadata,
        **metadata_kwargs,
    )


def dsq(
    subdomain_id: Any = "everywhere",
    domain: Any | None = None,
    *,
    quadrature_provider: Any | None = None,
    subdomain_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    **metadata_kwargs: Any,
) -> RuntimeMeasure:
    """Create a runtime-quadrature exterior-facet measure."""
    return runtime_measure(
        "ds",
        domain=domain,
        subdomain_id=subdomain_id,
        quadrature_provider=quadrature_provider,
        subdomain_data=subdomain_data,
        metadata=metadata,
        **metadata_kwargs,
    )


def dSq(
    subdomain_id: Any = "everywhere",
    domain: Any | None = None,
    *,
    quadrature_provider: Any | None = None,
    subdomain_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    **metadata_kwargs: Any,
) -> RuntimeMeasure:
    """Create a runtime-quadrature interior-facet measure."""
    return runtime_measure(
        "dS",
        domain=domain,
        subdomain_id=subdomain_id,
        quadrature_provider=quadrature_provider,
        subdomain_data=subdomain_data,
        metadata=metadata,
        **metadata_kwargs,
    )


def is_runtime_integral(integral: ufl.classes.Integral) -> bool:
    """Check if an integral is marked as a runtime integral.

    An integral is considered runtime if its ``subdomain_data`` contains a
    quadrature-rule object. The legacy ``quadrature_rule="runtime"`` metadata
    marker is also accepted.

    Args:
        integral: A UFL Integral object.

    Returns:
        True if the integral is marked as runtime, False otherwise.
    """
    return runtime_integral_mode(integral) is not RuntimeIntegralMode.STANDARD


def get_quadrature_provider(integral: ufl.classes.Integral) -> Any | None:
    """Get the quadrature provider from a runtime integral.

    Runtime measures store the caller's quadrature provider directly in
    ``subdomain_data``.

    Args:
        integral: A UFL Integral object.

    Returns:
        The quadrature provider object (subdomain_data), or None if not set.
    """
    return integral.subdomain_data()
