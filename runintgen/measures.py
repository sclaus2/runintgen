"""Runtime quadrature detection helpers for runintgen.

This module provides utilities to detect UFL integrals that carry runtime
quadrature data. New code should put quadrature-rule objects directly in UFL
``subdomain_data``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import ufl

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
            else:
                values.append(item)
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
    if subdomain_data is None:
        return False
    if is_runtime_quadrature_rule(subdomain_data):
        return False

    values = _iter_subdomain_data_values(subdomain_data)
    if not values:
        return False
    return any(not is_runtime_quadrature_rule(value) for value in values)


def runtime_integral_mode(integral: ufl.classes.Integral) -> RuntimeIntegralMode:
    """Return the code-generation mode requested by an integral."""
    subdomain_data = integral.subdomain_data()

    if has_runtime_quadrature(subdomain_data):
        if has_standard_subdomain_data(subdomain_data):
            return RuntimeIntegralMode.MIXED
        return RuntimeIntegralMode.RUNTIME

    return RuntimeIntegralMode.STANDARD


def is_runtime_integral(integral: ufl.classes.Integral) -> bool:
    """Check if an integral is marked as a runtime integral.

    An integral is considered runtime if its ``subdomain_data`` contains a
    quadrature-rule object.

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
