"""Runtime measures and quadrature helpers for runintgen.

This module provides utilities to tag UFL integrals as "runtime integrals"
using the standard UFL metadata mechanism with `quadrature_rule="runtime"`.

The recommended way to create runtime integrals is:

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=quadrature_provider,
                     metadata={"quadrature_rule": "runtime"})

Where `quadrature_provider` is any object that provides quadrature data at
assembly time (e.g., a C++ class wrapped via pybind11 or nanobind).

For facet integrals, use "dS" (interior) or "ds" (exterior):

    dS = ufl.Measure("dS", domain=mesh, subdomain_data=quadrature_provider,
                     metadata={"quadrature_rule": "runtime"})
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=quadrature_provider,
                     metadata={"quadrature_rule": "runtime"})
"""

from __future__ import annotations

from typing import Any

import ufl


# The key metadata value that marks an integral as runtime
RUNTIME_QUADRATURE_RULE = "runtime"


def is_runtime_integral(integral: ufl.classes.Integral) -> bool:
    """Check if an integral is marked as a runtime integral.

    An integral is considered "runtime" if its metadata contains
    {"quadrature_rule": "runtime"}.

    Args:
        integral: A UFL Integral object.

    Returns:
        True if the integral is marked as runtime, False otherwise.
    """
    md = dict(integral.metadata())
    return md.get("quadrature_rule") == RUNTIME_QUADRATURE_RULE


def get_quadrature_provider(integral: ufl.classes.Integral) -> Any | None:
    """Get the quadrature provider from a runtime integral.

    For runtime integrals, the subdomain_data contains the quadrature
    provider object (e.g., a C++ class).

    Args:
        integral: A UFL Integral object.

    Returns:
        The quadrature provider object (subdomain_data), or None if not set.
    """
    return integral.subdomain_data()
