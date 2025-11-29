"""Runtime measures and quadrature helpers for runintgen.

This module provides utilities to tag UFL integrals as "runtime integrals"
using subdomain_data and metadata on dx measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ufl


@dataclass(frozen=True)
class RuntimeQuadrature:
    """Marker object stored in subdomain_data to indicate runtime integration.

    The `payload` can carry arbitrary Python data (e.g. a symbolic name, rules,
    or configuration).

    Attributes:
        tag: A string identifier for the runtime quadrature.
        payload: Optional arbitrary data associated with the quadrature.
    """

    tag: str
    payload: Any | None = None


def runtime_dx(
    subdomain_id: int,
    domain: ufl.Mesh,
    tag: str = "runint",
    payload: Any | None = None,
    **metadata: Any,
) -> ufl.Measure:
    """Create a dx measure tagged as runtime.

    This is a convenience wrapper; internally we encode runtime info in
    `subdomain_data` and metadata.

    Args:
        subdomain_id: The subdomain identifier for this integral.
        domain: The UFL mesh domain.
        tag: A string tag for the runtime quadrature marker.
        payload: Optional arbitrary data to attach to the marker.
        **metadata: Additional metadata to pass to the measure.

    Returns:
        A UFL Measure configured for runtime integration.

    Example:
        >>> mesh = ufl.Mesh(...)
        >>> dx_rt = runtime_dx(1, mesh, tag="cut_cell", payload={"order": 3})
        >>> form = u * v * dx_rt
    """
    rq = RuntimeQuadrature(tag=tag, payload=payload)
    md = {"runintgen": True}
    md.update(metadata)
    return ufl.Measure(
        "dx",
        domain=domain,
        subdomain_id=subdomain_id,
        subdomain_data=rq,
        metadata=md,
    )


def is_runtime_integral(integral: ufl.classes.Integral) -> bool:
    """Check if an integral is marked as a runtime integral.

    An integral is considered "runtime" if:
    - Its subdomain_data is a RuntimeQuadrature instance, OR
    - Its metadata contains {"runintgen": True}

    Args:
        integral: A UFL Integral object.

    Returns:
        True if the integral is marked as runtime, False otherwise.
    """
    md = dict(integral.metadata())
    sd = integral.subdomain_data()
    return isinstance(sd, RuntimeQuadrature) or md.get("runintgen", False)
