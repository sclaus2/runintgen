"""Analysis module for runintgen.

This module handles the detection and analysis of runtime integrals
in UFL forms, and delegates to FFCX for IR computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import ufl

from ffcx.analysis import analyze_ufl_objects
from ffcx.ir.representation import compute_ir
from ffcx.options import get_options

from .measures import RuntimeQuadrature, is_runtime_integral

# Type alias for integral group keys
Key = tuple[Any, str, tuple[Any, ...]]  # (domain, integral_type, subdomain_ids)


@dataclass(frozen=True)
class RuntimeGroup:
    """A group of integrals sharing runtime quadrature.

    Attributes:
        domain: The UFL mesh domain.
        integral_type: The type of integral ("cell", "exterior_facet", etc.).
        subdomain_ids: Tuple of subdomain identifiers.
        marker: The RuntimeQuadrature marker for this group.
    """

    domain: Any  # ufl.Mesh - use Any for hashability
    integral_type: str
    subdomain_ids: tuple[Any, ...]
    marker: RuntimeQuadrature


@dataclass
class RuntimeInfo:
    """Container for analysis results.

    Attributes:
        ir: FFCX IR dictionary.
        groups: List of RuntimeGroup objects for runtime integrals.
        form_data: UFL FormData object.
        meta: Additional metadata.
    """

    ir: dict[str, Any]
    groups: list[RuntimeGroup] = field(default_factory=list)
    form_data: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


def build_runtime_info(form: ufl.Form, options: dict[str, Any]) -> RuntimeInfo:
    """Analyse form and collect runtime integrals + FFCX IR.

    This function:
    1. Computes UFL form_data with integral scaling disabled
    2. Scans integral_data for runtime markers
    3. Runs FFCX analysis and IR computation
    4. Returns a RuntimeInfo object

    Args:
        form: A UFL Form to analyze.
        options: Compilation options dictionary.

    Returns:
        RuntimeInfo containing IR, runtime groups, and metadata.
    """
    complex_mode = options.get("scalar_type", "float64") in ("complex64", "complex128")

    # 1. Compute form_data with disabled integral scaling (no detJ in integrand)
    form_data = ufl.algorithms.compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=False,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(ufl.classes.Jacobian,),
        do_apply_restrictions=True,
        do_append_everywhere_integrals=False,
        complex_mode=complex_mode,
    )

    # 2. First, extract runtime markers from the ORIGINAL form before processing
    #    (because compute_form_data may strip subdomain_data)
    original_markers: dict[tuple[str, int], RuntimeQuadrature] = {}
    for integral in form.integrals():
        sd = integral.subdomain_data()
        if isinstance(sd, RuntimeQuadrature):
            key = (integral.integral_type(), integral.subdomain_id())
            original_markers[key] = sd

    # 3. Scan integral_data for runtime markers (using metadata or original markers)
    groups: dict[Key, RuntimeGroup] = {}
    for itg_data in form_data.integral_data:
        domain = itg_data.domain
        itype = itg_data.integral_type
        subdomain_ids = tuple(itg_data.subdomain_id)
        key: Key = (domain, itype, subdomain_ids)

        marker: RuntimeQuadrature | None = None
        for integral in itg_data.integrals:
            if is_runtime_integral(integral):
                # First try to get marker from subdomain_data (may be None after processing)
                sd = integral.subdomain_data()
                if isinstance(sd, RuntimeQuadrature):
                    marker = sd
                    break
                # Otherwise, look up in original markers
                for sid in subdomain_ids:
                    orig_key = (itype, sid)
                    if orig_key in original_markers:
                        marker = original_markers[orig_key]
                        break
                # If still no marker but metadata says runtime, create a default one
                if marker is None and integral.metadata().get("runintgen"):
                    marker = RuntimeQuadrature(tag="runtime", payload=None)
                if marker is not None:
                    break

        if marker is not None:
            groups[key] = RuntimeGroup(
                domain=domain,
                integral_type=itype,
                subdomain_ids=subdomain_ids,
                marker=marker,
            )

    # 3. Let FFCX do the standard analysis/IR
    #    IMPORTANT: do not override FFCX options related to integral scaling here,
    #    we already set that in compute_form_data.
    scalar_type = np.dtype(options.get("scalar_type", "float64"))
    analysis = analyze_ufl_objects([form], scalar_type)

    # Get full FFCX options for IR computation
    ffcx_options = get_options(options)
    ir = compute_ir(analysis, {}, "runint", ffcx_options, visualise=False)

    # 4. Pack RuntimeInfo
    meta = {
        "form_rank": len(form.arguments()),
        "num_runtime_groups": len(groups),
    }

    return RuntimeInfo(
        ir=ir,
        groups=list(groups.values()),
        form_data=form_data,
        meta=meta,
    )
