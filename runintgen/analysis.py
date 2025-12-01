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

from .measures import is_runtime_integral

# Type alias for integral group keys
Key = tuple[Any, str, tuple[Any, ...]]  # (domain, integral_type, subdomain_ids)


@dataclass(frozen=True)
class RuntimeGroup:
    """A group of integrals sharing runtime quadrature.

    Attributes:
        domain: The UFL mesh domain.
        integral_type: The type of integral ("cell", "exterior_facet", etc.).
        subdomain_ids: Tuple of subdomain identifiers.
        quadrature_provider: The quadrature provider object (e.g., C++ class).
    """

    domain: Any  # ufl.Mesh - use Any for hashability
    integral_type: str
    subdomain_ids: tuple[Any, ...]
    quadrature_provider: Any = None


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


def _strip_runtime_metadata(form: ufl.Form) -> ufl.Form:
    """Strip quadrature_rule='runtime' metadata so FFCX doesn't choke on it.

    FFCX tries to interpret quadrature_rule as a basix quadrature type.
    We need to replace 'runtime' with a valid default (or remove it).
    """
    new_integrals = []
    for integral in form.integrals():
        md = dict(integral.metadata())
        if md.get("quadrature_rule") == "runtime":
            # Replace with default quadrature rule
            md["quadrature_rule"] = "default"
        new_integral = integral.reconstruct(metadata=md)
        new_integrals.append(new_integral)

    return ufl.Form(new_integrals)


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

    # 2. Extract quadrature providers from the ORIGINAL form before processing
    #    (because compute_form_data may strip subdomain_data)
    original_providers: dict[tuple[str, int], Any] = {}
    for integral in form.integrals():
        if is_runtime_integral(integral):
            orig_key = (integral.integral_type(), integral.subdomain_id())
            original_providers[orig_key] = integral.subdomain_data()

    # 3. Scan integral_data for runtime integrals
    groups: dict[Key, RuntimeGroup] = {}
    for itg_data in form_data.integral_data:
        domain = itg_data.domain
        itype = itg_data.integral_type
        subdomain_ids = tuple(itg_data.subdomain_id)
        key: Key = (domain, itype, subdomain_ids)

        quadrature_provider: Any = None

        for integral in itg_data.integrals:
            if is_runtime_integral(integral):
                # Get the quadrature provider from subdomain_data
                sd = integral.subdomain_data()
                if sd is not None:
                    quadrature_provider = sd
                else:
                    # If subdomain_data was stripped, look up in original data
                    for sid in subdomain_ids:
                        lookup_key = (itype, sid)
                        if lookup_key in original_providers:
                            quadrature_provider = original_providers[lookup_key]
                            break

                if quadrature_provider is not None:
                    break

        if quadrature_provider is not None:
            groups[key] = RuntimeGroup(
                domain=domain,
                integral_type=itype,
                subdomain_ids=subdomain_ids,
                quadrature_provider=quadrature_provider,
            )
        elif any(is_runtime_integral(i) for i in itg_data.integrals):
            # Runtime integral without provider (valid case)
            groups[key] = RuntimeGroup(
                domain=domain,
                integral_type=itype,
                subdomain_ids=subdomain_ids,
                quadrature_provider=None,
            )

    # 4. Let FFCX do the standard analysis/IR
    #    IMPORTANT: do not override FFCX options related to integral scaling here,
    #    we already set that in compute_form_data.
    #    Strip "runtime" quadrature_rule so FFCX doesn't try to interpret it.
    form_for_ffcx = _strip_runtime_metadata(form)
    scalar_type = np.dtype(options.get("scalar_type", "float64"))
    analysis = analyze_ufl_objects([form_for_ffcx], scalar_type)

    # Get full FFCX options for IR computation
    ffcx_options = get_options(options)
    ir = compute_ir(analysis, {}, "runint", ffcx_options, visualise=False)

    # 5. Pack RuntimeInfo
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
