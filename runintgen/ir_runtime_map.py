"""IR to runtime group mapping utilities.

This module provides helpers to match RuntimeGroup objects to specific
integral indices and names inside the FFCX IR.
"""

from __future__ import annotations

from typing import Any

from ffcx.ir.representation import DataIR

from .analysis import RuntimeGroup, RuntimeInfo


def map_runtime_groups_to_ir(
    runtime_info: RuntimeInfo,
) -> dict[RuntimeGroup, list[tuple[str, int]]]:
    """Map RuntimeGroup objects to their corresponding IR indices.

    Returns a mapping from RuntimeGroup to list of (integral_type, idx) pairs,
    where idx is the index into ir.forms[0].subdomain_ids[integral_type].

    Args:
        runtime_info: RuntimeInfo object containing IR and groups.

    Returns:
        Dictionary mapping each RuntimeGroup to its IR indices.
    """
    ir: DataIR = runtime_info.ir
    mapping: dict[RuntimeGroup, list[tuple[str, int]]] = {}

    # DataIR is a NamedTuple with: integrals, forms, expressions
    if not ir.forms:
        return mapping

    # Use the first form's IR (typically there's only one)
    form_ir = ir.forms[0]

    # Get subdomain_ids per integral type from FormIR
    # FormIR has: subdomain_ids: dict[str, list[int]]
    subdomain_ids_per_type: dict[str, list[int]] = form_ir.subdomain_ids

    for group in runtime_info.groups:
        itype = group.integral_type
        # For now, assume one subdomain_id per group (common case)
        if len(group.subdomain_ids) != 1:
            # Handle multiple subdomain_ids if needed
            continue

        gid = group.subdomain_ids[0]
        indices: list[tuple[str, int]] = []

        if itype in subdomain_ids_per_type:
            for idx, sid in enumerate(subdomain_ids_per_type[itype]):
                # Match either by exact ID or by special "everywhere"/"otherwise" -> -1
                if sid == gid:
                    indices.append((itype, idx))
                elif gid in ("everywhere", "otherwise") and sid == -1:
                    # UFL 'everywhere' integral is stored as -1 in IR
                    indices.append((itype, idx))
                elif isinstance(gid, int) and gid == sid:
                    indices.append((itype, idx))

        mapping[group] = indices

    return mapping


def get_integral_from_ir(
    ir: DataIR,
    integral_type: str,
    idx: int,
) -> Any:
    """Get an integral object from the IR by type and index.

    Args:
        ir: The FFCX DataIR (NamedTuple with integrals, forms, expressions).
        integral_type: The type of integral ("cell", "exterior_facet", etc.).
        idx: The index of the integral within that type.

    Returns:
        The integral IR object, or None if not found.
    """
    if not ir.forms:
        return None

    form_ir = ir.forms[0]

    # Get subdomain_ids for this integral type
    subdomain_ids = form_ir.subdomain_ids.get(integral_type, [])
    if idx >= len(subdomain_ids):
        return None

    # Find the matching integral in ir.integrals
    # Filter by integral type first, then by index within that type
    matching_integrals = []
    for integral_ir in ir.integrals:
        expr = integral_ir.expression
        # Check integral_type attribute on expression
        if hasattr(expr, "integral_type") and expr.integral_type == integral_type:
            matching_integrals.append(integral_ir)

    # Return the idx-th integral of this type
    if idx < len(matching_integrals):
        return matching_integrals[idx]

    return None
