"""C code generation glue for runtime integrals.

This module ties together the runtime integral generator, C formatter,
and templates to produce complete C kernel code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...analysis import RuntimeAnalysisInfo
from ...runtime_api import RuntimeKernelInfo
from ...runtime_tables import build_runtime_element_mapping_from_ir
from ..runtime_integrals import RuntimeIntegralGenerator
from .integrals_template import (
    factory_runtime_kernel,
    factory_runtime_kernel_decl,
    runintgen_data_struct,
)


def dtype_to_c_type(dtype: str) -> str:
    """Convert numpy dtype name to C type string."""
    mapping = {
        "float32": "float",
        "float64": "double",
        "complex64": "float _Complex",
        "complex128": "double _Complex",
    }
    return mapping.get(dtype, "double")


def dtype_to_scalar_dtype(dtype: str) -> str:
    """Get the scalar (real) dtype for a potentially complex dtype."""
    if dtype == "complex64":
        return "float32"
    elif dtype == "complex128":
        return "float64"
    return dtype


def generate_C_runtime_kernels(
    analysis: RuntimeAnalysisInfo,
    options: dict[str, Any],
) -> list[RuntimeKernelInfo]:
    """Generate C kernels for all runtime integrals in the analysis.

    Args:
        analysis: RuntimeAnalysisInfo object from analysis phase.
        options: Compilation options dictionary.

    Returns:
        List of RuntimeKernelInfo objects, one per runtime integral.
    """
    from ...runtime_tables import build_runtime_element_mapping

    ir = analysis.ir

    # Determine scalar and geometry types
    scalar_type = np.dtype(options.get("scalar_type", np.float64)).name
    scalar_c = dtype_to_c_type(scalar_type)
    geom_c = dtype_to_c_type(dtype_to_scalar_dtype(scalar_type))

    # Create runtime integral generator
    backend = None
    rig = RuntimeIntegralGenerator(ir, backend)

    # Build mapping from (itype, ir_index) to subdomain_id via groups
    key_to_subdomain: dict[tuple[str, int], int] = {}
    for group in analysis.groups:
        itype = group.integral_type
        subdomain_id = group.subdomain_ids[0] if group.subdomain_ids else 0
        # For simplicity, assign subdomain to any matching integral type
        for key_itype, key_idx in analysis.integral_infos.keys():
            if key_itype == itype:
                key_to_subdomain[(key_itype, key_idx)] = subdomain_id

    kernels: list[RuntimeKernelInfo] = []

    # Process each runtime integral
    for (itype, ir_index), integral_info in analysis.integral_infos.items():
        # Get the corresponding integral from IR
        ir_type_counts: dict[str, int] = {}
        integral_ir = None
        for iir in ir.integrals:
            t = iir.expression.integral_type
            idx = ir_type_counts.get(t, 0)
            ir_type_counts[t] = idx + 1
            if t == itype and idx == ir_index:
                integral_ir = iir
                break

        if integral_ir is None:
            continue

        # Build element mapping from analysis
        element_mapping = build_runtime_element_mapping(integral_info)

        # Generate the kernel body
        body_c = rig.generate_runtime(integral_ir, element_mapping)

        # Build function name - get subdomain_id from group mapping
        subdomain_id = key_to_subdomain.get((itype, ir_index), 0)
        func_name = f"runint_{itype}_{subdomain_id}_{ir_index}"

        # Format the complete C definition
        c_def = factory_runtime_kernel.format(
            factory_name=func_name,
            scalar=scalar_c,
            geom=geom_c,
            body=body_c,
        )

        # Format the declaration
        c_decl = factory_runtime_kernel_decl.format(
            factory_name=func_name,
            scalar=scalar_c,
            geom=geom_c,
        )

        # Collect element info for runtime
        element_info = [
            {
                "index": e.index,
                "element_id": e.element_id,
                "ndofs": e.ndofs,
                "ncomps": e.ncomps,
                "max_derivative_order": e.max_derivative_order,
                "is_argument": e.is_argument,
                "is_test": e.is_test,
                "is_trial": e.is_trial,
                "is_coefficient": e.is_coefficient,
                "is_coordinate": e.is_coordinate,
                "usages": [
                    {
                        "role": u.role.name,
                        "terminal_index": u.terminal_index,
                        "component": u.component,
                    }
                    for u in e.usages
                ],
            }
            for e in element_mapping.elements
        ]

        kernels.append(
            RuntimeKernelInfo(
                name=func_name,
                integral_type=itype,
                subdomain_id=subdomain_id,
                c_declaration=c_decl,
                c_definition=c_def,
                tensor_shape=None,
                table_info=element_info,
            )
        )

    return kernels


def get_runintgen_data_struct() -> str:
    """Return the C struct definition for runintgen_data.

    This should be included once in any translation unit using runtime kernels.
    """
    return runintgen_data_struct
