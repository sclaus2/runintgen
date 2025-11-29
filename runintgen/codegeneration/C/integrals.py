"""C code generation glue for runtime integrals.

This module ties together the runtime integral generator, C formatter,
and templates to produce complete C kernel code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...analysis import RuntimeInfo
from ...ir_runtime_map import get_integral_from_ir, map_runtime_groups_to_ir
from ...runtime_api import RuntimeKernelInfo
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
    runtime_info: RuntimeInfo,
    options: dict[str, Any],
) -> list[RuntimeKernelInfo]:
    """Generate C kernels for all runtime groups in `runtime_info`.

    Args:
        runtime_info: RuntimeInfo object from analysis phase.
        options: Compilation options dictionary.

    Returns:
        List of RuntimeKernelInfo objects, one per runtime integral.
    """
    ir = runtime_info.ir

    # Determine scalar and geometry types
    scalar_type = np.dtype(options.get("scalar_type", np.float64)).name
    scalar_c = dtype_to_c_type(scalar_type)
    geom_c = dtype_to_c_type(dtype_to_scalar_dtype(scalar_type))

    # Create runtime integral generator
    # Note: backend would normally be FFCXBackend, but we create a minimal version
    backend = (
        None  # Placeholder - will be FFCXBackend(ir, options) when fully implemented
    )
    rig = RuntimeIntegralGenerator(ir, backend)

    # Map runtime groups to IR indices
    mapping = map_runtime_groups_to_ir(runtime_info)

    kernels: list[RuntimeKernelInfo] = []

    for group, indices in mapping.items():
        for itype, idx in indices:
            # Get the corresponding integral from IR
            integral_ir = get_integral_from_ir(ir, itype, idx)

            # Generate the kernel body
            body_c = rig.generate_runtime(integral_ir)

            # Build function name
            subdomain_id = group.subdomain_ids[0] if group.subdomain_ids else 0
            func_name = f"tabulate_tensor_runint_{itype}_{subdomain_id}_{idx}"

            # Format the complete C definition
            c_def = factory_runtime_kernel.format(
                fname=func_name,
                scalar=scalar_c,
                geom=geom_c,
                body=body_c,
            )

            # Format the declaration
            c_decl = factory_runtime_kernel_decl.format(
                fname=func_name,
                scalar=scalar_c,
                geom=geom_c,
            )

            kernels.append(
                RuntimeKernelInfo(
                    name=func_name,
                    integral_type=itype,
                    subdomain_id=subdomain_id,
                    c_declaration=c_decl,
                    c_definition=c_def,
                    tensor_shape=None,
                )
            )

    return kernels


def get_runintgen_data_struct() -> str:
    """Return the C struct definition for runintgen_data.

    This should be included once in any translation unit using runtime kernels.
    """
    return runintgen_data_struct
