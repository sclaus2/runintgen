"""runintgen - Runtime integration kernel generator for FEniCSx/FFCx."""

from __future__ import annotations

from .fe_tables import (
    ComponentRequest,
    IntegralRuntimeMeta,
    extract_integral_metadata,
    print_integral_metadata,
)
from .measures import RuntimeQuadrature, is_runtime_integral, runtime_dx
from .runtime_api import RunintModule, RuntimeKernelInfo, compile_runtime_integrals

__all__ = [
    "ComponentRequest",
    "IntegralRuntimeMeta",
    "RuntimeKernelInfo",
    "RuntimeQuadrature",
    "RunintModule",
    "compile_runtime_integrals",
    "extract_integral_metadata",
    "is_runtime_integral",
    "print_integral_metadata",
    "runtime_dx",
]
