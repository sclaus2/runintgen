"""runintgen - Runtime integration kernel generator for FEniCSx/FFCx."""

from __future__ import annotations

from .codegeneration.C.integrals import get_runintgen_data_struct
from .dolfinx_utils import (
    CompiledKernel,
    RuntimeFormInfo,
    compile_runtime_kernels,
    create_dolfinx_form_with_runtime,
    set_runtime_data,
)
from .fe_tables import (
    ComponentRequest,
    IntegralRuntimeMeta,
    extract_integral_metadata,
    print_integral_metadata,
)
from .measures import (
    RUNTIME_QUADRATURE_RULE,
    get_quadrature_provider,
    is_runtime_integral,
)
from .runtime_api import RunintModule, RuntimeKernelInfo, compile_runtime_integrals
from .runtime_data import (
    CFFI_DEF,
    ElementTableInfo,
    QuadratureConfig,
    RuntimeDataBuilder,
    create_quadrature_config,
    tabulate_element,
)
from .runtime_tables import (
    DerivativeMapping,
    ElementUsage,
    RuntimeElementMapping,
    UniqueElementInfo,
    build_runtime_element_mapping,
)

__all__ = [
    "CFFI_DEF",
    "CompiledKernel",
    "ComponentRequest",
    "DerivativeMapping",
    "ElementTableInfo",
    "ElementUsage",
    "IntegralRuntimeMeta",
    "QuadratureConfig",
    "RUNTIME_QUADRATURE_RULE",
    "RuntimeDataBuilder",
    "RuntimeElementMapping",
    "RuntimeFormInfo",
    "RuntimeKernelInfo",
    "RunintModule",
    "UniqueElementInfo",
    "build_runtime_element_mapping",
    "compile_runtime_integrals",
    "compile_runtime_kernels",
    "create_dolfinx_form_with_runtime",
    "create_quadrature_config",
    "extract_integral_metadata",
    "get_quadrature_provider",
    "get_runintgen_data_struct",
    "is_runtime_integral",
    "print_integral_metadata",
    "set_runtime_data",
    "tabulate_element",
]
