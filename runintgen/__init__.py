"""runintgen - Runtime integration kernel generator for FEniCSx/FFCx."""

from __future__ import annotations

from .analysis import (
    ArgumentInfo,
    ArgumentRole,
    ElementInfo,
    RuntimeAnalysisInfo,
    RuntimeGroup,
    RuntimeInfo,
    RuntimeIntegralInfo,
    build_runtime_analysis,
    build_runtime_info,
)
from .codegeneration.C.integrals import get_runintgen_data_struct
from .dolfinx_utils import (
    CompiledKernel,
    RuntimeFormInfo,
    compile_runtime_kernels,
    create_dolfinx_form_with_runtime,
    set_runtime_data,
)
from .geometry import (
    generate_subcell_quadrature,
    map_quadrature_to_subcell,
    map_subcell_to_parent_reference,
    scale_weights_by_jacobian,
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
    RuntimeDataBuilder,
    create_quadrature_config,
    tabulate_element,
    to_intptr,
)
from .runtime_tables import (
    DerivativeMapping,
    ElementUsage,
    RuntimeElementMapping,
    UniqueElementInfo,
    build_runtime_element_mapping,
    build_runtime_element_mapping_from_ir,
)
from .tabulation import (
    PreparedTables,
    compute_detJ_triangle,
    prepare_runtime_data,
    prepare_runtime_data_for_cell,
    tabulate_from_table_info,
)

__all__ = [
    # Analysis types
    "ArgumentInfo",
    "ArgumentRole",
    "ElementInfo",
    "RuntimeAnalysisInfo",
    "RuntimeGroup",
    "RuntimeInfo",
    "RuntimeIntegralInfo",
    "build_runtime_analysis",
    "build_runtime_info",
    # Code generation
    "get_runintgen_data_struct",
    # DOLFINx utilities
    "CompiledKernel",
    "RuntimeFormInfo",
    "compile_runtime_kernels",
    "create_dolfinx_form_with_runtime",
    "set_runtime_data",
    # Geometry utilities
    "generate_subcell_quadrature",
    "map_quadrature_to_subcell",
    "map_subcell_to_parent_reference",
    "scale_weights_by_jacobian",
    # Measures
    "RUNTIME_QUADRATURE_RULE",
    "get_quadrature_provider",
    "is_runtime_integral",
    # Runtime API
    "RunintModule",
    "RuntimeKernelInfo",
    "compile_runtime_integrals",
    # Runtime data
    "CFFI_DEF",
    "ElementTableInfo",
    "RuntimeDataBuilder",
    "create_quadrature_config",
    "tabulate_element",
    "to_intptr",
    # Runtime tables
    "DerivativeMapping",
    "ElementUsage",
    "RuntimeElementMapping",
    "UniqueElementInfo",
    "build_runtime_element_mapping",
    "build_runtime_element_mapping_from_ir",
    # Tabulation utilities
    "PreparedTables",
    "compute_detJ_triangle",
    "prepare_runtime_data",
    "prepare_runtime_data_for_cell",
    "tabulate_from_table_info",
]
