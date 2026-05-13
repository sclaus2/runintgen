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
from .basix_runtime import (
    CustomData,
    create_custom_data,
    element_specs_from_metadata,
)
from .codegen_files import (
    RuntimeCodeFiles,
    format_runtime_abi_header,
    format_runtime_header,
    format_runtime_source,
    runtime_abi_header_path,
    write_runtime_code,
)
from .codegeneration.C.integrals import get_runintgen_data_struct
from .form_metadata import (
    BasixElementSpec,
    ElementKey,
    FormElementInfo,
    FormRuntimeMetadata,
    IntegralElementUsage,
    IntegralRuntimeLayout,
    Role,
    basix_element_spec_from_basix,
    build_form_runtime_metadata,
    element_key_from_basix,
    element_key_from_ufl,
    export_metadata_for_cpp,
)
from .geometry import (
    generate_subcell_quadrature,
    map_quadrature_to_subcell,
    map_subcell_to_parent_reference,
    scale_weights_by_jacobian,
)
from .measures import (
    RUNTIME_QUADRATURE_RULE,
    RuntimeIntegralMode,
    RuntimeMeasure,
    dSq,
    dsq,
    dxq,
    get_quadrature_provider,
    has_runtime_quadrature,
    has_standard_subdomain_data,
    is_runtime_integral,
    is_runtime_quadrature_rule,
    runtime_integral_mode,
    runtime_measure,
)
from .runtime_api import RunintModule, RuntimeKernelInfo, compile_runtime_integrals
from .runtime_data import (
    CFFI_DEF,
    RuntimeBasixElement,
    RuntimeContextBuilder,
    RuntimeEntityMap,
    RuntimeQuadraturePayload,
    RuntimeQuadratureRule,
    RuntimeQuadratureRules,
    RuntimeTableRequest,
    RuntimeTableView,
    as_runtime_quadrature_payload,
    as_runtime_quadrature_rules,
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
    # Basix runtime
    "CustomData",
    "create_custom_data",
    "element_specs_from_metadata",
    # Code generation
    "get_runintgen_data_struct",
    "RuntimeCodeFiles",
    "format_runtime_abi_header",
    "format_runtime_header",
    "format_runtime_source",
    "runtime_abi_header_path",
    "write_runtime_code",
    # Form metadata (Plan v2)
    "BasixElementSpec",
    "ElementKey",
    "FormElementInfo",
    "FormRuntimeMetadata",
    "IntegralElementUsage",
    "IntegralRuntimeLayout",
    "Role",
    "basix_element_spec_from_basix",
    "build_form_runtime_metadata",
    "element_key_from_basix",
    "element_key_from_ufl",
    "export_metadata_for_cpp",
    # Geometry utilities
    "generate_subcell_quadrature",
    "map_quadrature_to_subcell",
    "map_subcell_to_parent_reference",
    "scale_weights_by_jacobian",
    # Measures
    "RUNTIME_QUADRATURE_RULE",
    "RuntimeIntegralMode",
    "RuntimeMeasure",
    "dSq",
    "dsq",
    "dxq",
    "get_quadrature_provider",
    "has_standard_subdomain_data",
    "has_runtime_quadrature",
    "is_runtime_quadrature_rule",
    "is_runtime_integral",
    "runtime_integral_mode",
    "runtime_measure",
    # Runtime API
    "RunintModule",
    "RuntimeKernelInfo",
    "compile_runtime_integrals",
    # Runtime data
    "CFFI_DEF",
    "RuntimeBasixElement",
    "RuntimeContextBuilder",
    "RuntimeEntityMap",
    "RuntimeQuadraturePayload",
    "RuntimeQuadratureRule",
    "RuntimeQuadratureRules",
    "RuntimeTableRequest",
    "RuntimeTableView",
    "as_runtime_quadrature_payload",
    "as_runtime_quadrature_rules",
    "to_intptr",
    # Runtime tables
    "DerivativeMapping",
    "ElementUsage",
    "RuntimeElementMapping",
    "UniqueElementInfo",
    "build_runtime_element_mapping",
    "build_runtime_element_mapping_from_ir",
]
