"""C code generation glue for runtime integrals."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.C.integral import generator as standard_integral_generator
from ffcx.codegeneration.integral_generator import (
    IntegralGenerator as StandardIntegralGenerator,
)
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.options import get_options

from ...analysis import RuntimeAnalysisInfo
from ...form_metadata import FormRuntimeMetadata
from ...measures import RuntimeIntegralMode
from ...runtime_api import RuntimeKernelInfo
from ..runtime_integrals import RuntimeIntegralGenerator
from .integrals_template import (
    factory_mixed_tabulate_tensor,
    factory_runtime_kernel,
    factory_runtime_kernel_decl,
    factory_runtime_tabulate_tensor,
    factory_standard_tabulate_tensor,
    runintgen_data_struct,
)


def _runtime_options(options: dict[str, Any]) -> dict[str, Any]:
    """Return FFCx options suitable for runtime code generation."""
    ffcx_options = get_options(dict(options or {}))
    ffcx_options["sum_factorization"] = False
    return ffcx_options


def _subdomain_ids_by_integral(
    analysis: RuntimeAnalysisInfo,
) -> dict[tuple[str, int], int]:
    """Build a runtime integral key to subdomain id map."""
    return {
        key: info.subdomain_id for key, info in analysis.integral_infos.items()
    }


def _all_subdomain_ids_by_integral(ir: Any) -> dict[tuple[str, int], int]:
    """Build an integral key to first subdomain id map for all form integrals."""
    if not getattr(ir, "forms", None):
        return {}

    form_ir = ir.forms[0]
    ids_by_key: dict[tuple[str, int], int] = {}
    for integral_type, ids in form_ir.subdomain_ids.items():
        names = form_ir.integral_names[integral_type]
        pos = 0
        group_index = 0
        while pos < len(names):
            name = names[pos]
            sid = ids[pos]
            ids_by_key[(integral_type, group_index)] = int(sid)
            pos += 1
            while pos < len(names) and names[pos] == name:
                pos += 1
            group_index += 1
    return ids_by_key


def _domains_for_integral(integral_ir: Any) -> list[Any]:
    """Return sorted domains present in an FFCx integral IR."""
    domains = {cell for cell, _ in integral_ir.expression.integrand.keys()}
    return sorted(domains, key=lambda cell: cell.name)


def _integrals_by_key(ir: Any) -> dict[tuple[str, int], Any]:
    """Return FFCx integral IR objects keyed by type-local index."""
    integrals: dict[tuple[str, int], Any] = {}
    ir_type_counts: dict[str, int] = {}
    for integral_ir in ir.integrals:
        integral_type = integral_ir.expression.integral_type
        index = ir_type_counts.get(integral_type, 0)
        ir_type_counts[integral_type] = index + 1
        integrals[(integral_type, index)] = integral_ir
    return integrals


def _kernel_name(
    integral_ir: Any,
    domain: Any,
) -> str:
    """Return FFCx's UFCx integral object name."""
    return f"{integral_ir.expression.name}_{domain.name}"


def _standard_kernel_info(
    integral_ir: Any,
    domain: Any,
    *,
    integral_type: str,
    ir_index: int,
    subdomain_id: int,
    options: dict[str, Any],
    kernel_id: int,
) -> RuntimeKernelInfo:
    """Generate one standard FFCx kernel and return runintgen metadata."""
    c_decl, c_def = standard_integral_generator(integral_ir, domain, options)
    scalar_type = np.dtype(options.get("scalar_type", np.float64)).name
    geometry_type = np.dtype(dtype_to_scalar_dtype(scalar_type)).name
    return RuntimeKernelInfo(
        name=_kernel_name(integral_ir, domain),
        integral_type=integral_type,
        subdomain_id=subdomain_id,
        ir_index=ir_index,
        c_declaration=c_decl,
        c_definition=c_def,
        tensor_shape=tuple(integral_ir.expression.tensor_shape),
        table_info=[],
        table_slots={},
        domain=domain.name,
        kernel_id=kernel_id,
        scalar_type=scalar_type,
        geometry_type=geometry_type,
        base_name=integral_ir.expression.name,
        mode="standard",
    )


def _local_index_position(integral_type: str) -> int:
    """Return the appended entity-local-index slot for runtime rule lookup."""
    if integral_type == "interior_facet":
        return 2
    if integral_type == "exterior_facet":
        return 1
    return 0


def _local_index_expr(integral_type: str) -> str:
    """Return C expression for the runtime quadrature rule index."""
    position = _local_index_position(integral_type)
    return f"(entity_local_index == 0 ? 0 : entity_local_index[{position}])"


def _max_source_dof(table: Any) -> int:
    """Return the largest raw Basix dof index a table access can request."""
    num_dofs = int(table.shape[3])
    if num_dofs <= 0:
        return -1
    return num_dofs - 1


@dataclass(frozen=True)
class _ElementTableRequest:
    """One Basix tabulation request shared by multiple FFCx table references."""

    slot: int
    element_index: int
    max_derivative_order: int
    max_derivative_index: int
    max_source_dof: int
    max_component: int
    is_permuted: bool
    c_symbol: str


def _element_table_requests(runtime_tables: list[Any]) -> list[_ElementTableRequest]:
    """Group FFCx table references into one request per Basix element slot."""
    by_slot: dict[int, list[Any]] = {}
    for table in runtime_tables:
        by_slot.setdefault(int(table.slot), []).append(table)

    requests: list[_ElementTableRequest] = []
    for slot, tables in sorted(by_slot.items()):
        element_indices = {int(table.element_index) for table in tables}
        if len(element_indices) != 1:
            raise ValueError(
                "Runtime table references sharing a Basix tabulation slot "
                f"resolved to multiple form element indices: {sorted(element_indices)}."
            )

        requests.append(
            _ElementTableRequest(
                slot=slot,
                element_index=next(iter(element_indices)),
                max_derivative_order=max(
                    int(sum(table.derivative_counts)) for table in tables
                ),
                max_derivative_index=max(
                    int(table.derivative_index) for table in tables
                ),
                max_source_dof=max(_max_source_dof(table) for table in tables),
                max_component=0,
                is_permuted=any(bool(table.is_permuted) for table in tables),
                c_symbol=tables[0].c_symbol,
            )
        )

    return requests


def _table_requests(element_requests: list[_ElementTableRequest]) -> str:
    """Generate static C requests for per-element runtime FE tabulations."""
    lines = []
    for request in element_requests:
        lines.append(
            f"static const runintgen_table_request "
            f"rt_element_request_{request.slot} = {{\n"
            f"  .slot = {request.slot},\n"
            f"  .derivative_order = {request.max_derivative_order},\n"
            f"  .is_permuted = {int(request.is_permuted)},\n"
            f"}};"
        )
    return "\n".join(lines)


def _table_preparation(element_requests: list[_ElementTableRequest]) -> str:
    """Generate in-kernel Basix wrapper calls for runtime FE element slots."""
    lines = []
    for request in element_requests:
        lines.extend(
            [
                f"  if (form == 0 || elements == 0 || "
                f"form->num_elements <= {request.element_index})",
                "    return;",
                f"  if (elements[{request.element_index}].tabulate == 0)",
                "    return;",
                f"  runintgen_table_view rt_element_view_{request.slot};",
                f"  if (elements[{request.element_index}].tabulate(",
                f"          &elements[{request.element_index}], rule,",
                f"          &rt_element_request_{request.slot}, "
                f"&rt_element_view_{request.slot}) != 0)",
                "    return;",
                f"  const double* {request.c_symbol} = "
                f"rt_element_view_{request.slot}.values;",
                f"  if ({request.c_symbol} == 0)",
                "    return;",
                f"  const int {request.c_symbol}_num_dofs = "
                f"rt_element_view_{request.slot}.num_dofs;",
                f"  const int {request.c_symbol}_num_components = "
                f"rt_element_view_{request.slot}.num_components;",
                f"  if (rt_element_view_{request.slot}.num_points != rt_nq || "
                f"rt_element_view_{request.slot}.num_derivatives "
                f"<= {request.max_derivative_index} "
                f"|| {request.c_symbol}_num_dofs <= {request.max_source_dof} "
                f"|| {request.c_symbol}_num_components <= {request.max_component})",
                "    return;",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _enabled_coefficients(integral_ir: Any, factory_name: str) -> tuple[str, str]:
    """Return C code for FFCx enabled coefficient metadata."""
    if len(integral_ir.enabled_coefficients) == 0:
        return "", "NULL"

    values = ", ".join(
        "1" if item else "0" for item in integral_ir.enabled_coefficients
    )
    size = len(integral_ir.enabled_coefficients)
    init = f"bool enabled_coefficients_{factory_name}[{size}] = {{{values}}};"
    return init, f"enabled_coefficients_{factory_name}"


def _standard_body(
    integral_ir: Any,
    domain: Any,
    options: dict[str, Any],
) -> str:
    """Generate a standard FFCx tabulate_tensor body for one integral."""
    backend = FFCXBackend(integral_ir, options)
    generator = StandardIntegralGenerator(integral_ir, backend)
    parts = generator.generate(domain)
    formatter = Formatter(options["scalar_type"])
    return formatter(parts)


def _tabulate_tensor_functions(
    *,
    mode: RuntimeIntegralMode,
    factory_name: str,
    scalar: str,
    geom: str,
    local_index_expr: str,
    table_preparation: str,
    body: str,
    standard_body: str,
) -> str:
    """Return tabulate_tensor functions for one generated integral."""
    runtime_kwargs = {
        "factory_name": factory_name,
        "scalar": scalar,
        "geom": geom,
        "local_index_expr": local_index_expr,
        "form_context": (
            "  const runintgen_form_context* form = data->form;\n"
            "  const runintgen_basix_element* elements = "
            "(form == 0 ? 0 : form->elements);"
            if table_preparation
            else ""
        ),
        "runtime_points": (
            "  const double* rt_points = rule->points;\n"
            if "rt_points" in body
            else ""
        ),
        "table_preparation": table_preparation,
        "body": body,
    }
    if mode is RuntimeIntegralMode.RUNTIME:
        return factory_runtime_tabulate_tensor.format(
            static_prefix="",
            suffix="",
            **runtime_kwargs,
        )

    if mode is RuntimeIntegralMode.MIXED:
        return "\n".join(
            [
                factory_runtime_tabulate_tensor.format(
                    static_prefix="static ",
                    suffix="_runtime",
                    **runtime_kwargs,
                ),
                factory_standard_tabulate_tensor.format(
                    factory_name=factory_name,
                    scalar=scalar,
                    geom=geom,
                    standard_body=standard_body,
                ),
                factory_mixed_tabulate_tensor.format(
                    factory_name=factory_name,
                    scalar=scalar,
                    geom=geom,
                    local_index_expr=local_index_expr,
                ),
            ]
        )

    raise ValueError(f"Unsupported runtime integral mode: {mode!r}.")


def _tabulate_tensor_initializers(
    scalar_type: str, factory_name: str
) -> dict[str, str]:
    """Return UFCx tabulate function pointer initializers."""
    code = {
        "tabulate_tensor_float32": ".tabulate_tensor_float32 = NULL,",
        "tabulate_tensor_float64": ".tabulate_tensor_float64 = NULL,",
        "tabulate_tensor_complex64": ".tabulate_tensor_complex64 = NULL,",
        "tabulate_tensor_complex128": ".tabulate_tensor_complex128 = NULL,",
    }
    if sys.platform.startswith("win32"):
        code["tabulate_tensor_complex64"] = ""
        code["tabulate_tensor_complex128"] = ""
    code[f"tabulate_tensor_{scalar_type}"] = (
        f".tabulate_tensor_{scalar_type} = tabulate_tensor_{factory_name},"
    )
    return code


def _basix_hash(element: Any) -> int | None:
    """Return a Basix element hash when available."""
    basix_element = element
    if hasattr(element, "_element"):
        basix_element = element._element
    elif hasattr(element, "basix_element"):
        basix_element = element.basix_element

    if hasattr(basix_element, "basix_hash"):
        return int(basix_element.basix_hash())
    if hasattr(basix_element, "hash"):
        return int(basix_element.hash())
    return None


def _form_element_indices_by_hash(
    form_metadata: FormRuntimeMetadata | None,
) -> dict[int, int]:
    """Map Basix hashes to form-level element indices."""
    if form_metadata is None:
        return {}

    indices: dict[int, int] = {}
    for element in form_metadata.unique_elements:
        element_hash = _basix_hash(element.element)
        if element_hash is not None:
            indices[element_hash] = element.form_elem_index
    return indices


def _apply_form_element_indices(
    runtime_tables: list[Any],
    element_indices_by_hash: dict[int, int],
) -> list[Any]:
    """Return runtime table metadata with form-level element indices."""
    if not element_indices_by_hash:
        return runtime_tables

    resolved = []
    for table in runtime_tables:
        if table.element_hash in element_indices_by_hash:
            resolved.append(
                replace(
                    table,
                    element_index=element_indices_by_hash[table.element_hash],
                )
            )
        else:
            raise ValueError(
                "Could not resolve runtime table "
                f"{table.name!r} with Basix hash {table.element_hash!r} "
                "to a form-level element index."
            )
    return resolved


def generate_C_runtime_kernels(
    analysis: RuntimeAnalysisInfo,
    options: dict[str, Any],
    form_metadata: FormRuntimeMetadata | None = None,
) -> list[RuntimeKernelInfo]:
    """Generate C kernels for all runtime integrals in the analysis."""
    ir = analysis.ir
    ffcx_options = _runtime_options(options)

    scalar_type = np.dtype(ffcx_options.get("scalar_type", np.float64)).name
    geometry_type = np.dtype(dtype_to_scalar_dtype(scalar_type)).name
    scalar_c = dtype_to_c_type(scalar_type)
    geom_c = dtype_to_c_type(geometry_type)

    subdomain_by_key = _subdomain_ids_by_integral(analysis)
    standard_integrals = (
        _integrals_by_key(analysis.standard_ir)
        if analysis.standard_ir is not None
        else {}
    )
    element_indices_by_hash = _form_element_indices_by_hash(form_metadata)
    generator = RuntimeIntegralGenerator(ffcx_options)

    kernels: list[RuntimeKernelInfo] = []
    ir_type_counts: dict[str, int] = {}

    for integral_ir in ir.integrals:
        integral_type = integral_ir.expression.integral_type
        ir_index = ir_type_counts.get(integral_type, 0)
        ir_type_counts[integral_type] = ir_index + 1
        key = (integral_type, ir_index)

        if key not in analysis.integral_infos:
            continue

        subdomain_id = subdomain_by_key.get(key, 0)
        mode = analysis.integral_infos[key].mode
        domains = _domains_for_integral(integral_ir)
        for domain_id, domain in enumerate(domains):
            generated = generator.generate_runtime(integral_ir, domain)
            runtime_tables = _apply_form_element_indices(
                generated.runtime_tables, element_indices_by_hash
            )
            element_requests = _element_table_requests(runtime_tables)
            func_name = _kernel_name(integral_ir, domain)
            kernel_id = ir_index * 1024 + domain_id
            enabled_coefficients_init, enabled_coefficients = _enabled_coefficients(
                integral_ir, func_name
            )
            tabulate_tensor = _tabulate_tensor_initializers(scalar_type, func_name)
            standard_integral_ir = standard_integrals.get(key, integral_ir)
            local_index_expr = _local_index_expr(integral_type)
            table_preparation = _table_preparation(element_requests)
            standard_body = (
                _standard_body(standard_integral_ir, domain, ffcx_options)
                if mode is RuntimeIntegralMode.MIXED
                else ""
            )

            c_def = factory_runtime_kernel.format(
                factory_name=func_name,
                scalar=scalar_c,
                geom=geom_c,
                enabled_coefficients_init=enabled_coefficients_init,
                enabled_coefficients=enabled_coefficients,
                table_requests=_table_requests(element_requests),
                tabulate_tensor_functions=_tabulate_tensor_functions(
                    mode=mode,
                    factory_name=func_name,
                    scalar=scalar_c,
                    geom=geom_c,
                    local_index_expr=local_index_expr,
                    table_preparation=table_preparation,
                    body=generated.body,
                    standard_body=standard_body,
                ),
                tabulate_tensor_float32=tabulate_tensor["tabulate_tensor_float32"],
                tabulate_tensor_float64=tabulate_tensor["tabulate_tensor_float64"],
                tabulate_tensor_complex64=tabulate_tensor[
                    "tabulate_tensor_complex64"
                ],
                tabulate_tensor_complex128=tabulate_tensor[
                    "tabulate_tensor_complex128"
                ],
                needs_facet_permutations=(
                    "true"
                    if integral_ir.expression.needs_facet_permutations
                    else "false"
                ),
                coordinate_element_hash=(
                    f"UINT64_C({integral_ir.expression.coordinate_element_hash})"
                ),
                domain=int(domain),
            )

            c_decl = factory_runtime_kernel_decl.format(
                factory_name=func_name,
            )

            max_derivative_by_slot = {
                request.slot: request.max_derivative_order
                for request in element_requests
            }
            table_info = [
                {
                    **table.to_dict(),
                    "element_max_derivative_order": max_derivative_by_slot[
                        table.slot
                    ],
                }
                for table in runtime_tables
            ]
            table_slots = {table.name: table.slot for table in runtime_tables}

            kernels.append(
                RuntimeKernelInfo(
                    name=func_name,
                    integral_type=integral_type,
                    subdomain_id=subdomain_id,
                    ir_index=ir_index,
                    c_declaration=c_decl,
                    c_definition=c_def,
                    tensor_shape=tuple(integral_ir.expression.tensor_shape),
                    table_info=table_info,
                    table_slots=table_slots,
                    domain=domain.name,
                    kernel_id=kernel_id,
                    scalar_type=scalar_type,
                    geometry_type=geometry_type,
                    base_name=integral_ir.expression.name,
                    mode=mode.value,
                )
            )

    return kernels


def generate_C_combined_kernels(
    analysis: RuntimeAnalysisInfo,
    options: dict[str, Any],
    form_metadata: FormRuntimeMetadata | None = None,
) -> list[RuntimeKernelInfo]:
    """Generate C kernels for standard, runtime, and mixed integrals."""
    ffcx_options = _runtime_options(options)
    runtime_keys = set(analysis.integral_infos)
    runtime_kernels = {
        (kernel.integral_type, kernel.ir_index, kernel.domain): kernel
        for kernel in generate_C_runtime_kernels(analysis, options, form_metadata)
    }

    runtime_integrals = _integrals_by_key(analysis.ir)
    subdomain_by_key = _all_subdomain_ids_by_integral(analysis.standard_ir)

    kernels: list[RuntimeKernelInfo] = []
    ir_type_counts: dict[str, int] = {}
    for integral_ir in analysis.standard_ir.integrals:
        integral_type = integral_ir.expression.integral_type
        ir_index = ir_type_counts.get(integral_type, 0)
        ir_type_counts[integral_type] = ir_index + 1
        key = (integral_type, ir_index)

        source_ir = runtime_integrals[key] if key in runtime_keys else integral_ir
        domains = _domains_for_integral(source_ir)
        for domain_id, domain in enumerate(domains):
            if key in runtime_keys:
                kernel = runtime_kernels[(integral_type, ir_index, domain.name)]
            else:
                kernel = _standard_kernel_info(
                    integral_ir,
                    domain,
                    integral_type=integral_type,
                    ir_index=ir_index,
                    subdomain_id=subdomain_by_key.get(key, -1),
                    options=ffcx_options,
                    kernel_id=ir_index * 1024 + domain_id,
                )
            kernels.append(kernel)

    return kernels


def get_runintgen_data_struct() -> str:
    """Return the C ABI definition required by generated runtime kernels."""
    return runintgen_data_struct
