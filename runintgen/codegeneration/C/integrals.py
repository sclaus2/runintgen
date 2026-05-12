"""C code generation glue for runtime integrals."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

import numpy as np
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.options import get_options

from ...analysis import RuntimeAnalysisInfo
from ...form_metadata import FormRuntimeMetadata
from ...runtime_api import RuntimeKernelInfo
from ..runtime_integrals import RuntimeIntegralGenerator
from .integrals_template import (
    factory_runtime_kernel,
    factory_runtime_kernel_decl,
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


def _domains_for_integral(integral_ir: Any) -> list[Any]:
    """Return sorted domains present in an FFCx integral IR."""
    domains = {cell for cell, _ in integral_ir.expression.integrand.keys()}
    return sorted(domains, key=lambda cell: cell.name)


def _kernel_name(
    integral_ir: Any,
    domain: Any,
) -> str:
    """Return FFCx's UFCx integral object name."""
    return f"{integral_ir.expression.name}_{domain.name}"


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


def _padded_derivative_counts(counts: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Pad derivative counts to the fixed ABI size."""
    padded = list(counts[:4])
    padded.extend([0] * (4 - len(padded)))
    return tuple(padded)  # type: ignore[return-value]


def _max_source_dof(table: Any) -> int:
    """Return the largest raw Basix dof index a table access can request."""
    num_dofs = int(table.shape[3])
    if num_dofs <= 0:
        return -1

    offset = -1 if table.offset is None else int(table.offset)
    block_size = -1 if table.block_size is None else int(table.block_size)
    last = num_dofs - 1
    if offset >= 0 and block_size > 0:
        return offset + block_size * last
    if offset >= 0:
        return offset + last
    return last


def _table_requests(runtime_tables: list[Any]) -> str:
    """Generate static C table requests for runtime FE table slots."""
    lines = []
    for table in runtime_tables:
        counts = _padded_derivative_counts(table.derivative_counts)
        derivative_order = sum(table.derivative_counts)
        flat_component = -1 if table.flat_component is None else table.flat_component
        offset = -1 if table.offset is None else table.offset
        block_size = -1 if table.block_size is None else table.block_size
        lines.append(
            f"static const runintgen_table_request "
            f"rt_request_{table.name} = {{\n"
            f"  .slot = {table.slot},\n"
            f"  .element_index = {table.element_index},\n"
            f"  .derivative_order = {derivative_order},\n"
            f"  .derivative_counts = "
            f"{{{counts[0]}, {counts[1]}, {counts[2]}, {counts[3]}}},\n"
            f"  .flat_component = {flat_component},\n"
            f"  .num_permutations = {table.shape[0]},\n"
            f"  .num_entities = {table.shape[1]},\n"
            f"  .num_dofs = {table.shape[3]},\n"
            f"  .block_size = {block_size},\n"
            f"  .offset = {offset},\n"
            f"  .is_uniform = {int(table.is_uniform)},\n"
            f"  .is_permuted = {int(table.is_permuted)},\n"
            f"}};"
        )
    return "\n".join(lines)


def _table_preparation(runtime_tables: list[Any]) -> str:
    """Generate in-kernel Basix wrapper calls for runtime FE table slots."""
    lines = []
    for table in runtime_tables:
        component = 0 if table.flat_component is None else int(table.flat_component)
        max_source_dof = _max_source_dof(table)
        lines.extend(
            [
                f"  if (form == 0 || elements == 0 || "
                f"form->num_elements <= {table.element_index})",
                "    return;",
                f"  if (elements[{table.element_index}].tabulate == 0)",
                "    return;",
                f"  runintgen_table_view rt_view_{table.slot};",
                f"  if (elements[{table.element_index}].tabulate(",
                f"          &elements[{table.element_index}], rule,",
                f"          &rt_request_{table.name}, &rt_view_{table.slot}) != 0)",
                "    return;",
                f"  const double* {table.c_symbol} = rt_view_{table.slot}.values;",
                f"  if ({table.c_symbol} == 0)",
                "    return;",
                f"  const int {table.c_symbol}_num_dofs = "
                f"rt_view_{table.slot}.num_dofs;",
                f"  const int {table.c_symbol}_num_components = "
                f"rt_view_{table.slot}.num_components;",
                f"  if (rt_view_{table.slot}.num_points != rt_nq || "
                f"rt_view_{table.slot}.num_derivatives <= {table.derivative_index} "
                f"|| {table.c_symbol}_num_dofs <= {max_source_dof} "
                f"|| {table.c_symbol}_num_components <= {component})",
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
        domains = _domains_for_integral(integral_ir)
        for domain_id, domain in enumerate(domains):
            generated = generator.generate_runtime(integral_ir, domain)
            runtime_tables = _apply_form_element_indices(
                generated.runtime_tables, element_indices_by_hash
            )
            func_name = _kernel_name(integral_ir, domain)
            kernel_id = ir_index * 1024 + domain_id
            enabled_coefficients_init, enabled_coefficients = _enabled_coefficients(
                integral_ir, func_name
            )
            tabulate_tensor = _tabulate_tensor_initializers(scalar_type, func_name)

            c_def = factory_runtime_kernel.format(
                factory_name=func_name,
                scalar=scalar_c,
                geom=geom_c,
                enabled_coefficients_init=enabled_coefficients_init,
                enabled_coefficients=enabled_coefficients,
                local_index_expr=_local_index_expr(integral_type),
                table_requests=_table_requests(runtime_tables),
                table_preparation=_table_preparation(runtime_tables),
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
                body=generated.body,
            )

            c_decl = factory_runtime_kernel_decl.format(
                factory_name=func_name,
            )

            table_info = [table.to_dict() for table in runtime_tables]
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
                )
            )

    return kernels


def get_runintgen_data_struct() -> str:
    """Return the C ABI definition required by generated runtime kernels."""
    return runintgen_data_struct
