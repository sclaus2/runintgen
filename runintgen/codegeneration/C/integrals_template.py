"""C templates for runtime kernels."""

from __future__ import annotations

from ...cpp_headers import runtime_abi_cdef

runintgen_data_struct = runtime_abi_cdef()

factory_runtime_kernel = r"""
// Code for runtime integral {factory_name}

{enabled_coefficients_init}

{table_requests}

{tabulate_tensor_functions}

ufcx_integral {factory_name} =
{{
  .enabled_coefficients = {enabled_coefficients},
  {tabulate_tensor_float32}
  {tabulate_tensor_float64}
  {tabulate_tensor_complex64}
  {tabulate_tensor_complex128}
  .needs_facet_permutations = {needs_facet_permutations},
  .coordinate_element_hash = {coordinate_element_hash},
  .domain = {domain},
}};

// End of code for runtime integral {factory_name}
"""

factory_runtime_tabulate_tensor = r"""
{static_prefix}void tabulate_tensor_{factory_name}{suffix}({scalar}* restrict A,
                                    const {scalar}* restrict w,
                                    const {scalar}* restrict c,
                                    const {geom}* restrict coordinate_dofs,
                                    const int* restrict entity_local_index,
                                    const uint8_t* restrict quadrature_permutation,
                                    void* custom_data)
{{
  const runintgen_context* data = (const runintgen_context*)custom_data;
  const int local_index = {local_index_expr};
  if (data == 0 || data->quadrature == 0 || data->entities == 0)
    return;
  const runintgen_quadrature_rules* quadrature = data->quadrature;
  const runintgen_entity_map* entities = data->entities;
  if (local_index < 0 || local_index >= entities->num_entities)
    return;
  if (entities->rule_indices == 0 || quadrature->offsets == 0
      || quadrature->points == 0 || quadrature->weights == 0)
    return;
  const int rule_index = entities->rule_indices[local_index];
  if (rule_index < 0 || rule_index >= quadrature->num_rules)
    return;
  const int64_t q0 = quadrature->offsets[rule_index];
  const int64_t q1 = quadrature->offsets[rule_index + 1];
  if (q0 < 0 || q1 < q0)
    return;
  const runintgen_quadrature_rule rule_storage = {{
      .nq = (int)(q1 - q0),
      .tdim = quadrature->tdim,
      .points = quadrature->points + q0 * quadrature->tdim,
      .weights = quadrature->weights + q0,
  }};
  const runintgen_quadrature_rule* rule = &rule_storage;
{form_context}

  const int rt_nq = rule->nq;
  const double* rt_weights = rule->weights;
{runtime_points}
{table_preparation}

{body}
}}
"""

factory_standard_tabulate_tensor = r"""
static void tabulate_tensor_{factory_name}_standard({scalar}* restrict A,
                                    const {scalar}* restrict w,
                                    const {scalar}* restrict c,
                                    const {geom}* restrict coordinate_dofs,
                                    const int* restrict entity_local_index,
                                    const uint8_t* restrict quadrature_permutation,
                                    void* custom_data)
{{
{standard_body}
}}
"""

factory_mixed_tabulate_tensor = r"""
void tabulate_tensor_{factory_name}({scalar}* restrict A,
                                    const {scalar}* restrict w,
                                    const {scalar}* restrict c,
                                    const {geom}* restrict coordinate_dofs,
                                    const int* restrict entity_local_index,
                                    const uint8_t* restrict quadrature_permutation,
                                    void* custom_data)
{{
  const runintgen_context* data = (const runintgen_context*)custom_data;
  const int local_index = {local_index_expr};
  const runintgen_entity_map* entities = (data == 0 ? 0 : data->entities);
  if (entities != 0 && entities->is_cut != 0 && local_index >= 0
      && local_index < entities->num_entities
      && entities->is_cut[local_index] != 0)
  {{
    tabulate_tensor_{factory_name}_runtime(
        A, w, c, coordinate_dofs, entity_local_index, quadrature_permutation,
        custom_data);
    return;
  }}

  tabulate_tensor_{factory_name}_standard(
      A, w, c, coordinate_dofs, entity_local_index, quadrature_permutation,
      custom_data);
}}
"""

factory_runtime_kernel_decl = "extern ufcx_integral {factory_name};\n"
