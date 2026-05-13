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
  if (data == 0 || data->rules == 0)
    return;
  if (local_index < 0 || local_index >= data->num_rules)
    return;
  const runintgen_quadrature_rule* rule = &data->rules[local_index];
  const runintgen_form_context* form = data->form;
  const runintgen_basix_element* elements = (form == 0 ? 0 : form->elements);

  const int rt_nq = rule->nq;
  const double* rt_weights = rule->weights;
  const double* rt_points = rule->points;

{table_preparation}

  (void)w;
  (void)c;
  (void)entity_local_index;
  (void)quadrature_permutation;
  (void)form;
  (void)elements;
  (void)rt_points;

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
  if (data != 0 && data->is_cut != 0 && local_index >= 0
      && local_index < data->num_rules && data->is_cut[local_index] != 0)
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
