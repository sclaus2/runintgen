"""C templates for runtime kernels."""

from __future__ import annotations

from ...runtime_abi import RUNTIME_ABI_SOURCE

runintgen_data_struct = RUNTIME_ABI_SOURCE

factory_runtime_kernel = r"""
// Code for runtime integral {factory_name}

{enabled_coefficients_init}

{table_requests}

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
  if (data == 0 || data->rules == 0)
    return;
  if (local_index < 0 || local_index >= data->num_rules)
    return;
  const runintgen_quadrature_rule* rule = &data->rules[local_index];
  const runintgen_basix_element* elements = data->elements;

  const int rt_nq = rule->nq;
  const double* rt_weights = rule->weights;
  const double* rt_points = rule->points;

{table_preparation}

  (void)w;
  (void)c;
  (void)entity_local_index;
  (void)quadrature_permutation;
  (void)elements;
  (void)rt_points;

{body}
}}

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

factory_runtime_kernel_decl = "extern ufcx_integral {factory_name};\n"
