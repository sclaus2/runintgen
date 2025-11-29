"""C templates for runtime kernels.

This module provides C string templates for runtime tabulation functions
that match standard UFCx signatures, but interpret custom_data as a
runintgen_data struct.
"""

from __future__ import annotations

# C struct definition for runtime data
# This is emitted once per translation unit
runintgen_data_struct = r"""
typedef struct
{
  int num_parts;                    // Number of quadrature partitions
  const int* num_q_per_part;        // [num_parts] - points per partition
  const double* points;             // [sum_q * gdim] - quadrature points
  const double* weights;            // [sum_q] - quadrature weights
  const double* FE;                 // Flattened FE basis tables
  const size_t* FE_shape;           // Shape description for FE tables
} runintgen_data;
"""

# Template for runtime kernel function
# Placeholders:
#   {fname}  - function name
#   {scalar} - scalar type (double, float, etc.)
#   {geom}   - geometry type (usually same as scalar for real cases)
#   {body}   - generated kernel body
factory_runtime_kernel = r"""
void {fname}(
    {scalar}* restrict A,
    const {scalar}* restrict w,
    const {scalar}* restrict c,
    const {geom}* restrict coordinate_dofs,
    const int* restrict entity_local_index,
    const uint8_t* restrict quadrature_permutation,
    const int* restrict num_points,
    const {geom}* restrict points,
    const {geom}* restrict weights,
    void* restrict custom_data)
{{
  // Cast custom_data to runintgen_data struct
  const runintgen_data* data = (const runintgen_data*)custom_data;
  const int num_parts = data->num_parts;
  const int* num_q_per_part = data->num_q_per_part;
  const {geom}* rpoints = data->points;
  const {geom}* rweights = data->weights;
  const {scalar}* FE = data->FE;
  const size_t* FE_shape = data->FE_shape;

{body}
}}
"""

# Declaration template for header files
factory_runtime_kernel_decl = r"""void {fname}(
    {scalar}* A,
    const {scalar}* w,
    const {scalar}* c,
    const {geom}* coordinate_dofs,
    const int* entity_local_index,
    const uint8_t* quadrature_permutation,
    const int* num_points,
    const {geom}* points,
    const {geom}* weights,
    void* custom_data);
"""
