"""C templates for runtime kernels.

This module provides C string templates for runtime tabulation functions
that use a clean runtime interface with explicit arguments for
quadrature points, weights, and FE tables.

The runtime kernel signature is designed to be:
1. Compatible with DOLFINx's custom integral infrastructure
2. Easy to call from Python/C++ with runtime-computed quadrature
3. Clear about what data is needed at runtime
4. Support per-cell quadrature rules with table caching
"""

from __future__ import annotations

# C struct definition for runtime data
# This is passed via custom_data pointer
# The caller is responsible for populating this structure
#
# Simplified single-config design:
# - One quadrature rule (points/weights) per runintgen_data
# - Element tables tabulated at those points
# - For per-cell quadrature, create one runintgen_data per unique quadrature rule
#   and use DOLFINx's custom integral infrastructure to route cells appropriately
#
runintgen_data_struct = r"""
// Per-element table information
typedef struct
{
  int ndofs;              // Number of DOFs for this element
  int nderivs;            // Number of derivative levels in table
  const double* table;    // [nderivs * nq * ndofs] - basix tabulation output
  // Access: table[deriv_idx * nq * ndofs + q * ndofs + dof]
  // Derivative ordering follows basix: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
} runintgen_element;

// Runtime data structure (single quadrature configuration)
typedef struct
{
  int nq;                             // Number of quadrature points
  const double* points;               // [nq * tdim] - reference coordinates
  const double* weights;              // [nq] - quadrature weights
  int nelements;                      // Number of unique elements
  const runintgen_element* elements;  // [nelements] array of element info
} runintgen_data;
"""

# Template for runtime kernel function
# The kernel unpacks runintgen_data from custom_data and performs integration
# Placeholders:
#   {factory_name}  - integral name (e.g., "integral_a_cell_triangle")
#   {scalar} - scalar type (double, float, etc.)
#   {geom}   - geometry type (usually same as scalar for real cases)
#   {body}   - generated kernel body
#
# The function is named tabulate_tensor_{factory_name} to match FFCX naming
# convention, enabling use with DOLFINx's standard assembler infrastructure.
#
factory_runtime_kernel = r"""
#include <math.h>
#include <stdint.h>

void tabulate_tensor_{factory_name}(
    {scalar}* restrict A,
    const {scalar}* restrict w,
    const {scalar}* restrict c,
    const {geom}* restrict coordinate_dofs,
    const int* restrict entity_local_index,
    const uint8_t* restrict quadrature_permutation,
    void* restrict custom_data)
{{
  // Unpack runtime data
  const runintgen_data* data = (const runintgen_data*)custom_data;
  const int nq = data->nq;

  // Suppress unused variable warnings for standard signature params
  (void)w; (void)c; (void)quadrature_permutation; (void)entity_local_index;

{body}
}}
"""

# Declaration template for header files
factory_runtime_kernel_decl = r"""void tabulate_tensor_{factory_name}(
    {scalar}* A,
    const {scalar}* w,
    const {scalar}* c,
    const {geom}* coordinate_dofs,
    const int* entity_local_index,
    const uint8_t* quadrature_permutation,
    void* custom_data);
"""

# Template for the table info structure that describes required tables
# This is generated at compile time to tell the runtime what tables to provide
table_info_struct = r"""
typedef struct
{
  int index;              // Index in the rtables array
  const char* role;       // "argument", "coefficient", "jacobian", "coordinate"
  int terminal_index;     // Argument number or coefficient index
  int component;          // Flat component index
  int deriv_order;        // Sum of derivative counts
  int deriv_x;            // Derivative count in x direction
  int deriv_y;            // Derivative count in y direction
  int ndofs;              // Number of DOFs in the table
} runintgen_table_info;
"""
