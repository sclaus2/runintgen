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
# Design for per-cell quadrature with caching:
# - Multiple "quadrature configurations" can be defined
# - Each configuration has its own quadrature points/weights and tabulated tables
# - A cell-to-config map allows different cells to use different quadrature
# - Cells sharing the same quadrature share the same tabulated tables (caching)
#
# For single-config usage (all cells same quadrature):
# - Set num_configs=1, active_config=0, cell_config_map=NULL
#
# For multi-config usage (per-cell quadrature):
# - Set num_configs to number of unique quadrature rules
# - Set cell_config_map to map cell indices to config indices
# - The kernel uses entity_local_index[0] as the cell index
#
runintgen_data_struct = r"""
// Per-element table information for a single quadrature configuration
typedef struct
{
  int ndofs;              // Number of DOFs for this element
  int nderivs;            // Number of derivative levels in table
  const double* table;    // [nderivs, nq, ndofs] - basix tabulation output
  // Access: table[deriv_idx * nq * ndofs + q * ndofs + dof]
  // Derivative ordering follows basix: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
} runintgen_element;

// A single quadrature configuration (points, weights, and tabulated tables)
typedef struct
{
  // Quadrature data for this configuration
  int nq;                             // Number of quadrature points
  const double* points;               // [nq * tdim] - reference coordinates
  const double* weights;              // [nq] - quadrature weights

  // Per-element table data (tabulated at this quadrature's points)
  int nelements;                      // Number of unique elements
  const runintgen_element* elements;  // [nelements] array of element info
} runintgen_quadrature_config;

// Main runtime data structure
// Supports both single-config and multi-config (per-cell quadrature) modes
typedef struct
{
  // Number of quadrature configurations
  int num_configs;

  // Array of quadrature configurations
  // configs[i] contains quadrature rule i and its tabulated tables
  const runintgen_quadrature_config* configs;

  // For single-config mode: set active_config to the config index (0)
  // For multi-config mode: set to -1 and use cell_config_map
  int active_config;

  // Per-cell configuration map (optional, for multi-config mode)
  // If non-NULL: cell_config_map[cell_index] gives the config index for that cell
  // If NULL: use active_config for all cells
  const int* cell_config_map;

} runintgen_data;

// Legacy single-config alias for backwards compatibility
// This is equivalent to runintgen_quadrature_config
typedef runintgen_quadrature_config runintgen_data_single;
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
# The kernel supports both single-config and multi-config modes:
# - Single-config: data->active_config >= 0, uses that config for all cells
# - Multi-config: data->active_config < 0, uses cell_config_map[cell_index]
#
# In multi-config mode, entity_local_index[0] is used as the cell index
# to look up the appropriate quadrature configuration.
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
  const runintgen_data* rdata = (const runintgen_data*)custom_data;

  // Select quadrature configuration
  // - If active_config >= 0: use that config (single-config mode)
  // - If active_config < 0: use cell_config_map[cell_index] (multi-config mode)
  int config_idx;
  if (rdata->active_config >= 0) {{
    config_idx = rdata->active_config;
  }} else {{
    // Use entity_local_index[0] as cell index to look up config
    int cell_idx = entity_local_index[0];
    config_idx = rdata->cell_config_map[cell_idx];
  }}

  const runintgen_quadrature_config* data = &rdata->configs[config_idx];
  const int nq = data->nq;

  // Suppress unused variable warnings for standard signature params
  (void)w; (void)c; (void)quadrature_permutation;

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
