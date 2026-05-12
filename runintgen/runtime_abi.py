"""C ABI declarations for runtime integral kernels.

The generated kernels are C. They cannot call C++ Basix methods directly, so
``custom_data`` carries reference quadrature rules plus opaque Basix element
handles. Each element exposes a small C function pointer that tabulates one
generated FFCx table-request slot into wrapper-owned scratch.
"""

from __future__ import annotations

RUNTIME_ABI_TYPES = r"""
typedef struct
{
  int nq;
  int tdim;
  const double* points;
  const double* weights;
} runintgen_quadrature_rule;

typedef struct
{
  const double* values;
  int num_permutations;
  int num_entities;
  int num_points;
  int num_dofs;
} runintgen_table_view;

typedef struct
{
  int slot;
  int element_index;
  int derivative_order;
  int derivative_counts[4];
  int flat_component;
  int num_permutations;
  int num_entities;
  int num_dofs;
  int block_size;
  int offset;
  int is_uniform;
  int is_permuted;
} runintgen_table_request;

typedef struct runintgen_basix_element runintgen_basix_element;
typedef struct runintgen_context runintgen_context;

typedef int (*runintgen_element_tabulate_fn)(
    const runintgen_basix_element* element,
    const runintgen_quadrature_rule* rule,
    const runintgen_table_request* request,
    runintgen_table_view* view);

struct runintgen_basix_element
{
  const void* handle;
  runintgen_element_tabulate_fn tabulate;
};

struct runintgen_context
{
  int num_rules;
  const runintgen_quadrature_rule* rules;
  int num_elements;
  const runintgen_basix_element* elements;
  void* scratch;
};
"""

RUNTIME_ABI_CDEF = RUNTIME_ABI_TYPES

RUNTIME_ABI_SOURCE = RUNTIME_ABI_TYPES
