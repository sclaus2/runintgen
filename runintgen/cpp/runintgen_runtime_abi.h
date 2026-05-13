#ifndef RUNINTGEN_RUNTIME_ABI_H
#define RUNINTGEN_RUNTIME_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
  RUNINTGEN_ELEMENT_TEST = 0,
  RUNINTGEN_ELEMENT_TRIAL = 1,
  RUNINTGEN_ELEMENT_COEFFICIENT = 2,
  RUNINTGEN_ELEMENT_GEOMETRY = 3
} runintgen_element_role;

typedef struct
{
  int form_element_index;
  runintgen_element_role role;
  int role_index;
  uint64_t basix_hash;
  int family;
  int cell_type;
  int degree;
  int value_rank;
  int value_shape[4];
  int block_size;
  int discontinuous;
} runintgen_form_element_descriptor;

typedef struct
{
  int num_elements;
  const runintgen_form_element_descriptor* elements;
} runintgen_form_descriptor;

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
  int num_derivatives;
  int num_points;
  int num_dofs;
  int num_components;
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
typedef struct runintgen_form_context runintgen_form_context;
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

struct runintgen_form_context
{
  int num_elements;
  const runintgen_basix_element* elements;
  const runintgen_form_descriptor* descriptor;
  void* scratch;
};

struct runintgen_context
{
  int num_rules;
  const runintgen_quadrature_rule* rules;
  const uint8_t* is_cut;
  const runintgen_form_context* form;
};

#ifdef __cplusplus
}
#endif

#endif // RUNINTGEN_RUNTIME_ABI_H
