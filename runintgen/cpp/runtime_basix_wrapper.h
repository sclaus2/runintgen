#ifndef RUNINTGEN_RUNTIME_BASIX_WRAPPER_H
#define RUNINTGEN_RUNTIME_BASIX_WRAPPER_H

#include "runintgen_runtime_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct runintgen_form_context_owner runintgen_form_context_owner;

runintgen_form_context_owner* runintgen_form_context_create(
    const runintgen_form_descriptor* descriptor);

void runintgen_form_context_destroy(runintgen_form_context_owner* owner);

const runintgen_form_context* runintgen_form_context_get(
    const runintgen_form_context_owner* owner);

int runintgen_form_context_set_basix_element(
    runintgen_form_context_owner* owner,
    int form_element_index,
    const void* basix_element);

int runintgen_form_context_set_element(
    runintgen_form_context_owner* owner,
    int form_element_index,
    const void* handle,
    runintgen_element_tabulate_fn tabulate);

int runintgen_basix_tabulate(
    const runintgen_basix_element* element,
    const runintgen_quadrature_rule* rule,
    const runintgen_table_request* request,
    runintgen_table_view* view);

#ifdef __cplusplus
}
#endif

#endif  // RUNINTGEN_RUNTIME_BASIX_WRAPPER_H
