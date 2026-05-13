#include "runtime_basix_wrapper.h"
#include "runtime_tabulation_helpers.h"

#include <basix/finite-element.h>

#include <cstddef>
#include <deque>
#include <vector>

namespace
{
struct BasixElementHandle
{
  const basix::FiniteElement<double>* element = nullptr;
  int block_size = 1;
  std::deque<std::vector<double>> basis_by_slot;
};

int descriptor_block_size(const runintgen_form_descriptor* descriptor,
                          int form_element_index)
{
  if (descriptor == nullptr || descriptor->elements == nullptr
      || descriptor->num_elements <= 0)
    return 1;

  for (int i = 0; i < descriptor->num_elements; ++i)
  {
    const runintgen_form_element_descriptor& element = descriptor->elements[i];
    if (element.form_element_index == form_element_index)
      return element.block_size > 1 ? element.block_size : 1;
  }
  return 1;
}
} // namespace

struct runintgen_form_context_owner
{
  explicit runintgen_form_context_owner(
      const runintgen_form_descriptor* descriptor)
      : descriptor(descriptor), handles(descriptor == nullptr ? 0
                                                              : descriptor->num_elements),
        elements(handles.size())
  {
    context.num_elements = static_cast<int>(elements.size());
    context.elements = elements.data();
    context.descriptor = descriptor;
    context.scratch = nullptr;
  }

  const runintgen_form_descriptor* descriptor = nullptr;
  std::vector<BasixElementHandle> handles;
  std::vector<runintgen_basix_element> elements;
  runintgen_form_context context{};
};

extern "C" runintgen_form_context_owner* runintgen_form_context_create(
    const runintgen_form_descriptor* descriptor)
{
  try
  {
    return new runintgen_form_context_owner(descriptor);
  }
  catch (...)
  {
    return nullptr;
  }
}

extern "C" void runintgen_form_context_destroy(
    runintgen_form_context_owner* owner)
{
  delete owner;
}

extern "C" const runintgen_form_context* runintgen_form_context_get(
    const runintgen_form_context_owner* owner)
{
  return owner == nullptr ? nullptr : &owner->context;
}

extern "C" int runintgen_form_context_set_basix_element(
    runintgen_form_context_owner* owner, int form_element_index,
    const void* basix_element)
{
  if (owner == nullptr || basix_element == nullptr || form_element_index < 0)
    return 1;

  const auto index = static_cast<std::size_t>(form_element_index);
  if (index >= owner->handles.size())
    return 2;

  owner->handles[index].element
      = static_cast<const basix::FiniteElement<double>*>(basix_element);
  owner->handles[index].block_size
      = descriptor_block_size(owner->descriptor, form_element_index);
  return runintgen_form_context_set_element(
      owner, form_element_index, &owner->handles[index],
      &runintgen_basix_tabulate);
}

extern "C" int runintgen_form_context_set_element(
    runintgen_form_context_owner* owner, int form_element_index,
    const void* handle, runintgen_element_tabulate_fn tabulate)
{
  if (owner == nullptr || handle == nullptr || tabulate == nullptr
      || form_element_index < 0)
    return 1;

  const auto index = static_cast<std::size_t>(form_element_index);
  if (index >= owner->elements.size())
    return 2;

  owner->elements[index].handle = handle;
  owner->elements[index].tabulate = tabulate;
  return 0;
}

extern "C" int runintgen_basix_tabulate(
    const runintgen_basix_element* element,
    const runintgen_quadrature_rule* rule,
    const runintgen_table_request* request,
    runintgen_table_view* view)
{
  if (element == nullptr || element->handle == nullptr || rule == nullptr
      || request == nullptr || view == nullptr)
    return 1;

  auto* handle
      = static_cast<BasixElementHandle*>(const_cast<void*>(element->handle));
  if (handle->element == nullptr)
    return 4;

  return runintgen::detail::tabulate_runtime_table(
      *handle->element, rule, request, view,
      [handle](int slot) -> std::vector<double>& {
        const auto index = static_cast<std::size_t>(slot);
        if (index >= handle->basis_by_slot.size())
          handle->basis_by_slot.resize(index + 1);
        return handle->basis_by_slot[index];
      },
      handle->block_size);
}
