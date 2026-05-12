#include "runtime_basix_wrapper.h"
#include "runtime_tabulation_helpers.h"

#include <basix/finite-element.h>

#include <cstddef>
#include <deque>
#include <exception>
#include <span>
#include <vector>

namespace
{
struct BasixElementHandle
{
  const basix::FiniteElement<double>* element = nullptr;
  std::deque<std::vector<double>> basis_by_slot;
};
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
  if (rule->nq < 0 || rule->tdim <= 0 || rule->points == nullptr)
    return 2;
  if (request->derivative_order < 0)
    return 2;
  if (request->is_permuted)
    return 3;

  auto* handle
      = static_cast<BasixElementHandle*>(const_cast<void*>(element->handle));
  if (handle->element == nullptr)
    return 4;

  try
  {
    const basix::FiniteElement<double>& finite_element = *handle->element;
    const auto shape = finite_element.tabulate_shape(
        static_cast<std::size_t>(request->derivative_order),
        static_cast<std::size_t>(rule->nq));

    if (request->slot < 0)
      return 6;
    const auto slot = static_cast<std::size_t>(request->slot);
    if (slot >= handle->basis_by_slot.size())
      handle->basis_by_slot.resize(slot + 1);

    std::vector<double>& basis = handle->basis_by_slot[slot];
    basis.resize(runintgen::detail::table_shape_size(shape));
    finite_element.tabulate(
        request->derivative_order,
        std::span<const double>(
            rule->points, static_cast<std::size_t>(rule->nq * rule->tdim)),
        {static_cast<std::size_t>(rule->nq),
         static_cast<std::size_t>(rule->tdim)},
        std::span<double>(basis));

    const int derivative
        = runintgen::detail::derivative_index(request->derivative_counts,
                                             rule->tdim);
    if (derivative < 0 || static_cast<std::size_t>(derivative) >= shape[0])
      return 5;

    const int num_dofs = request->num_dofs;
    if (num_dofs <= 0)
      return 6;

    const std::size_t component
        = request->flat_component < 0
              ? 0
              : static_cast<std::size_t>(request->flat_component);
    if (component >= shape[3])
      return 7;

    if (runintgen::detail::source_dof(*request, num_dofs - 1) >= shape[2])
      return 8;

    view->values = basis.data();
    view->num_derivatives = static_cast<int>(shape[0]);
    view->num_points = static_cast<int>(shape[1]);
    view->num_dofs = static_cast<int>(shape[2]);
    view->num_components = static_cast<int>(shape[3]);
    return 0;
  }
  catch (const std::exception&)
  {
    return 9;
  }
  catch (...)
  {
    return 10;
  }
}
