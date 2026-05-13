#ifndef RUNINTGEN_RUNTIME_TABULATION_HELPERS_H
#define RUNINTGEN_RUNTIME_TABULATION_HELPERS_H

#include "runintgen_runtime_abi.h"

#include <array>
#include <cstddef>
#include <exception>
#include <span>
#include <utility>
#include <vector>

namespace runintgen::detail
{
inline std::size_t table_shape_size(const std::array<std::size_t, 4>& shape)
{
  return shape[0] * shape[1] * shape[2] * shape[3];
}

template <typename Element>
inline void tabulate_basis(const Element& element, int derivative_order,
                           const runintgen_quadrature_rule& rule,
                           std::vector<double>& basis,
                           const std::array<std::size_t, 4>& shape)
{
  basis.resize(table_shape_size(shape));
  element.tabulate(
      derivative_order,
      std::span<const double>(
          rule.points, static_cast<std::size_t>(rule.nq * rule.tdim)),
      {static_cast<std::size_t>(rule.nq), static_cast<std::size_t>(rule.tdim)},
      std::span<double>(basis));
}

template <typename Element, typename BasisForSlot>
inline int tabulate_runtime_table(const Element& element,
                                  const runintgen_quadrature_rule* rule,
                                  const runintgen_table_request* request,
                                  runintgen_table_view* view,
                                  BasisForSlot&& basis_for_slot,
                                  int block_size = 1)
{
  if (rule == nullptr || request == nullptr || view == nullptr)
    return 1;
  if (rule->nq < 0 || rule->tdim <= 0 || rule->points == nullptr)
    return 2;
  if (request->derivative_order < 0)
    return 2;
  if (request->is_permuted)
    return 3;
  if (request->slot < 0)
    return 6;

  try
  {
    const auto shape = element.tabulate_shape(
        static_cast<std::size_t>(request->derivative_order),
        static_cast<std::size_t>(rule->nq));

    (void)block_size;
    std::vector<double>& basis
        = std::forward<BasisForSlot>(basis_for_slot)(request->slot);
    const std::array<std::size_t, 4> view_shape = shape;
    tabulate_basis(element, request->derivative_order, *rule, basis, shape);

    view->values = basis.data();
    view->num_derivatives = static_cast<int>(view_shape[0]);
    view->num_points = static_cast<int>(view_shape[1]);
    view->num_dofs = static_cast<int>(view_shape[2]);
    view->num_components = static_cast<int>(view_shape[3]);
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
} // namespace runintgen::detail

#endif // RUNINTGEN_RUNTIME_TABULATION_HELPERS_H
