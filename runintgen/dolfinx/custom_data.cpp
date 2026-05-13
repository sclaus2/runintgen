#include "custom_data.h"
#include "../cpp/runtime_tabulation_helpers.h"

#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
struct CoordinateElementHandle
{
  const dolfinx::fem::CoordinateElement<double>* element = nullptr;
  std::deque<std::vector<double>> basis_by_slot;
};

struct FormContextDeleter
{
  void operator()(runintgen_form_context_owner* owner) const noexcept
  {
    runintgen_form_context_destroy(owner);
  }
};

using FormContextOwner
    = std::unique_ptr<runintgen_form_context_owner, FormContextDeleter>;

int coordinate_tabulate(const runintgen_basix_element* element,
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
      = static_cast<CoordinateElementHandle*>(const_cast<void*>(element->handle));
  if (handle->element == nullptr)
    return 4;

  try
  {
    const auto shape = handle->element->tabulate_shape(
        static_cast<std::size_t>(request->derivative_order),
        static_cast<std::size_t>(rule->nq));

    if (request->slot < 0)
      return 6;
    const auto slot = static_cast<std::size_t>(request->slot);
    if (slot >= handle->basis_by_slot.size())
      handle->basis_by_slot.resize(slot + 1);

    std::vector<double>& basis = handle->basis_by_slot[slot];
    basis.resize(runintgen::detail::table_shape_size(shape));
    handle->element->tabulate(
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

template <typename T>
const basix::FiniteElement<double>& matching_basix_element(
    const dolfinx::fem::FunctionSpace<T>& space,
    const runintgen_form_element_descriptor& descriptor)
{
  const auto cell_types = space.mesh()->topology()->cell_types();
  if (cell_types.empty())
    throw std::runtime_error("Cannot resolve Basix element on mesh with no cells.");

  for (std::size_t i = 0; i < cell_types.size(); ++i)
  {
    const auto element = space.elements(static_cast<int>(i));
    if (!element)
      continue;

    const basix::FiniteElement<double>& basix_element
        = element->basix_element();
    if (descriptor.basix_hash == 0
        || basix_element.hash() == descriptor.basix_hash)
      return basix_element;
  }

  throw std::runtime_error(
      "Could not find DOLFINx argument/coefficient element with Basix hash "
      + std::to_string(descriptor.basix_hash) + ".");
}

const dolfinx::fem::CoordinateElement<double>& matching_coordinate_element(
    const dolfinx::mesh::Geometry<double>& geometry,
    const runintgen_form_element_descriptor& descriptor)
{
  const auto& cmaps = geometry.cmaps();
  if (cmaps.empty())
    throw std::runtime_error("Cannot resolve coordinate element on empty geometry.");

  for (const auto& cmap : cmaps)
  {
    if (descriptor.basix_hash == 0 || cmap.hash() == descriptor.basix_hash)
      return cmap;
  }

  throw std::runtime_error(
      "Could not find DOLFINx coordinate element with Basix hash "
      + std::to_string(descriptor.basix_hash) + ".");
}

const basix::FiniteElement<double>& resolve_basix_element(
    const dolfinx::fem::Form<double, double>& form,
    const runintgen_form_element_descriptor& descriptor)
{
  switch (descriptor.role)
  {
  case RUNINTGEN_ELEMENT_TEST:
  case RUNINTGEN_ELEMENT_TRIAL:
  {
    const auto& spaces = form.function_spaces();
    if (descriptor.role_index < 0
        || static_cast<std::size_t>(descriptor.role_index) >= spaces.size())
    {
      throw std::runtime_error("Runintgen descriptor references an invalid "
                               "DOLFINx argument space.");
    }
    if (!spaces[descriptor.role_index])
      throw std::runtime_error("Runintgen descriptor references a null "
                               "DOLFINx argument space.");
    return matching_basix_element(*spaces[descriptor.role_index], descriptor);
  }
  case RUNINTGEN_ELEMENT_COEFFICIENT:
  {
    const auto& coefficients = form.coefficients();
    if (descriptor.role_index < 0
        || static_cast<std::size_t>(descriptor.role_index) >= coefficients.size())
    {
      throw std::runtime_error("Runintgen descriptor references an invalid "
                               "DOLFINx coefficient.");
    }
    if (!coefficients[descriptor.role_index]
        || !coefficients[descriptor.role_index]->function_space())
    {
      throw std::runtime_error("Runintgen descriptor references a coefficient "
                               "without a function space.");
    }
    return matching_basix_element(
        *coefficients[descriptor.role_index]->function_space(), descriptor);
  }
  case RUNINTGEN_ELEMENT_GEOMETRY:
    throw std::runtime_error("Geometry descriptors are not Basix finite "
                             "element descriptors.");
  }

  throw std::runtime_error("Unknown runintgen form element role.");
}
} // namespace

namespace runintgen::dolfinx
{
class CustomData::Impl
{
public:
  Impl(const ::dolfinx::fem::Form<double, double>& form,
       const runintgen_form_descriptor& descriptor,
       std::vector<QuadratureRule> rules,
       std::vector<std::uint8_t> is_cut)
      : _spaces(form.function_spaces()), _coefficients(form.coefficients()),
        _mesh(form.mesh()), _rules_owned(std::move(rules)),
        _is_cut_owned(std::move(is_cut)),
        _form_owner(runintgen_form_context_create(&descriptor))
  {
    if (_form_owner == nullptr)
      throw std::runtime_error("Failed to create runintgen form context.");

    initialise_rules();
    initialise_cut_flags();
    initialise_form_elements(form, descriptor);

    _context.num_rules = static_cast<int>(_rules.size());
    _context.rules = _rules.data();
    _context.is_cut = _is_cut_owned.empty() ? nullptr : _is_cut_owned.data();
    _context.form = runintgen_form_context_get(_form_owner.get());
  }

  const runintgen_context* context() const noexcept { return &_context; }
  void* custom_data() noexcept { return &_context; }

private:
  void initialise_rules()
  {
    _rules.reserve(_rules_owned.size());
    for (const QuadratureRule& rule : _rules_owned)
    {
      if (rule.tdim <= 0)
        throw std::runtime_error("Runintgen quadrature rule has invalid tdim.");
      if (rule.weights.empty())
        throw std::runtime_error("Runintgen quadrature rule has no weights.");
      if (rule.points.size() != rule.weights.size() * rule.tdim)
      {
        throw std::runtime_error("Runintgen quadrature points must have size "
                                 "nq * tdim.");
      }

      _rules.push_back(
          {static_cast<int>(rule.weights.size()), rule.tdim, rule.points.data(),
           rule.weights.data()});
    }
  }

  void initialise_cut_flags()
  {
    if (_is_cut_owned.empty())
      _is_cut_owned.assign(_rules_owned.size(), std::uint8_t{1});
    if (_is_cut_owned.size() != _rules_owned.size())
    {
      throw std::runtime_error(
          "Runintgen cut-flag array must have one entry per quadrature rule.");
    }
  }

  void initialise_form_elements(
      const ::dolfinx::fem::Form<double, double>& form,
      const runintgen_form_descriptor& descriptor)
  {
    if (descriptor.num_elements < 0)
      throw std::runtime_error("Runintgen descriptor has negative element count.");
    if (descriptor.num_elements > 0 && descriptor.elements == nullptr)
      throw std::runtime_error("Runintgen descriptor has null element list.");

    _coordinate_handles.reserve(
        static_cast<std::size_t>(descriptor.num_elements));
    for (int i = 0; i < descriptor.num_elements; ++i)
    {
      const runintgen_form_element_descriptor& element = descriptor.elements[i];
      if (element.form_element_index < 0
          || element.form_element_index >= descriptor.num_elements)
      {
        throw std::runtime_error(
            "Runintgen descriptor has invalid form element index.");
      }

      int error = 0;
      if (element.role == RUNINTGEN_ELEMENT_GEOMETRY)
      {
        if (!_mesh)
          throw std::runtime_error(
              "Cannot resolve geometry element without a mesh.");
        const auto& cmap
            = matching_coordinate_element(_mesh->geometry(), element);
        _coordinate_handles.push_back({&cmap, {}});
        error = runintgen_form_context_set_element(
            _form_owner.get(), element.form_element_index,
            &_coordinate_handles.back(), &coordinate_tabulate);
      }
      else
      {
        const basix::FiniteElement<double>& basix_element
            = resolve_basix_element(form, element);
        error = runintgen_form_context_set_basix_element(
            _form_owner.get(), element.form_element_index, &basix_element);
      }

      if (error != 0)
      {
        throw std::runtime_error(
            "Failed to register DOLFINx element in runintgen form context.");
      }
    }
  }

  std::vector<std::shared_ptr<
      const ::dolfinx::fem::FunctionSpace<double>>>
      _spaces;
  std::vector<std::shared_ptr<
      const ::dolfinx::fem::Function<double, double>>>
      _coefficients;
  std::shared_ptr<const ::dolfinx::mesh::Mesh<double>> _mesh;
  std::vector<QuadratureRule> _rules_owned;
  std::vector<std::uint8_t> _is_cut_owned;
  std::vector<runintgen_quadrature_rule> _rules;
  std::vector<CoordinateElementHandle> _coordinate_handles;
  FormContextOwner _form_owner;
  runintgen_context _context{};
};

CustomData::CustomData(const ::dolfinx::fem::Form<double, double>& form,
                       const runintgen_form_descriptor& descriptor,
                       std::vector<QuadratureRule> rules,
                       std::vector<std::uint8_t> is_cut)
    : _impl(std::make_unique<Impl>(form, descriptor, std::move(rules),
                                   std::move(is_cut)))
{
}

CustomData::~CustomData() = default;

CustomData::CustomData(CustomData&&) noexcept = default;

CustomData& CustomData::operator=(CustomData&&) noexcept = default;

const runintgen_context* CustomData::context() const noexcept
{
  return _impl == nullptr ? nullptr : _impl->context();
}

void* CustomData::custom_data() noexcept
{
  return _impl == nullptr ? nullptr : _impl->custom_data();
}

std::unique_ptr<CustomData>
create_custom_data(const ::dolfinx::fem::Form<double, double>& form,
                   const runintgen_form_descriptor& descriptor,
                   std::vector<QuadratureRule> rules,
                   std::vector<std::uint8_t> is_cut)
{
  return std::make_unique<CustomData>(form, descriptor, std::move(rules),
                                      std::move(is_cut));
}
} // namespace runintgen::dolfinx
