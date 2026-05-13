#include "runintgen_runtime_abi.h"
#include "runtime_tabulation_helpers.h"

#include <basix/finite-element.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace
{
using Array1D = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using Array2D = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;

struct ElementSpec
{
  int family = 0;
  int cell_type = 0;
  int degree = 0;
  int lagrange_variant = 0;
  int dpc_variant = 0;
  bool discontinuous = false;
  std::vector<int> dof_ordering;
  std::string dtype = "float64";
  int block_size = 1;
  std::uint64_t basix_hash = 0;
};

struct BasixElementHandle
{
  basix::FiniteElement<double> element;
  int block_size = 1;
};

struct QuadratureRuleStorage
{
  std::vector<double> points;
  std::vector<double> weights;
};

struct TableCacheKey
{
  const BasixElementHandle* handle = nullptr;
  int slot = 0;

  bool operator==(const TableCacheKey& other) const noexcept
  {
    return handle == other.handle && slot == other.slot;
  }
};

struct TableCacheKeyHash
{
  std::size_t operator()(const TableCacheKey& key) const noexcept
  {
    const auto ptr = reinterpret_cast<std::uintptr_t>(key.handle);
    return std::hash<std::uintptr_t>{}(ptr)
           ^ (std::hash<int>{}(key.slot) + 0x9e3779b9 + (ptr << 6)
              + (ptr >> 2));
  }
};

nb::handle dict_item(nb::dict data, const char* key)
{
  PyObject* item = PyDict_GetItemString(data.ptr(), key);
  if (item == nullptr)
    throw std::runtime_error(std::string("Missing required key '") + key + "'.");
  return nb::handle(item);
}

template <typename T>
T dict_cast(nb::dict data, const char* key)
{
  return nb::cast<T>(dict_item(data, key));
}

template <typename T>
T dict_cast_or(nb::dict data, const char* key, T default_value)
{
  PyObject* item = PyDict_GetItemString(data.ptr(), key);
  if (item == nullptr)
    return default_value;
  return nb::cast<T>(nb::handle(item));
}

ElementSpec parse_element_spec(nb::handle item)
{
  nb::dict data = nb::cast<nb::dict>(item);
  ElementSpec spec;
  spec.family = dict_cast<int>(data, "family");
  spec.cell_type = dict_cast<int>(data, "cell_type");
  spec.degree = dict_cast<int>(data, "degree");
  spec.lagrange_variant = dict_cast_or<int>(data, "lagrange_variant", 0);
  spec.dpc_variant = dict_cast_or<int>(data, "dpc_variant", 0);
  spec.discontinuous = dict_cast_or<bool>(data, "discontinuous", false);
  spec.dof_ordering = dict_cast_or<std::vector<int>>(data, "dof_ordering", {});
  spec.dtype = dict_cast_or<std::string>(data, "dtype", "float64");
  spec.block_size = std::max(1, dict_cast_or<int>(data, "block_size", 1));
  spec.basix_hash = dict_cast_or<std::uint64_t>(data, "basix_hash", 0);
  return spec;
}

basix::FiniteElement<double> create_element(const ElementSpec& spec)
{
  if (spec.dtype != "float64")
  {
    throw std::runtime_error(
        "runintgen Basix runtime currently supports only float64 elements.");
  }
  if (spec.family == static_cast<int>(basix::element::family::custom))
  {
    throw std::runtime_error(
        "Custom Basix elements are not yet supported by the Basix runtime.");
  }

  auto element = basix::create_element<double>(
      static_cast<basix::element::family>(spec.family),
      static_cast<basix::cell::type>(spec.cell_type), spec.degree,
      static_cast<basix::element::lagrange_variant>(spec.lagrange_variant),
      static_cast<basix::element::dpc_variant>(spec.dpc_variant),
      spec.discontinuous, spec.dof_ordering);

  if (spec.basix_hash != 0 && element.hash() != spec.basix_hash)
  {
    throw std::runtime_error("Reconstructed Basix element hash does not match "
                             "runintgen form metadata.");
  }
  return element;
}

void expand_blocked_basis(const std::vector<double>& raw,
                          const std::array<std::size_t, 4>& shape,
                          int block_size, std::vector<double>& expanded)
{
  const std::size_t num_derivatives = shape[0];
  const std::size_t num_points = shape[1];
  const std::size_t scalar_dofs = shape[2];
  const std::size_t scalar_components = shape[3];
  const std::size_t block = static_cast<std::size_t>(block_size);
  const std::size_t out_dofs = scalar_dofs * block;
  const std::size_t out_components = block;

  expanded.assign(num_derivatives * num_points * out_dofs * out_components,
                  0.0);

  for (std::size_t d = 0; d < num_derivatives; ++d)
  {
    for (std::size_t q = 0; q < num_points; ++q)
    {
      for (std::size_t i = 0; i < scalar_dofs; ++i)
      {
        const double value
            = raw[((d * num_points + q) * scalar_dofs + i)
                  * scalar_components];
        for (std::size_t c = 0; c < block; ++c)
        {
          for (std::size_t out_component = 0; out_component < out_components;
               ++out_component)
          {
            const std::size_t out_dof = i * block + c;
            expanded[((d * num_points + q) * out_dofs + out_dof)
                         * out_components
                     + out_component]
                = value;
          }
        }
      }
    }
  }
}

int basix_runtime_tabulate(const runintgen_basix_element* element,
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

  auto* handle = static_cast<const BasixElementHandle*>(element->handle);

  try
  {
    const auto shape = handle->element.tabulate_shape(
        static_cast<std::size_t>(request->derivative_order),
        static_cast<std::size_t>(rule->nq));
    const int derivative
        = runintgen::detail::derivative_index(request->derivative_counts,
                                             rule->tdim);
    if (derivative < 0 || static_cast<std::size_t>(derivative) >= shape[0])
      return 5;

    if (request->slot < 0 || request->num_dofs <= 0)
      return 6;

    thread_local std::unordered_map<TableCacheKey, std::vector<double>,
                                    TableCacheKeyHash>
        cache;
    std::vector<double>& basis = cache[{handle, request->slot}];

    std::array<std::size_t, 4> view_shape = shape;
    if (handle->block_size > 1)
    {
      std::vector<double> raw(runintgen::detail::table_shape_size(shape));
      handle->element.tabulate(
          request->derivative_order,
          std::span<const double>(
              rule->points, static_cast<std::size_t>(rule->nq * rule->tdim)),
          {static_cast<std::size_t>(rule->nq),
           static_cast<std::size_t>(rule->tdim)},
          std::span<double>(raw));

      expand_blocked_basis(raw, shape, handle->block_size, basis);
      view_shape[2] *= static_cast<std::size_t>(handle->block_size);
      view_shape[3] = static_cast<std::size_t>(handle->block_size);
    }
    else
    {
      basis.resize(runintgen::detail::table_shape_size(shape));
      handle->element.tabulate(
          request->derivative_order,
          std::span<const double>(
              rule->points, static_cast<std::size_t>(rule->nq * rule->tdim)),
          {static_cast<std::size_t>(rule->nq),
           static_cast<std::size_t>(rule->tdim)},
          std::span<double>(basis));
    }

    const std::size_t component
        = request->flat_component < 0
              ? 0
              : static_cast<std::size_t>(request->flat_component);
    if (component >= view_shape[3])
      return 7;

    if (runintgen::detail::source_dof(*request, request->num_dofs - 1)
        >= view_shape[2])
      return 8;

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

class CustomData
{
public:
  CustomData(nb::list element_specs, nb::list quadrature, nb::object is_cut)
  {
    initialise_elements(element_specs);
    initialise_rules(quadrature);
    initialise_cut_flags(std::move(is_cut));

    _form.num_elements = static_cast<int>(_abi_elements.size());
    _form.elements = _abi_elements.empty() ? nullptr : _abi_elements.data();
    _form.descriptor = nullptr;
    _form.scratch = nullptr;

    _context.num_rules = static_cast<int>(_abi_rules.size());
    _context.rules = _abi_rules.empty() ? nullptr : _abi_rules.data();
    _context.is_cut = _is_cut.empty() ? nullptr : _is_cut.data();
    _context.form = &_form;
  }

  std::uintptr_t ptr() noexcept
  {
    return reinterpret_cast<std::uintptr_t>(&_context);
  }

  int num_elements() const noexcept
  {
    return static_cast<int>(_abi_elements.size());
  }

  int num_rules() const noexcept { return static_cast<int>(_abi_rules.size()); }

private:
  void initialise_elements(nb::list element_specs)
  {
    const std::size_t n = nb::len(element_specs);
    _handles.reserve(n);
    _abi_elements.resize(n);

    for (std::size_t i = 0; i < n; ++i)
    {
      ElementSpec spec = parse_element_spec(element_specs[i]);
      _handles.push_back({create_element(spec), spec.block_size});
    }

    for (std::size_t i = 0; i < n; ++i)
    {
      _abi_elements[i].handle = &_handles[i];
      _abi_elements[i].tabulate = &basix_runtime_tabulate;
    }
  }

  void initialise_rules(nb::list quadrature)
  {
    const std::size_t n = nb::len(quadrature);
    _rule_storage.reserve(n);
    _abi_rules.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
      nb::dict rule = nb::cast<nb::dict>(quadrature[i]);
      Array2D points = nb::cast<Array2D>(dict_item(rule, "points"));
      Array1D weights = nb::cast<Array1D>(dict_item(rule, "weights"));
      if (points.shape(0) != weights.shape(0))
      {
        throw std::runtime_error(
            "Runtime quadrature points and weights disagree on nq.");
      }

      QuadratureRuleStorage storage;
      storage.points.assign(points.data(), points.data() + points.size());
      storage.weights.assign(weights.data(), weights.data() + weights.size());
      _rule_storage.push_back(std::move(storage));
      const QuadratureRuleStorage& owned = _rule_storage.back();
      _abi_rules.push_back(
          {static_cast<int>(weights.shape(0)), static_cast<int>(points.shape(1)),
           owned.points.data(), owned.weights.data()});
    }
  }

  void initialise_cut_flags(nb::object is_cut)
  {
    if (is_cut.is_none())
    {
      _is_cut.assign(_abi_rules.size(), std::uint8_t{1});
      return;
    }

    nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<1>, nb::c_contig>
        flags = nb::cast<
            nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<1>,
                        nb::c_contig>>(is_cut);
    if (flags.shape(0) != _abi_rules.size())
      throw std::runtime_error("is_cut must have one entry per quadrature rule.");
    _is_cut.assign(flags.data(), flags.data() + flags.size());
  }

  std::vector<BasixElementHandle> _handles;
  std::vector<runintgen_basix_element> _abi_elements;
  std::vector<QuadratureRuleStorage> _rule_storage;
  std::vector<runintgen_quadrature_rule> _abi_rules;
  std::vector<std::uint8_t> _is_cut;
  runintgen_form_context _form{};
  runintgen_context _context{};
};
} // namespace

NB_MODULE(_basix_runtime, m)
{
  nb::class_<CustomData>(m, "CustomData")
      .def(nb::init<nb::list, nb::list, nb::object>(), nb::arg("element_specs"),
           nb::arg("quadrature"), nb::arg("is_cut") = nb::none())
      .def_prop_ro("ptr", &CustomData::ptr)
      .def_prop_ro("num_elements", &CustomData::num_elements)
      .def_prop_ro("num_rules", &CustomData::num_rules)
      .def("__int__", &CustomData::ptr);
}
