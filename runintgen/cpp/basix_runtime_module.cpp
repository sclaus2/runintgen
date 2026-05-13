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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace
{
using FloatArray = nb::ndarray<nb::numpy, const double, nb::c_contig>;
using Array1D = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using Int32Array
    = nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig>;
using Int64Array
    = nb::ndarray<nb::numpy, const std::int64_t, nb::ndim<1>, nb::c_contig>;
using UInt8Array
    = nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<1>, nb::c_contig>;

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

template <typename T>
T object_cast(nb::handle data, const char* key)
{
  if (PyDict_Check(data.ptr()))
    return nb::cast<T>(dict_item(nb::cast<nb::dict>(data), key));
  if (!PyObject_HasAttrString(data.ptr(), key))
    throw std::runtime_error(std::string("Missing required attribute '") + key
                             + "'.");
  return nb::cast<T>(nb::getattr(data, key));
}

bool object_has_non_none(nb::handle data, const char* key)
{
  if (PyDict_Check(data.ptr()))
  {
    PyObject* item = PyDict_GetItemString(data.ptr(), key);
    return item != nullptr && item != Py_None;
  }
  if (!PyObject_HasAttrString(data.ptr(), key))
    return false;
  return !nb::getattr(data, key).is_none();
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

int basix_runtime_tabulate(const runintgen_basix_element* element,
                           const runintgen_quadrature_rule* rule,
                           const runintgen_table_request* request,
                           runintgen_table_view* view)
{
  if (element == nullptr || element->handle == nullptr || rule == nullptr
      || request == nullptr || view == nullptr)
    return 1;

  auto* handle = static_cast<const BasixElementHandle*>(element->handle);

  thread_local std::unordered_map<TableCacheKey, std::vector<double>,
                                  TableCacheKeyHash>
      cache;
  return runintgen::detail::tabulate_runtime_table(
      handle->element, rule, request, view,
      [handle](int slot) -> std::vector<double>& {
        return cache[{handle, slot}];
      },
      handle->block_size);
}

class CustomData
{
public:
  CustomData(nb::list element_specs, nb::object quadrature)
  {
    initialise_elements(element_specs);
    initialise_rules(quadrature);

    _form.num_elements = static_cast<int>(_abi_elements.size());
    _form.elements = _abi_elements.empty() ? nullptr : _abi_elements.data();
    _form.descriptor = nullptr;
    _form.scratch = nullptr;

    _context.quadrature = &_abi_quadrature;
    _context.entities = &_abi_entities;
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

  int num_rules() const noexcept { return _abi_quadrature.num_rules; }

  int num_entities() const noexcept { return _abi_entities.num_entities; }

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

  void initialise_rules(nb::object quadrature)
  {
    _quadrature_owner = std::move(quadrature);

    const int tdim = object_cast<int>(_quadrature_owner, "tdim");
    FloatArray points = object_cast<FloatArray>(_quadrature_owner, "points");
    Array1D weights = object_cast<Array1D>(_quadrature_owner, "weights");
    Int64Array offsets = object_cast<Int64Array>(_quadrature_owner, "offsets");
    Int32Array entity_indices
        = object_cast<Int32Array>(_quadrature_owner, "entity_indices");
    UInt8Array is_cut = object_cast<UInt8Array>(_quadrature_owner, "is_cut");
    Int32Array rule_indices
        = object_cast<Int32Array>(_quadrature_owner, "rule_indices");

    if (tdim <= 0)
      throw std::runtime_error("Runtime quadrature tdim must be positive.");
    if (points.ndim() == 2)
    {
      if (points.shape(1) != tdim || points.shape(0) != weights.shape(0))
        throw std::runtime_error(
            "Runtime quadrature points shape disagrees with tdim/weights.");
    }
    else if (points.ndim() == 1)
    {
      if (points.size() != static_cast<std::size_t>(weights.shape(0) * tdim))
      {
        throw std::runtime_error(
            "Flat runtime quadrature points size disagrees with weights/tdim.");
      }
    }
    else
    {
      throw std::runtime_error(
          "Runtime quadrature points must be flat or two-dimensional.");
    }
    if (offsets.shape(0) < 1 || offsets.data()[0] != 0)
      throw std::runtime_error("Runtime quadrature offsets must start at zero.");
    for (std::size_t i = 1; i < offsets.shape(0); ++i)
    {
      if (offsets.data()[i] < offsets.data()[i - 1])
        throw std::runtime_error(
            "Runtime quadrature offsets must be nondecreasing.");
    }
    if (offsets.data()[offsets.shape(0) - 1] != weights.shape(0))
      throw std::runtime_error(
          "Runtime quadrature offsets must end at total nq.");

    const int num_rules = static_cast<int>(offsets.shape(0) - 1);
    if (entity_indices.shape(0) != is_cut.shape(0)
        || entity_indices.shape(0) != rule_indices.shape(0))
    {
      throw std::runtime_error(
          "Runtime entity-map arrays must have equal length.");
    }

    for (std::size_t i = 0; i < rule_indices.shape(0); ++i)
    {
      const bool cut = is_cut.data()[i] != 0;
      const std::int32_t rule_index = rule_indices.data()[i];
      if (cut && (rule_index < 0 || rule_index >= num_rules))
        throw std::runtime_error("Cut entity rule index is out of range.");
      if (!cut && rule_index >= 0)
        throw std::runtime_error("Standard entities must use rule index -1.");
    }

    _abi_quadrature.tdim = tdim;
    _abi_quadrature.num_rules = num_rules;
    _abi_quadrature.offsets = offsets.data();
    _abi_quadrature.points = points.data();
    _abi_quadrature.weights = weights.data();
    _abi_quadrature.parent_map = nullptr;
    if (object_has_non_none(_quadrature_owner, "parent_map"))
    {
      Int32Array parent_map
          = object_cast<Int32Array>(_quadrature_owner, "parent_map");
      if (parent_map.shape(0) != static_cast<std::size_t>(num_rules))
      {
        throw std::runtime_error(
            "Runtime quadrature parent_map must have one entry per rule.");
      }
      _abi_quadrature.parent_map = parent_map.data();
    }

    _abi_entities.num_entities = static_cast<int>(entity_indices.shape(0));
    _abi_entities.entity_indices = entity_indices.data();
    _abi_entities.is_cut = is_cut.data();
    _abi_entities.rule_indices = rule_indices.data();
  }

  std::vector<BasixElementHandle> _handles;
  std::vector<runintgen_basix_element> _abi_elements;
  nb::object _quadrature_owner;
  runintgen_quadrature_rules _abi_quadrature{};
  runintgen_entity_map _abi_entities{};
  runintgen_form_context _form{};
  runintgen_context _context{};
};
} // namespace

NB_MODULE(_basix_runtime, m)
{
  nb::class_<CustomData>(m, "CustomData")
      .def(nb::init<nb::list, nb::object>(), nb::arg("element_specs"),
           nb::arg("quadrature"))
      .def_prop_ro("ptr", &CustomData::ptr)
      .def_prop_ro("num_elements", &CustomData::num_elements)
      .def_prop_ro("num_rules", &CustomData::num_rules)
      .def_prop_ro("num_entities", &CustomData::num_entities)
      .def("__int__", &CustomData::ptr);
}
