#include "runintgen_runtime_abi.h"
#include "runtime_tabulation_helpers.h"

#include <basix/finite-element.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
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
using Array2D = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>;
using Int32Array
    = nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig>;
using Int32Array2D
    = nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<2>, nb::c_contig>;
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

nb::object vector_to_numpy(std::vector<double> values,
                           std::initializer_list<std::size_t> shape)
{
  auto* owner = new std::vector<double>(std::move(values));
  nb::capsule capsule(owner, [](void* data) noexcept {
    delete static_cast<std::vector<double>*>(data);
  });
  return nb::ndarray<nb::numpy, double>(owner->data(), shape, capsule).cast();
}

basix::FiniteElement<double>
create_coordinate_element(int cell_type, int degree, int lagrange_variant)
{
  if (degree <= 0)
    throw std::runtime_error("Coordinate element degree must be positive.");

  return basix::create_element<double>(
      basix::element::family::P, static_cast<basix::cell::type>(cell_type),
      degree,
      static_cast<basix::element::lagrange_variant>(lagrange_variant),
      basix::element::dpc_variant::unset, false, {});
}

void validate_geometry_inputs(int tdim, int gdim, const FloatArray& points,
                              const Array1D& weights,
                              const Int32Array& parent_map,
                              const Array2D& geometry_x,
                              const Int32Array2D& geometry_dofmap)
{
  if (tdim <= 0)
    throw std::runtime_error("tdim must be positive.");
  if (gdim <= 0)
    throw std::runtime_error("gdim must be positive.");
  if (points.ndim() == 2)
  {
    if (points.shape(1) != static_cast<std::size_t>(tdim))
      throw std::runtime_error("points second dimension must equal tdim.");
    if (points.shape(0) != weights.shape(0)
        && parent_map.shape(0) == 0)
    {
      throw std::runtime_error(
          "points and weights disagree for empty parent_map.");
    }
  }
  else if (points.ndim() == 1)
  {
    if (points.size() != static_cast<std::size_t>(weights.shape(0) * tdim))
      throw std::runtime_error("flat points size disagrees with weights/tdim.");
  }
  else
  {
    throw std::runtime_error("points must be flat or two-dimensional.");
  }
  if (geometry_x.ndim() != 2 || geometry_x.shape(1) < static_cast<std::size_t>(gdim))
    throw std::runtime_error("geometry_x shape disagrees with gdim.");
  if (geometry_dofmap.ndim() != 2)
    throw std::runtime_error("geometry_dofmap must be two-dimensional.");
}

std::pair<const double*, std::size_t>
point_data_and_count(const FloatArray& points, int tdim)
{
  if (points.ndim() == 2)
    return {points.data(), points.shape(0)};
  return {points.data(), points.size() / static_cast<std::size_t>(tdim)};
}

double jacobian_measure(const std::vector<double>& j, int gdim, int tdim)
{
  if (tdim == 1)
  {
    double norm2 = 0.0;
    for (int g = 0; g < gdim; ++g)
      norm2 += j[g] * j[g];
    return std::sqrt(norm2);
  }
  if (tdim == 2)
  {
    double g00 = 0.0;
    double g01 = 0.0;
    double g11 = 0.0;
    for (int g = 0; g < gdim; ++g)
    {
      const double j0 = j[g * tdim];
      const double j1 = j[g * tdim + 1];
      g00 += j0 * j0;
      g01 += j0 * j1;
      g11 += j1 * j1;
    }
    return std::sqrt(std::max(0.0, g00 * g11 - g01 * g01));
  }
  if (tdim == 3)
  {
    double g00 = 0.0;
    double g01 = 0.0;
    double g02 = 0.0;
    double g11 = 0.0;
    double g12 = 0.0;
    double g22 = 0.0;
    for (int g = 0; g < gdim; ++g)
    {
      const double j0 = j[g * tdim];
      const double j1 = j[g * tdim + 1];
      const double j2 = j[g * tdim + 2];
      g00 += j0 * j0;
      g01 += j0 * j1;
      g02 += j0 * j2;
      g11 += j1 * j1;
      g12 += j1 * j2;
      g22 += j2 * j2;
    }
    const double det = g00 * (g11 * g22 - g12 * g12)
                       - g01 * (g01 * g22 - g12 * g02)
                       + g02 * (g01 * g12 - g11 * g02);
    return std::sqrt(std::max(0.0, det));
  }
  throw std::runtime_error("Only tdim 1, 2, and 3 are supported.");
}

void map_entity_points(const std::vector<double>& basis,
                       const std::array<std::size_t, 4>& shape,
                       const double* weights, const std::int32_t* dofs,
                       const double* geometry_x, std::size_t geometry_stride,
                       int tdim, int gdim, std::size_t local_q0,
                       std::size_t local_q1, std::size_t global_q0,
                       std::vector<double>& physical_points,
                       std::vector<double>* scaled_weights)
{
  const std::size_t nq = shape[1];
  const std::size_t ndofs = shape[2];
  const std::size_t value_size = shape[3];
  if (value_size != 1)
    throw std::runtime_error("Coordinate basis value_size must be one.");

  std::vector<double> jacobian(
      static_cast<std::size_t>(std::max(1, gdim * tdim)));
  for (std::size_t q = local_q0; q < local_q1; ++q)
  {
    const std::size_t global_q = global_q0 + (q - local_q0);
    for (int g = 0; g < gdim; ++g)
    {
      double value = 0.0;
      for (std::size_t dof = 0; dof < ndofs; ++dof)
      {
        const double phi = basis[((0 * nq + q) * ndofs + dof) * value_size];
        value += phi * geometry_x[static_cast<std::size_t>(dofs[dof])
                                  * geometry_stride
                                  + static_cast<std::size_t>(g)];
      }
      physical_points[static_cast<std::size_t>(g) * physical_points.size()
                      / static_cast<std::size_t>(gdim)
                      + global_q]
          = value;
    }

    if (scaled_weights != nullptr)
    {
      std::fill(jacobian.begin(), jacobian.end(), 0.0);
      for (int d = 0; d < tdim; ++d)
      {
        for (int g = 0; g < gdim; ++g)
        {
          double value = 0.0;
          for (std::size_t dof = 0; dof < ndofs; ++dof)
          {
            const double dphi
                = basis[(((1 + d) * nq + q) * ndofs + dof) * value_size];
            value += dphi * geometry_x[static_cast<std::size_t>(dofs[dof])
                                        * geometry_stride
                                        + static_cast<std::size_t>(g)];
          }
          jacobian[static_cast<std::size_t>(g * tdim + d)] = value;
        }
      }
      (*scaled_weights)[global_q] = weights[q] * jacobian_measure(jacobian, gdim, tdim);
    }
  }
}

nb::tuple map_per_entity_geometry(int cell_type, int degree, int lagrange_variant,
                                  int tdim, int gdim,
                                  const FloatArray& points,
                                  const Array1D& weights,
                                  const Int64Array& offsets,
                                  const Int32Array& parent_map,
                                  const Array2D& geometry_x,
                                  const Int32Array2D& geometry_dofmap,
                                  bool scale_weights)
{
  validate_geometry_inputs(tdim, gdim, points, weights, parent_map, geometry_x,
                           geometry_dofmap);
  if (offsets.shape(0) != parent_map.shape(0) + 1)
    throw std::runtime_error("offsets must have one more entry than parent_map.");
  if (offsets.shape(0) == 0 || offsets.data()[0] != 0)
    throw std::runtime_error("offsets must start at zero.");
  for (std::size_t i = 1; i < offsets.shape(0); ++i)
  {
    if (offsets.data()[i] < offsets.data()[i - 1])
      throw std::runtime_error("offsets must be nondecreasing.");
  }
  if (offsets.data()[offsets.shape(0) - 1] != weights.shape(0))
    throw std::runtime_error("offsets[-1] must equal weights.size.");

  const auto [point_data, total_points] = point_data_and_count(points, tdim);
  if (total_points != weights.shape(0))
    throw std::runtime_error("points and weights disagree on total nq.");
  auto element = create_coordinate_element(cell_type, degree, lagrange_variant);
  const int nd = scale_weights ? 1 : 0;
  auto [basis, shape] = element.tabulate(
      nd, std::span<const double>(point_data, total_points * tdim),
      {total_points, static_cast<std::size_t>(tdim)});
  if (shape[2] != geometry_dofmap.shape(1))
    throw std::runtime_error(
        "Coordinate element dimension disagrees with geometry_dofmap.");

  std::vector<double> physical_points(
      static_cast<std::size_t>(gdim) * total_points);
  std::vector<double> scaled_weights(scale_weights ? total_points : 0);
  std::vector<double>* scaled_ptr = scale_weights ? &scaled_weights : nullptr;
  const double* raw_weights = weights.data();

  for (std::size_t rule = 0; rule < parent_map.shape(0); ++rule)
  {
    const std::int32_t cell = parent_map.data()[rule];
    if (cell < 0 || static_cast<std::size_t>(cell) >= geometry_dofmap.shape(0))
      throw std::runtime_error("parent_map contains invalid local cells.");
    const std::int64_t q0 = offsets.data()[rule];
    const std::int64_t q1 = offsets.data()[rule + 1];
    map_entity_points(basis, shape, raw_weights, &geometry_dofmap(cell, 0),
                      geometry_x.data(), geometry_x.shape(1), tdim, gdim,
                      static_cast<std::size_t>(q0), static_cast<std::size_t>(q1),
                      static_cast<std::size_t>(q0), physical_points, scaled_ptr);
  }

  nb::object py_points = vector_to_numpy(
      std::move(physical_points), {static_cast<std::size_t>(gdim), total_points});
  nb::object py_weights = nb::none();
  if (scale_weights)
  {
    py_weights
        = vector_to_numpy(std::move(scaled_weights), {total_points});
  }
  return nb::make_tuple(py_points, py_weights);
}

nb::tuple map_shared_geometry(int cell_type, int degree, int lagrange_variant,
                              int tdim, int gdim, const FloatArray& points,
                              const Array1D& weights,
                              const Int32Array& parent_map,
                              const Array2D& geometry_x,
                              const Int32Array2D& geometry_dofmap,
                              bool scale_weights)
{
  validate_geometry_inputs(tdim, gdim, points, weights, parent_map, geometry_x,
                           geometry_dofmap);
  const auto [point_data, nq] = point_data_and_count(points, tdim);
  if (nq != weights.shape(0))
    throw std::runtime_error("shared points and weights disagree on nq.");

  auto element = create_coordinate_element(cell_type, degree, lagrange_variant);
  const int nd = scale_weights ? 1 : 0;
  auto [basis, shape] = element.tabulate(
      nd, std::span<const double>(point_data, nq * tdim),
      {nq, static_cast<std::size_t>(tdim)});
  if (shape[2] != geometry_dofmap.shape(1))
    throw std::runtime_error(
        "Coordinate element dimension disagrees with geometry_dofmap.");

  const std::size_t total_points = nq * parent_map.shape(0);
  std::vector<double> physical_points(
      static_cast<std::size_t>(gdim) * total_points);
  std::vector<double> scaled_weights(scale_weights ? total_points : 0);
  std::vector<double>* scaled_ptr = scale_weights ? &scaled_weights : nullptr;

  for (std::size_t entity = 0; entity < parent_map.shape(0); ++entity)
  {
    const std::int32_t cell = parent_map.data()[entity];
    if (cell < 0 || static_cast<std::size_t>(cell) >= geometry_dofmap.shape(0))
      throw std::runtime_error("parent_map contains invalid local cells.");
    map_entity_points(basis, shape, weights.data(), &geometry_dofmap(cell, 0),
                      geometry_x.data(), geometry_x.shape(1), tdim, gdim, 0, nq,
                      entity * nq, physical_points, scaled_ptr);
  }

  nb::object py_points = vector_to_numpy(
      std::move(physical_points), {static_cast<std::size_t>(gdim), total_points});
  nb::object py_weights = nb::none();
  if (scale_weights)
  {
    py_weights
        = vector_to_numpy(std::move(scaled_weights), {total_points});
  }
  return nb::make_tuple(py_points, py_weights);
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
    initialise_quadrature_functions();

    _form.num_elements = static_cast<int>(_abi_elements.size());
    _form.elements = _abi_elements.empty() ? nullptr : _abi_elements.data();
    _form.descriptor = nullptr;
    _form.scratch = nullptr;

    _context.quadrature = &_abi_quadrature;
    _context.entities = &_abi_entities;
    _context.quadrature_functions
        = _abi_quadrature_functions.num_functions == 0
              ? nullptr
              : &_abi_quadrature_functions;
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

  void initialise_quadrature_functions()
  {
    _abi_quadrature_functions.num_functions = 0;
    _abi_quadrature_functions.functions = nullptr;

    if (!object_has_non_none(_quadrature_owner, "quadrature_functions"))
      return;

    nb::object q_owner = nb::getattr(_quadrature_owner, "quadrature_functions");
    if (q_owner.is_none())
      return;

    nb::list functions = nb::cast<nb::list>(nb::getattr(q_owner, "functions"));
    const std::size_t n = nb::len(functions);
    _abi_quadrature_function_values.clear();
    _abi_quadrature_function_values.reserve(n);
    _abi_quadrature_functions_storage.resize(n);

    for (std::size_t i = 0; i < n; ++i)
    {
      nb::handle item = functions[i];
      FloatArray values = object_cast<FloatArray>(item, "values");
      const int value_size = object_cast<int>(item, "value_size");
      if (value_size <= 0)
        throw std::runtime_error(
            "QuadratureFunction value_size must be positive.");
      if (values.ndim() == 1)
      {
        if (value_size != 1)
          throw std::runtime_error(
              "Vector/tensor QuadratureFunction values must be 2D.");
      }
      else if (values.ndim() == 2)
      {
        if (values.shape(1) != static_cast<std::size_t>(value_size))
          throw std::runtime_error(
              "QuadratureFunction values shape disagrees with value_size.");
      }
      else
      {
        throw std::runtime_error(
            "QuadratureFunction values must be one- or two-dimensional.");
      }

      _abi_quadrature_function_values.push_back(values);
      _abi_quadrature_functions_storage[i].values = values.data();
      _abi_quadrature_functions_storage[i].value_size = value_size;
      _abi_quadrature_functions_storage[i].num_points
          = static_cast<int>(values.shape(0));
    }

    _quadrature_functions_owner = std::move(q_owner);
    _abi_quadrature_functions.num_functions = static_cast<int>(n);
    _abi_quadrature_functions.functions
        = _abi_quadrature_functions_storage.empty()
              ? nullptr
              : _abi_quadrature_functions_storage.data();
  }

  std::vector<BasixElementHandle> _handles;
  std::vector<runintgen_basix_element> _abi_elements;
  nb::object _quadrature_owner;
  nb::object _quadrature_functions_owner;
  std::vector<FloatArray> _abi_quadrature_function_values;
  std::vector<runintgen_quadrature_function>
      _abi_quadrature_functions_storage;
  runintgen_quadrature_functions _abi_quadrature_functions{};
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
  m.def("map_per_entity_geometry", &map_per_entity_geometry, nb::arg("cell_type"),
        nb::arg("degree"), nb::arg("lagrange_variant"), nb::arg("tdim"),
        nb::arg("gdim"), nb::arg("points"), nb::arg("weights"),
        nb::arg("offsets"), nb::arg("parent_map"), nb::arg("geometry_x"),
        nb::arg("geometry_dofmap"), nb::arg("scale_weights") = false);
  m.def("map_shared_geometry", &map_shared_geometry, nb::arg("cell_type"),
        nb::arg("degree"), nb::arg("lagrange_variant"), nb::arg("tdim"),
        nb::arg("gdim"), nb::arg("points"), nb::arg("weights"),
        nb::arg("parent_map"), nb::arg("geometry_x"),
        nb::arg("geometry_dofmap"), nb::arg("scale_weights") = false);
}
