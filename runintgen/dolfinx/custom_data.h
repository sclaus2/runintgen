#ifndef RUNINTGEN_DOLFINX_CUSTOM_DATA_H
#define RUNINTGEN_DOLFINX_CUSTOM_DATA_H

#include "../cpp/runtime_basix_wrapper.h"

#include <dolfinx/fem/Form.h>

#include <memory>
#include <vector>

namespace runintgen::dolfinx
{
/// Runtime quadrature rule owned by a DOLFINx custom-data context.
struct QuadratureRule
{
  /// Reference quadrature points in row-major shape `(nq, tdim)`.
  std::vector<double> points;

  /// Runtime weights. The caller is responsible for measure scaling.
  std::vector<double> weights;

  /// Topological dimension of each reference quadrature point.
  int tdim = 0;
};

/// Owns the runintgen custom data passed to DOLFINx assemblers.
///
/// The object resolves Basix elements from a DOLFINx form using the
/// `runintgen_form_descriptor` emitted next to generated runtime kernels. Keep
/// the object alive until assembly is complete; `custom_data()` returns the
/// pointer that should be passed through the DOLFINx Form integral data.
class CustomData
{
public:
  CustomData(const ::dolfinx::fem::Form<double, double>& form,
             const runintgen_form_descriptor& descriptor,
             std::vector<QuadratureRule> rules);

  ~CustomData();

  CustomData(const CustomData&) = delete;
  CustomData& operator=(const CustomData&) = delete;
  CustomData(CustomData&&) noexcept;
  CustomData& operator=(CustomData&&) noexcept;

  /// Return the typed runtime context.
  const runintgen_context* context() const noexcept;

  /// Return the pointer passed as UFCx `custom_data`.
  void* custom_data() noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

/// Build a DOLFINx custom-data owner for a generated runintgen descriptor.
std::unique_ptr<CustomData>
create_custom_data(const ::dolfinx::fem::Form<double, double>& form,
                   const runintgen_form_descriptor& descriptor,
                   std::vector<QuadratureRule> rules);
} // namespace runintgen::dolfinx

#endif // RUNINTGEN_DOLFINX_CUSTOM_DATA_H
