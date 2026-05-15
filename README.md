# runintgen

Runtime quadrature kernel generator for FEniCSx/FFCx.

## Overview

`runintgen` compiles UFL integrals marked for runtime quadrature into
UFCx-compatible C kernels. The generated kernels keep the standard UFCx
`tabulate_tensor` signature and receive runtime context through
`void* custom_data`.

This is intended for integration rules that are not known when FFCx compiles the
form, for example:

- Cut-cell and unfitted finite element methods.
- Polygonal or subcell integration.
- Per-entity custom quadrature rules.
- Singular or externally supplied integration data.

The current generator is backed by FFCx IR. It adapts FFCx's
`IntegralGenerator`, keeps FFCx table-reference metadata, emits static
piecewise tables where FFCx would, and redirects runtime-varying FE tables to
Basix tabulation callbacks supplied by the runtime context.

## Installation

For this development branch, use Basix, UFL, and FFCx from their `main`
branches so the Python codegen imports and Basix C++ ABI stay consistent:

```bash
# Install build tools
python -m pip install --upgrade pip setuptools wheel scikit-build-core nanobind numpy cffi

# Install FEniCSx main packages directly from GitHub
python -m pip install --upgrade --force-reinstall --no-deps \
  "fenics-basix @ git+https://github.com/FEniCS/basix.git@main" \
  "fenics-ufl @ git+https://github.com/FEniCS/ufl.git@main" \
  "fenics-ffcx @ git+https://github.com/FEniCS/ffcx.git@main"

# Install runintgen against that stack
cd PATH_TO/runintgen
python -m pip install --no-build-isolation --no-deps --force-reinstall .
```

The default install builds the Basix-only runtime extension used to construct
``custom_data`` without depending on DOLFINx. It requires a C++20 compiler and a
Basix installation discoverable by CMake. In conda environments, the build
prefers the compiler from the Python environment so it matches the Basix
library ABI.

## Requirements

- Python >= 3.10.
- NumPy.
- FEniCSx Python packages: Basix >= 0.11.0.dev0,
  UFL >= 2025.3.0.dev0, and FFCx >= 0.11.0.dev0.
- UFCx headers and a C/C++ compiler when compiling generated kernels.
- DOLFINx is optional and only needed when constructing DOLFINx forms.
- `cffi` is optional and mainly useful for Python-side ABI tests/prototypes.

## Quick Start

```python
from pathlib import Path

import numpy as np
import ufl
from basix.ufl import element

from runintgen import (
    RuntimeQuadratureRule,
    compile_runtime_integrals,
    write_runtime_code,
)

# P1 coordinate map, P2 solution space.
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

quadrature = RuntimeQuadratureRule(
    points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
    weights=np.array([0.5], dtype=np.float64),
)
dx_rt = ufl.Measure("dx", domain=mesh, subdomain_id=1, subdomain_data=quadrature)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

module = compile_runtime_integrals(a, options={"scalar_type": np.float64})
files = write_runtime_code(
    module,
    prefix="laplace_p2",
    output_dir=Path("generated/runtime_codegen"),
)

print(files.header)
print(files.source)
print(files.abi_header)  # Canonical shared ABI header in runintgen/cpp.

for kernel in module.kernels:
    print(kernel.name, kernel.integral_type, kernel.subdomain_id)
    for table in kernel.table_info or []:
        print(
            table["slot"],
            table["role"],
            table["element_index"],
            table["derivative_counts"],
        )

runtime_data = module.create_custom_data(quadrature)
print(runtime_data.ptr)  # Pointer passed as UFCx custom_data.
```

`write_runtime_code` writes:

- `<prefix>.h`, declaring the generated UFCx integral object and form
  descriptor.
- `<prefix>.c`, defining the generated kernel and descriptor.
- A reference to the shared ABI header
  `runintgen/cpp/runintgen_runtime_abi.h`.

Add the generated output directory, `runintgen/cpp`, UFCx, and FFCx/DOLFINx
include directories to the downstream C/C++ build.

## Runtime Measures

Runtime integrals are selected by the contents of UFL `subdomain_data`.
There are three cases:

- Quadrature rule only: runintgen generates a runtime-only UFCx integral.
- Entity data plus quadrature rule: runintgen generates a mixed UFCx integral
  that uses `custom_data->is_cut[entity_index]` to choose the runtime or
  standard branch.
- Entity data only: the integral is standard-only and is left to FFCx/DOLFINx.

```python
from runintgen import RuntimeQuadratureRule

inside_entities = [0, 1, 4]
inside_quadrature = RuntimeQuadratureRule(points=points, weights=weights)

dx = ufl.Measure(
    "dx",
    domain=mesh,
    subdomain_data=[(0, inside_entities), (0, inside_quadrature)],
)
```

Ordinary entity data without quadrature-rule payloads remains a standard UFL
measure for FFCx/DOLFINx. The legacy
`metadata={"quadrature_rule": "runtime"}` marker is still accepted, but new code
should prefer `subdomain_data` because measure metadata is easy to lose when UFL
reconfigures measures.

## Runtime ABI

Generated kernels cast `custom_data` to `const runintgen_context*`. The ABI is
defined in `runintgen/cpp/runintgen_runtime_abi.h` and exposed in Python as
`CFFI_DEF`.

The important pieces are:

```c
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
  int derivative_order;
  int is_permuted;
} runintgen_table_request;

typedef int (*runintgen_element_tabulate_fn)(
    const runintgen_basix_element* element,
    const runintgen_quadrature_rule* rule,
    const runintgen_table_request* request,
    runintgen_table_view* view);

struct runintgen_context
{
  int num_rules;
  const runintgen_quadrature_rule* rules;
  const uint8_t* is_cut;
  const runintgen_form_context* form;
};
```

Runtime weights must already include the relevant measure scaling, such as
`|detJ|`, facet scaling, or cut-cell scaling. Generated C does not create
quadrature rules and does not multiply weights by `detJ`. Reference-to-physical
mapping, such as `J^-T grad_ref`, remains in generated C when required by the
FFCx IR. For non-cut entities, the generated mixed wrapper calls the standard
FFCx branch, which uses the normal compile-time quadrature and measure scaling.

## Runtime Table Calls

For FE tables backed by runtime quadrature, generated C emits one
`runintgen_table_request` per Basix element used by the integral. The request
uses the maximum derivative order required by all FFCx table references mapped
to that element, and calls the element's `tabulate` function pointer once. The
returned `runintgen_table_view` is treated as raw Basix tabulation storage with
layout:

```text
[derivative][point][dof][component]
```

`RuntimeKernelInfo.table_info` records the mapping from each FFCx table
reference to its runtime Basix element slot. It includes the slot, FFCx table
name, derivative counts, Basix derivative index, form element index, flat
component, block offset/stride metadata, role, terminal index, and the maximum
derivative order used for the shared element tabulation.

## Basix-Only Custom Data

`runintgen.basix_runtime.CustomData` builds the runtime context from
`module.form_metadata` and one or more `RuntimeQuadratureRule` objects:

```python
runtime_data = module.create_custom_data(
    RuntimeQuadratureRule(points=points, weights=weights)
)

kernel(..., runtime_data.ptr)
```

The generated C kernels do not link to Basix. They call the `tabulate` function
pointer stored in `custom_data`; the installed `_basix_runtime` extension owns
the C++ Basix elements and performs tabulation. Keep the `CustomData` object
alive until assembly or kernel calls are complete.

## Form Metadata

`compile_runtime_integrals` also builds form-level metadata:

```python
metadata = module.form_metadata
for element_info in metadata.unique_elements:
    print(
        element_info.form_elem_index,
        element_info.role.name,
        element_info.index,
        element_info.element_key,
    )
```

Generated sources embed this metadata as a `runintgen_form_descriptor`. The C++
wrapper uses it to resolve active elements from DOLFINx argument spaces,
coefficient function spaces, and mesh coordinate elements. This keeps generated
C free of Basix and DOLFINx C++ casts.

## DOLFINx JIT Integration

For DOLFINx users, the preferred entry point is `runintgen.dolfinx.form`.
It mirrors `dolfinx.fem.form` and returns a normal DOLFINx `Form`:

```python
from runintgen.dolfinx import form as runintgen_form

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
runtime_rules = RuntimeQuadratureRules(
    tdim=2,
    points=points,
    weights=weights,
    offsets=offsets,
    parent_map=np.arange(num_cells, dtype=np.int32),
)
dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=runtime_rules)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

A = dolfinx.fem.assemble_matrix(runintgen_form(a))
```

The DOLFINx integration currently targets the patched DOLFINx branch with
per-integral `custom_data` support. Runtime quadrature support is intentionally
limited to `float64` cell integrals for the first implementation. Standard-only
forms passed to `runintgen.dolfinx.form` delegate to `dolfinx.fem.form`.

Lower-level APIs such as `compile_runtime_integrals`, `create_custom_data`, and
`runintgen.jit.compile_forms` remain available for diagnostics, generated-code
inspection, and non-DOLFINx integration experiments.

## Geometry Helpers

`runintgen.geometry` contains small utilities for building runtime quadrature
rules on subcells:

- `map_quadrature_to_subcell`.
- `map_subcell_to_parent_reference`.
- `scale_weights_by_jacobian`.
- `generate_subcell_quadrature`.

They support the tested triangle and tetrahedron affine mapping workflows and
leave final runtime-rule ownership to the caller or C++ bridge.


## Architecture

```text
UFL form with quadrature-rule payloads in subdomain_data
  |
  v
runintgen.analysis.build_runtime_info
  |  selects runtime/mixed integrals and preserves FFCx IR context
  v
FFCx analysis and IR computation for runtime and standard branches
  |
  v
runintgen.codegeneration.runtime_integrals
  |  emits runtime-only or mixed runtime/standard branch wrappers
  v
RunintModule
  |  kernels, table requests, scalar/geometry types, form metadata
  v
write_runtime_code
  |  generated .h/.c plus shared runtime ABI header
  v
DOLFINx/C++ or CFFI runtime context
  |  per-entity is_cut flags, quadrature rules, and Basix callbacks
  v
UFCx kernel called with void* custom_data
```

## API Reference

### Measures

- `RUNTIME_QUADRATURE_RULE`: Constant `"runtime"` for UFL metadata.
- `RuntimeIntegralMode`: Classification enum with `STANDARD`, `RUNTIME`, and
  `MIXED`.
- `is_runtime_quadrature_rule(value)`: Check the structural quadrature-rule
  protocol, currently `points` plus `weights`.
- `has_runtime_quadrature(subdomain_data)`: Check whether UFL subdomain data
  contains a quadrature-rule payload.
- `has_standard_subdomain_data(subdomain_data)`: Check whether UFL subdomain
  data contains ordinary entity payloads.
- `runtime_integral_mode(integral)`: Classify an integral as standard-only,
  runtime-only, or mixed.
- `is_runtime_integral(integral)`: Check whether an integral uses runtime
  quadrature.
- `get_quadrature_provider(integral)`: Return the integral's
  `subdomain_data`.

### Compilation

- `compile_runtime_integrals(form, options=None)`: Compile runtime-marked
  integrals and return a `RunintModule`.
- `runintgen.jit.compile_forms(forms, options=None, ...)`: Compile combined
  standard/runtime forms into full UFCx form objects with CFFI.
- `RunintModule`: Holds generated kernels, module metadata, and
  `form_metadata`.
- `RuntimeKernelInfo`: Holds one kernel's C declaration/definition, UFCx
  integral name, table requests, tensor shape, scalar type, and geometry type.

### DOLFINx

- `runintgen.dolfinx.form(form, ...)`: Preferred DOLFINx entry point. Returns a
  normal `dolfinx.fem.Form`, delegating standard-only forms to DOLFINx and using
  runintgen JIT when runtime integrals are present.
- `runintgen.dolfinx.compile_form(comm, form, ...)`: Compile a UFL form into a
  public `CompiledRunintForm` without binding runtime DOLFINx data.
- `runintgen.dolfinx.create_form(compiled, V, mesh, ...)`: Build a DOLFINx form
  from a compiled runintgen form.
- `runintgen.dolfinx.has_runtime_custom_data_support()`: Check whether the
  loaded DOLFINx build exposes the required custom-data support.

### Generated Files

- `write_runtime_code(module, prefix, output_dir)`: Write generated header and
  source files.
- `RuntimeCodeFiles`: Paths returned by `write_runtime_code`.
- `format_runtime_header`, `format_runtime_source`, `format_runtime_abi_header`:
  Formatting helpers for downstream build tooling.
- `runtime_abi_header_path()`: Path to the shared ABI header.

### Runtime ABI

- `get_runintgen_data_struct()`: Return the C ABI definition string.
- `CFFI_DEF`: CFFI declaration string for the ABI.
- `RuntimeQuadratureRule`: Python container for reference points and pre-scaled
  weights.
- `RuntimeBasixElement`: Python-side opaque element handle and tabulate
  callback pointer.
- `RuntimeTableRequest`, `RuntimeTableView`: Python mirrors of ABI table
  request/view structures.
- `RuntimeContextBuilder`: CFFI helper for building `runintgen_context*`.
- `to_intptr(ffi, data)`: Convert a CFFI pointer to an integer handle.

### Form Metadata

- `build_form_runtime_metadata(runtime_info)`: Build form-level element
  metadata.
- `FormRuntimeMetadata`, `FormElementInfo`, `IntegralRuntimeLayout`,
  `IntegralElementUsage`, `ElementKey`, `Role`: Metadata structures used by the
  generated descriptor and C++ bridge.
- `element_key_from_basix`, `element_key_from_ufl`: Build structured element
  keys.
- `export_metadata_for_cpp(metadata)`: JSON-serialisable metadata export.

### Runtime Table Metadata

- `DerivativeMapping`: Basix derivative-index helpers.
- `RuntimeElementMapping`, `UniqueElementInfo`, `ElementUsage`: Compatibility
  metadata for table analysis.
- `build_runtime_element_mapping`, `build_runtime_element_mapping_from_ir`:
  Mapping helpers used by tests and migration code.

### Geometry

- `generate_subcell_quadrature`.
- `map_quadrature_to_subcell`.
- `map_subcell_to_parent_reference`.
- `scale_weights_by_jacobian`.

## Examples

- `examples/write_runtime_codegen.py`: Generate inspectable ABI-backed C files
  for P1 mass, P2 Laplace, and coefficient mass forms.
- `examples/simple_example.py`: Compile and inspect a form with mixed standard
  and runtime integrals.
- `generated/runtime_codegen/`: Checked-in examples of the current generated
  headers and sources.


## License

MIT License - Copyright (c) 2025 ONERA

See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is developed at ONERA and builds upon the FEniCSx project:

- [FEniCSx](https://fenicsproject.org/)
- [Basix](https://github.com/FEniCS/basix)
- [UFL](https://github.com/FEniCS/ufl)
- [FFCx](https://github.com/FEniCS/ffcx)
