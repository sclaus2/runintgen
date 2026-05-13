# runintgen

Runtime quadrature kernel generator for FEniCSx/FFCx.

## Overview

`runintgen` compiles UFL integrals marked with runtime quadrature metadata into
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

```bash
pip install -e .
```

For optional C++/nanobind experiments:

```bash
pip install -e ".[cpp]"
```

## Requirements

- Python >= 3.10.
- NumPy.
- FEniCSx Python packages: UFL, FFCx, and Basix.
- UFCx headers and a C/C++ compiler when compiling generated kernels.
- DOLFINx is optional, but needed for the C++ `runintgen/dolfinx` custom-data
  bridge.
- `cffi` is optional and mainly useful for Python-side ABI tests/prototypes.

## Quick Start

```python
from pathlib import Path

import numpy as np
import ufl
from basix.ufl import element

from runintgen import (
    RUNTIME_QUADRATURE_RULE,
    compile_runtime_integrals,
    write_runtime_code,
)

# P1 coordinate map, P2 solution space.
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dx_rt = ufl.Measure(
    "dx",
    domain=mesh,
    subdomain_id=1,
    metadata={"quadrature_rule": RUNTIME_QUADRATURE_RULE},
)

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

Runtime integrals are ordinary UFL integrals with
`metadata={"quadrature_rule": "runtime"}`:

```python
dx_rt = ufl.Measure(
    "dx",
    domain=mesh,
    subdomain_data=provider,
    metadata={"quadrature_rule": "runtime"},
)

ds_rt = ufl.Measure("ds", domain=mesh, metadata={"quadrature_rule": "runtime"})
dS_rt = ufl.Measure("dS", domain=mesh, metadata={"quadrature_rule": "runtime"})
```

`subdomain_data` is preserved for analysis via `get_quadrature_provider`, but
the generated C kernel receives runtime quadrature through `custom_data`.

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

typedef int (*runintgen_element_tabulate_fn)(
    const runintgen_basix_element* element,
    const runintgen_quadrature_rule* rule,
    const runintgen_table_request* request,
    runintgen_table_view* view);

struct runintgen_context
{
  int num_rules;
  const runintgen_quadrature_rule* rules;
  const runintgen_form_context* form;
};
```

Runtime weights must already include the relevant measure scaling, such as
`|detJ|`, facet scaling, or cut-cell scaling. Generated C does not create
quadrature rules and does not multiply weights by `detJ`. Reference-to-physical
mapping, such as `J^-T grad_ref`, remains in generated C when required by the
FFCx IR.

## Runtime Table Calls

For non-piecewise FE tables, generated C emits one `runintgen_table_request` per
FFCx table reference and calls the element's `tabulate` function pointer. The
returned `runintgen_table_view` is treated as raw Basix tabulation storage with
layout:

```text
[derivative][point][dof][component]
```

`RuntimeKernelInfo.table_info` mirrors these requests. It includes the runtime
slot, FFCx table name, derivative counts, Basix derivative index, form element
index, flat component, block offset/stride metadata, role, and terminal index.

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

## DOLFINx Bridge

The C++ helper in `runintgen/dolfinx/custom_data.{h,cpp}` owns a
`runintgen_context`, resolves Basix/coordinate elements from a DOLFINx form, and
registers tabulation callbacks.

Sketch:

```cpp
#include "laplace_p2.h"
#include "runintgen/dolfinx/custom_data.h"

std::vector<runintgen::dolfinx::QuadratureRule> rules = /* per-entity rules */;

auto data = runintgen::dolfinx::create_custom_data(
    form, runintgen_form_descriptor_laplace_p2, std::move(rules));

void* custom_data = data->custom_data();
```

Keep the returned owner alive for the whole assembly. The generated kernels
select the active quadrature rule from `entity_local_index` using the current
DOLFINx runtime-data convention:

- Cell integrals: `entity_local_index[0]`.
- Exterior facet integrals: `entity_local_index[1]`.
- Interior facet integrals: `entity_local_index[2]`.

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
UFL form with metadata={"quadrature_rule": "runtime"}
  |
  v
runintgen.analysis.build_runtime_info
  |  selects runtime integrals and preserves FFCx IR context
  v
FFCx analysis and IR computation
  |
  v
runintgen.codegeneration.runtime_integrals
  |  adapts FFCx IntegralGenerator for runtime weights, points, and FE tables
  v
RunintModule
  |  kernels, table requests, scalar/geometry types, form metadata
  v
write_runtime_code
  |  generated .h/.c plus shared runtime ABI header
  v
DOLFINx/C++ or CFFI runtime context
  |  per-entity quadrature rules and Basix tabulate callbacks
  v
UFCx kernel called with void* custom_data
```

## API Reference

### Measures

- `RUNTIME_QUADRATURE_RULE`: Constant `"runtime"` for UFL metadata.
- `is_runtime_integral(integral)`: Check whether an integral uses runtime
  quadrature.
- `get_quadrature_provider(integral)`: Return the integral's
  `subdomain_data`.

### Compilation

- `compile_runtime_integrals(form, options=None)`: Compile runtime-marked
  integrals and return a `RunintModule`.
- `RunintModule`: Holds generated kernels, module metadata, and
  `form_metadata`.
- `RuntimeKernelInfo`: Holds one kernel's C declaration/definition, UFCx
  integral name, table requests, tensor shape, scalar type, and geometry type.

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
