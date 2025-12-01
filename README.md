# runintgen

Runtime integration kernel generator for FEniCSx/FFCx.

## Overview

**runintgen** is a Python library that extends FFCx (FEniCSx Form Compiler) to support runtime integration kernels. This enables the use of custom quadrature rules that are determined at runtime rather than compile time, which is essential for applications like:

- Cut cell methods (CutFEM, unfitted finite elements)
- Polygonal finite element methods
- Custom quadrature for singular integrands

## Installation

```bash
pip install .
```

Or in development mode:

```bash
pip install -e .
```

## Requirements

- Python >= 3.10
- FEniCSx stack (basix, ufl, ffcx)
- NumPy
- cffi (for JIT compilation)

## Quick Start

```python
import ufl
from basix.ufl import element
from runintgen import compile_runtime_integrals

# Define mesh and function space
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create a runtime measure using standard UFL with special metadata
dx_rt = ufl.Measure("dx", domain=mesh, subdomain_id=1,
                    metadata={"quadrature_rule": "runtime"})

# Define a bilinear form using the runtime measure
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

# Compile runtime integration kernels
module = compile_runtime_integrals(a)

# Access kernel information
for kernel in module.kernels:
    print(f"Kernel: {kernel.name}")
    print(f"Integral type: {kernel.integral_type}")
    print(f"Subdomain ID: {kernel.subdomain_id}")
```

## Key Concepts

### Runtime Measures

Use standard UFL measures with `metadata={"quadrature_rule": "runtime"}` to mark integrals for runtime quadrature:

```python
import ufl

# Cell integral with runtime quadrature
dx_rt = ufl.Measure("dx", domain=mesh, subdomain_id=1,
                    metadata={"quadrature_rule": "runtime"})

# Exterior facet integral with runtime quadrature  
ds_rt = ufl.Measure("ds", domain=mesh, subdomain_id=2,
                    metadata={"quadrature_rule": "runtime"})

# You can also pass a quadrature provider object via subdomain_data
provider = MyQuadratureProvider()
dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=provider,
                    metadata={"quadrature_rule": "runtime"})
```

The constant `RUNTIME_QUADRATURE_RULE = "runtime"` is provided for convenience:

```python
from runintgen import RUNTIME_QUADRATURE_RULE

dx_rt = ufl.Measure("dx", domain=mesh,
                    metadata={"quadrature_rule": RUNTIME_QUADRATURE_RULE})
```

### FE Table Metadata Extraction

Extract information about which finite elements, components, and derivatives are needed:

```python
from runintgen import extract_integral_metadata
from runintgen.analysis import build_runtime_info
from ffcx.options import get_options

# Build runtime info from a form
runtime_info = build_runtime_info(a, get_options())

# Extract metadata for each runtime integral
metadata = extract_integral_metadata(runtime_info)

for group, meta in metadata.items():
    print(f"Integral type: {group.integral_type}")
    print(f"Component requests: {len(meta.component_requests)}")
    for req in meta.component_requests:
        print(f"  - {req.role}[{req.index}]: deriv={req.local_derivatives}")
```

### Runtime Data Structures

The compiled kernels use a runtime data structure to receive quadrature and FE tables:

```c
typedef struct {
    int ndofs;
    int ncomps;
    double* table;  // [nderivs * nq * ndofs]
} runintgen_element;

typedef struct {
    int nq;
    const double* weights;
    const double* points;
    int nelements;
    runintgen_element* elements;
} runintgen_quadrature_config;

typedef struct {
    int active_config;      // >= 0: single-config mode, < 0: multi-config mode
    int nconfigs;
    runintgen_quadrature_config* configs;
    int* cell_config_map;   // For multi-config: maps cell index to config
} runintgen_data;
```

Build runtime data using the Python API:

```python
from runintgen import RuntimeDataBuilder, QuadratureConfig, tabulate_element
import basix

# Create quadrature rule
qpts, qwts = basix.make_quadrature(basix.CellType.triangle, 2)

# Create P1 element and tabulate
P1 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

# Build runtime data
builder = RuntimeDataBuilder(ffi)  # ffi from cffi

config = QuadratureConfig(points=qpts, weights=qwts)
config.elements.append(tabulate_element(P1, qpts, max_deriv_order=1))
config.elements.append(tabulate_element(P1, qpts, max_deriv_order=1))  # coord element

builder.add_config(config)
runtime_data = builder.build()  # Single-config mode

# For multi-config (different cells use different quadrature):
cell_config_map = np.array([0, 0, 1, 0, 1], dtype=np.int32)  
runtime_data = builder.build(cell_config_map=cell_config_map)
```

### DOLFINx Integration

Use runtime kernels with the DOLFINx assembler:

```python
from mpi4py import MPI
from dolfinx import mesh, fem
from runintgen import (
    compile_runtime_kernels,
    create_dolfinx_form_with_runtime,
    set_runtime_data,
    RuntimeDataBuilder,
    QuadratureConfig,
    tabulate_element,
)

# Create mesh and function space
msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(msh, ("Lagrange", 1))

# Define UFL form with runtime quadrature
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
dx_rt = ufl.Measure("dx", domain=msh.ufl_domain(),
                    metadata={"quadrature_rule": "runtime"})
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

# Compile kernels (JIT compiles C code)
runtime_info = compile_runtime_kernels(a)

# Set up quadrature data
builder = RuntimeDataBuilder(runtime_info.ffi)
config = QuadratureConfig(points=qpts, weights=qwts)
# ... add element tables ...
builder.add_config(config)
rdata = builder.build()

# Create DOLFINx form with custom kernel
form = create_dolfinx_form_with_runtime([V, V], runtime_info, msh)

# Set runtime data pointer
data_ptr = int(runtime_info.ffi.cast("intptr_t", rdata))
set_runtime_data(form, fem.IntegralType.cell, subdomain_id, data_ptr)

# Assemble
A = fem.assemble_matrix(form)
A.scatter_reverse()
```

See `examples/dolfinx_integration.py` for a complete working example.

### Component Requests

Each `ComponentRequest` contains:

- `element`: The basix finite element
- `role`: "argument", "coefficient", "jacobian", or "coordinate"
- `index`: Argument number (0=test, 1=trial) or coefficient index
- `component`: Flat component index for vector/tensor elements
- `max_deriv`: Maximum derivative order needed
- `local_derivatives`: Tuple of derivative counts per reference direction

## Architecture

```
UFL form with runtime measure (metadata={"quadrature_rule": "runtime"})
  │
  v
runintgen.analysis.build_runtime_info
  │  (scan form_data.integral_data, identify runtime integrals)
  v
FFCX analysis + IR (with stripped runtime metadata)
  │
  v
runintgen.fe_tables.extract_integral_metadata
  │  (extract element/component/derivative requirements)
  v
runintgen.codegeneration.generate_C_runtime_kernels
  │  (generate C code for runtime integrals)
  v
C runtime kernels + metadata
  │
  v
JIT compile with cffi → function pointers
  │
  v
DOLFINx Form with custom kernel + custom_data
```

## API Reference

### Measures

- `RUNTIME_QUADRATURE_RULE`: Constant `"runtime"` for metadata
- `is_runtime_integral(integral)`: Check if an integral uses runtime quadrature
- `get_quadrature_provider(integral)`: Get the subdomain_data provider object

### Compilation

- `compile_runtime_integrals(form, options=None)`: Compile a form, returns `RunintModule`
- `compile_runtime_kernels(form, options=None)`: Compile and JIT, returns `RuntimeFormInfo`
- `RunintModule`: Container for compiled runtime kernels (C code)
- `RuntimeKernelInfo`: Information about a single runtime kernel
- `RuntimeFormInfo`: Compiled kernels with function pointers
- `CompiledKernel`: A compiled kernel ready for DOLFINx

### DOLFINx Integration

- `create_dolfinx_form_with_runtime(spaces, runtime_info, mesh)`: Create DOLFINx Form
- `set_runtime_data(form, integral_type, subdomain_id, data_ptr)`: Set custom_data

### Runtime Data

- `RuntimeDataBuilder`: Builder for runtime quadrature data structures
- `QuadratureConfig`: Configuration for a quadrature rule
- `tabulate_element(element, points, max_deriv_order)`: Tabulate element at points
- `get_runintgen_data_struct()`: Get C struct definition

### Metadata Extraction

- `extract_integral_metadata(runtime_info)`: Extract FE table metadata from IR
- `ComponentRequest`: Request for a specific element component
- `IntegralRuntimeMeta`: Aggregated metadata for an integral

## Examples

- `examples/simple_example.py`: Basic usage showing kernel generation
- `examples/dolfinx_integration.py`: Full DOLFINx integration with assembly

## License

MIT License - Copyright (c) 2025 ONERA

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

This project is developed at ONERA and builds upon the FEniCSx project:

- [FEniCSx](https://fenicsproject.org/)
- [Basix](https://github.com/FEniCS/basix)
- [UFL](https://github.com/FEniCS/ufl)
- [FFCx](https://github.com/FEniCS/ffcx)
