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
import numpy as np
import basix
import ufl
from basix.ufl import element

from runintgen import (
    compile_runtime_integrals,
    prepare_runtime_data_for_cell,
)

# Define mesh and function space
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create a runtime measure using standard UFL with special metadata
dx_rt = ufl.Measure("dx", domain=mesh,
                    metadata={"quadrature_rule": "runtime"})

# Define a bilinear form using the runtime measure
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

# Compile runtime integration kernels
module = compile_runtime_integrals(a)

# Access kernel information
for kernel in module.kernels:
    print(f"Kernel: {kernel.name}")
    print(f"Integral type: {kernel.integral_type}")
    print(f"Required element tables: {len(kernel.table_info)}")

    # Inspect element table requirements
    for t in kernel.table_info:
        print(f"  - ndofs={t['ndofs']}, max_deriv={t['max_derivative_order']}")
```

## Key Concepts

### Runtime Measures

Use standard UFL measures with `metadata={"quadrature_rule": "runtime"}` to mark integrals for runtime quadrature:

```python
import ufl

# Cell integral with runtime quadrature
dx_rt = ufl.Measure("dx", domain=mesh,
                    metadata={"quadrature_rule": "runtime"})

# Exterior facet integral with runtime quadrature  
ds_rt = ufl.Measure("ds", domain=mesh,
                    metadata={"quadrature_rule": "runtime"})

# Interior facet integral with runtime quadrature
dS_rt = ufl.Measure("dS", domain=mesh,
                    metadata={"quadrature_rule": "runtime"})
```

The constant `RUNTIME_QUADRATURE_RULE = "runtime"` is provided for convenience:

```python
from runintgen import RUNTIME_QUADRATURE_RULE

dx_rt = ufl.Measure("dx", domain=mesh,
                    metadata={"quadrature_rule": RUNTIME_QUADRATURE_RULE})
```

### Runtime Data Structures

The compiled kernels use a simplified runtime data structure:

```c
typedef struct {
  int ndofs;
  int nderivs;
  const double* table;  // [nderivs, nq, ndofs] flattened
} runintgen_element;

typedef struct {
  int nq;                              // Number of quadrature points
  const double* points;                // [nq * tdim] quadrature points
  const double* weights;               // [nq] weights (should include |detJ|)
  int nelements;                       // Number of element tables
  const runintgen_element* elements;   // Array of element tables
} runintgen_data;
```

**Important**: The weights should already include `|detJ|` (Jacobian determinant). The generated kernel does NOT multiply by `|detJ|` internally—this is the caller's responsibility.

### Preparing Runtime Data

Use the tabulation utilities to automatically prepare runtime data from kernel metadata:

```python
from runintgen import (
    compile_runtime_integrals,
    prepare_runtime_data_for_cell,
)

# Compile the form
module = compile_runtime_integrals(form)
kernel_info = module.kernels[0]

# Define cell coordinates
coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

# Prepare runtime data (auto-computes |detJ| and scales weights)
prepared = prepare_runtime_data_for_cell(
    kernel_info, coords, quadrature_degree=4
)

# Build the C data structure
data = prepared.builder.build()
```

For more control, use `prepare_runtime_data()` with pre-scaled weights:

```python
from runintgen import prepare_runtime_data, compute_detJ_triangle
import basix

# Get quadrature rule
points, weights = basix.make_quadrature(basix.CellType.triangle, 4)

# Compute and apply |detJ|
detJ = compute_detJ_triangle(coords)
weights_scaled = weights * np.abs(detJ)

# Prepare runtime data
prepared = prepare_runtime_data(kernel_info, points, weights_scaled)
data = prepared.builder.build()
```

### Low-Level RuntimeDataBuilder API

For maximum control, use `RuntimeDataBuilder` directly:

```python
from runintgen import RuntimeDataBuilder, ElementTableInfo, CFFI_DEF
import cffi

# Create FFI instance
ffi = cffi.FFI()
ffi.cdef(CFFI_DEF)

# Create builder
builder = RuntimeDataBuilder(ffi)

# Set quadrature (weights should include |detJ|)
builder.set_quadrature(points, weights_scaled)

# Add element tables
builder.add_element_table(ElementTableInfo(
    ndofs=3,
    nderivs=3,  # (0,0), (1,0), (0,1)
    table=table_array  # shape [nderivs, nq, ndofs]
))

# Build the structure
data = builder.build()
```

### Kernel Table Info

Each `RuntimeKernelInfo` contains `table_info` describing required element tables:

```python
for kernel in module.kernels:
    for t in kernel.table_info:
        print(f"Element {t['index']}:")
        print(f"  ndofs: {t['ndofs']}")
        print(f"  ncomps: {t['ncomps']}")
        print(f"  max_derivative_order: {t['max_derivative_order']}")
        print(f"  is_argument: {t['is_argument']}")
        print(f"  is_coordinate: {t['is_coordinate']}")
        print(f"  usages: {[u['role'] for u in t['usages']]}")
```

## Architecture

```
UFL form with runtime measure (metadata={"quadrature_rule": "runtime"})
  │
  v
runintgen.analysis.build_runtime_analysis
  │  (scan form, identify runtime integrals, extract element info)
  v
FFCX analysis + IR computation
  │
  v
runintgen.codegeneration.generate_C_runtime_kernels
  │  (generate C code with runtime quadrature/tables)
  v
RunintModule with RuntimeKernelInfo
  │  (contains C code + table_info metadata)
  v
prepare_runtime_data / RuntimeDataBuilder
  │  (tabulate elements, build C structures)
  v
Call kernel with runintgen_data*
```

## API Reference

### Measures

- `RUNTIME_QUADRATURE_RULE`: Constant `"runtime"` for metadata
- `is_runtime_integral(integral)`: Check if an integral uses runtime quadrature
- `get_quadrature_provider(integral)`: Get the subdomain_data provider object

### Compilation

- `compile_runtime_integrals(form, options=None)`: Compile a form, returns `RunintModule`
- `RunintModule`: Container for compiled runtime kernels
- `RuntimeKernelInfo`: Information about a single runtime kernel (name, C code, table_info)

### Tabulation Utilities

- `prepare_runtime_data(kernel_info, points, weights)`: Prepare data from kernel metadata
- `prepare_runtime_data_for_cell(kernel_info, coords, quadrature_degree)`: Auto-compute |detJ|
- `tabulate_from_table_info(table_info, points)`: Tabulate single element from metadata
- `compute_detJ_triangle(coords)`: Compute Jacobian determinant for P1 triangle
- `PreparedTables`: Container with builder, ffi, and table arrays

### Runtime Data

- `RuntimeDataBuilder`: Builder for runtime data structures
- `ElementTableInfo`: Container for element tabulation data
- `tabulate_element(element, points, max_deriv_order)`: Tabulate basix element
- `CFFI_DEF`: C struct definitions for CFFI
- `to_intptr(ffi, data)`: Convert pointer to integer for FFI

### Analysis

- `build_runtime_analysis(form, options)`: Full analysis returning `RuntimeAnalysisInfo`
- `RuntimeAnalysisInfo`: Contains IR, runtime groups, and integral metadata
- `ArgumentInfo`, `ElementInfo`, `RuntimeIntegralInfo`: Analysis data structures

### Code Generation

- `get_runintgen_data_struct()`: Get C struct definition string

## Examples

- `examples/laplacian_runtime.py`: Basic P2 Laplacian kernel generation
- `examples/elasticity_runtime.py`: Vector-valued linear elasticity
- `examples/stokes_runtime.py`: Mixed Stokes problem (Taylor-Hood P2-P1)
- `examples/tabulation_example.py`: Generic tabulation workflow

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
