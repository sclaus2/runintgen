# runintgen

Runtime integration kernel generator for FEniCSx/FFCx.

## Overview

**runintgen** is a Python library that extends FFCx (FEniCSx Form Compiler) to support runtime-defined integration kernels. This enables the use of custom quadrature rules that are determined at runtime rather than compile time, which is essential for applications like:

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

## Quick Start

```python
import ufl
from basix.ufl import element
from runintgen import runtime_dx, compile_runtime_integrals

# Define mesh and function space
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create a runtime measure with a tag for identification
dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="cut_cell")

# Define a bilinear form using the runtime measure
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

# Compile runtime integration kernels
module = compile_runtime_integrals(a)

# Access kernel information
for kernel in module.kernels:
    print(f"Kernel: {kernel.name}, tag: {kernel.tag}")
```

## Key Concepts

### Runtime Measures

Use `runtime_dx()` to create a measure that marks integrals for runtime quadrature:

```python
from runintgen import runtime_dx

# Basic usage
dx_rt = runtime_dx(subdomain_id=1, domain=mesh)

# With a tag for identification
dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="my_quadrature")

# With custom payload data
dx_rt = runtime_dx(
    subdomain_id=1, 
    domain=mesh, 
    tag="custom", 
    payload={"order": 5, "type": "gauss"}
)
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
UFL form with runtime_dx measure
  |  (dx integrals carry subdomain_data = RuntimeQuadrature)
  v
runintgen.analysis.build_runtime_info
  |  (scan form_data.integral_data, identify runtime integrals)
  v
FFCX analysis + IR (unchanged)
  v
runintgen.fe_tables.extract_integral_metadata
  |  (extract element/component/derivative requirements)
  v
runintgen.codegen.generate_runtime_kernels
  |  (generate C code for runtime integrals)
  v
C runtime kernels + metadata
```

## API Reference

### Measures

- `runtime_dx(subdomain_id, domain, tag=None, payload=None, metadata=None)`: Create a runtime cell measure
- `RuntimeQuadrature`: Dataclass holding runtime quadrature metadata
- `is_runtime_integral(integral)`: Check if an integral uses runtime quadrature

### Compilation

- `compile_runtime_integrals(form, options=None)`: Compile a form with runtime integrals
- `RunintModule`: Container for compiled runtime kernels
- `RuntimeKernelInfo`: Information about a single runtime kernel

### Metadata Extraction

- `extract_integral_metadata(runtime_info)`: Extract FE table metadata from IR
- `ComponentRequest`: Request for a specific element component
- `IntegralRuntimeMeta`: Aggregated metadata for an integral

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
