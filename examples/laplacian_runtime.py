#!/usr/bin/env python3
"""Example: Generate runtime integration kernel for Laplacian.

This example demonstrates how to use runintgen to generate C kernels
for runtime integration, where quadrature points and FE tables are
provided at runtime instead of being embedded in the kernel.

The generated kernel signature is:
    void kernel(
        double* A,                    // Output tensor
        const double* w,              // Coefficients (unused for bilinear form)
        const double* c,              // Constants (unused here)
        const double* coordinate_dofs,// Cell vertex coordinates
        const int* entity_local_index,// Not used
        const uint8_t* quadrature_permutation, // Not used
        void* custom_data             // Pointer to runintgen_data struct
    );

The caller must populate the runintgen_data struct with:
    - nq: number of quadrature points
    - points: quadrature points in reference coordinates [nq * gdim]
    - weights: quadrature weights [nq]
    - ntables: number of FE tables
    - tables: array of FE table pointers
    - table_ndofs: DOFs per table
"""

import ufl
from basix.ufl import element

from runintgen import (
    compile_runtime_integrals,
    get_runintgen_data_struct,
    runtime_dx,
)


def main():
    # Define mesh and function space
    # P1 mesh (linear triangles) with P2 solution space
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V_el = element("Lagrange", "triangle", 2)
    V = ufl.FunctionSpace(mesh, V_el)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Create a runtime measure
    # The tag can be used to identify different quadrature schemes
    dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="custom_quadrature")

    # Define the Laplacian bilinear form
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

    # Compile runtime integration kernels
    module = compile_runtime_integrals(a)

    # Print the C struct definition
    print("=" * 70)
    print("RUNTIME DATA STRUCT DEFINITION")
    print("=" * 70)
    print(get_runintgen_data_struct())

    # Print kernel information
    for kernel in module.kernels:
        print("=" * 70)
        print(f"KERNEL: {kernel.name}")
        print("=" * 70)
        print(f"Integral type: {kernel.integral_type}")
        print(f"Subdomain ID: {kernel.subdomain_id}")
        print(f"Tag: {kernel.tag}")
        print()

        print("Required FE tables:")
        for t in kernel.table_info:
            print(f"  Table {t['index']}:")
            print(f"    FFCX name: {t['name']}")
            print(f"    Role: {t['role']}[{t['terminal_index']}]")
            print(f"    Derivative: {t['derivative']}")
            print(f"    DOFs: {t['ndofs']}")
        print()

        print("C declaration:")
        print(kernel.c_declaration)
        print()

        print("C definition:")
        print(kernel.c_definition)

    # Explain how to use the kernel
    print("=" * 70)
    print("USAGE")
    print("=" * 70)
    print("""
To use the generated kernel:

1. Include the runintgen_data struct definition in your C/C++ code.

2. At runtime, compute your custom quadrature points and weights.

3. Evaluate the FE basis functions at those quadrature points using basix:
   
   import basix
   import numpy as np
   
   # For P2 Lagrange on triangle
   elem = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 2)
   
   # Evaluate derivatives at quadrature points
   # For derivative (1,0) = d/dX:
   dphi_dX = elem.tabulate(1, quad_points)[1, :, :, 0]  # shape: (nq, ndofs)
   # For derivative (0,1) = d/dY:
   dphi_dY = elem.tabulate(1, quad_points)[2, :, :, 0]  # shape: (nq, ndofs)

4. Populate the runintgen_data struct and call the kernel.
""")


if __name__ == "__main__":
    main()
