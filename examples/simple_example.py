"""Simple example demonstrating runintgen usage.

This example requires basix to be installed for FFCX compilation.
"""

from __future__ import annotations

import basix.ufl

import ufl

from runintgen import compile_runtime_integrals


def main():
    """Demonstrate basic runintgen workflow."""
    # Create a simple mesh domain using basix elements (required for FFCX)
    cell = ufl.triangle
    coord_elem = basix.ufl.element("Lagrange", cell.cellname, 1, shape=(2,))
    mesh = ufl.Mesh(coord_elem)

    # Create a function space element
    elem = basix.ufl.element("Lagrange", cell.cellname, 1)
    V = ufl.FunctionSpace(mesh, elem)

    # Create trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Create a runtime measure using metadata={"quadrature_rule": "runtime"}
    dx_rt = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=1,
        metadata={"quadrature_rule": "runtime"},
    )

    # Also use a standard measure for comparison
    dx = ufl.Measure("dx", domain=mesh)

    # Define a bilinear form with both standard and runtime integrals
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.inner(u, v) * dx_rt

    # Compile the runtime integrals
    print("Compiling runtime integrals...")
    module = compile_runtime_integrals(a)

    print(f"\nCompiled {len(module.kernels)} runtime kernel(s)")
    print(f"Form rank: {module.meta.get('form_rank')}")
    print(f"Number of runtime groups: {module.meta.get('num_runtime_groups')}")

    # Print kernel information
    for kernel in module.kernels:
        print(f"\n--- Kernel: {kernel.name} ---")
        print(f"Integral type: {kernel.integral_type}")
        print(f"Subdomain ID: {kernel.subdomain_id}")
        print("\nC Declaration:")
        print(kernel.c_declaration)
        print("\nC Definition:")
        print(kernel.c_definition)


if __name__ == "__main__":
    main()
