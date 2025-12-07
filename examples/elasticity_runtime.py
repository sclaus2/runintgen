#!/usr/bin/env python3
"""Example: Vector-valued problem (linear elasticity).

This example demonstrates runtime quadrature for a vector-valued problem
using linear elasticity with vector elements.

We define the bilinear form for linear elasticity:
    a(u, v) = ∫ σ(u) : ε(v) dx

where:
    ε(u) = 1/2 (∇u + (∇u)^T)   (strain tensor)
    σ(u) = λ tr(ε(u)) I + 2μ ε(u)  (stress tensor)
"""

import ufl
from basix.ufl import element

from runintgen import compile_runtime_integrals


def epsilon(u):
    """Symmetric gradient (strain tensor)."""
    return ufl.sym(ufl.grad(u))


def sigma(u, lam, mu):
    """Stress tensor for linear elasticity."""
    return lam * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


def main():
    print("=" * 70)
    print("VECTOR-VALUED PROBLEM: LINEAR ELASTICITY")
    print("=" * 70)

    # Define mesh and vector function space
    # P1 mesh (linear triangles)
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))

    # Vector P1 elements for displacement
    V_el = element("Lagrange", "triangle", 1, shape=(2,))
    V = ufl.FunctionSpace(mesh, V_el)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Material parameters (Lamé parameters)
    lam = ufl.Constant(mesh)  # First Lamé parameter
    mu = ufl.Constant(mesh)  # Second Lamé parameter (shear modulus)

    # Runtime measure
    dx_rt = ufl.Measure(
        "dx", domain=mesh, subdomain_id=1, metadata={"quadrature_rule": "runtime"}
    )

    # Linear elasticity bilinear form
    a = ufl.inner(sigma(u, lam, mu), epsilon(v)) * dx_rt

    # Compile runtime kernels
    module = compile_runtime_integrals(a)

    # Print kernel information
    for kernel in module.kernels:
        print(f"\nKERNEL: {kernel.name}")
        print(f"Integral type: {kernel.integral_type}")
        print(f"Subdomain ID: {kernel.subdomain_id}")
        print()

        print("Required FE elements:")
        for t in kernel.table_info:
            print(f"  Element {t['index']}:")
            print(f"    DOFs: {t['ndofs']}")
            print(f"    Components: {t['ncomps']}")
            print(f"    Max derivative order: {t['max_derivative_order']}")
            print(f"    Is argument: {t['is_argument']}")
            print(f"    Is coordinate: {t['is_coordinate']}")
            print(f"    Usages: {[u['role'] for u in t['usages']]}")
        print()

        print("C declaration:")
        print(kernel.c_declaration)
        print()

        # Print first 50 lines of C code
        lines = kernel.c_definition.split("\n")
        print(f"C definition (first 50 lines of {len(lines)}):")
        print("\n".join(lines[:50]))
        if len(lines) > 50:
            print(f"  ... ({len(lines) - 50} more lines)")


if __name__ == "__main__":
    main()
