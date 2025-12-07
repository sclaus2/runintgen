#!/usr/bin/env python3
"""Example: Stokes mixed problem with Taylor-Hood elements.

This example demonstrates runtime quadrature for a mixed problem
using the Stokes equations with Taylor-Hood (P2-P1) elements.

The Stokes equations in weak form:
    a((u, p), (v, q)) = ∫ ∇u : ∇v dx - ∫ p div(v) dx - ∫ div(u) q dx

where:
    u: velocity (vector P2)
    p: pressure (scalar P1)
    v, q: test functions

This produces a saddle point system with separate trial/test function spaces.
"""

import ufl
from basix.ufl import element, mixed_element

from runintgen import compile_runtime_integrals


def main():
    print("=" * 70)
    print("MIXED PROBLEM: STOKES EQUATIONS (TAYLOR-HOOD P2-P1)")
    print("=" * 70)

    # Define mesh
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))

    # Taylor-Hood elements: P2 velocity, P1 pressure
    P2_vec = element("Lagrange", "triangle", 2, shape=(2,))  # Vector P2 for velocity
    P1 = element("Lagrange", "triangle", 1)  # Scalar P1 for pressure

    # Mixed element
    TH = mixed_element([P2_vec, P1])
    W = ufl.FunctionSpace(mesh, TH)

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Runtime measure
    dx_rt = ufl.Measure(
        "dx", domain=mesh, subdomain_id=1, metadata={"quadrature_rule": "runtime"}
    )

    # Viscosity
    nu = ufl.Constant(mesh)

    # Stokes bilinear form
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt
        - p * ufl.div(v) * dx_rt
        - ufl.div(u) * q * dx_rt
    )

    # Compile runtime kernels
    print("\nCompiling Stokes bilinear form...")
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


def stokes_velocity_only():
    """Alternative: Stokes with separate velocity-velocity and velocity-pressure blocks."""
    print("\n" + "=" * 70)
    print("ALTERNATIVE: SEPARATE STOKES BLOCKS")
    print("=" * 70)

    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))

    # Velocity space (P2 vector)
    V_el = element("Lagrange", "triangle", 2, shape=(2,))
    V = ufl.FunctionSpace(mesh, V_el)

    # Pressure space (P1 scalar)
    Q_el = element("Lagrange", "triangle", 1)
    Q = ufl.FunctionSpace(mesh, Q_el)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    # Runtime measure
    dx_rt = ufl.Measure(
        "dx", domain=mesh, subdomain_id=1, metadata={"quadrature_rule": "runtime"}
    )

    nu = ufl.Constant(mesh)

    # Velocity-velocity block: a_uu = ν ∫ ∇u : ∇v dx
    a_uu = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

    # Pressure-velocity block: a_up = -∫ p div(v) dx
    a_up = -p * ufl.div(v) * dx_rt

    # Velocity-pressure block: a_pu = -∫ div(u) q dx
    a_pu = -ufl.div(u) * q * dx_rt

    print("\n--- Block a_uu (velocity-velocity) ---")
    module_uu = compile_runtime_integrals(a_uu)
    for kernel in module_uu.kernels:
        print(f"Kernel: {kernel.name}")
        print(f"Elements: {len(kernel.table_info)}")
        for t in kernel.table_info:
            print(
                f"  - ndofs={t['ndofs']}, ncomps={t['ncomps']}, "
                f"is_arg={t['is_argument']}, max_deriv={t['max_derivative_order']}"
            )

    print("\n--- Block a_up (pressure-velocity) ---")
    module_up = compile_runtime_integrals(a_up)
    for kernel in module_up.kernels:
        print(f"Kernel: {kernel.name}")
        print(f"Elements: {len(kernel.table_info)}")
        for t in kernel.table_info:
            print(
                f"  - ndofs={t['ndofs']}, ncomps={t['ncomps']}, "
                f"is_arg={t['is_argument']}, max_deriv={t['max_derivative_order']}"
            )

    print("\n--- Block a_pu (velocity-pressure) ---")
    module_pu = compile_runtime_integrals(a_pu)
    for kernel in module_pu.kernels:
        print(f"Kernel: {kernel.name}")
        print(f"Elements: {len(kernel.table_info)}")
        for t in kernel.table_info:
            print(
                f"  - ndofs={t['ndofs']}, ncomps={t['ncomps']}, "
                f"is_arg={t['is_argument']}, max_deriv={t['max_derivative_order']}"
            )


if __name__ == "__main__":
    main()
    stokes_velocity_only()
