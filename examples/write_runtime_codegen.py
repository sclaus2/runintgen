#!/usr/bin/env python3
"""Write inspectable runintgen generated C files.

Run from any directory:

    python /path/to/runintgen/examples/write_runtime_codegen.py

The generated ``.h`` and ``.c`` files are written to the current working
directory by default, plus one shared runtime ABI header.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import ufl
from basix.ufl import element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runintgen import compile_runtime_integrals, write_runtime_code

SCALAR_TYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def create_forms() -> dict[str, ufl.Form]:
    """Create example runtime-marked UFL forms."""
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))

    v1 = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v2 = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))

    u1 = ufl.TrialFunction(v1)
    test1 = ufl.TestFunction(v1)
    u2 = ufl.TrialFunction(v2)
    test2 = ufl.TestFunction(v2)
    kappa = ufl.Coefficient(v1)

    dx_mass = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=1,
        metadata={"quadrature_rule": "runtime"},
    )
    dx_laplace = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=2,
        metadata={"quadrature_rule": "runtime"},
    )
    dx_coefficient = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=3,
        metadata={"quadrature_rule": "runtime"},
    )

    return {
        "mass_p1": ufl.inner(u1, test1) * dx_mass,
        "laplace_p2": ufl.inner(ufl.grad(u2), ufl.grad(test2)) * dx_laplace,
        "coefficient_mass_p1": kappa * ufl.inner(u1, test1) * dx_coefficient,
    }


def main() -> None:
    """Run the code generation example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated files. Defaults to the current directory.",
    )
    parser.add_argument(
        "--scalar-type",
        choices=sorted(SCALAR_TYPES),
        default="float64",
    )
    args = parser.parse_args()

    forms = create_forms()
    options = {"scalar_type": SCALAR_TYPES[args.scalar_type]}

    print(f"Writing generated runtime kernels to {args.output_dir}")
    abi_header = None
    for name, form in forms.items():
        module = compile_runtime_integrals(form, options=options)
        files = write_runtime_code(module, prefix=name, output_dir=args.output_dir)
        abi_header = files.abi_header
        print(f"\n{name}")
        print(f"  header:   {files.header}")
        print(f"  source:   {files.source}")
        for kernel in module.kernels:
            print(
                "  kernel:   "
                f"{kernel.name} "
                f"scalar={kernel.scalar_type} "
                f"geometry={kernel.geometry_type} "
                f"tables={len(kernel.table_info or [])}"
            )
    if abi_header is not None:
        print(f"\nshared ABI header: {abi_header}")


if __name__ == "__main__":
    main()
