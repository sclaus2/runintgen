"""Example: DOLFINx-backed QuadratureFunction values.

This example shows the FEniCSx-style path:

- create a normal DOLFINx ``Function`` with runintgen quadrature semantics;
- fill its background DOF vector by interpolation;
- let the DOLFINx adapter evaluate it at runtime quadrature points while
  creating ``custom_data``.

The generated kernel still reads ``q_function_0`` from ``custom_data``. The
background Function is only used in the preprocessing step.
"""

from __future__ import annotations

import numpy as np
import ufl

try:
    import basix
    from basix.ufl import element
    from dolfinx import fem
    from dolfinx.mesh import create_unit_square
    from mpi4py import MPI
except ImportError:  # pragma: no cover - example fallback
    basix = None
    element = None
    fem = None
    create_unit_square = None
    MPI = None

from runintgen import QuadratureRules, compile_runtime_integrals, dxq
from runintgen.dolfinx import QuadratureFunction, create_custom_data


def main() -> None:
    """Create custom_data from a DOLFINx background Function."""
    if MPI is None:
        raise RuntimeError("This example requires DOLFINx and mpi4py.")

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, dtype=np.float64)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    alpha = QuadratureFunction(mesh, name="alpha")
    alpha.interpolate(lambda x: np.sin(x[0]) + np.cos(x[1]))

    reference_points, reference_weights = basix.make_quadrature(
        basix.CellType.triangle,
        2,
    )
    cells = np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local,
        dtype=np.int32,
    )
    points = np.ascontiguousarray(np.tile(reference_points, (cells.size, 1)))
    weights = np.ascontiguousarray(np.tile(reference_weights, cells.size))
    offsets = np.arange(
        cells.size + 1,
        dtype=np.int64,
    ) * reference_weights.size
    rules = QuadratureRules(
        kind="per_entity",
        tdim=mesh.topology.dim,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=cells,
    )

    form = alpha * ufl.inner(u, v) * dxq(
        domain=mesh.ufl_domain(),
        quadrature_provider=rules,
    )
    module = compile_runtime_integrals(form)
    data = create_custom_data(module, mesh, rules)

    print("q-functions:", [info.label for info in module.quadrature_functions])
    print("custom_data pointer:", int(data.ptr))
    print(
        "generated q-function load:",
        "q_function_0[q0 + iq]" in module.kernels[0].c_definition,
    )


if __name__ == "__main__":
    main()
