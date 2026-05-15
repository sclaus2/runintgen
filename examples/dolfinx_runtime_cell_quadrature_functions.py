"""Comprehensive runtime-cell QuadratureFunction example.

This is the runtime-only v1 path:

- standard branches may use ordinary DOLFINx coefficients;
- ``QuadratureFunction`` terminals appear only in ``dx_rt``/``dxq`` branches;
- runtime quadrature values are prepared before assembly and read from
  ``custom_data`` by generated kernels.

The runtime branch combines three value sources:

- a DOLFINx-backed ``QuadratureFunction`` evaluated at quadrature points;
- a pure NumPy callable evaluated on component-first physical points;
- an explicit provider-owned vector array attached with ``set_values``.
"""

from __future__ import annotations

import numpy as np
import ufl

try:
    import basix
    from dolfinx import fem
    from dolfinx.mesh import create_unit_square
    from mpi4py import MPI
except ImportError:  # pragma: no cover - example fallback
    basix = None
    fem = None
    create_unit_square = None
    MPI = None

import runintgen as rig
import runintgen.dolfinx as rigx


def _runtime_cell_rules(mesh) -> rig.QuadratureRules:
    """Create one per-cell runtime rule with physical points attached."""
    reference_points, reference_weights = basix.make_quadrature(
        basix.CellType.triangle,
        2,
    )
    cells = np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local,
        dtype=np.int32,
    )
    nq = int(reference_weights.size)

    # Runtime kernels expect integration-scaled weights. For this affine
    # 2x2 unit-square triangle mesh, every cell map has |detJ| = 0.25.
    scaled_reference_weights = np.ascontiguousarray(reference_weights * 0.25)

    rules = rig.QuadratureRules(
        kind="per_entity",
        tdim=mesh.topology.dim,
        points=np.ascontiguousarray(np.tile(reference_points, (cells.size, 1))),
        weights=np.ascontiguousarray(np.tile(scaled_reference_weights, cells.size)),
        offsets=np.arange(cells.size + 1, dtype=np.int64) * nq,
        parent_map=cells,
    )
    return rigx.compute_physical_points(mesh, rules)


def main() -> None:
    """Assemble a mixed standard/runtime cell form."""
    if MPI is None:
        raise RuntimeError("This example requires DOLFINx and mpi4py.")

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, dtype=np.float64)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    rules = _runtime_cell_rules(mesh)
    xq = rules.physical_points

    # Ordinary standard coefficient: this remains in the normal DOLFINx/FFCx
    # coefficient/constant path and appears only in the standard branch.
    kappa = fem.Constant(mesh, np.float64(0.25))

    # DOLFINx-backed q-function: values are stored in a concrete Function, then
    # evaluated at runtime quadrature points while custom_data is created.
    alpha = rigx.QuadratureFunction(mesh, name="alpha")
    alpha.interpolate(lambda x: 1.0 + x[0] + 0.5 * x[1])

    # Callable q-function: evaluated once on component-first physical points.
    beta = rigx.QuadratureFunction(
        mesh,
        lambda x: 0.5 + np.sin(x[0]) * np.cos(x[1]),
        name="beta",
    )

    # Explicit provider-owned vector q-function. The array is point-major with
    # component as the innermost stride: values[global_q, component].
    direction = rigx.QuadratureFunction(
        mesh,
        name="direction",
        shape=(mesh.geometry.dim,),
    )
    direction_values = np.ascontiguousarray(
        np.column_stack(
            (
                1.0 + 0.0 * xq[0],
                0.25 + 0.0 * xq[1],
            )
        ),
        dtype=np.float64,
    )
    direction.set_values(rules, direction_values)

    # Preferred FEniCSx-style spelling: put the runtime QuadratureRules directly
    # into UFL subdomain_data. runintgen detects the rule object and compiles
    # this branch as a runtime cell integral.
    dx_rt = ufl.dx(domain=mesh.ufl_domain(), subdomain_data=rules)
    standard_branch = kappa * ufl.inner(u, v) * ufl.dx(domain=mesh.ufl_domain())
    runtime_branch = (
        alpha * ufl.inner(ufl.grad(u), ufl.grad(v))
        + beta * ufl.inner(u, v)
        + ufl.dot(direction, ufl.grad(u)) * v
    ) * dx_rt
    form = standard_branch + runtime_branch

    runint_form = rigx.form(form)
    matrix = fem.assemble_matrix(runint_form)
    compiled = runint_form._runintgen_compiled_form
    labels = [info.label for info in compiled.jit_info.module.quadrature_functions]

    print("runtime q-functions:", labels)
    print("runtime rules:", rules.num_rules)
    print("runtime quadrature points:", rules.total_points)
    print("assembled matrix entries:", matrix.data.size)
    print("assembled matrix data norm:", float(np.linalg.norm(matrix.data)))


if __name__ == "__main__":
    main()
