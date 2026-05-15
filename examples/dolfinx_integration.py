"""Example: assemble runintgen runtime and mixed forms through DOLFINx.

This example uses the production-style API:

    runintgen.dolfinx.form(a_ufl)

The returned object is a normal ``dolfinx.fem.Form`` and can be assembled with
the standard DOLFINx assembly functions. This example compares a runtime-only
form, a mixed form with half the local cells on standard quadrature and half on
runtime quadrature, and the all-standard DOLFINx form. The example requires the
patched DOLFINx build that supports per-integral ``custom_data``.
"""

from __future__ import annotations

import numpy as np
import ufl

try:
    import basix
    from dolfinx import fem
    from dolfinx.fem import functionspace
    from dolfinx.mesh import create_unit_square
    from mpi4py import MPI
except ImportError:  # pragma: no cover - example fallback
    MPI = None
    basix = None
    fem = None
    functionspace = None
    create_unit_square = None

from runintgen.dolfinx import form as runtime_form
from runintgen.runtime_data import QuadratureRules


def _cell_jacobian_determinants(mesh) -> np.ndarray:
    """Return affine cell ``abs(detJ)`` values for a linear triangle mesh."""
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_dofs = mesh.geometry.dofmap[:num_cells, :3]
    x = mesh.geometry.x
    coords = x[cell_dofs, :2]
    return np.abs(
        (coords[:, 1, 0] - coords[:, 0, 0])
        * (coords[:, 2, 1] - coords[:, 0, 1])
        - (coords[:, 2, 0] - coords[:, 0, 0])
        * (coords[:, 1, 1] - coords[:, 0, 1])
    )


def _local_cells(mesh) -> np.ndarray:
    """Return local cell indices as DOLFINx integration entities."""
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    return np.arange(num_cells, dtype=np.int32)


def _runtime_rules_for_cells(
    mesh,
    cells: np.ndarray | None = None,
) -> QuadratureRules:
    """Create one runtime quadrature rule per selected cell."""
    if cells is None:
        cells = _local_cells(mesh)
    cells = np.ascontiguousarray(cells, dtype=np.int32)
    points, weights = basix.make_quadrature(basix.CellType.triangle, 2)
    detJ = _cell_jacobian_determinants(mesh)[cells]
    nq = len(weights)
    return QuadratureRules(
        kind="per_entity",
        tdim=2,
        points=np.ascontiguousarray(np.tile(points, (len(cells), 1))),
        weights=np.ascontiguousarray((detJ[:, None] * weights[None, :]).ravel()),
        offsets=np.arange(len(cells) + 1, dtype=np.int64) * nq,
        parent_map=cells,
    )


def _split_standard_runtime_cells(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Split local cells into standard and runtime integration subsets."""
    cells = _local_cells(mesh)
    midpoint = len(cells) // 2
    return (
        np.ascontiguousarray(cells[:midpoint], dtype=np.int32),
        np.ascontiguousarray(cells[midpoint:], dtype=np.int32),
    )


def main() -> None:
    """Assemble and compare a runtime P1 Laplace matrix."""
    if MPI is None:
        print("DOLFINx is not available.")
        return

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, dtype=np.float64)
    V = functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    quadrature = _runtime_rules_for_cells(mesh)
    dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=quadrature)

    a_runtime = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt
    A_runtime = fem.assemble_matrix(runtime_form(a_runtime))
    A_runtime.scatter_reverse()

    standard_cells, runtime_cells = _split_standard_runtime_cells(mesh)
    mixed_quadrature = _runtime_rules_for_cells(mesh, runtime_cells)
    dx_mixed = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=0,
        subdomain_data=[standard_cells, mixed_quadrature],
    )
    a_mixed = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_mixed
    A_mixed = fem.assemble_matrix(runtime_form(a_mixed))
    A_mixed.scatter_reverse()

    a_standard = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A_standard = fem.assemble_matrix(fem.form(a_standard))
    A_standard.scatter_reverse()

    runtime_norm = np.sqrt(A_runtime.squared_norm())
    mixed_norm = np.sqrt(A_mixed.squared_norm())
    standard_norm = np.sqrt(A_standard.squared_norm())
    print(
        "Mixed cell split: "
        f"{len(standard_cells)} standard, {len(runtime_cells)} runtime"
    )
    print(f"Runtime matrix norm: {runtime_norm:.12e}")
    print(f"Mixed matrix norm: {mixed_norm:.12e}")
    print(f"Standard matrix norm: {standard_norm:.12e}")
    print(f"Runtime match: {np.isclose(runtime_norm, standard_norm)}")
    print(f"Mixed match: {np.isclose(mixed_norm, standard_norm)}")


if __name__ == "__main__":
    main()
