"""Compare direct-entity runtime assembly with MeshTags standard assembly.

This example exercises the runtime-integral path where FFCx/DOLFINx sees two
cell integral ids. The standard reference form uses ordinary DOLFINx
``MeshTags``. The runintgen form uses the preferred direct entity payload:

    subdomain_data=[standard_entities, runtime_rules]

for each subdomain id. This keeps entity ownership explicit and avoids relying
on MeshTags for runtime quadrature routing.
"""

from __future__ import annotations

import argparse

import numpy as np
import ufl

try:
    import basix
    from dolfinx import fem
    from dolfinx.fem import functionspace
    from dolfinx.mesh import create_unit_square, meshtags
    from mpi4py import MPI
except ImportError:  # pragma: no cover - example fallback
    MPI = None
    basix = None
    fem = None
    functionspace = None
    create_unit_square = None
    meshtags = None

import runintgen.dolfinx as rigx
from runintgen.runtime_data import QuadratureRules

DEFAULT_ACTIVE_SUBDOMAIN_IDS = (0, 1)


def _local_cells(mesh) -> np.ndarray:
    """Return owned local cell indices."""
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    return np.arange(num_cells, dtype=np.int32)


def _cell_vertices(mesh) -> np.ndarray:
    """Return local affine-cell vertex coordinates."""
    cells = _local_cells(mesh)
    cell_dofs = mesh.geometry.dofmap[cells, :3]
    return mesh.geometry.x[cell_dofs, : mesh.topology.dim]


def _cell_jacobian_determinants(mesh) -> np.ndarray:
    """Return affine cell ``abs(detJ)`` values."""
    x = _cell_vertices(mesh)
    return np.abs(
        (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1])
        - (x[:, 2, 0] - x[:, 0, 0]) * (x[:, 1, 1] - x[:, 0, 1])
    )


def _two_subdomain_entities(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Split cells into two entity arrays by x-coordinate of their midpoint."""
    cells = _local_cells(mesh)
    midpoints = _cell_vertices(mesh).mean(axis=1)
    return (
        np.ascontiguousarray(cells[midpoints[:, 0] < 0.5], dtype=np.int32),
        np.ascontiguousarray(cells[midpoints[:, 0] >= 0.5], dtype=np.int32),
    )


def _mesh_tags_from_entities(
    mesh,
    entities_0: np.ndarray,
    entities_1: np.ndarray,
):
    """Create DOLFINx MeshTags for the standard reference assembly."""
    cells = np.ascontiguousarray(np.concatenate([entities_0, entities_1]))
    values = np.ascontiguousarray(
        np.concatenate(
            [
                np.zeros(entities_0.size, dtype=np.int32),
                np.ones(entities_1.size, dtype=np.int32),
            ]
        )
    )
    order = np.argsort(cells)
    return meshtags(mesh, mesh.topology.dim, cells[order], values[order])


def _split_standard_runtime_entities(
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one subdomain into direct standard entities and runtime entities."""
    return (
        np.ascontiguousarray(cells[::2], dtype=np.int32),
        np.ascontiguousarray(cells[1::2], dtype=np.int32),
    )


def _runtime_rules_for_cells(mesh, cells: np.ndarray) -> QuadratureRules:
    """Create one runtime quadrature rule per selected cell."""
    cells = np.ascontiguousarray(cells, dtype=np.int32)
    points, weights = basix.make_quadrature(basix.CellType.triangle, 2)
    detj = _cell_jacobian_determinants(mesh)[cells]
    nq = int(weights.size)
    return QuadratureRules(
        kind="per_entity",
        tdim=mesh.topology.dim,
        points=np.ascontiguousarray(np.tile(points, (cells.size, 1))),
        weights=np.ascontiguousarray((detj[:, None] * weights[None, :]).ravel()),
        offsets=np.arange(cells.size + 1, dtype=np.int64) * nq,
        parent_map=cells,
    )


def _assemble_dense(form) -> np.ndarray:
    """Assemble a DOLFINx form and return a dense local matrix."""
    matrix = fem.assemble_matrix(form)
    matrix.scatter_reverse()
    return matrix.to_dense()


def _compiled_integral_summary(form) -> list[tuple[int, str, bool]]:
    """Return ``(subdomain_id, kernel_mode, needs_custom_data)`` for a form."""
    compiled = form._runintgen_compiled_form
    return [
        (info.subdomain_id, info.kernel.mode, info.needs_custom_data)
        for info in compiled.jit_info.integral_infos
    ]


def _sum_terms(terms: list[ufl.Form]) -> ufl.Form:
    """Return the sum of a non-empty list of UFL form terms."""
    if not terms:
        raise ValueError("At least one form term is required.")
    result = terms[0]
    for term in terms[1:]:
        result += term
    return result


def _parse_active_subdomain_ids(value: str) -> tuple[int, ...]:
    """Parse comma-separated active subdomain ids from the command line."""
    try:
        ids = tuple(int(part) for part in value.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "active subdomain ids must be comma-separated integers"
        ) from exc
    if not ids:
        raise argparse.ArgumentTypeError("at least one subdomain id is required")
    if not set(ids).issubset({0, 1}):
        raise argparse.ArgumentTypeError("subdomain ids must be 0 and/or 1")
    return ids


def _parse_args() -> argparse.Namespace:
    """Return command-line arguments for the example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-subdomain-ids",
        default=DEFAULT_ACTIVE_SUBDOMAIN_IDS,
        type=_parse_active_subdomain_ids,
        help="Comma-separated subdomain ids to include in the assembled form.",
    )
    return parser.parse_args()


def main(active_subdomain_ids: tuple[int, ...] = DEFAULT_ACTIVE_SUBDOMAIN_IDS) -> None:
    """Assemble runtime and standard forms over two subdomain ids."""
    if MPI is None:
        print("DOLFINx is not available.")
        return

    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=np.float64)
    V = functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    cells_0, cells_1 = _two_subdomain_entities(mesh)
    cell_tags = _mesh_tags_from_entities(mesh, cells_0, cells_1)
    standard_0, runtime_0 = _split_standard_runtime_entities(cells_0)
    standard_1, runtime_1 = _split_standard_runtime_entities(cells_1)
    standard_entities = {0: standard_0, 1: standard_1}
    runtime_entities = {0: runtime_0, 1: runtime_1}

    rules_0 = _runtime_rules_for_cells(mesh, runtime_0)
    rules_1 = _runtime_rules_for_cells(mesh, runtime_1)

    integrand = ufl.inner(ufl.grad(u), ufl.grad(v)) + 0.25 * ufl.inner(u, v)

    dx_standard = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    active_ids = tuple(active_subdomain_ids)
    if not set(active_ids).issubset({0, 1}):
        raise ValueError("active_subdomain_ids must contain only 0 and/or 1.")
    a_standard_full = integrand * dx_standard(0) + integrand * dx_standard(1)
    a_standard = _sum_terms([integrand * dx_standard(sid) for sid in active_ids])

    runtime_measures = {
        0: ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_id=0,
            subdomain_data=[standard_0, rules_0],
        ),
        1: ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_id=1,
            subdomain_data=[standard_1, rules_1],
        ),
    }
    a_runtime = _sum_terms(
        [integrand * runtime_measures[sid] for sid in active_ids]
    )

    runtime_form = rigx.form(a_runtime)
    standard_form = rigx.form(a_standard)
    standard_full_form = rigx.form(a_standard_full)
    A_runtime = _assemble_dense(runtime_form)
    A_standard = _assemble_dense(standard_form)
    A_standard_full = _assemble_dense(standard_full_form)

    active_error = np.linalg.norm(A_runtime - A_standard, ord=np.inf)
    full_error = np.linalg.norm(A_runtime - A_standard_full, ord=np.inf)
    active_reference = np.linalg.norm(A_standard)
    full_reference = np.linalg.norm(A_standard_full)
    active_standard_entities = sum(standard_entities[sid].size for sid in active_ids)
    active_runtime_entities = sum(runtime_entities[sid].size for sid in active_ids)
    np.testing.assert_allclose(A_runtime, A_standard, rtol=1.0e-12, atol=1.0e-14)

    runtime_integral_summary = _compiled_integral_summary(runtime_form)
    runtime_ids = [
        subdomain_id
        for subdomain_id, _, needs_custom_data in runtime_integral_summary
        if needs_custom_data
    ]
    runtime_mixed_ids = [
        subdomain_id
        for subdomain_id, kernel_mode, _ in runtime_integral_summary
        if kernel_mode == "mixed"
    ]
    runtime_standard_only_ids = [
        subdomain_id
        for subdomain_id, _, needs_custom_data in runtime_integral_summary
        if not needs_custom_data
    ]

    print(f"active subdomain ids: {active_ids}")
    print(f"subdomain 0 cells: {cells_0.size}")
    print(f"subdomain 1 cells: {cells_1.size}")
    print(f"active direct standard entities: {active_standard_entities}")
    print(f"active runtime-rule entities: {active_runtime_entities}")
    print(f"runtime-form compiled integrals: {runtime_integral_summary}")
    print(f"runtime-form ids needing custom data: {runtime_ids}")
    print(f"runtime-form mixed ids: {runtime_mixed_ids}")
    print(f"runtime-form separate standard-only ids: {runtime_standard_only_ids}")
    print(f"active MeshTags standard-form ids: {active_ids}")
    print("full MeshTags standard-form ids: (0, 1)")
    print(f"active standard matrix Frobenius norm: {active_reference:.12e}")
    print(f"full standard matrix Frobenius norm: {full_reference:.12e}")
    print(f"active runtime vs active standard max difference: {active_error:.12e}")
    print(f"active runtime vs full standard max difference: {full_error:.12e}")


if __name__ == "__main__":
    args = _parse_args()
    main(args.active_subdomain_ids)
