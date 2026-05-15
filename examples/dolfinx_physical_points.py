"""Example: attach physical points to QuadratureRules with DOLFINx.

This example is intentionally limited to the mapping step. The returned rules
can then be used by callable ``QuadratureFunction`` sources that expect
component-first physical coordinates ``x`` with ``x[0]``, ``x[1]``, ...
"""

from __future__ import annotations

import numpy as np

try:
    import basix
    from dolfinx.mesh import create_unit_square
    from mpi4py import MPI
except ImportError:  # pragma: no cover - example fallback
    basix = None
    create_unit_square = None
    MPI = None

from runintgen import QuadratureRules
from runintgen.dolfinx import compute_physical_points


def main() -> None:
    """Map a repeated reference rule to every local cell."""
    if MPI is None:
        raise RuntimeError("This example requires DOLFINx and mpi4py.")

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, dtype=np.float64)
    cells = np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local,
        dtype=np.int32,
    )
    reference_points, reference_weights = basix.make_quadrature(
        basix.CellType.triangle,
        2,
    )

    rules = QuadratureRules(
        kind="shared",
        tdim=mesh.topology.dim,
        points=np.ascontiguousarray(reference_points),
        weights=np.ascontiguousarray(reference_weights),
        parent_map=cells,
    )

    mapped_rules = compute_physical_points(mesh, rules)

    print("rule_id preserved:", mapped_rules.rule_id == rules.rule_id)
    print("physical_points shape:", mapped_rules.physical_points.shape)
    print("first point:", mapped_rules.physical_points[:, 0].tolist())


if __name__ == "__main__":
    main()
