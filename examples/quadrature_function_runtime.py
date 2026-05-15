"""Example: QuadratureFunction values in runtime kernels.

The important API distinction is:

- ``QuadratureFunction`` supplies the source of quadrature-point values.
- ``module.create_custom_data(rules)`` resolves that source for the active
  ``QuadratureRules`` object and stores the prepared value references.
- The generated kernel reads those already prepared values through
  ``custom_data`` during the quadrature loop.

This example intentionally stays UFL/Basix-only. It creates one scalar
QuadratureFunction from explicit provider-owned values and another from a NumPy
callable evaluated on component-first physical points.
"""

from __future__ import annotations

import numpy as np
import ufl
from basix.ufl import element

from runintgen import (
    QuadratureFunction,
    QuadratureRules,
    compile_runtime_integrals,
    dxq,
)
from runintgen.runtime_data import build_quadrature_function_value_set


def _symbolic_problem():
    """Return a simple symbolic P1 mass-form setup."""
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    return mesh, u, v


def _runtime_rules() -> QuadratureRules:
    """Return a two-point per-entity rule with component-first physical points."""
    points = np.array([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    offsets = np.array([0, 2], dtype=np.int64)

    # Storage organisation:
    # - reference points stay point-major: points[q, reference_component]
    # - physical points are component-first: physical_points[g, global_q]
    physical_points = np.ascontiguousarray(points.T)

    return QuadratureRules(
        kind="per_entity",
        tdim=2,
        gdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=np.array([0], dtype=np.int32),
        physical_points=physical_points,
    )


def explicit_values_example() -> None:
    """Attach provider-owned quadrature values with set_values."""
    mesh, u, v = _symbolic_problem()
    alpha = QuadratureFunction(mesh)
    rules = _runtime_rules()
    alpha_values = np.array([2.0, 3.0], dtype=np.float64)
    alpha.set_values(rules, alpha_values)

    form = alpha * ufl.inner(u, v) * dxq(domain=mesh, quadrature_provider=rules)
    module = compile_runtime_integrals(form)
    values = build_quadrature_function_value_set(module.quadrature_functions, rules)

    print("Explicit q-function labels:", [q.label for q in module.quadrature_functions])
    print("Explicit values borrowed:", values.functions[0].values is alpha_values)
    print(
        "Kernel uses q_function_0:",
        "q_function_0[q0 + iq]" in module.kernels[0].c_definition,
    )


def callable_source_example() -> None:
    """Evaluate a callable once on the rule's physical quadrature points."""
    mesh, u, v = _symbolic_problem()

    def q_function(x: np.ndarray) -> np.ndarray:
        return np.sin(x[0]) + np.cos(x[1])

    alpha = QuadratureFunction(mesh, q_function)
    rules = _runtime_rules()

    form = alpha * ufl.inner(u, v) * dxq(domain=mesh, quadrature_provider=rules)
    module = compile_runtime_integrals(form)
    values = build_quadrature_function_value_set(module.quadrature_functions, rules)

    print("Callable values:", values.functions[0].values.tolist())
    print("Callable used physical_points shape:", rules.physical_points.shape)


def main() -> None:
    """Run both API sketches."""
    explicit_values_example()
    callable_source_example()


if __name__ == "__main__":
    main()
