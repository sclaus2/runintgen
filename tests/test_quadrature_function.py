"""Tests for QuadratureFunction runtime integration support."""

from __future__ import annotations

import cffi
import numpy as np
import pytest
import ufl
from basix.ufl import element

from runintgen import (
    CFFI_DEF,
    QuadratureFunction,
    QuadratureRules,
    RuntimeContextBuilder,
    RuntimeEntityMap,
    RuntimeQuadraturePayload,
    compile_runtime_integrals,
    evaluate_quadrature_function,
)
from runintgen.jit import compile_forms
from runintgen.runtime_data import build_quadrature_function_value_set


def _mesh() -> ufl.Mesh:
    """Return a symbolic triangle mesh."""
    return ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))


class _RuntimeRule:
    points = ()
    weights = ()


def _dx_runtime(mesh: ufl.Mesh) -> ufl.Measure:
    return ufl.Measure("dx", domain=mesh, subdomain_data=_RuntimeRule())


def _dS_runtime(mesh: ufl.Mesh) -> ufl.Measure:
    return ufl.Measure("dS", domain=mesh, subdomain_data=_RuntimeRule())


def _rules(*, physical_points: np.ndarray | None = None) -> QuadratureRules:
    """Return a small runtime quadrature rule set."""
    points = np.array([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)
    weights = np.array([0.25, 0.25], dtype=np.float64)
    offsets = np.array([0, 2], dtype=np.int32)
    kwargs = {}
    if physical_points is not None:
        kwargs["gdim"] = 2
        kwargs["physical_points"] = physical_points
    return QuadratureRules(
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        **kwargs,
    )


def _interior_payload() -> RuntimeQuadraturePayload:
    """Return a two-sided runtime interior-facet payload."""
    rules = QuadratureRules(
        tdim=2,
        points=np.array([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64),
        secondary_points=np.array([[0.8, 0.3], [0.4, 0.2]], dtype=np.float64),
        weights=np.array([0.25, 0.25], dtype=np.float64),
        offsets=np.array([0, 2], dtype=np.int32),
        parent_map=np.array([10], dtype=np.int32),
    )
    entities = RuntimeEntityMap(
        entity_indices=np.array([[10, 1, 20, 2]], dtype=np.int32),
        is_cut=np.array([1], dtype=np.uint8),
        rule_indices=np.array([0], dtype=np.int32),
    )
    return RuntimeQuadraturePayload(rules=rules, entities=entities)


def test_quadrature_function_mesh_constructor_defaults_to_scalar_dg0() -> None:
    """Passing a mesh should create a scalar UFL coefficient."""
    alpha = QuadratureFunction(_mesh())

    assert alpha.ufl_shape == ()
    assert alpha._runintgen_quadrature_function.name is None
    assert alpha._runintgen_quadrature_function.value_size == 1


def test_callable_source_is_evaluated_on_component_first_points() -> None:
    """Callable sources should consume physical_points with x[component]."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, lambda x: x[0] + x[1])
    module = compile_runtime_integrals(alpha * ufl.inner(u, v) * _dx_runtime(mesh))
    rules = _rules(physical_points=np.ascontiguousarray([[0.2, 0.6], [0.3, 0.2]]))

    values = build_quadrature_function_value_set(module.quadrature_functions, rules)

    assert values is not None
    np.testing.assert_allclose(values.functions[0].values, [0.5, 0.8])


def test_explicit_values_are_borrowed_by_rule_id() -> None:
    """Explicit set_values arrays should be used without copying."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh)
    module = compile_runtime_integrals(alpha * ufl.inner(u, v) * _dx_runtime(mesh))
    rules = _rules()
    explicit = np.array([2.0, 3.0], dtype=np.float64)
    alpha.set_values(rules, explicit)

    values = build_quadrature_function_value_set(module.quadrature_functions, rules)

    assert values is not None
    assert values.functions[0].values is explicit


def test_generated_runtime_kernel_loads_quadrature_function_from_custom_data() -> None:
    """Generated C should not interpolate QuadratureFunction through w."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh)

    module = compile_runtime_integrals(alpha * ufl.inner(u, v) * _dx_runtime(mesh))
    kernel = module.kernels[0]

    assert len(module.quadrature_functions) == 1
    assert kernel.quadrature_function_slots == [0]
    assert "q_function_0[q0 + iq]" in kernel.c_definition
    assert "enabled_coefficients_" in kernel.c_definition
    assert "= {0};" in kernel.c_definition
    assert "coefficient" not in kernel.table_slots.values()


def test_cffi_context_builder_exposes_quadrature_function_values() -> None:
    """CFFI context helper should expose q-function value pointers."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh)
    module = compile_runtime_integrals(alpha * ufl.inner(u, v) * _dx_runtime(mesh))
    rules = _rules()
    explicit = np.array([2.0, 3.0], dtype=np.float64)
    alpha.set_values(rules, explicit)
    values = build_quadrature_function_value_set(module.quadrature_functions, rules)

    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)
    builder = RuntimeContextBuilder(ffi)
    ctx = builder.build_context(rules, quadrature_functions=values)

    assert ctx.quadrature_functions.num_functions == 1
    assert ctx.quadrature_functions.functions[0].value_size == 1
    assert ctx.quadrature_functions.functions[0].num_points == 2
    assert int(ffi.cast("intptr_t", ctx.quadrature_functions.functions[0].values)) == (
        explicit.ctypes.data
    )


def test_fallback_evaluator_supplies_background_values() -> None:
    """Backend adapters may supply values when no explicit/source data exists."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh)
    module = compile_runtime_integrals(alpha * ufl.inner(u, v) * _dx_runtime(mesh))
    rules = _rules(physical_points=np.ascontiguousarray([[0.2, 0.6], [0.3, 0.2]]))

    def evaluator(info, active_rules):
        assert info.label == "quadrature_function_0"
        assert active_rules is rules
        return np.array([4.0, 5.0], dtype=np.float64)

    values = build_quadrature_function_value_set(
        module.quadrature_functions,
        rules,
        fallback_evaluator=evaluator,
    )

    assert values is not None
    np.testing.assert_allclose(values.functions[0].values, [4.0, 5.0])


def test_vector_quadrature_function_uses_component_inner_stride() -> None:
    """Vector component accesses should use point-major component-innermost layout."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1, shape=(2,)))
    v = ufl.TestFunction(V)
    normal = QuadratureFunction(mesh, shape=(2,))

    module = compile_runtime_integrals(ufl.dot(normal, v) * _dx_runtime(mesh))
    kernel = module.kernels[0]

    assert "q_function_0[(q0 + iq) * 2]" in kernel.c_definition
    assert "q_function_0[(q0 + iq) * 2 + 1]" in kernel.c_definition


def test_restricted_quadrature_function_occurrences_use_distinct_slots() -> None:
    """Plus and minus restrictions should request separate value slots."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")

    module = compile_runtime_integrals(
        (alpha("+") * v("+") + alpha("-") * v("-")) * _dS_runtime(mesh)
    )
    kernel = module.kernels[0]

    assert [info.label for info in module.quadrature_functions] == [
        "alpha(+)",
        "alpha(-)",
    ]
    assert [info.restriction for info in module.quadrature_functions] == ["+", "-"]
    assert kernel.quadrature_function_slots == [0, 1]
    assert "q_function_0[q0 + iq]" in kernel.c_definition
    assert "q_function_1[q0 + iq]" in kernel.c_definition


def test_restricted_quadrature_function_evaluator_requests_are_distinct() -> None:
    """Context-aware evaluators should receive side-specific rule views."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    payload = _interior_payload()
    seen = []

    class RestrictionEvaluator:
        def evaluate(self, context):
            seen.append(
                (
                    context.info.restriction,
                    context.rules.points.copy(),
                    context.rules.parent_map.copy(),
                    dict(context.metadata or {}),
                )
            )
            if context.info.restriction == "+":
                return np.array([1.0, 2.0], dtype=np.float64)
            if context.info.restriction == "-":
                return np.array([3.0, 4.0], dtype=np.float64)
            raise AssertionError("expected restricted QuadratureFunction request")

    alpha = QuadratureFunction(mesh, RestrictionEvaluator(), name="alpha")
    module = compile_runtime_integrals(
        (alpha("+") * v("+") + alpha("-") * v("-")) * _dS_runtime(mesh)
    )

    values = build_quadrature_function_value_set(module.quadrature_functions, payload)

    assert values is not None
    np.testing.assert_allclose(values.functions[0].values, [1.0, 2.0])
    np.testing.assert_allclose(values.functions[1].values, [3.0, 4.0])
    assert [item[0] for item in seen] == ["+", "-"]
    np.testing.assert_allclose(seen[0][1], payload.rules.points)
    np.testing.assert_allclose(seen[1][1], payload.rules.secondary_points)
    np.testing.assert_array_equal(seen[0][2], [10])
    np.testing.assert_array_equal(seen[1][2], [20])
    assert seen[0][3]["point_set"] == 0
    assert seen[1][3]["point_set"] == 1
    np.testing.assert_array_equal(seen[0][3]["local_facets"], [1])
    np.testing.assert_array_equal(seen[1][3]["local_facets"], [2])


def test_restricted_quadrature_function_requires_side_context() -> None:
    """Side-restricted slots should not be evaluated from side-neutral rules."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")
    module = compile_runtime_integrals(alpha("+") * v("+") * _dS_runtime(mesh))

    with pytest.raises(ValueError, match="RuntimeQuadraturePayload"):
        build_quadrature_function_value_set(module.quadrature_functions, _rules())


def test_multiple_quadrature_functions_in_pointwise_expression() -> None:
    """Pointwise algebra should load every QuadratureFunction from custom_data."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")
    beta = QuadratureFunction(mesh, name="beta")

    module = compile_runtime_integrals(
        (alpha + beta) * ufl.inner(u, v) * _dx_runtime(mesh)
    )
    kernel = module.kernels[0]

    assert [info.label for info in module.quadrature_functions] == [
        "alpha",
        "beta",
    ]
    assert kernel.quadrature_function_slots == [0, 1]
    assert "q_function_0[q0 + iq]" in kernel.c_definition
    assert "q_function_1[q0 + iq]" in kernel.c_definition
    assert "= {0, 0};" in kernel.c_definition


def test_quadrature_function_algebra_stays_inside_quadrature_loop() -> None:
    """Q-function dependent algebra must not be hoisted outside the iq loop."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    c = ufl.Constant(mesh)
    alpha = QuadratureFunction(mesh, name="alpha")

    module = compile_runtime_integrals(
        (c + alpha) * ufl.inner(u, v) * _dx_runtime(mesh)
    )
    kernel = module.kernels[0]
    loop_pos = kernel.c_definition.find("for (int iq")
    access_pos = kernel.c_definition.find("q_function_0[q0 + iq]")

    assert loop_pos >= 0
    assert access_pos > loop_pos


def test_quadrature_function_derivative_is_rejected() -> None:
    """Derivatives of QuadratureFunction are not defined in v1."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")

    with pytest.raises(
        NotImplementedError,
        match="Derivatives and averages of QuadratureFunction",
    ):
        compile_runtime_integrals(
            ufl.inner(ufl.grad(alpha), ufl.grad(v)) * _dx_runtime(mesh)
        )


def test_set_values_supplies_side_specific_interior_facet_values() -> None:
    """Explicit values attached per restriction feed the matching slots."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")
    module = compile_runtime_integrals(
        (alpha("+") * v("+") + alpha("-") * v("-")) * _dS_runtime(mesh)
    )
    payload = _interior_payload()
    plus = np.array([1.0, 2.0], dtype=np.float64)
    minus = np.array([3.0, 4.0], dtype=np.float64)
    alpha.set_values(payload.rules, plus, restriction="+")
    alpha.set_values(payload.rules, minus, restriction="-")

    values = build_quadrature_function_value_set(module.quadrature_functions, payload)

    assert values is not None
    assert values.functions[0].values is plus
    assert values.functions[1].values is minus


def test_side_neutral_values_serve_sides_without_specific_values() -> None:
    """Values attached without a restriction are the default for both sides."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")
    module = compile_runtime_integrals(
        (alpha("+") * v("+") + alpha("-") * v("-")) * _dS_runtime(mesh)
    )
    payload = _interior_payload()
    both = np.array([5.0, 6.0], dtype=np.float64)
    minus = np.array([7.0, 8.0], dtype=np.float64)
    alpha.set_values(payload.rules, both)

    shared = build_quadrature_function_value_set(module.quadrature_functions, payload)
    alpha.set_values(payload.rules, minus, restriction="-")
    overridden = build_quadrature_function_value_set(
        module.quadrature_functions, payload
    )

    assert shared.functions[0].values is both
    assert shared.functions[1].values is both
    assert overridden.functions[0].values is both
    assert overridden.functions[1].values is minus


def test_set_values_rejects_unknown_restriction() -> None:
    """Only interior-facet sides are valid restrictions."""
    alpha = QuadratureFunction(_mesh(), name="alpha")

    with pytest.raises(ValueError, match="restriction"):
        alpha.set_values(_rules(), np.zeros(2), restriction="+-")


def test_evaluator_cache_is_keyed_by_side_not_slot() -> None:
    """Cached values of one side must not serve the other side in another form.

    Both modules place their single restriction of ``alpha`` in slot 0, and the
    evaluator's ``cache_key`` ignores the side, as the CutFEMx normal and the
    TDC Weingarten evaluators do.
    """
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    payload = _interior_payload()
    requests = []

    class SideAgnosticCacheKey:
        def cache_key(self, context):
            return "one-key-for-both-sides"

        def evaluate(self, context):
            requests.append(context.restriction)
            sign = 1.0 if context.restriction == "+" else -1.0
            return np.full(context.rules.total_points, sign)

    alpha = QuadratureFunction(mesh, SideAgnosticCacheKey(), name="alpha")
    plus_module = compile_runtime_integrals(alpha("+") * v("+") * _dS_runtime(mesh))
    minus_module = compile_runtime_integrals(alpha("-") * v("-") * _dS_runtime(mesh))
    assert [info.slot for info in plus_module.quadrature_functions] == [0]
    assert [info.slot for info in minus_module.quadrature_functions] == [0]

    plus = build_quadrature_function_value_set(
        plus_module.quadrature_functions, payload
    )
    minus = build_quadrature_function_value_set(
        minus_module.quadrature_functions, payload
    )
    minus_again = build_quadrature_function_value_set(
        minus_module.quadrature_functions, payload
    )

    np.testing.assert_allclose(plus.functions[0].values, [1.0, 1.0])
    np.testing.assert_allclose(minus.functions[0].values, [-1.0, -1.0])
    assert minus_again.functions[0].values is minus.functions[0].values
    assert requests == ["+", "-"]


def test_evaluator_cache_does_not_keep_unused_values_alive() -> None:
    """Values of rule sets no custom_data uses any more are released."""
    import gc

    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    class Constant:
        def evaluate(self, context):
            return np.ones(context.rules.total_points)

    alpha = QuadratureFunction(mesh, Constant(), name="alpha")
    module = compile_runtime_integrals(alpha * ufl.inner(u, v) * _dx_runtime(mesh))
    values = build_quadrature_function_value_set(module.quadrature_functions, _rules())
    assert len(alpha._runintgen_cache) == 1

    del values
    gc.collect()

    assert len(alpha._runintgen_cache) == 0


def test_evaluators_resolve_input_quadrature_functions_on_the_same_side() -> None:
    """Composite evaluators receive their inputs' values for the active side."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    v = ufl.TestFunction(V)
    payload = _interior_payload()
    normal = QuadratureFunction(mesh, name="normal", shape=(2,))
    plus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    normal.set_values(payload.rules, plus, restriction="+")
    normal.set_values(payload.rules, -plus, restriction="-")

    class Doubled:
        def evaluate(self, context):
            return 2.0 * evaluate_quadrature_function(normal, context)

    twice = QuadratureFunction(mesh, Doubled(), name="twice", shape=(2,))
    module = compile_runtime_integrals(
        (twice("+")[0] * v("+") + twice("-")[1] * v("-")) * _dS_runtime(mesh)
    )

    values = build_quadrature_function_value_set(module.quadrature_functions, payload)

    np.testing.assert_allclose(values.functions[0].values, 2.0 * plus)
    np.testing.assert_allclose(values.functions[1].values, -2.0 * plus)


def _tabulate_runtime_scalar(module, form_index, payload) -> float:
    """Run the runtime kernel of a scalar form on the first entity of payload."""
    info = module._runintgen_jit.forms[form_index]
    kernel = info.integral_infos[0].kernel
    values = build_quadrature_function_value_set(
        info.module.quadrature_functions,
        payload,
        active_slots=kernel.quadrature_function_slots,
    )
    ffi = module.ffi
    builder = RuntimeContextBuilder(ffi)  # owns the C structs behind ctx
    ctx = builder.build_context(payload, quadrature_functions=values)
    tabulate = info.ufcx_form.form_integrals[0].tabulate_tensor_float64

    A = np.zeros(1, dtype=np.float64)
    w = np.zeros(1, dtype=np.float64)
    c = np.zeros(1, dtype=np.float64)
    coordinate_dofs = np.zeros(2 * 3 * 3, dtype=np.float64)
    entity_local_index = np.array([1, 2, 0], dtype=np.intc)
    permutation = np.zeros(2, dtype=np.uint8)
    tabulate(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coordinate_dofs.ctypes.data),
        ffi.cast("int*", entity_local_index.ctypes.data),
        ffi.cast("uint8_t*", permutation.ctypes.data),
        ffi.cast("void*", ctx),
    )
    return float(A[0])


def test_interior_facet_kernel_combines_plus_and_minus_values() -> None:
    """Products of both restrictions of one QuadratureFunction are pointwise."""
    mesh = _mesh()
    payload = _interior_payload()
    q = QuadratureFunction(mesh, name="q", shape=(2,))
    q.set_values(
        payload.rules, np.array([[1.0, 2.0], [3.0, 4.0]]), restriction="+"
    )
    q.set_values(
        payload.rules, np.array([[5.0, 6.0], [7.0, 8.0]]), restriction="-"
    )
    dS = ufl.Measure("dS", domain=mesh, subdomain_data=payload.rules)
    cases = [
        (ufl.dot(q("+"), q("-")) * dS, 0.25 * (17.0 + 53.0)),
        (ufl.dot(q("+") + q("-"), q("+") + q("-")) * dS, 0.25 * (100.0 + 244.0)),
        (ufl.dot(ufl.jump(q), ufl.jump(q)) * dS, 0.25 * (32.0 + 32.0)),
        (q("-")[1] * q("+")[0] * dS, 0.25 * (6.0 * 1.0 + 8.0 * 3.0)),
        (q("-")[1] * dS, 0.25 * (6.0 + 8.0)),
    ]

    _, module, _ = compile_forms([form for form, _ in cases])

    for index, (form, expected) in enumerate(cases):
        assert _tabulate_runtime_scalar(module, index, payload) == pytest.approx(
            expected
        ), str(form)


def test_compile_runtime_integrals_rejects_standard_quadrature_function() -> None:
    """Standard integrals cannot load quadrature values in either compile path."""
    mesh = _mesh()
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")
    form = alpha * ufl.inner(u, v) * ufl.dx(domain=mesh)
    form += ufl.inner(u, v) * _dx_runtime(mesh)

    with pytest.raises(NotImplementedError, match="standard integrals"):
        compile_runtime_integrals(form)
