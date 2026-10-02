"""Tests for the runintgen loop optimizer."""

from __future__ import annotations

import ffcx.codegeneration.jit
import ffcx.codegeneration.lnodes as L
import numpy as np
import pytest
import ufl
from basix.ufl import element

from runintgen.codegeneration.optimizer import depends_on, grouped_licm
from runintgen.jit import compile_forms


def _index(name: str) -> L.Symbol:
    return L.Symbol(name, dtype=L.DataType.INT)


def test_depends_on_inspects_flat_runtime_table_indices():
    """Runtime table accesses hide the dof index inside a flat index."""
    i, j, iq = _index("i"), _index("j"), _index("iq")
    nq, ndofs = _index("rt_nq"), _index("rt_element_0_num_dofs")
    table = L.Symbol("rt_element_0", dtype=L.DataType.REAL)
    access = table[((L.LiteralInt(2) * nq + iq) * ndofs + i) * L.LiteralInt(1)]

    assert depends_on(access, i)
    assert depends_on(access, iq)
    assert not depends_on(access, j)
    assert not depends_on(L.Symbol("fw0", dtype=L.DataType.SCALAR), i)


def _rank2_section(terms):
    """Section ``for j: for i: A[3 i + j] += f(i) * g(j) * w`` for ``terms``."""
    i, j = _index("i"), _index("j")
    A = L.Symbol("A", dtype=L.DataType.SCALAR)
    body = []
    for f, g, w in terms:
        f_i = L.Symbol(f, dtype=L.DataType.REAL)[i]
        g_j = L.Symbol(g, dtype=L.DataType.REAL)[j]
        fw = L.Symbol(w, dtype=L.DataType.SCALAR)
        body.append(L.AssignAdd(A[L.LiteralInt(3) * i + j], L.Product([fw, f_i, g_j])))
    loop = L.ForRange(j, 0, 3, [L.ForRange(i, 0, 3, body)])
    return L.Section("Tensor Computation", [loop], [], [], [A], [L.Annotation.licm])


def test_grouped_licm_sums_terms_sharing_inner_factor():
    """Terms with the same inner factor share one hoisted temporary."""
    section = _rank2_section(
        [("F0", "F1", "fw0"), ("F0", "F2", "fw1"), ("F3", "F1", "fw2")]
    )
    optimized = grouped_licm(section)

    loop = optimized.statements[-1]
    inner = loop.body.statements[0]
    assignments = [s.expr for s in inner.body.statements]
    temps = [s for s in optimized.statements if isinstance(s, L.ArrayDecl)]
    assert len(assignments) == 2
    assert len(temps) == 2
    assert L.Annotation.licm not in optimized.annotations

    # Evaluate both versions on random data.
    rng = np.random.default_rng(1)
    arrays = {name: rng.standard_normal(3) for name in ("F0", "F1", "F2", "F3")}
    scalars = {name: rng.standard_normal() for name in ("fw0", "fw1", "fw2")}

    def evaluate(statements):
        env = {"A": np.zeros(9), **arrays, **scalars}

        def value(e):
            if isinstance(e, L.Symbol):
                return env[e.name]
            if isinstance(e, (L.LiteralInt, L.LiteralFloat)):
                return e.value
            if isinstance(e, L.ArrayAccess):
                return env[e.array.name][int(value(e.indices[0]))]
            if isinstance(e, L.Product):
                return np.prod([value(a) for a in e.args])
            if isinstance(e, L.Sum):
                return sum(value(a) for a in e.args)
            if isinstance(e, L.Mul):
                return value(e.lhs) * value(e.rhs)
            if isinstance(e, L.Add):
                return value(e.lhs) + value(e.rhs)
            raise NotImplementedError(type(e))

        def run(statement):
            if isinstance(statement, L.StatementList):
                for s in statement.statements:
                    run(s)
            elif isinstance(statement, L.ForRange):
                for k in range(statement.begin.value, statement.end.value):
                    env[statement.index.name] = k
                    run(statement.body)
            elif isinstance(statement, L.ArrayDecl):
                env[statement.symbol.name] = np.zeros(statement.sizes[0])
            else:
                expr = statement.expr
                lhs = expr.lhs
                target = env[lhs.array.name]
                index = int(value(lhs.indices[0]))
                if isinstance(expr, L.AssignAdd):
                    target[index] += value(expr.rhs)
                else:
                    target[index] = value(expr.rhs)

        for s in statements:
            run(s)
        return env["A"]

    np.testing.assert_allclose(
        evaluate(optimized.statements), evaluate(section.statements), rtol=1e-14
    )


def _tabulate(module, form, coordinate_dofs, shape, local_index=None):
    ffi = module.ffi
    kernel = form.form_integrals[0].tabulate_tensor_float64
    A = np.zeros(shape)
    w = np.zeros(1)
    c = np.zeros(1)
    coords = np.ascontiguousarray(coordinate_dofs, dtype=np.float64)
    if local_index is None:
        entity, perm = ffi.NULL, ffi.NULL
    else:
        entity = ffi.new("int[]", local_index)
        perm = ffi.new("uint8_t[]", [0, 0])
    kernel(
        ffi.cast("double *", A.ctypes.data),
        ffi.cast("double *", w.ctypes.data),
        ffi.cast("double *", c.ctypes.data),
        ffi.cast("double *", coords.ctypes.data),
        entity,
        perm,
        ffi.NULL,
    )
    return A


_HEX = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.1, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [1.2, 1.0, 0.0],
        [0.1, 0.0, 1.0],
        [1.0, 0.1, 1.1],
        [0.0, 1.1, 0.9],
        [1.1, 1.2, 1.2],
    ]
)


@pytest.mark.parametrize("integral_type", ["cell", "interior_facet"])
def test_standard_kernels_match_ffcx(integral_type, tmp_path):
    """Optimized standard kernels reproduce FFCx kernels on a non-affine cell."""
    cell = "hexahedron"
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(3,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1, shape=(3,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps = ufl.sym(ufl.grad(u))
    if integral_type == "cell":
        form = (
            ufl.inner(eps, ufl.sym(ufl.grad(v)))
            + ufl.inner(ufl.grad(ufl.grad(u)), ufl.grad(ufl.grad(v)))
        ) * ufl.dx(domain=mesh)
        coords = _HEX
        local_index = None
        shape = (24, 24)
    else:
        n = ufl.FacetNormal(mesh)
        form = (
            ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n))
            + ufl.inner(
                ufl.jump(ufl.grad(ufl.grad(u))), ufl.jump(ufl.grad(ufl.grad(v)))
            )
        ) * ufl.dS(domain=mesh)
        coords = np.concatenate([_HEX, _HEX + [0.0, 0.0, 1.2]])
        local_index = [5, 0]
        shape = (48, 48)

    forms, module, _ = compile_forms([form], cache_dir=tmp_path / "runintgen")
    ref_forms, ref_module, _ = ffcx.codegeneration.jit.compile_forms(
        [form], cache_dir=tmp_path / "ffcx"
    )
    A = _tabulate(module, forms[0], coords, shape, local_index)
    A_ref = _tabulate(ref_module, ref_forms[0], coords, shape, local_index)

    assert np.abs(A_ref).max() > 0.0
    np.testing.assert_allclose(A, A_ref, rtol=1e-12, atol=1e-12 * np.abs(A_ref).max())
