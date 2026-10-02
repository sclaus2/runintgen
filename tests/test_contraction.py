"""Tests for the quadrature-chunked contraction of runtime kernels.

Each runtime kernel is compared with the standard kernel of the same form,
whose code generation does not use the contraction. The runtime rule is the
standard rule of the same degree, so the element tensors agree to rounding.
"""

from __future__ import annotations

import importlib.util

import basix
import ffcx.codegeneration.lnodes as L
import numpy as np
import pytest
import ufl
from basix.ufl import element, mixed_element

from runintgen.codegeneration import contraction
from runintgen.codegeneration.optimizer import expression_key
from runintgen.jit import compile_forms
from runintgen.runtime_data import (
    QuadratureRules,
    RuntimeEntityMap,
    RuntimeQuadraturePayload,
)

_basix_runtime = pytest.importorskip("runintgen._basix_runtime")

_CELLS = {
    "tetrahedron": (
        basix.CellType.tetrahedron,
        np.array([[0.1, 0.0, 0.0], [1.0, 0.2, 0.0], [0.3, 1.1, 0.1], [0.2, 0.3, 0.9]]),
    ),
    # A parallelepiped: affine
    "hexahedron": (
        basix.CellType.hexahedron,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.2, 0.9, 0.0],
                [1.2, 1.0, 0.0],
                [0.1, 0.2, 0.8],
                [1.1, 0.3, 0.8],
                [0.3, 1.1, 0.8],
                [1.3, 1.2, 0.8],
            ]
        ),
    ),
}
# Not a parallelepiped: the Jacobian varies within the cell
_SKEWED_HEXAHEDRON = _CELLS["hexahedron"][1] + np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0.2, 0.1, 0],
        [0, 0, 0],
        [0, 0, 0.1],
        [0, 0, 0],
        [0.1, 0.2, 0.3],
    ]
)


class _RuntimeRule:
    points = ()
    weights = ()


def _epsilon(w):
    return ufl.sym(ufl.grad(w))


def _forms(cell: str, kind: str, degree: int, quadrature_degree: int):
    """Return the standard and runtime versions of one bilinear form."""
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(3,)))
    dx = ufl.Measure(
        "dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree}
    )
    dx_runtime = ufl.Measure("dx", domain=mesh, subdomain_data=_RuntimeRule())
    coefficients = []
    if kind == "stokes":
        W = ufl.FunctionSpace(
            mesh,
            mixed_element(
                [
                    element("Lagrange", cell, degree, shape=(3,)),
                    element("Lagrange", cell, degree - 1),
                ]
            ),
        )
        (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
        integrand = (
            ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - q * ufl.div(u)
        )
    else:
        shape = (3,) if kind == "elasticity" else ()
        V = ufl.FunctionSpace(mesh, element("Lagrange", cell, degree, shape=shape))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        if kind == "laplace":
            integrand = ufl.inner(ufl.grad(u), ufl.grad(v))
        elif kind == "complex laplace":
            integrand = (1.0 + 2.0j) * ufl.inner(ufl.grad(u), ufl.grad(v))
        elif kind == "mass":
            integrand = u * v
        elif kind == "coefficient":
            c = ufl.Coefficient(ufl.FunctionSpace(mesh, element("Lagrange", cell, 1)))
            coefficients.append(c)
            integrand = c * ufl.inner(ufl.grad(u), ufl.grad(v))
        elif kind == "elasticity":
            integrand = 2.0 * ufl.inner(_epsilon(u), _epsilon(v)) + 1.5 * ufl.div(
                u
            ) * ufl.div(v)
        else:
            raise ValueError(kind)
    return integrand * dx, integrand * dx_runtime, len(coefficients)


def _tabulate(module, form_index, coordinates, dtype, num_coefficients, payload=None):
    """Tabulate the first integral of a form on one cell."""
    info = module._runintgen_jit.forms[form_index]
    ufcx_form = info.ufcx_form
    shape = [
        a.ufl_function_space().ufl_element().dim for a in info.ufl_form.arguments()
    ]
    ffi = module.ffi
    A = np.zeros(shape, dtype=dtype)
    w = np.linspace(1.0, 2.0, max(num_coefficients * 4, 1)).astype(dtype)
    c = np.zeros(1, dtype=dtype)
    x = np.ascontiguousarray(coordinates, dtype=np.float64)
    entity = np.zeros(1, dtype=np.intc)
    permutation = np.zeros(1, dtype=np.uint8)
    custom_data = ffi.NULL
    owner = None
    if payload is not None:
        from runintgen.basix_runtime import CustomData

        owner = CustomData(info.module, payload)
        custom_data = ffi.cast("void*", owner.ptr)
    c_type = "double _Complex" if np.iscomplexobj(A) else "double"
    name = (
        "tabulate_tensor_complex128"
        if np.iscomplexobj(A)
        else "tabulate_tensor_float64"
    )
    getattr(ufcx_form.form_integrals[0], name)(
        ffi.cast(f"{c_type}*", A.ctypes.data),
        ffi.cast(f"{c_type}*", w.ctypes.data),
        ffi.cast(f"{c_type}*", c.ctypes.data),
        ffi.cast("double*", x.ctypes.data),
        ffi.cast("int*", entity.ctypes.data),
        ffi.cast("uint8_t*", permutation.ctypes.data),
        custom_data,
    )
    del owner
    return A


def _payload(cell_type: basix.CellType, coordinates, quadrature_degree: int):
    """Return the standard rule as runtime rule of one cell.

    Runtime weights include the measure scaling |det J| of the cell.
    """
    points, weights = basix.make_quadrature(cell_type, quadrature_degree)
    geometry = basix.create_element(basix.ElementFamily.P, cell_type, 1)
    derivatives = geometry.tabulate(1, points)[1:, :, :, 0]
    jacobians = np.einsum("kn,dqk->qnd", coordinates, derivatives)
    weights = weights * np.abs(np.linalg.det(jacobians))
    rules = QuadratureRules(
        tdim=points.shape[1],
        points=np.ascontiguousarray(points),
        weights=np.ascontiguousarray(weights),
        offsets=np.array([0, weights.size], dtype=np.int32),
        parent_map=np.array([0], dtype=np.int32),
    )
    entities = RuntimeEntityMap(
        entity_indices=np.array([0], dtype=np.int32),
        is_cut=np.array([1], dtype=np.uint8),
        rule_indices=np.array([0], dtype=np.int32),
    )
    return RuntimeQuadraturePayload(rules=rules, entities=entities)


# (cell, form, degree, quadrature degree, kernel calls of the contraction)
_CASES = [
    # Scalar forms sum the terms of each test factor (3 contractions)
    ("tetrahedron", "laplace", 1, 2, 3),
    ("tetrahedron", "laplace", 2, 2, 3),
    ("tetrahedron", "mass", 2, 4, 1),
    # The coefficient varies at the points: no point sums
    ("tetrahedron", "coefficient", 2, 3, 3),
    # One point sum per pair of derivatives: 6 of 9 by symmetry
    ("tetrahedron", "elasticity", 2, 2, 6),
    # 6 velocity pairs, 3 velocity-pressure pairs, the transposes reused
    ("tetrahedron", "stokes", 2, 2, 9),
]


@pytest.mark.parametrize("cell,kind,degree,quadrature_degree,calls", _CASES)
def test_runtime_kernel_matches_standard_kernel(
    cell, kind, degree, quadrature_degree, calls, tmp_path
):
    """Runtime kernels with the contraction agree with standard kernels."""
    cell_type, coordinates = _CELLS[cell]
    standard, runtime, num_coefficients = _forms(cell, kind, degree, quadrature_degree)
    _, module, (_, code) = compile_forms([standard, runtime], cache_dir=tmp_path)

    assert code.count(f"{'runintgen_contract_'}") >= calls
    expected = _tabulate(module, 0, coordinates, np.float64, num_coefficients)
    A = _tabulate(
        module,
        1,
        coordinates,
        np.float64,
        num_coefficients,
        _payload(cell_type, coordinates, quadrature_degree),
    )
    assert np.abs(expected).max() > 0
    np.testing.assert_allclose(
        A, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


def _contraction_calls(code: str) -> int:
    return sum(
        1
        for line in code.splitlines()
        if line.strip().startswith("runintgen_contract_")
        and line.rstrip().endswith(";")
    )


@pytest.mark.parametrize("cell,kind,degree,quadrature_degree,calls", _CASES)
def test_runtime_kernel_plan(cell, kind, degree, quadrature_degree, calls, tmp_path):
    """The generated kernels use the cheaper plan for each form."""
    _, runtime, _ = _forms(cell, kind, degree, quadrature_degree)
    _, _, (_, code) = compile_forms([runtime], cache_dir=tmp_path)
    assert _contraction_calls(code) == calls


@pytest.mark.parametrize(
    "affine,coordinates,calls",
    [
        # Point sums of the 6 symmetric pairs of reference derivatives
        (True, _CELLS["hexahedron"][1], 6),
        # The Jacobian varies: 27 rows (3 per block) summed per test factor
        (None, _SKEWED_HEXAHEDRON, 27),
    ],
)
def test_hexahedron_elasticity_kernel(affine, coordinates, calls, tmp_path):
    """Affine and non-affine hexahedra, with the geometry option."""
    standard, runtime, _ = _forms("hexahedron", "elasticity", 2, 4)
    options = {} if affine is None else {"runintgen_affine_geometry": affine}
    _, module, (_, code) = compile_forms(
        [standard, runtime], options=options, cache_dir=tmp_path
    )
    assert _contraction_calls(code) == calls
    expected = _tabulate(module, 0, coordinates, np.float64, 0)
    A = _tabulate(
        module,
        1,
        coordinates,
        np.float64,
        0,
        _payload(basix.CellType.hexahedron, coordinates, 4),
    )
    np.testing.assert_allclose(
        A, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


def test_complex_runtime_kernel(tmp_path):
    """A complex factor multiplies real point sums or complex staged rows."""
    standard, runtime, _ = _forms("tetrahedron", "complex laplace", 2, 2)
    _, module, _ = compile_forms(
        [standard, runtime],
        options={"scalar_type": np.complex128},
        cache_dir=tmp_path,
    )
    coordinates = _CELLS["tetrahedron"][1]
    expected = _tabulate(module, 0, coordinates, np.complex128, 0)
    A = _tabulate(
        module,
        1,
        coordinates,
        np.complex128,
        0,
        _payload(basix.CellType.tetrahedron, coordinates, 2),
    )
    assert np.abs(expected.imag).max() > 0
    np.testing.assert_allclose(
        A, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


def test_expression_key_anonymises_index_and_accepts_unhashable_nodes():
    """Keys compare structure; the anonymous index matches any symbol."""
    table = L.Symbol("FE", dtype=L.DataType.REAL)
    i = L.Symbol("i", dtype=L.DataType.INT)
    j = L.Symbol("j", dtype=L.DataType.INT)
    q = L.Symbol("iq", dtype=L.DataType.INT)
    # LiteralFloat and Neg nodes are not hashable
    a = L.Product([L.LiteralFloat(2.0), L.Neg(table[L.Sum([L.Product([q, 4]), i])])])
    b = L.Product([L.LiteralFloat(2.0), L.Neg(table[L.Sum([L.Product([q, 4]), j])])])

    assert expression_key(a, i) == expression_key(b, j)
    assert expression_key(a, i) != expression_key(b, i)
    assert expression_key(a) != expression_key(b)
    assert hash(expression_key(a, i)) == hash(expression_key(b, j))


@pytest.mark.parametrize("tile", [contraction._WIDE_TILE, contraction._NARROW_TILE])
def test_tiled_contraction(tile, tmp_path):
    """Both register tiles, with remainders, add X^T Y to the accumulator."""
    import cffi

    sizes = [(1, 1), (3, 3), (4, 4), (10, 4), (5, 13), (27, 27)]
    ffi = cffi.FFI()
    source = ""
    for ni, nj in sizes:
        body = contraction._tiled_contraction(
            ni, nj, "double", "double", "double", tile
        )
        signature = (
            f"void contract_{ni}x{nj}(double* restrict acc, const int nk, "
            "const double* restrict X, const double* restrict Y)"
        )
        ffi.cdef(signature.replace("restrict ", "") + ";")
        source += f"{signature}\n{{\n{body}}}\n"
    name = f"_contraction_test_{tile[0]}x{tile[1]}"
    ffi.set_source(name, source)
    library = ffi.compile(tmpdir=str(tmp_path))
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(1)
    for ni, nj in sizes:
        for nk in (1, 7):
            X, Y = rng.random((nk, ni)), rng.random((nk, nj))
            acc = rng.random((ni, nj))
            expected = acc + X.T @ Y
            getattr(module.lib, f"contract_{ni}x{nj}")(
                module.ffi.cast("double*", acc.ctypes.data),
                nk,
                module.ffi.cast("double*", X.ctypes.data),
                module.ffi.cast("double*", Y.ctypes.data),
            )
            np.testing.assert_allclose(acc, expected, rtol=1e-14)


def test_runtime_kernel_with_two_quadrature_rules(tmp_path):
    """Integrals of two degrees give two contractions in one kernel."""
    mesh = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "tetrahedron", 2, shape=(3,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def form(**measure):
        dx = ufl.Measure("dx", domain=mesh, **measure)
        return ufl.inner(_epsilon(u), _epsilon(v)) * dx(degree=2) + ufl.inner(
            u, v
        ) * dx(degree=4)

    standard, runtime = form(), form(subdomain_data=_RuntimeRule())
    _, module, (_, code) = compile_forms([standard, runtime], cache_dir=tmp_path)
    assert code.count("for (int rt_chunk") == 2

    cell_type, coordinates = _CELLS["tetrahedron"]
    expected = _tabulate(module, 0, coordinates, np.float64, 0)
    A = _tabulate(
        module, 1, coordinates, np.float64, 0, _payload(cell_type, coordinates, 4)
    )
    np.testing.assert_allclose(
        A, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


def test_chunked_contraction_can_be_disabled(tmp_path):
    """With the option off, the kernel updates the tensor at every point."""
    standard, runtime, _ = _forms("tetrahedron", "elasticity", 2, 2)
    _, module, (_, code) = compile_forms(
        [standard, runtime],
        options={"runintgen_chunked_contraction": False},
        cache_dir=tmp_path,
    )
    assert "runintgen_contract_" not in code
    cell_type, coordinates = _CELLS["tetrahedron"]
    expected = _tabulate(module, 0, coordinates, np.float64, 0)
    A = _tabulate(
        module, 1, coordinates, np.float64, 0, _payload(cell_type, coordinates, 2)
    )
    np.testing.assert_allclose(
        A, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )
