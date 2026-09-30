"""Tests for runintgen's combined CFFI JIT layer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import ufl
from basix.ufl import element

import runintgen
from runintgen import QuadratureFunction
from runintgen.jit import compile_forms
from runintgen.runtime_data import QuadratureRules

_SCALAR_KERNEL_SLOT = {
    np.dtype(np.float32): "tabulate_tensor_float32",
    np.dtype(np.float64): "tabulate_tensor_float64",
    np.dtype(np.complex64): "tabulate_tensor_complex64",
    np.dtype(np.complex128): "tabulate_tensor_complex128",
}

_SCALAR_C_TYPE = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.complex64): "float _Complex",
    np.dtype(np.complex128): "double _Complex",
}

_GEOMETRY_C_TYPE = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}


def _space():
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    return mesh, V


def _runtime_rules() -> QuadratureRules:
    return QuadratureRules(
        tdim=2,
        points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
        weights=np.array([0.5], dtype=np.float64),
        offsets=np.array([0, 1], dtype=np.int32),
        parent_map=np.array([0], dtype=np.int32),
    )


def test_compile_standard_form_exposes_ufcx_form():
    """Standard-only forms can be compiled into a full UFCx form."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    form = ufl.inner(u, v) * ufl.dx(domain=mesh)

    forms, module, code = compile_forms([form])

    assert len(forms) == 1
    assert code[0] is not None
    assert forms[0].rank == 2
    assert forms[0].form_integral_offsets[0] == 0
    assert forms[0].form_integral_offsets[1] == 1
    assert module._runintgen_jit.kernels[0].mode == "standard"


def test_compile_runtime_form_exposes_runtime_metadata():
    """Runtime-only forms compile to a UFCx form with runtime sidecar data."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=_runtime_rules())
    form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

    forms, module, _ = compile_forms([form])

    assert forms[0].rank == 2
    assert module._runintgen_jit.kernels[0].mode == "runtime"
    assert module._runintgen_jit.forms[0].module.form_metadata is not None
    assert module._runintgen_jit.forms[0].integral_infos[0].needs_custom_data


@pytest.mark.parametrize(
    "scalar_dtype",
    [np.float32, np.float64, np.complex64, np.complex128],
)
@pytest.mark.parametrize("geometry_dtype", [np.float32, np.float64])
def test_compile_runtime_form_exposes_scalar_geometry_kernel(
    scalar_dtype, geometry_dtype
):
    """Runtime kernels support independent PDE scalar and geometry dtypes."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=_runtime_rules())
    form = ufl.inner(u, v) * dx_rt

    scalar_dtype = np.dtype(scalar_dtype)
    geometry_dtype = np.dtype(geometry_dtype)
    forms, module, code = compile_forms(
        [form],
        options={
            "scalar_type": scalar_dtype.type,
            "geometry_type": geometry_dtype.type,
        },
    )
    integral = forms[0].form_integrals[0]
    slot = _SCALAR_KERNEL_SLOT[scalar_dtype]
    scalar_c = _SCALAR_C_TYPE[scalar_dtype]
    geometry_c = _GEOMETRY_C_TYPE[geometry_dtype]

    assert module._runintgen_jit.kernels[0].scalar_type == scalar_dtype.name
    assert module._runintgen_jit.kernels[0].geometry_type == geometry_dtype.name
    assert getattr(integral, slot) != module.ffi.NULL
    assert f"{scalar_c}* restrict A" in code[1]
    assert f"const {geometry_c}* restrict coordinate_dofs" in code[1]


def test_compile_combined_standard_and_runtime_form():
    """One JIT module can contain standard-only and runtime kernels."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=3,
        subdomain_data=_runtime_rules(),
    )
    form = (
        ufl.inner(u, v) * ufl.dx(domain=mesh)
        + ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt
    )

    forms, module, _ = compile_forms([form])
    modes = {kernel.mode for kernel in module._runintgen_jit.kernels}

    assert forms[0].form_integral_offsets[1] == 2
    assert modes == {"standard", "runtime"}


def test_compile_runtime_subdomain_tuple_uses_distinct_kernels():
    """Runtime form slots with grouped subdomain ids use per-id kernels."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=(1, 2),
        subdomain_data=_runtime_rules(),
    )
    form = ufl.inner(u, v) * dx_rt

    forms, module, _ = compile_forms([form])
    kernels = module._runintgen_jit.kernels
    infos = module._runintgen_jit.forms[0].integral_infos

    assert forms[0].form_integral_offsets[1] == 2
    assert [kernel.subdomain_id for kernel in kernels] == [1, 2]
    assert len({kernel.name for kernel in kernels}) == 2
    assert [info.subdomain_id for info in infos] == [1, 2]
    assert [info.kernel.subdomain_id for info in infos] == [1, 2]


def test_runtime_subdomain_ids_affect_jit_cache_key(tmp_path):
    """Changing runtime subdomain ids must produce a distinct cached module."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_0 = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=0,
        subdomain_data=_runtime_rules(),
    )
    dx_1 = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=1,
        subdomain_data=_runtime_rules(),
    )
    form_two_ids = ufl.inner(u, v) * dx_0 + ufl.inner(u, v) * dx_1
    form_one_id = ufl.inner(u, v) * dx_0

    _, module_two_ids, _ = compile_forms([form_two_ids], cache_dir=tmp_path)
    _, module_one_id, code_one_id = compile_forms([form_one_id], cache_dir=tmp_path)

    info_two_ids = module_two_ids._runintgen_jit
    info_one_id = module_one_id._runintgen_jit
    assert info_two_ids.module_name != info_one_id.module_name
    assert [kernel.subdomain_id for kernel in info_two_ids.kernels] == [0, 1]
    assert [kernel.subdomain_id for kernel in info_one_id.kernels] == [0]
    assert code_one_id != (None, None)


def test_compile_same_integrand_standard_and_runtime_ids_split_kernels():
    """Standard ids remain standard when FFCx groups them with runtime ids."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_runtime = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=2,
        subdomain_data=_runtime_rules(),
    )
    form = (
        ufl.inner(u, v) * ufl.Measure("dx", domain=mesh, subdomain_id=1)
        + ufl.inner(u, v) * dx_runtime
    )

    forms, module, _ = compile_forms([form])
    infos = module._runintgen_jit.forms[0].integral_infos

    assert forms[0].form_integral_offsets[1] == 2
    assert [info.subdomain_id for info in infos] == [1, 2]
    assert [info.needs_custom_data for info in infos] == [False, True]
    assert [info.kernel.mode for info in infos] == ["standard", "runtime"]


def test_compile_mixed_entity_runtime_form():
    """Mixed entity/runtime integrals compile into one mixed UFCx integral."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    standard_entities = np.array([1, 2], dtype=np.int32)
    runtime_rules = QuadratureRules(
        tdim=2,
        points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
        weights=np.array([0.5], dtype=np.float64),
        offsets=np.array([0, 1], dtype=np.int32),
        parent_map=np.array([8], dtype=np.int32),
    )
    dx_mixed = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=0,
        subdomain_data=[standard_entities, runtime_rules],
    )

    forms, module, _ = compile_forms([ufl.inner(u, v) * dx_mixed])

    assert forms[0].form_integral_offsets[1] == 1
    assert module._runintgen_jit.kernels[0].mode == "mixed"
    assert module._runintgen_jit.forms[0].integral_infos[0].needs_custom_data


def test_standard_quadrature_function_is_rejected_until_supported():
    """Standard kernels must not silently interpolate QuadratureFunction."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    alpha = QuadratureFunction(mesh, name="alpha")
    form = alpha * ufl.inner(u, v) * ufl.dx(domain=mesh)

    with pytest.raises(NotImplementedError, match="standard integrals"):
        compile_forms([form])


def _interior_facet_rules() -> QuadratureRules:
    return QuadratureRules(
        tdim=2,
        points=np.array([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64),
        secondary_points=np.array([[0.8, 0.3], [0.4, 0.2]], dtype=np.float64),
        weights=np.array([0.25, 0.25], dtype=np.float64),
        offsets=np.array([0, 2], dtype=np.int32),
        parent_map=np.array([10], dtype=np.int32),
    )


def test_cached_module_rebinds_quadrature_function_terminals(tmp_path):
    """A cache hit must evaluate the current form's QuadratureFunctions.

    Both forms have one UFL signature. Before rebinding, the second form reused
    the first form's terminals, so dot(mu('+'), mu('-')) was evaluated with the
    values of n.
    """
    mesh, _ = _space()
    dS_rt = ufl.Measure("dS", domain=mesh, subdomain_data=_interior_facet_rules())
    n = QuadratureFunction(mesh, name="normal", shape=(2,))
    mu = QuadratureFunction(mesh, name="conormal", shape=(2,))

    _, module_n, _ = compile_forms(
        [ufl.dot(n("+"), n("-")) * dS_rt], cache_dir=tmp_path
    )
    sidecar_n = module_n._runintgen_jit
    _, module_mu, code_mu = compile_forms(
        [ufl.dot(mu("+"), mu("-")) * dS_rt], cache_dir=tmp_path
    )
    sidecar_mu = module_mu._runintgen_jit

    assert sidecar_mu.module_name == sidecar_n.module_name
    assert code_mu == (None, None)
    infos_n = sidecar_n.forms[0].module.quadrature_functions
    infos_mu = sidecar_mu.forms[0].module.quadrature_functions
    assert all(info.terminal is n for info in infos_n)
    assert all(info.terminal is mu for info in infos_mu)
    assert [info.label for info in infos_mu] == ["conormal(+)", "conormal(-)"]
    assert [info.slot for info in infos_mu] == [info.slot for info in infos_n]


def test_cached_module_rebinds_quadrature_functions_by_coefficient_position(
    tmp_path,
):
    """Rebinding follows the signature numbering, not names or first use."""
    mesh, V = _space()
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=_runtime_rules())
    a1 = QuadratureFunction(mesh, name="a")
    b1 = QuadratureFunction(mesh, name="b")
    a2 = QuadratureFunction(mesh, name="b")
    b2 = QuadratureFunction(mesh, name="a")

    compile_forms([b1**2 * a1 * v * dx_rt], cache_dir=tmp_path)
    _, module, code = compile_forms([b2**2 * a2 * v * dx_rt], cache_dir=tmp_path)

    infos = module._runintgen_jit.forms[0].module.quadrature_functions
    assert code == (None, None)
    assert [info.terminal for info in infos] == [a2, b2]
    assert infos[0].terminal is a2 and infos[1].terminal is b2
    assert [info.label for info in infos] == ["b", "a"]


def test_quadrature_function_and_coefficient_do_not_share_cached_module(tmp_path):
    """A QuadratureFunction and an ordinary coefficient on one element differ.

    UFL signs both alike, but only the QuadratureFunction kernel loads values
    from custom_data; sharing a module silently read the other source.
    """
    mesh, V = _space()
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=_runtime_rules())
    q = QuadratureFunction(mesh, name="q")
    f = ufl.Coefficient(q.ufl_function_space())
    assert (f * v * dx_rt).signature() == (q * v * dx_rt).signature()

    _, module_f, _ = compile_forms([f * v * dx_rt], cache_dir=tmp_path)
    name_f = module_f._runintgen_jit.module_name
    _, module_q, code_q = compile_forms([q * v * dx_rt], cache_dir=tmp_path)

    assert module_q._runintgen_jit.module_name != name_f
    assert code_q != (None, None)
    assert "q_function_0" in code_q[1]
    infos = module_q._runintgen_jit.forms[0].module.quadrature_functions
    assert [info.terminal for info in infos] == [q]


def test_mixed_integral_rejects_quadrature_function():
    """Standard entities of a mixed integral cannot load quadrature values.

    Their FFCx body read the QuadratureFunction from ``w``, where its slot is
    disabled, instead of raising.
    """
    mesh, V = _space()
    v = ufl.TestFunction(V)
    dx_mixed = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=0,
        subdomain_data=[np.array([1, 2], dtype=np.int32), _runtime_rules()],
    )
    q = QuadratureFunction(mesh, name="q")

    with pytest.raises(NotImplementedError, match="mixed standard/runtime"):
        compile_forms([q * v * dx_mixed])


_CONCURRENT_COMPILE_SCRIPT = """
import json, os, sys, time
from pathlib import Path

import numpy as np
import ufl
from basix.ufl import element

from runintgen.jit import compile_forms
from runintgen.runtime_data import QuadratureRules

cache_dir, barrier, nprocs, timeout = sys.argv[1:]
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
rules = QuadratureRules(
    tdim=2,
    points=np.array([[1.0 / 3.0, 1.0 / 3.0]]),
    weights=np.array([0.5]),
    offsets=np.array([0, 1], dtype=np.int32),
    parent_map=np.array([0], dtype=np.int32),
)
dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=rules)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

# Start compiling together once every process has finished importing.
Path(barrier, str(os.getpid())).touch()
while len(os.listdir(barrier)) < int(nprocs):
    time.sleep(0.01)
try:
    forms, _, code = compile_forms([form], cache_dir=cache_dir, timeout=int(timeout))
    result = {"rank": forms[0].rank, "compiled": code != (None, None)}
except Exception as exc:
    result = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(result), flush=True)
os._exit(0)  # skip interpreter (and possible MPI) teardown
"""


def _compile_concurrently(tmp_path: Path, nprocs: int, timeout: int):
    """Compile one runtime form in ``nprocs`` processes at the same time."""
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    cache_dir = tmp_path / "cache"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(runintgen.__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    args = [str(cache_dir), str(barrier), str(nprocs), str(timeout)]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CONCURRENT_COMPILE_SCRIPT, *args],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(nprocs)
    ]
    outputs = [proc.communicate(timeout=600)[0] for proc in procs]
    return [json.loads(out.splitlines()[-1]) for out in outputs], cache_dir


def test_concurrent_compiles_share_one_build(tmp_path):
    """Processes compiling the same form at once wait for a single build."""
    results, cache_dir = _compile_concurrently(tmp_path, nprocs=3, timeout=600)

    assert all(result.get("rank") == 2 for result in results), results
    assert sum(result["compiled"] for result in results) == 1
    assert all(path.is_file() for path in cache_dir.iterdir())


def test_concurrent_compiles_that_stop_waiting_all_succeed(tmp_path):
    """Processes that stop waiting build too and publish atomically.

    With the FFCx claim-file protocol, a compile outlasting ``timeout`` raised
    TimeoutError in every waiting process.
    """
    results, cache_dir = _compile_concurrently(tmp_path, nprocs=3, timeout=0)

    assert all(result.get("rank") == 2 for result in results), results
    assert any(result["compiled"] for result in results)
    assert all(path.is_file() for path in cache_dir.iterdir())


def test_killed_compile_leftovers_do_not_block_compile(tmp_path):
    """Files left by a killed compile neither stall nor break later compiles."""
    mesh, V = _space()
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=_runtime_rules())
    form = ufl.inner(u, v) * dx_rt
    _, module, _ = compile_forms([form], cache_dir=tmp_path / "first")
    module_name = module._runintgen_jit.module_name

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_dir.joinpath(module_name + ".c").touch()
    cache_dir.joinpath(module_name + ".lock").touch()
    cache_dir.joinpath(module_name + "-build-killed").mkdir()

    forms, _, code = compile_forms([form], cache_dir=cache_dir, timeout=1)
    assert forms[0].rank == 2
    assert code != (None, None)

    _, _, cached_code = compile_forms([form], cache_dir=cache_dir, timeout=1)
    assert cached_code == (None, None)
