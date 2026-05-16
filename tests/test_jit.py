"""Tests for runintgen's combined CFFI JIT layer."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from basix.ufl import element

from runintgen import QuadratureFunction
from runintgen.jit import compile_forms
from runintgen.runtime_data import QuadratureRules


def _space():
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    return mesh, V


def _runtime_rules() -> QuadratureRules:
    return QuadratureRules(
        tdim=2,
        points=np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
        weights=np.array([0.5], dtype=np.float64),
        offsets=np.array([0, 1], dtype=np.int64),
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
        offsets=np.array([0, 1], dtype=np.int64),
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
