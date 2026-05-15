"""Example: form-level mixed standard/runtime entity data.

This demonstrates the intended separation:

- ``QuadratureRules`` stores only runtime quadrature for cut entities.
- The UFL measure carries the mix as ``subdomain_data``.
- runintgen derives the full loop entity map with ``is_cut`` flags.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path

import cffi
import numpy as np
import ufl
from basix.ufl import element

from runintgen import (
    CFFI_DEF,
    QuadratureRules,
    compile_runtime_integrals,
    get_runintgen_data_struct,
)
from runintgen.runtime_data import as_runtime_quadrature_payload


def _strip_ufcx_integral_object(c_definition: str, kernel_name: str) -> str:
    """Remove the UFCx integral object so CFFI can compile only the function."""
    pattern = (
        rf"\nufcx_integral\s+{re.escape(kernel_name)}\s*=\s*"
        rf"\{{.*?\n\}};\n"
    )
    return re.sub(pattern, "\n", c_definition, flags=re.DOTALL)


def _compile_tabulate_tensor(kernel):
    """JIT-compile one generated tabulate_tensor function with CFFI."""
    c_code = "\n".join(
        [
            "#include <complex.h>",
            "#include <math.h>",
            "#include <stdbool.h>",
            "#include <stdint.h>",
            "",
            get_runintgen_data_struct(),
            _strip_ufcx_integral_object(kernel.c_definition, kernel.name),
        ]
    )
    declaration = f"""
void tabulate_tensor_{kernel.name}(double* restrict A,
                                   const double* restrict w,
                                   const double* restrict c,
                                   const double* restrict coordinate_dofs,
                                   const int* restrict entity_local_index,
                                   const uint8_t* restrict quadrature_permutation,
                                   void* custom_data);
"""

    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)
    ffi.cdef(declaration)

    with tempfile.TemporaryDirectory(prefix="runintgen_mixed_example_") as tmpdir:
        module_name = f"_mixed_runtime_{kernel.name[:16]}"
        ffi.set_source(module_name, c_code, extra_compile_args=["-std=c17", "-O2"])
        ffi.compile(tmpdir=tmpdir, verbose=False)

        so_files = list(Path(tmpdir).glob("*.so")) + list(Path(tmpdir).glob("*.pyd"))
        if not so_files:
            raise RuntimeError("CFFI did not produce a compiled extension.")
        spec = importlib.util.spec_from_file_location(module_name, so_files[0])
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the compiled CFFI extension.")
        compiled = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled)

    return getattr(compiled.lib, f"tabulate_tensor_{kernel.name}"), compiled.ffi


def _call_kernel(kernel_fn, ffi, custom_data, loop_index: int) -> np.ndarray:
    """Call the generated tensor kernel for one loop index."""
    A = np.zeros((3, 3), dtype=np.float64)
    w = np.zeros(0, dtype=np.float64)
    c = np.zeros(0, dtype=np.float64)
    coordinate_dofs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    ).ravel()
    entity_local_index = np.array([loop_index], dtype=np.int32)
    quadrature_permutation = np.zeros(1, dtype=np.uint8)

    kernel_fn(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coordinate_dofs.ctypes.data),
        ffi.cast("int*", entity_local_index.ctypes.data),
        ffi.cast("uint8_t*", quadrature_permutation.ctypes.data),
        ffi.cast("void*", custom_data.ptr),
    )
    return A


def main() -> None:
    """Compile a mixed runtime/standard cell integral."""
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # These entities should use the standard FFCx integral branch.
    standard_entities = np.array([1, 4, 5, 6], dtype=np.int32)

    # These entities have per-entity quadrature. The i-th parent_map entry
    # corresponds to the i-th rule slice in offsets.
    parent_map = np.array([8, 10, 11], dtype=np.int32)
    points = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0],
            [0.2, 0.2],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.5, 0.25],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.5, 0.2, 0.2, 0.2, 0.15], dtype=np.float64)
    offsets = np.array([0, 1, 3, 5], dtype=np.int64)
    runtime_rules = QuadratureRules(
        kind="per_entity",
        tdim=2,
        points=points,
        weights=weights,
        offsets=offsets,
        parent_map=parent_map,
    )

    # This is the form-level mix: the rule object is per-entity, while the
    # measure payload carries both standard entities and per-entity rules.
    dx_mixed = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_id=0,
        subdomain_data=[standard_entities, runtime_rules],
    )
    form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_mixed

    payload = as_runtime_quadrature_payload(dx_mixed.subdomain_data())
    print("Loop entity indices:", payload.entity_indices.tolist())
    print("is_cut:", payload.is_cut.tolist())
    print("rule_indices:", payload.rule_indices.tolist())

    module = compile_runtime_integrals(form)
    custom_data = module.create_custom_data()
    kernel = module.kernels[0]
    print("Generated kernel:", kernel.name)
    print("Generated mixed branches:", f"{kernel.name}_runtime" in kernel.c_definition)
    print("custom_data pointer:", custom_data.ptr)

    kernel_fn, ffi = _compile_tabulate_tensor(kernel)
    standard_A = _call_kernel(kernel_fn, ffi, custom_data, loop_index=0)
    runtime_A = _call_kernel(kernel_fn, ffi, custom_data, loop_index=4)

    print("Standard branch tensor:")
    print(standard_A)
    print("Runtime branch tensor:")
    print(runtime_A)
    np.testing.assert_allclose(runtime_A, standard_A, rtol=1e-12, atol=1e-14)
    print("Branch tensors match for the unit triangle.")


if __name__ == "__main__":
    main()
