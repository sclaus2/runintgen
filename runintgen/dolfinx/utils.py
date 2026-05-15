"""DOLFINx integration utilities for runintgen.

This module contains Python helpers that depend on DOLFINx. The C++ adapter
for resolving DOLFINx form elements into runintgen custom data lives next to
this module in ``custom_data.h`` and ``custom_data.cpp``.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Sequence

import cffi
import numpy as np
import numpy.typing as npt
import ufl

if TYPE_CHECKING:
    from dolfinx.fem import Form, FunctionSpace

    from runintgen.form_metadata import FormRuntimeMetadata
    from runintgen.jit import JITFormInfo
    from runintgen.runtime_data import QuadratureRules

fem: Any | None = None
IntegralType: Any | None = None
HAS_DOLFINX = importlib.util.find_spec("dolfinx") is not None
_DOLFINX_IMPORT_ERROR: BaseException | None = None


def _require_dolfinx() -> tuple[Any, Any]:
    """Load DOLFINx lazily and return ``(fem, IntegralType)``."""
    global HAS_DOLFINX, IntegralType, fem, _DOLFINX_IMPORT_ERROR

    if fem is not None and IntegralType is not None:
        return fem, IntegralType

    try:
        from dolfinx import fem as dolfinx_fem
        from dolfinx.fem import IntegralType as dolfinx_integral_type
    except Exception as exc:
        fem = None
        IntegralType = None
        HAS_DOLFINX = False
        _DOLFINX_IMPORT_ERROR = exc
    else:
        fem = dolfinx_fem
        IntegralType = dolfinx_integral_type
        HAS_DOLFINX = True
        _DOLFINX_IMPORT_ERROR = None
        return fem, IntegralType

    if not HAS_DOLFINX:
        raise ImportError("DOLFINx is required for this function") from (
            _DOLFINX_IMPORT_ERROR
        )
    raise ImportError("DOLFINx could not be loaded") from _DOLFINX_IMPORT_ERROR


def _quadrature_function_set_values(
    self: Any,
    quadrature: Any,
    values: npt.ArrayLike,
) -> None:
    """Attach explicit provider-owned values to a DOLFINx quadrature function."""
    rule_id = getattr(quadrature, "rule_id", None)
    if rule_id is None:
        raise TypeError("quadrature must carry a stable rule_id.")
    self._runintgen_values[str(rule_id)] = values


def _quadrature_function_is_cellwise_constant(self: Any) -> bool:
    """Return false so UFL does not erase quadrature-function derivatives."""
    return False


def _is_function_space(value: Any) -> bool:
    """Return whether ``value`` behaves like a DOLFINx/UFL function space."""
    return (
        hasattr(value, "ufl_domain")
        and hasattr(value, "ufl_element")
        and hasattr(value, "mesh")
    )


def _as_dolfinx_mesh(value: Any) -> Any:
    """Return a concrete DOLFINx mesh from a mesh-like object."""
    if isinstance(value, ufl.Mesh):
        mesh = value.ufl_cargo()
        if mesh is None:
            raise ValueError(
                "runintgen.dolfinx.QuadratureFunction requires a concrete "
                "DOLFINx mesh, not a cargo-free UFL mesh."
            )
        return mesh
    return value


def QuadratureFunction(
    space_or_mesh: Any,
    source: Any | None = None,
    *,
    name: str | None = None,
    space: Any | None = None,
    shape: tuple[int, ...] = (),
    dtype: npt.DTypeLike | None = None,
) -> Any:
    """Create a DOLFINx Function with runintgen quadrature semantics.

    The returned object is a normal ``dolfinx.fem.Function`` and can therefore
    be used naturally in UFL forms. runintgen attaches quadrature-function
    metadata and a ``set_values(rules, values)`` method to that function. If no
    explicit values or callable source are supplied, the DOLFINx adapter
    evaluates the background Function at quadrature physical points before
    assembly.
    """
    dolfinx_fem, _ = _require_dolfinx()
    from runintgen.quadrature_function import QuadratureFunctionSpec

    if space is not None:
        function_space = space
    elif _is_function_space(space_or_mesh):
        function_space = space_or_mesh
    else:
        mesh = _as_dolfinx_mesh(space_or_mesh)
        element = ("DG", 0, shape) if shape else ("DG", 0)
        function_space = dolfinx_fem.functionspace(mesh, element)

    function = dolfinx_fem.Function(
        function_space,
        name=name,
        dtype=dtype,
    )

    value_shape = tuple(shape or function.ufl_shape)
    if shape and value_shape != tuple(function.ufl_shape):
        raise ValueError(
            "The supplied shape does not match the DOLFINx FunctionSpace "
            f"shape {tuple(function.ufl_shape)}."
        )
    value_size = int(prod(value_shape)) if value_shape else 1
    function._runintgen_quadrature_function = QuadratureFunctionSpec(
        name=name,
        value_shape=value_shape,
        value_size=value_size,
    )
    function._runintgen_source = source
    function._runintgen_values = {}
    function.set_values = MethodType(_quadrature_function_set_values, function)
    function.is_cellwise_constant = MethodType(
        _quadrature_function_is_cellwise_constant,
        function,
    )
    return function


@dataclass
class CompiledKernel:
    """A compiled runtime kernel ready for use with DOLFINx.

    Attributes:
        name: The kernel function name.
        integral_type: Type of integral ("cell", "exterior_facet", etc.).
        subdomain_id: The subdomain identifier.
        kernel_ptr: Function pointer as an integer.
        ffi: The CFFI object used to compile the module.
        module: The compiled CFFI module; keep it alive with the kernel.
        table_info: Information about required element tables.
    """

    name: str
    integral_type: str
    subdomain_id: int
    kernel_ptr: int
    ffi: cffi.FFI
    module: Any
    table_info: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RuntimeFormInfo:
    """Compiled runtime kernels and their shared CFFI object."""

    kernels: list[CompiledKernel] = field(default_factory=list)
    ffi: cffi.FFI | None = None
    form_metadata: "FormRuntimeMetadata | None" = None


@dataclass
class CompiledRunintForm:
    """Compiled runintgen form without associated DOLFINx runtime objects."""

    ufl_form: ufl.Form
    ufcx_form: Any
    module: Any
    code: tuple[str | None, str | None]
    dtype: npt.DTypeLike
    jit_info: "JITFormInfo"


def compile_runtime_kernels(
    form: Any,
    options: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> RuntimeFormInfo:
    """Compile all runtime integrals in a UFL form.

    Args:
        form: A UFL form containing runtime integrals.
        options: Optional FFCX compilation options.
        cache_dir: Optional directory for compiled CFFI modules.

    Returns:
        RuntimeFormInfo containing compiled kernels and CFFI state.
    """
    from runintgen.codegeneration.C.integrals import get_runintgen_data_struct
    from runintgen.runtime_api import compile_runtime_integrals
    from runintgen.runtime_data import CFFI_DEF

    module = compile_runtime_integrals(form, options)

    if not module.kernels:
        return RuntimeFormInfo(form_metadata=module.form_metadata)

    struct_def = get_runintgen_data_struct()

    c_parts = [
        "#include <complex.h>",
        "#include <math.h>",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        struct_def,
    ]

    header_parts = []

    for kernel in module.kernels:
        c_parts.append(_strip_ufcx_integral_object(kernel.c_definition, kernel.name))
        header_parts.append(_tabulate_tensor_declaration(kernel))

    full_c_code = "\n".join(c_parts)
    full_header = "\n".join(header_parts)

    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)
    ffi.cdef(full_header)

    if cache_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="runintgen_")
    else:
        tmpdir = str(cache_dir)
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    module_name = f"runtime_kernels_{id(form)}"
    ffi.set_source(
        module_name,
        full_c_code,
        extra_compile_args=["-std=c17", "-O2", "-fPIC"],
    )
    ffi.compile(tmpdir=tmpdir, verbose=False)

    import importlib.machinery
    import importlib.util

    finder = importlib.machinery.FileFinder(
        tmpdir,
        (
            importlib.machinery.ExtensionFileLoader,
            importlib.machinery.EXTENSION_SUFFIXES,
        ),
    )
    finder.invalidate_caches()
    spec = finder.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load compiled runintgen kernel module.")
    compiled_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled_module)

    compiled_kernels = []
    for kernel in module.kernels:
        func_name = f"tabulate_tensor_{kernel.name}"
        func = getattr(compiled_module.lib, func_name)
        kernel_ptr = int(ffi.cast("intptr_t", func))

        compiled_kernels.append(
            CompiledKernel(
                name=kernel.name,
                integral_type=kernel.integral_type,
                subdomain_id=kernel.subdomain_id,
                kernel_ptr=kernel_ptr,
                ffi=ffi,
                module=compiled_module,
                table_info=kernel.table_info or [],
            )
        )

    return RuntimeFormInfo(
        kernels=compiled_kernels,
        ffi=ffi,
        form_metadata=module.form_metadata,
    )


def has_runtime_custom_data_support() -> bool:
    """Return whether the loaded DOLFINx exposes required custom-data support."""
    try:
        dolfinx_fem, _ = _require_dolfinx()
        from dolfinx import cpp as _cpp
    except Exception:
        return False

    return (
        hasattr(_cpp.fem, "Form_float64")
        and hasattr(dolfinx_fem, "Form")
    )


def _has_runtime_integrals(form: Any) -> bool:
    """Return whether a form tree contains runintgen runtime integrals."""
    from runintgen.measures import is_runtime_integral

    if isinstance(form, ufl.Form):
        return any(is_runtime_integral(integral) for integral in form.integrals())
    if isinstance(form, ufl.ZeroBaseForm):
        return False
    if isinstance(form, Sequence) and not isinstance(form, (str, bytes)):
        return any(_has_runtime_integrals(item) for item in form)
    return False


def _reject_standard_quadrature_functions(form_object: ufl.Form) -> None:
    """Reject standard-integral quadrature functions until supported."""
    from runintgen.measures import is_runtime_integral
    from runintgen.quadrature_function import integral_quadrature_functions

    for integral in form_object.integrals():
        if is_runtime_integral(integral):
            continue
        functions = integral_quadrature_functions(integral)
        if not functions:
            continue
        labels = []
        for function in functions:
            spec = getattr(function, "_runintgen_quadrature_function")
            labels.append(spec.name or "<unnamed>")
        raise NotImplementedError(
            "QuadratureFunction in standard DOLFINx integrals is not "
            "implemented yet. Use a runtime measure with QuadratureRules, or "
            "provide the quantity as an ordinary DOLFINx Function if standard "
            "coefficient interpolation is intended. Affected quadrature "
            f"functions: {', '.join(labels)}."
        )


def _as_cpp_object(obj: Any) -> Any:
    """Return the wrapped C++ object when ``obj`` is a Python DOLFINx wrapper."""
    return getattr(obj, "_cpp_object", obj)


def jit(
    comm: Any,
    ufl_object: Any,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
) -> tuple[Any, Any, tuple[str | None, str | None]]:
    """Compile one UFL form with runintgen using DOLFINx JIT options."""
    if not isinstance(ufl_object, ufl.Form):
        raise TypeError(type(ufl_object))

    import ffcx
    from dolfinx import jit as dolfinx_jit

    import runintgen.jit as runintgen_jit

    def _local_jit(
        form_object: ufl.Form,
        *,
        form_compiler_options: dict[str, Any] | None = None,
        jit_options: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, tuple[str | None, str | None]]:
        p_ffcx = ffcx.get_options(form_compiler_options)
        p_jit = dolfinx_jit.get_options(jit_options)
        compiled_forms, module, code = runintgen_jit.compile_forms(
            [form_object],
            options=p_ffcx,
            **p_jit,
        )
        return compiled_forms[0], module, code

    mpi_jit = dolfinx_jit.mpi_jit_decorator(_local_jit)
    return mpi_jit(
        comm,
        ufl_object,
        form_compiler_options=form_compiler_options,
        jit_options=jit_options,
    )


def compile_form(
    comm: Any,
    form: ufl.Form,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
) -> CompiledRunintForm:
    """Compile a UFL form without binding DOLFINx functions/domains."""
    import ffcx

    if form_compiler_options is None:
        form_compiler_options = {}
    p_ffcx = ffcx.get_options(form_compiler_options)
    ufcx_form, module, code = jit(
        comm,
        form,
        form_compiler_options=p_ffcx,
        jit_options=jit_options,
    )
    sidecar = module._runintgen_jit.forms[0]
    return CompiledRunintForm(
        ufl_form=form,
        ufcx_form=ufcx_form,
        module=module,
        code=code,
        dtype=np.dtype(p_ffcx["scalar_type"]),
        jit_info=sidecar,
    )


def _normalised_subdomain_id(value: Any) -> int:
    """Return a DOLFINx-compatible integral id."""
    return -1 if value in ("otherwise", "everywhere") else int(value)


def _default_cell_entities(mesh: Any) -> npt.NDArray[np.int32]:
    """Return owned cell indices for a mesh."""
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    return np.arange(num_cells, dtype=np.int32)


def _points_as_2d(
    points: npt.NDArray[np.float64],
    tdim: int,
) -> npt.NDArray[np.float64]:
    """Return reference points with shape ``(num_points, tdim)``."""
    if points.ndim == 2:
        return points
    return points.reshape((-1, tdim))


def _as_contiguous_array(value: Any, dtype: npt.DTypeLike) -> np.ndarray:
    """Return a C-contiguous array, copying only when the provider requires it."""
    array = np.asarray(value, dtype=dtype)
    if array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array)


def _basix_cell_type_id(mesh: Any) -> int | None:
    """Return the Basix cell type id for a DOLFINx mesh if available."""
    if not hasattr(mesh, "basix_cell"):
        return None
    return int(mesh.basix_cell())


def _compiled_physical_points(mesh: Any, rules: "QuadratureRules", gdim: int) -> Any:
    """Use the DOLFINx-free C++ geometry helper when possible."""
    cell_type = _basix_cell_type_id(mesh)
    if cell_type is None:
        return None

    try:
        from runintgen import _basix_runtime
    except ImportError:
        return None

    geometry = mesh.geometry
    cmap = geometry.cmap()
    geometry_x = _as_contiguous_array(geometry.x, np.float64)
    geometry_dofmap = _as_contiguous_array(geometry.dofmap, np.int32)
    degree = int(cmap.degree)
    variant = int(cmap.variant)

    if rules.kind == "per_entity":
        if rules.offsets is None:
            raise ValueError("per-entity QuadratureRules require offsets.")
        physical_points, _ = _basix_runtime.map_per_entity_geometry(
            cell_type,
            degree,
            variant,
            rules.tdim,
            gdim,
            rules.points,
            rules.weights,
            rules.offsets,
            rules.parent_map,
            geometry_x,
            geometry_dofmap,
            False,
        )
    elif rules.kind == "shared":
        physical_points, _ = _basix_runtime.map_shared_geometry(
            cell_type,
            degree,
            variant,
            rules.tdim,
            gdim,
            rules.points,
            rules.weights,
            rules.parent_map,
            geometry_x,
            geometry_dofmap,
            False,
        )
    else:  # pragma: no cover - QuadratureRules validates this.
        raise ValueError(f"Unsupported QuadratureRules kind {rules.kind!r}.")
    return np.asarray(physical_points, dtype=np.float64)


def compute_physical_points(
    mesh: Any,
    rules: "QuadratureRules",
) -> "QuadratureRules":
    """Return rules with physical quadrature points attached.

    The input ``rules`` object is not mutated. Its reference points, weights,
    offsets, parent map, and rule id are reused in the returned
    ``QuadratureRules`` object; only the component-first physical point array is
    newly allocated.

    Args:
        mesh: DOLFINx mesh whose geometry map should be used.
        rules: ``QuadratureRules`` with ``parent_map`` identifying local cells.

    Returns:
        A ``QuadratureRules`` object with ``physical_points`` of shape
        ``(gdim, total_nq)``.
    """
    from runintgen.runtime_data import QuadratureRules

    if not isinstance(rules, QuadratureRules):
        raise TypeError("rules must be a runintgen.QuadratureRules object.")
    if rules.parent_map is None:
        raise ValueError(
            "compute_physical_points requires QuadratureRules.parent_map."
        )

    geometry = mesh.geometry
    dofmap = geometry.dofmap
    x = geometry.x
    gdim = int(getattr(geometry, "dim", x.shape[1]))
    if rules.gdim is not None and int(rules.gdim) != gdim:
        raise ValueError(
            f"QuadratureRules.gdim={rules.gdim} does not match mesh geometry "
            f"dimension {gdim}."
        )

    if np.dtype(x.dtype) != np.dtype(np.float64):
        raise NotImplementedError(
            "runintgen.dolfinx.compute_physical_points currently supports "
            "float64 mesh geometry."
        )

    parent_map = np.asarray(rules.parent_map, dtype=np.int32)
    if np.any(parent_map < 0) or np.any(parent_map >= dofmap.shape[0]):
        raise ValueError("QuadratureRules.parent_map contains invalid local cells.")

    physical_points = _compiled_physical_points(mesh, rules, gdim)
    if physical_points is None:
        cmap = geometry.cmap()
        ref_points = _points_as_2d(rules.points, rules.tdim)
        physical_points = np.empty((gdim, rules.total_points), dtype=np.float64)

        if rules.kind == "per_entity":
            if rules.offsets is None:
                raise ValueError("per-entity QuadratureRules require offsets.")
            for rule_index, cell in enumerate(parent_map):
                q0 = int(rules.offsets[rule_index])
                q1 = int(rules.offsets[rule_index + 1])
                if q0 == q1:
                    continue
                cell_geometry = np.asarray(
                    x[dofmap[int(cell)], :gdim],
                    dtype=np.float64,
                )
                physical_points[:, q0:q1] = cmap.push_forward(
                    ref_points[q0:q1],
                    cell_geometry,
                ).T
        elif rules.kind == "shared":
            nq = int(ref_points.shape[0])
            for entity_index, cell in enumerate(parent_map):
                q0 = entity_index * nq
                q1 = q0 + nq
                cell_geometry = np.asarray(
                    x[dofmap[int(cell)], :gdim],
                    dtype=np.float64,
                )
                physical_points[:, q0:q1] = cmap.push_forward(
                    ref_points,
                    cell_geometry,
                ).T
        else:  # pragma: no cover - QuadratureRules validates this.
            raise ValueError(f"Unsupported QuadratureRules kind {rules.kind!r}.")

    return QuadratureRules(
        tdim=rules.tdim,
        points=rules.points,
        weights=rules.weights,
        offsets=rules.offsets,
        parent_map=rules.parent_map,
        rule_id=rules.rule_id,
        kind=rules.kind,
        gdim=gdim,
        physical_points=physical_points,
    )


def _cells_for_quadrature_points(rules: "QuadratureRules") -> npt.NDArray[np.int32]:
    """Return one parent cell index per flattened quadrature point."""
    if rules.parent_map is None:
        raise ValueError(
            "DOLFINx QuadratureFunction evaluation requires "
            "QuadratureRules.parent_map."
        )
    if rules.kind == "per_entity":
        if rules.offsets is None:
            raise ValueError("per-entity QuadratureRules require offsets.")
        counts = np.diff(rules.offsets).astype(np.int64, copy=False)
        return np.repeat(rules.parent_map, counts).astype(np.int32, copy=False)
    if rules.kind == "shared":
        return np.repeat(rules.parent_map, rules.weights.size).astype(
            np.int32,
            copy=False,
        )
    raise ValueError(f"Unsupported QuadratureRules kind {rules.kind!r}.")


def _points_for_dolfinx_eval(rules: "QuadratureRules") -> npt.NDArray[np.float64]:
    """Return component-major physical points as DOLFINx ``(n, 3)`` input."""
    if rules.physical_points is None:
        raise ValueError(
            "DOLFINx QuadratureFunction evaluation requires physical_points."
        )
    points = np.asarray(rules.physical_points, dtype=np.float64).T
    if points.shape[1] == 3:
        return np.ascontiguousarray(points, dtype=np.float64)
    if points.shape[1] > 3:
        raise ValueError("DOLFINx Function.eval expects at most 3 coordinates.")
    padded = np.zeros((points.shape[0], 3), dtype=np.float64)
    padded[:, : points.shape[1]] = points
    return padded


def _evaluate_background_quadrature_function(
    info: Any,
    rules: "QuadratureRules",
) -> npt.NDArray[np.float64]:
    """Evaluate a DOLFINx Function at quadrature physical points."""
    function = info.terminal
    if not hasattr(function, "eval") or not hasattr(function, "function_space"):
        raise ValueError(
            f"QuadratureFunction {info.label!r} has no explicit values, no "
            "callable source, and no DOLFINx background Function evaluator."
        )

    if rules.physical_points is None:
        rules = compute_physical_points(function.function_space.mesh, rules)

    points = _points_for_dolfinx_eval(rules)
    cells = _cells_for_quadrature_points(rules)
    if points.shape[0] != cells.size:
        raise ValueError("Quadrature physical points and parent cells disagree.")

    values = np.asarray(function.eval(points, cells), dtype=np.float64)
    if rules.total_points == 1:
        values = values.reshape(1, info.value_size)
    return values


def _needs_physical_points_for_q_functions(
    module: Any,
    rules: "QuadratureRules",
) -> bool:
    """Return whether q-function sources need physical points for these rules."""
    from runintgen.quadrature_function import (
        quadrature_function_source,
        quadrature_function_values,
    )

    for info in getattr(module, "quadrature_functions", []) or []:
        explicit = quadrature_function_values(info.terminal)
        if str(rules.rule_id) in explicit:
            continue
        if quadrature_function_source(info.terminal) is not None:
            return True
        if hasattr(info.terminal, "eval") and hasattr(info.terminal, "function_space"):
            return True
    return False


def _quadrature_for_custom_data(
    mesh: Any,
    module: Any,
    provider: Any,
) -> Any:
    """Attach physical points when DOLFINx q-function evaluation needs them."""
    from runintgen.runtime_data import (
        RuntimeQuadraturePayload,
        as_runtime_quadrature_payload,
    )

    payload = as_runtime_quadrature_payload(provider)
    rules = payload.rules
    if (
        rules.physical_points is None
        and _needs_physical_points_for_q_functions(module, rules)
    ):
        rules = compute_physical_points(mesh, rules)
        return RuntimeQuadraturePayload(
            rules=rules,
            entities=payload.entities,
            quadrature_functions=payload.quadrature_functions,
        )
    return provider


def create_custom_data(
    module: Any,
    mesh: Any,
    quadrature: Any,
    *,
    is_cut: npt.ArrayLike | None = None,
) -> Any:
    """Create runtime custom data with DOLFINx q-function evaluation.

    This is the DOLFINx-aware counterpart of ``runintgen.create_custom_data``.
    It keeps DOLFINx optional by living in ``runintgen.dolfinx`` and only uses
    DOLFINx-specific behavior for default background Function evaluation and
    physical-point mapping.
    """
    from runintgen.basix_runtime import CustomData

    quadrature = _quadrature_for_custom_data(mesh, module, quadrature)
    return CustomData(
        module,
        quadrature=quadrature,
        is_cut=is_cut,
        quadrature_function_evaluator=_evaluate_background_quadrature_function,
    )


def _active_coefficients(
    ffi_obj: cffi.FFI,
    ufcx_form: Any,
    integral: Any,
) -> npt.NDArray[np.int8]:
    """Return active coefficient indices for one UFCx integral."""
    if integral.enabled_coefficients == ffi_obj.NULL:
        return np.array([], dtype=np.int8)
    values = [
        i
        for i in range(int(ufcx_form.num_coefficients))
        if bool(integral.enabled_coefficients[i])
    ]
    return np.asarray(values, dtype=np.int8)


def _runtime_providers(jit_info: "JITFormInfo") -> dict[tuple[str, int], Any]:
    """Return runtime quadrature providers keyed by integral type/subdomain id."""
    providers: dict[tuple[str, int], Any] = {}
    for group in jit_info.analysis.groups:
        for sid in group.subdomain_ids:
            providers[(group.integral_type, _normalised_subdomain_id(sid))] = (
                group.quadrature_provider
            )
    return providers


def _runtime_domain_and_custom_data(
    *,
    compiled: CompiledRunintForm,
    mesh: Any,
    integral_type: str,
    subdomain_id: int,
    kernel_idx: int,
    custom_data: int | dict[tuple[Any, ...], int | Any | None] | Any | None,
    cache: dict[tuple[int, str, int], Any],
) -> tuple[npt.NDArray[np.int32], int | None, Any | None]:
    """Return runtime entity domain and custom-data pointer for one integral."""
    from runintgen.basix_runtime import CustomData
    from runintgen.runtime_data import as_runtime_quadrature_payload

    if integral_type != "cell":
        raise NotImplementedError(
            "runintgen.dolfinx.form currently supports runtime cell "
            f"integrals only, got {integral_type!r}."
        )

    providers = _runtime_providers(compiled.jit_info)
    provider = providers.get((integral_type, subdomain_id))
    if provider is None:
        raise RuntimeError(
            "Could not find runtime quadrature provider for "
            f"{integral_type} subdomain {subdomain_id}."
        )

    payload = as_runtime_quadrature_payload(provider)
    if payload.parent_map is None:
        raise RuntimeError(
            "DOLFINx runtime quadrature requires QuadratureRules.parent_map "
            "so per-entity rules can be mapped to mesh entities."
        )

    data_ptr = _resolve_custom_data(
        custom_data,
        integral_type,
        subdomain_id,
        kernel_idx,
    )
    owner = None
    if data_ptr is None:
        cache_key = (id(provider), integral_type, subdomain_id)
        owner = cache.get(cache_key)
        if owner is None:
            quadrature = _quadrature_for_custom_data(
                mesh,
                compiled.jit_info.module,
                provider,
            )
            owner = CustomData(
                compiled.jit_info.module,
                quadrature=quadrature,
                quadrature_function_evaluator=(
                    _evaluate_background_quadrature_function
                ),
            )
            cache[cache_key] = owner
        data_ptr = int(owner.ptr)

    entities = np.ascontiguousarray(payload.entity_indices, dtype=np.int32)
    return entities, data_ptr, owner


def _standard_domain(
    *,
    mesh: Any,
    integral_type: Any,
    subdomain_id: int,
    subdomains: dict[Any, list[tuple[int, npt.NDArray[np.int32]]]] | None,
) -> npt.NDArray[np.int32]:
    """Return a standard DOLFINx integration domain."""
    if subdomain_id == -1 and integral_type == _ufl_to_dolfinx_integral_type("cell"):
        return _default_cell_entities(mesh)

    if subdomains is not None:
        for sid, entities in subdomains.get(integral_type, []):
            if int(sid) == int(subdomain_id):
                return np.ascontiguousarray(entities, dtype=np.int32).ravel()

    raise RuntimeError(
        "Could not resolve integration domain for standard integral "
        f"{integral_type} id {subdomain_id}."
    )


def _build_subdomains(
    form_object: ufl.Form,
    ufcx_form: Any,
) -> dict[Any, list[tuple[int, npt.NDArray[np.int32]]]]:
    """Build standard DOLFINx integration domains where possible."""
    from dolfinx.fem.forms import get_integration_domains

    sd = form_object.subdomain_data()
    if not sd:
        return {}
    (domain,) = list(sd.keys())
    subdomains: dict[Any, list[tuple[int, npt.NDArray[np.int32]]]] = {}
    domain_data = sd.get(domain)
    offsets = [
        int(ufcx_form.form_integral_offsets[i]) for i in range(len(IntegralType) + 1)
    ]
    for integral_type_name, subdomain_data in domain_data.items():
        integral_type = _ufl_to_dolfinx_integral_type(integral_type_name)
        type_index = _integral_type_index(integral_type)
        ids = [
            int(ufcx_form.form_integral_ids[j])
            for j in range(
                offsets[type_index],
                offsets[type_index + 1],
            )
        ]
        ids = sorted({sid for sid in ids if sid != -1})
        if not ids:
            continue
        try:
            data = subdomain_data[0]
            if all(d is data for d in subdomain_data if d is not None):
                subdomains[integral_type] = get_integration_domains(
                    integral_type,
                    data,
                    ids,
                )
        except Exception:
            continue
    return subdomains


def create_form(
    compiled: CompiledRunintForm,
    V: list["FunctionSpace"],
    msh: Any,
    subdomains: dict[Any, list[tuple[int, npt.NDArray[np.int32]]]] | None = None,
    coefficient_map: dict[Any, Any] | None = None,
    constant_map: dict[Any, Any] | None = None,
    entity_maps: Sequence[Any] | None = None,
    custom_data: int | dict[tuple[Any, ...], int | Any | None] | Any | None = None,
) -> "Form":
    """Create a DOLFINx Form from a compiled runintgen form."""
    dolfinx_fem, _ = _require_dolfinx()
    from dolfinx import cpp as _cpp

    if np.dtype(compiled.dtype) != np.dtype(np.float64):
        raise NotImplementedError("runintgen.dolfinx currently supports float64 only.")

    if subdomains is None:
        subdomains = _build_subdomains(compiled.ufl_form, compiled.ufcx_form)

    original_coefficients = compiled.ufl_form.coefficients()
    coefficients = []
    coefficient_map = coefficient_map or {}
    for i in range(int(compiled.ufcx_form.num_coefficients)):
        original_index = int(compiled.ufcx_form.original_coefficient_positions[i])
        original = original_coefficients[original_index]
        coeff = coefficient_map.get(original, original)
        coefficients.append(_as_cpp_object(coeff))

    original_constants = compiled.ufl_form.constants()
    constants = []
    constant_map = constant_map or {}
    for constant in original_constants:
        constants.append(_as_cpp_object(constant_map.get(constant, constant)))

    integrals: dict[Any, list[Any]] = {}
    owners: list[Any] = []
    owner_cache: dict[tuple[int, str, int], Any] = {}
    ffi_obj = compiled.module.ffi

    offsets = [
        int(compiled.ufcx_form.form_integral_offsets[i])
        for i in range(len(IntegralType) + 1)
    ]
    info_by_position = {
        info.integral_position: info for info in compiled.jit_info.integral_infos
    }
    for type_index in range(len(IntegralType)):
        integral_type = IntegralType(type_index)
        for j in range(offsets[type_index], offsets[type_index + 1]):
            integral_index = j - offsets[type_index]
            info = info_by_position[j]
            integral = compiled.ufcx_form.form_integrals[j]
            kernel_fn = integral.tabulate_tensor_float64
            if kernel_fn == ffi_obj.NULL:
                raise RuntimeError("Generated float64 UFCx kernel is NULL.")
            kernel_ptr = int(ffi_obj.cast("uintptr_t", kernel_fn))
            subdomain_id = int(compiled.ufcx_form.form_integral_ids[j])
            active_coeffs = _active_coefficients(
                ffi_obj,
                compiled.ufcx_form,
                integral,
            )

            if info.needs_custom_data:
                entities, data_ptr, owner = _runtime_domain_and_custom_data(
                    compiled=compiled,
                    mesh=msh,
                    integral_type=info.integral_type,
                    subdomain_id=subdomain_id,
                    kernel_idx=0,
                    custom_data=custom_data,
                    cache=owner_cache,
                )
                if owner is not None:
                    owners.append(owner)
            else:
                entities = _standard_domain(
                    mesh=msh,
                    integral_type=integral_type,
                    subdomain_id=subdomain_id,
                    subdomains=subdomains,
                )
                data_ptr = _resolve_custom_data(
                    custom_data,
                    integral_type,
                    subdomain_id,
                    0,
                )

            integrals.setdefault(integral_type, []).append(
                (integral_index, kernel_ptr, entities, active_coeffs, data_ptr)
            )

    cpp_form = _cpp.fem.Form_float64(
        [_as_cpp_object(space) for space in V],
        integrals,
        coefficients,
        constants,
        any(
            bool(compiled.ufcx_form.form_integrals[j].needs_facet_permutations)
            for j in range(offsets[0], offsets[-1])
        ),
        (
            []
            if entity_maps is None
            else [_as_cpp_object(entity_map) for entity_map in entity_maps]
        ),
        mesh=_as_cpp_object(msh),
    )
    result = dolfinx_fem.Form(
        cpp_form,
        compiled.ufcx_form,
        compiled.code,
        compiled.module,
    )
    result._runintgen_compiled_form = compiled
    result._runintgen_custom_data = owners
    return result


def form(
    form_object: Any,
    dtype: npt.DTypeLike | None = None,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
    jit_comm: Any | None = None,
    entity_maps: Sequence[Any] | None = None,
    custom_data: int | dict[tuple[Any, ...], int | Any | None] | Any | None = None,
) -> Any:
    """Create a DOLFINx Form using runintgen when runtime integrals are present."""
    dolfinx_fem, _ = _require_dolfinx()
    from dolfinx import default_scalar_type

    if not _has_runtime_integrals(form_object):
        return dolfinx_fem.form(
            form_object,
            dtype=default_scalar_type if dtype is None else dtype,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            jit_comm=jit_comm,
            entity_maps=entity_maps,
        )

    if not has_runtime_custom_data_support():
        raise RuntimeError(
            "runintgen runtime forms require a patched DOLFINx build with "
            "per-integral custom_data support."
        )

    dtype = default_scalar_type if dtype is None else dtype
    if np.dtype(dtype) != np.dtype(np.float64):
        raise NotImplementedError("runintgen.dolfinx currently supports float64 only.")

    if form_compiler_options is None:
        form_compiler_options = {}
    form_compiler_options = dict(form_compiler_options)
    form_compiler_options["scalar_type"] = dtype

    def _create_one(ufl_form: Any) -> Any:
        if isinstance(ufl_form, ufl.ZeroBaseForm):
            return dolfinx_fem.form(
                ufl_form,
                dtype=dtype,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                jit_comm=jit_comm,
                entity_maps=entity_maps,
            )
        if not isinstance(ufl_form, ufl.Form):
            return ufl_form
        _reject_standard_quadrature_functions(ufl_form)
        if not _has_runtime_integrals(ufl_form):
            return dolfinx_fem.form(
                ufl_form,
                dtype=dtype,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                jit_comm=jit_comm,
                entity_maps=entity_maps,
            )

        sd = ufl_form.subdomain_data()
        (domain,) = list(sd.keys())
        msh = domain.ufl_cargo()
        if msh is None:
            raise RuntimeError("Expecting to find a DOLFINx Mesh in the form.")
        comm = msh.comm if jit_comm is None else jit_comm
        compiled = compile_form(
            comm,
            ufl_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        V = [arg.ufl_function_space() for arg in ufl_form.arguments()]
        if form_compiler_options.get("part", "full") == "diagonal":
            V = [V[0]]
        return create_form(
            compiled,
            V,
            msh,
            entity_maps=entity_maps,
            custom_data=custom_data,
        )

    def _create_tree(value: Any) -> Any:
        if isinstance(value, (ufl.Form, ufl.ZeroBaseForm)):
            return _create_one(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_create_tree(item) for item in value]
        return value

    if isinstance(form_object, (ufl.Form, ufl.ZeroBaseForm)):
        return _create_one(form_object)
    if isinstance(form_object, Sequence) and not isinstance(form_object, (str, bytes)):
        return [_create_tree(item) for item in form_object]
    return form_object


def _c_scalar_type(dtype_name: str | None) -> str:
    """Return the C scalar type used in a generated kernel signature."""
    mapping = {
        "float32": "float",
        "float64": "double",
        "complex64": "float _Complex",
        "complex128": "double _Complex",
    }
    return mapping.get(dtype_name or "float64", "double")


def _tabulate_tensor_declaration(kernel: Any) -> str:
    """Return the CFFI declaration for one generated tabulate function."""
    scalar = _c_scalar_type(kernel.scalar_type)
    geom = _c_scalar_type(kernel.geometry_type)
    return f"""
void tabulate_tensor_{kernel.name}({scalar}* restrict A,
                                   const {scalar}* restrict w,
                                   const {scalar}* restrict c,
                                   const {geom}* restrict coordinate_dofs,
                                   const int* restrict entity_local_index,
                                   const uint8_t* restrict quadrature_permutation,
                                   void* custom_data);
"""


def _strip_ufcx_integral_object(c_definition: str, kernel_name: str) -> str:
    """Remove the UFCx integral object from CFFI/JIT source code."""
    pattern = (
        rf"\nufcx_integral\s+{re.escape(kernel_name)}\s*=\s*"
        rf"\{{.*?\n\}};\n"
    )
    return re.sub(pattern, "\n", c_definition, flags=re.DOTALL)


def create_dolfinx_form_with_runtime(
    function_spaces: list["FunctionSpace"],
    runtime_info: RuntimeFormInfo,
    mesh: Any,
    cells: npt.NDArray[np.int32] | None = None,
    coefficients: list[Any] | None = None,
    constants: list[Any] | None = None,
    custom_data: int | dict[tuple[Any, ...], int | None] | None = None,
) -> "Form":
    """Create a DOLFINx Form using compiled runtime kernels.

    Args:
        function_spaces: DOLFINx function spaces, ordered as DOLFINx expects
            for the form arguments.
        runtime_info: Compiled runtime kernel information.
        mesh: The DOLFINx mesh.
        cells: Optional cell indices. If None, all local cells are used.
        coefficients: Optional coefficient Functions.
        constants: Optional Constants.
        custom_data: Runtime context pointer for all kernels, or a dictionary
            keyed by ``(integral_type, subdomain_id, kernel_idx)``.

    Returns:
        A DOLFINx Form ready for assembly.
    """
    dolfinx_fem, _ = _require_dolfinx()

    from dolfinx import cpp as _cpp

    tdim = mesh.topology.dim
    if cells is None:
        num_cells = mesh.topology.index_map(tdim).size_local
        cells = np.arange(num_cells, dtype=np.int32)

    integrals: dict[Any, list] = {}

    integral_indices: dict[Any, int] = {}
    for kernel_idx, kernel in enumerate(runtime_info.kernels):
        itype = _ufl_to_dolfinx_integral_type(kernel.integral_type)
        integral_index = integral_indices.get(itype, 0)
        integral_indices[itype] = integral_index + 1

        if itype not in integrals:
            integrals[itype] = []

        active_coeffs = np.array([], dtype=np.int8)
        data_ptr = _resolve_custom_data(
            custom_data, itype, kernel.subdomain_id, kernel_idx
        )

        integrals[itype].append(
            (integral_index, kernel.kernel_ptr, cells, active_coeffs, data_ptr)
        )

    spaces = [_as_cpp_object(V) for V in function_spaces]
    coeffs = [_as_cpp_object(c) for c in (coefficients or [])]
    consts = [_as_cpp_object(c) for c in (constants or [])]

    dtype = mesh.geometry.x.dtype
    if np.issubdtype(dtype, np.float64):
        form_class = _cpp.fem.Form_float64
    elif np.issubdtype(dtype, np.float32):
        form_class = _cpp.fem.Form_float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    cpp_form = form_class(
        spaces,
        integrals,
        coeffs,
        consts,
        False,
        [],
        mesh=_as_cpp_object(mesh),
    )

    form = dolfinx_fem.Form(cpp_form)
    try:
        form._runintgen_custom_data = custom_data
        form._runintgen_runtime_info = runtime_info
    except Exception:
        pass
    return form


def _ufl_to_dolfinx_integral_type(ufl_type: str) -> "IntegralType":
    """Convert UFL integral type string to DOLFINx IntegralType."""
    _, integral_type = _require_dolfinx()
    mapping = {
        "cell": integral_type.cell,
        "exterior_facet": integral_type.exterior_facet,
        "interior_facet": integral_type.interior_facet,
        "vertex": integral_type.vertex,
    }
    return mapping[ufl_type]


def _integral_type_index(integral_type: Any) -> int:
    """Return the UFCx/DOLFINx enum index for an integral type."""
    try:
        return int(integral_type)
    except (TypeError, ValueError):
        name = getattr(integral_type, "name", None)
        order = ("cell", "exterior_facet", "interior_facet", "vertex", "ridge")
        if name in order:
            return order.index(name)
        raise


def _resolve_custom_data(
    custom_data: int | dict[tuple[Any, ...], int | None] | None,
    integral_type: Any,
    subdomain_id: int,
    kernel_idx: int,
) -> int | None:
    """Resolve a custom data pointer for one generated kernel."""
    if custom_data is None:
        return None
    if not isinstance(custom_data, dict):
        return int(custom_data)

    integral_type_variants = [integral_type]
    if isinstance(integral_type, str):
        try:
            integral_type_variants.append(_ufl_to_dolfinx_integral_type(integral_type))
        except Exception:
            pass
    else:
        name = getattr(integral_type, "name", None)
        if name is not None:
            integral_type_variants.append(name)

    keys = []
    for type_variant in integral_type_variants:
        keys.extend(
            [
                (type_variant, subdomain_id, kernel_idx),
                (type_variant, subdomain_id),
            ]
        )
    keys.extend([(subdomain_id, kernel_idx), (subdomain_id,)])
    for key in keys:
        if key in custom_data:
            value = custom_data[key]
            return None if value is None else int(value)
    return None


def set_runtime_data(
    form: "Form",
    integral_type: "IntegralType",
    subdomain_id: int,
    runtime_data_ptr: int,
    kernel_idx: int = 0,
) -> None:
    """Raise with guidance for the old mutable custom-data workflow."""
    _require_dolfinx()

    raise RuntimeError(
        "DOLFINx custom_data is set at Form construction time. Pass "
        "custom_data to create_dolfinx_form_with_runtime instead."
    )


__all__ = [
    "CompiledRunintForm",
    "CompiledKernel",
    "HAS_DOLFINX",
    "QuadratureFunction",
    "RuntimeFormInfo",
    "compile_runtime_kernels",
    "compile_form",
    "compute_physical_points",
    "create_custom_data",
    "create_form",
    "create_dolfinx_form_with_runtime",
    "form",
    "has_runtime_custom_data_support",
    "jit",
    "set_runtime_data",
]
