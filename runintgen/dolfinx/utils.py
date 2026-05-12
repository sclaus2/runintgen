"""DOLFINx integration utilities for runintgen.

This module contains Python helpers that depend on DOLFINx. The C++ adapter
for resolving DOLFINx form elements into runintgen custom data lives next to
this module in ``custom_data.h`` and ``custom_data.cpp``.
"""

from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cffi
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from dolfinx.fem import Form, FunctionSpace

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
        return RuntimeFormInfo()

    struct_def = get_runintgen_data_struct()

    c_parts = [
        "#include <math.h>",
        "#include <stdint.h>",
        "",
        struct_def,
    ]

    header_parts = []

    for kernel in module.kernels:
        c_parts.append(kernel.c_definition)
        header_parts.append(kernel.c_declaration.strip().rstrip(";") + ";")

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

    return RuntimeFormInfo(kernels=compiled_kernels, ffi=ffi)


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

    for kernel_idx, kernel in enumerate(runtime_info.kernels):
        itype = _ufl_to_dolfinx_integral_type(kernel.integral_type)

        if itype not in integrals:
            integrals[itype] = []

        active_coeffs = np.array([], dtype=np.int8)
        data_ptr = _resolve_custom_data(
            custom_data, itype, kernel.subdomain_id, kernel_idx
        )

        integrals[itype].append(
            (kernel.subdomain_id, kernel.kernel_ptr, cells, active_coeffs, data_ptr)
        )

    spaces = [V._cpp_object for V in function_spaces]
    coeffs = [c._cpp_object for c in (coefficients or [])]
    consts = [c._cpp_object for c in (constants or [])]

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
        mesh=mesh._cpp_object,
    )

    return dolfinx_fem.Form(cpp_form)


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

    keys = (
        (integral_type, subdomain_id, kernel_idx),
        (integral_type, subdomain_id),
        (subdomain_id, kernel_idx),
        (subdomain_id,),
    )
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
    "CompiledKernel",
    "HAS_DOLFINX",
    "RuntimeFormInfo",
    "compile_runtime_kernels",
    "create_dolfinx_form_with_runtime",
    "set_runtime_data",
]
