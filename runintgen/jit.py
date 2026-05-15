"""CFFI JIT compiler for combined runintgen/FFCx forms."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import io
import logging
import os
import sys
import sysconfig
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cffi
import ffcx
import ffcx.codegeneration
import ffcx.naming
import numpy as np
import ufl
from ffcx.codegeneration.C import file as ffcx_file
from ffcx.codegeneration.C import form as ffcx_form
from ffcx.codegeneration.C.file_template import libraries as _ffcx_libraries
from ffcx.codegeneration.jit import (
    UFC_FORM_DECL,
    UFC_HEADER_DECL,
    UFC_INTEGRAL_DECL,
    get_cached_module,
)

from .analysis import RuntimeAnalysisInfo, build_runtime_info
from .codegeneration.C.integrals import (
    _domains_for_integral,
    generate_C_combined_kernels,
)
from .cpp_headers import runtime_abi_header_text
from .form_metadata import FormRuntimeMetadata, build_form_runtime_metadata
from .runtime_api import RunintModule, RuntimeKernelInfo
from .runtime_data import CFFI_DEF

logger = logging.getLogger("runintgen")
root_logger = logging.getLogger()


@dataclass(frozen=True)
class JITIntegralInfo:
    """Metadata for one generated integral in UFCx form order."""

    form_index: int
    integral_type: str
    integral_position: int
    subdomain_id: int
    kernel: RuntimeKernelInfo
    needs_custom_data: bool


@dataclass
class JITFormInfo:
    """Sidecar metadata for one generated runintgen form."""

    ufl_form: ufl.Form
    analysis: RuntimeAnalysisInfo
    form_metadata: FormRuntimeMetadata
    module: RunintModule
    form_name: str
    integral_infos: list[JITIntegralInfo] = field(default_factory=list)


@dataclass
class JITModuleInfo:
    """Sidecar metadata for a compiled runintgen CFFI module."""

    forms: list[JITFormInfo]
    kernels: list[RuntimeKernelInfo]
    module_name: str
    cache_dir: Path
    options: dict[str, Any]


def _compute_option_signature(options: dict[str, Any]) -> str:
    """Return an option signature compatible with FFCx-style JIT names."""
    return str(sorted(options.items()))


def _compilation_signature(
    cffi_extra_compile_args: list[str],
    cffi_debug: bool,
) -> str:
    """Return a platform/Python/compiler signature for CFFI modules."""
    if sys.platform.startswith("win32"):
        return (
            str(cffi_extra_compile_args)
            + str(cffi_debug)
            + str(sysconfig.get_config_var("EXT_SUFFIX"))
        )
    return (
        str(cffi_extra_compile_args)
        + str(cffi_debug)
        + str(sysconfig.get_config_var("CFLAGS"))
        + str(sysconfig.get_config_var("SOABI"))
    )


def _runtime_signature() -> str:
    """Return a signature for runintgen codegen inputs."""
    digest = hashlib.sha1()
    digest.update(runtime_abi_header_text().encode("utf-8"))
    digest.update(b"runintgen-combined-jit-v1")
    return digest.hexdigest()


def _jit_libraries(extra_libraries: list[str] | None) -> list[str]:
    """Return C libraries for the JIT build."""
    if extra_libraries is None:
        return list(_ffcx_libraries)
    return list(_ffcx_libraries) + list(extra_libraries)


def _load_objects(
    cache_dir: Path,
    module_name: str,
    object_names: list[str],
) -> tuple[list[Any], Any]:
    """Load compiled CFFI module objects by name."""
    finder = importlib.machinery.FileFinder(
        str(cache_dir),
        (
            importlib.machinery.ExtensionFileLoader,
            importlib.machinery.EXTENSION_SUFFIXES,
        ),
    )
    finder.invalidate_caches()
    spec = finder.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError("Unable to find JIT module.")

    compiled_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled_module)
    objects = [getattr(compiled_module.lib, name) for name in object_names]
    return objects, compiled_module


def _form_integral_segments(names: list[str]) -> list[tuple[int, int]]:
    """Return contiguous form-integral name segments."""
    segments = []
    pos = 0
    while pos < len(names):
        end = pos + 1
        while end < len(names) and names[end] == names[pos]:
            end += 1
        segments.append((pos, end))
        pos = end
    return segments


def _combined_form_ir(analysis: RuntimeAnalysisInfo, kernels: list[RuntimeKernelInfo]):
    """Return a standard FFCx FormIR pointing at combined kernel names."""
    form_ir = analysis.standard_ir.forms[0]
    runtime_base_names = {
        (kernel.integral_type, kernel.ir_index): kernel.base_name
        for kernel in kernels
        if kernel.mode != "standard" and kernel.base_name is not None
    }
    runtime_domains = {
        (kernel.integral_type, kernel.ir_index): _domains_for_integral(
            _integrals_by_key(analysis.ir)[(kernel.integral_type, kernel.ir_index)]
        )
        for kernel in kernels
        if kernel.mode != "standard"
    }

    integral_names = {key: list(value) for key, value in form_ir.integral_names.items()}
    integral_domains = {
        key: list(value) for key, value in form_ir.integral_domains.items()
    }
    for integral_type, names in integral_names.items():
        for ir_index, (start, end) in enumerate(_form_integral_segments(names)):
            key = (integral_type, ir_index)
            base_name = runtime_base_names.get(key)
            if base_name is None:
                continue
            for pos in range(start, end):
                integral_names[integral_type][pos] = base_name
                integral_domains[integral_type][pos] = runtime_domains[key]

    return form_ir._replace(
        integral_names=integral_names,
        integral_domains=integral_domains,
    )


def _integrals_by_key(ir: Any) -> dict[tuple[str, int], Any]:
    """Return FFCx integral IR objects keyed by type-local index."""
    integrals: dict[tuple[str, int], Any] = {}
    ir_type_counts: dict[str, int] = {}
    for integral_ir in ir.integrals:
        integral_type = integral_ir.expression.integral_type
        index = ir_type_counts.get(integral_type, 0)
        ir_type_counts[integral_type] = index + 1
        integrals[(integral_type, index)] = integral_ir
    return integrals


def _form_integral_infos(
    *,
    form_index: int,
    form_ir: Any,
    kernels: list[RuntimeKernelInfo],
) -> list[JITIntegralInfo]:
    """Return generated-integral metadata in UFCx form order."""
    kernels_by_key_domain = {
        (kernel.integral_type, kernel.ir_index, kernel.domain): kernel
        for kernel in kernels
    }
    infos: list[JITIntegralInfo] = []
    integral_types = ("cell", "exterior_facet", "interior_facet", "vertex", "ridge")
    for integral_type in integral_types:
        names = form_ir.integral_names[integral_type]
        ids = form_ir.subdomain_ids[integral_type]
        domains = form_ir.integral_domains[integral_type]
        for ir_index, (start, end) in enumerate(_form_integral_segments(names)):
            for pos in range(start, end):
                for domain in domains[pos]:
                    kernel = kernels_by_key_domain[
                        (integral_type, ir_index, domain.name)
                    ]
                    infos.append(
                        JITIntegralInfo(
                            form_index=form_index,
                            integral_type=integral_type,
                            integral_position=len(infos),
                            subdomain_id=int(ids[pos]),
                            kernel=kernel,
                            needs_custom_data=kernel.mode != "standard",
                        )
                    )
    return infos


def _compile_form_source(
    form: ufl.Form,
    *,
    form_index: int,
    module_name: str,
    form_name: str,
    options: dict[str, Any],
) -> tuple[str, str, JITFormInfo]:
    """Generate declaration/source code and metadata for one UFL form."""
    analysis = build_runtime_info(form, options, prefix=module_name)
    form_metadata = build_form_runtime_metadata(analysis)
    kernels = generate_C_combined_kernels(analysis, options, form_metadata)
    combined_form_ir = _combined_form_ir(analysis, kernels)
    combined_form_ir = combined_form_ir._replace(
        name=form_name,
        name_from_uflfile=f"form_{module_name}_{form_index}",
    )
    form_decl, form_impl = ffcx_form.generator(combined_form_ir, options)

    providers = [
        group.quadrature_provider
        for group in analysis.groups
        if group.quadrature_provider is not None
    ]
    unique_provider_ids = {id(provider) for provider in providers}
    quadrature_provider = providers[0] if len(unique_provider_ids) == 1 else None
    module = RunintModule(
        kernels=kernels,
        meta=analysis.meta,
        form_metadata=form_metadata,
        quadrature_provider=quadrature_provider,
    )
    form_info = JITFormInfo(
        ufl_form=form,
        analysis=analysis,
        form_metadata=form_metadata,
        module=module,
        form_name=combined_form_ir.name,
        integral_infos=_form_integral_infos(
            form_index=form_index,
            form_ir=combined_form_ir,
            kernels=kernels,
        ),
    )

    declarations = "\n".join(kernel.c_declaration for kernel in kernels)
    implementations = "\n\n".join(kernel.c_definition for kernel in kernels)
    return (
        "\n".join([declarations, form_decl]),
        "\n\n".join([implementations, form_impl]),
        form_info,
    )


def _generate_code(
    forms: list[ufl.Form],
    module_name: str,
    options: dict[str, Any],
) -> tuple[str, str, JITModuleInfo]:
    """Generate full CFFI declaration and source strings."""
    file_pre, file_post = ffcx_file.generator(options)
    decl_parts = [
        UFC_HEADER_DECL.format(np.dtype(options["scalar_type"]).name),
        UFC_INTEGRAL_DECL,
        UFC_FORM_DECL,
        CFFI_DEF,
    ]
    impl_parts = [
        file_pre[1],
        CFFI_DEF,
    ]

    form_infos = []
    kernels = []
    for form_index, form in enumerate(forms):
        form_decl, form_impl, form_info = _compile_form_source(
            form,
            form_index=form_index,
            module_name=module_name,
            form_name=ffcx.naming.form_name(form, form_index, module_name),
            options=options,
        )
        decl_parts.append(form_decl)
        impl_parts.append(form_impl)
        form_infos.append(form_info)
        kernels.extend(form_info.module.kernels)

    impl_parts.append(file_post[1])
    sidecar = JITModuleInfo(
        forms=form_infos,
        kernels=kernels,
        module_name=module_name,
        cache_dir=Path(),
        options=dict(options),
    )
    return "\n".join(decl_parts), "\n\n".join(impl_parts), sidecar


def _compile_objects(
    decl: str,
    impl: str,
    *,
    module_name: str,
    cache_dir: Path,
    cffi_extra_compile_args: list[str],
    cffi_verbose: bool,
    cffi_debug: bool,
    cffi_libraries: list[str] | None,
) -> None:
    """Compile generated C source with CFFI."""
    if sys.platform.startswith("win32"):
        cffi_base_compile_args = ["-std:c17"]
    else:
        cffi_base_compile_args = ["-std=c17"]
    cffi_final_compile_args = cffi_base_compile_args + cffi_extra_compile_args

    ffibuilder = cffi.FFI()
    ffibuilder.set_source(
        module_name,
        impl,
        include_dirs=[ffcx.codegeneration.get_include_path()],
        extra_compile_args=cffi_final_compile_args,
        libraries=_jit_libraries(cffi_libraries),
    )
    ffibuilder.cdef(decl)

    cache_dir.mkdir(exist_ok=True, parents=True)
    c_filename = cache_dir.joinpath(module_name + ".c")
    ready_name = c_filename.with_suffix(".c.cached")

    t0 = time.time()
    out = io.StringIO()
    old_handlers = root_logger.handlers.copy()
    root_logger.handlers = [logging.StreamHandler(out)]
    try:
        with redirect_stdout(out):
            ffibuilder.compile(tmpdir=cache_dir, verbose=True, debug=cffi_debug)
    finally:
        root_logger.handlers = old_handlers

    build_log = out.getvalue()
    if cffi_verbose:
        print(build_log)

    logger.info("runintgen JIT C compiler finished in %.4f", time.time() - t0)
    with open(ready_name, "x") as ready_file:
        ready_file.write(build_log)


def _attach_sidecar(module: Any, sidecar: JITModuleInfo, cache_dir: Path) -> None:
    """Attach runintgen metadata to a compiled CFFI module."""
    sidecar.cache_dir = cache_dir
    module._runintgen_jit = sidecar
    for form_info in sidecar.forms:
        try:
            form_info.ufcx_form = getattr(module.lib, form_info.form_name)
        except Exception:
            pass


def compile_forms(
    forms: list[ufl.Form],
    options: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    timeout: int = 10,
    cffi_extra_compile_args: list[str] | None = None,
    cffi_verbose: bool = False,
    cffi_debug: bool = False,
    cffi_libraries: list[str] | None = None,
    visualise: bool = False,
) -> tuple[list[Any], Any, tuple[str | None, str | None]]:
    """Compile UFL forms into runintgen/FFCx UFCx form objects."""
    if visualise:
        raise NotImplementedError("runintgen JIT visualisation is not implemented.")

    cffi_extra_compile_args = list(cffi_extra_compile_args or [])
    p = ffcx.options.get_options(options or {})
    p["sum_factorization"] = False

    if np.issubdtype(np.dtype(p["scalar_type"]), np.complexfloating):
        raise NotImplementedError("runintgen.dolfinx currently supports float64 only.")

    dtype = np.dtype(p["scalar_type"])
    if dtype != np.dtype(np.float64):
        raise NotImplementedError("runintgen.dolfinx currently supports float64 only.")

    signature_tag = (
        _compute_option_signature(p)
        + _compilation_signature(cffi_extra_compile_args, cffi_debug)
        + _runtime_signature()
    )
    module_name = "librunintgen_forms_" + ffcx.naming.compute_signature(
        forms,
        signature_tag,
    )
    form_names = [
        ffcx.naming.form_name(form, i, module_name) for i, form in enumerate(forms)
    ]

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        obj, mod = get_cached_module(module_name, form_names, cache_dir, timeout)
        if obj is not None:
            _, _, sidecar = _generate_code(forms, module_name, p)
            _attach_sidecar(mod, sidecar, cache_dir)
            return obj, mod, (None, None)
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="runintgen_jit_"))

    try:
        decl, impl, sidecar = _generate_code(forms, module_name, p)
        _compile_objects(
            decl,
            impl,
            module_name=module_name,
            cache_dir=cache_dir,
            cffi_extra_compile_args=cffi_extra_compile_args,
            cffi_verbose=cffi_verbose,
            cffi_debug=cffi_debug,
            cffi_libraries=cffi_libraries,
        )
    except Exception:
        try:
            c_filename = cache_dir.joinpath(module_name + ".c")
            os.replace(c_filename, c_filename.with_suffix(".c.failed"))
        except Exception:
            pass
        raise

    obj, module = _load_objects(cache_dir, module_name, form_names)
    _attach_sidecar(module, sidecar, cache_dir)
    return obj, module, (decl, impl)


__all__ = [
    "JITFormInfo",
    "JITIntegralInfo",
    "JITModuleInfo",
    "compile_forms",
]
