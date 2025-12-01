"""DOLFINx integration utilities for runintgen.

This module provides convenience functions for using runtime integration
kernels with the DOLFINx finite element library.

The main workflow is:
1. Define a UFL form with runtime integrals using:
   dx = ufl.Measure("dx", metadata={"quadrature_rule": "runtime"}, ...)

2. Use compile_form_with_runtime() to compile both FFCX standard integrals
   and runintgen runtime kernels, returning a DOLFINx Form ready for assembly.

3. Provide runtime quadrature data via QuadratureConfig and RuntimeDataBuilder.

Example:
    >>> from dolfinx import mesh, fem
    >>> from mpi4py import MPI
    >>> import ufl
    >>> from runintgen.dolfinx_utils import compile_form_with_runtime
    >>>
    >>> msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    >>> V = fem.functionspace(msh, ("Lagrange", 1))
    >>> u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    >>>
    >>> # Define form with runtime quadrature
    >>> dx_rt = ufl.Measure("dx", domain=msh.ufl_domain(),
    ...                     metadata={"quadrature_rule": "runtime"})
    >>> a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt
    >>>
    >>> # Compile and get form with kernel
    >>> form, runtime_info = compile_form_with_runtime(a, V)
    >>>
    >>> # Set up quadrature data
    >>> set_runtime_quadrature(form, quadrature_configs, ...)
    >>>
    >>> # Assemble
    >>> A = fem.assemble_matrix(form)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cffi
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from dolfinx.fem import Form, FunctionSpace

# Check if DOLFINx is available
try:
    from dolfinx import fem
    from dolfinx.fem import IntegralType

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


@dataclass
class CompiledKernel:
    """A compiled runtime kernel ready for use with DOLFINx.

    Attributes:
        name: The kernel function name.
        integral_type: Type of integral ("cell", "exterior_facet", etc.).
        subdomain_id: The subdomain identifier.
        kernel_ptr: Function pointer (as int) for DOLFINx.
        ffi: The cffi FFI object (needed to create runtime data).
        module: The compiled cffi module (keep alive).
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
    """Information about a form with runtime integrals.

    Attributes:
        kernels: List of compiled kernels.
        ffi: Shared FFI object.
    """

    kernels: list[CompiledKernel] = field(default_factory=list)
    ffi: cffi.FFI | None = None


def compile_runtime_kernels(
    form,
    options: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> RuntimeFormInfo:
    """Compile all runtime integrals in a UFL form.

    This function:
    1. Analyzes the form for runtime integrals (marked with
       metadata={"quadrature_rule": "runtime"})
    2. Generates C code for the runtime kernels
    3. JIT compiles the code to a shared library
    4. Returns compiled kernel information ready for DOLFINx

    Args:
        form: A UFL form containing runtime integrals.
        options: Optional FFCX compilation options.
        cache_dir: Optional directory to cache compiled modules.

    Returns:
        RuntimeFormInfo containing compiled kernels and FFI.

    Example:
        >>> runtime_info = compile_runtime_kernels(a_ufl)
        >>> for k in runtime_info.kernels:
        ...     print(f"Kernel: {k.name}, pointer: {hex(k.kernel_ptr)}")
    """
    from .codegeneration.C.integrals import get_runintgen_data_struct
    from .runtime_api import compile_runtime_integrals
    from .runtime_data import CFFI_DEF

    # Compile with runintgen
    module = compile_runtime_integrals(form, options)

    if not module.kernels:
        return RuntimeFormInfo()

    # Build combined C code
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
        # Extract just the function signature for cffi cdef
        header_parts.append(kernel.c_declaration.strip().rstrip(";") + ";")

    full_c_code = "\n".join(c_parts)
    full_header = "\n".join(header_parts)

    # Compile with CFFI
    ffi = cffi.FFI()
    ffi.cdef(CFFI_DEF)
    ffi.cdef(full_header)

    # Create temporary directory or use cache
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

    # Load the compiled module
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
    compiled_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled_module)

    # Build kernel info
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
    function_spaces: list,
    runtime_info: RuntimeFormInfo,
    mesh,
    cells: npt.NDArray[np.int32] | None = None,
    coefficients: list | None = None,
    constants: list | None = None,
) -> "Form":
    """Create a DOLFINx Form using compiled runtime kernels.

    This creates a Form that uses runtime kernels instead of FFCX-generated
    kernels. The custom_data pointer will need to be set with runtime
    quadrature data before assembly.

    Args:
        function_spaces: List of DOLFINx function spaces [test, trial] or [test].
        runtime_info: Compiled runtime kernel information.
        mesh: The DOLFINx mesh.
        cells: Optional array of cell indices to integrate over.
            If None, all cells are used.
        coefficients: Optional list of coefficients.
        constants: Optional list of constants.

    Returns:
        A DOLFINx Form ready for assembly (after setting custom_data).

    Example:
        >>> form = create_dolfinx_form_with_runtime(
        ...     [V, V], runtime_info, mesh
        ... )
        >>> form._cpp_object.set_custom_data(
        ...     IntegralType.cell, subdomain_id, 0, runtime_data_ptr
        ... )
        >>> A = fem.assemble_matrix(form)
    """
    if not HAS_DOLFINX:
        raise ImportError("DOLFINx is required for this function")

    from dolfinx import cpp as _cpp

    # Get mesh info
    tdim = mesh.topology.dim
    if cells is None:
        num_cells = mesh.topology.index_map(tdim).size_local
        cells = np.arange(num_cells, dtype=np.int32)

    # Build integrals dict for DOLFINx
    integrals: dict[IntegralType, list] = {}

    for kernel in runtime_info.kernels:
        itype = _ufl_to_dolfinx_integral_type(kernel.integral_type)

        if itype not in integrals:
            integrals[itype] = []

        # Empty active coefficients for now (could be extracted from kernel info)
        active_coeffs = np.array([], dtype=np.int8)

        integrals[itype].append(
            (kernel.subdomain_id, kernel.kernel_ptr, cells, active_coeffs)
        )

    # Create the Form
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
        False,  # needs_permutation_data
        [],  # entity_maps
        mesh=mesh._cpp_object,
    )

    return fem.Form(cpp_form)


def _ufl_to_dolfinx_integral_type(ufl_type: str) -> "IntegralType":
    """Convert UFL integral type string to DOLFINx IntegralType."""
    mapping = {
        "cell": IntegralType.cell,
        "exterior_facet": IntegralType.exterior_facet,
        "interior_facet": IntegralType.interior_facet,
        "vertex": IntegralType.vertex,
    }
    return mapping[ufl_type]


def set_runtime_data(
    form: "Form",
    integral_type: "IntegralType",
    subdomain_id: int,
    runtime_data_ptr: int,
    kernel_idx: int = 0,
) -> None:
    """Set the runtime data pointer for a runtime integral.

    Args:
        form: The DOLFINx Form.
        integral_type: The integral type.
        subdomain_id: The subdomain ID.
        runtime_data_ptr: Pointer to the runintgen_data struct (as int).
        kernel_idx: Kernel index (default 0).

    Example:
        >>> data_ptr = int(ffi.cast("intptr_t", rdata))
        >>> set_runtime_data(form, IntegralType.cell, 1, data_ptr)
    """
    if not HAS_DOLFINX:
        raise ImportError("DOLFINx is required for this function")

    form._cpp_object.set_custom_data(
        integral_type, subdomain_id, kernel_idx, runtime_data_ptr
    )
