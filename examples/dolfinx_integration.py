"""Example: Using runintgen with DOLFINx assembler.

This example demonstrates how to use runtime-generated kernels with the
standard DOLFINx assembly infrastructure, including:

1. Single-config mode: All cells use the same quadrature
2. Multi-config mode: Different cells use different quadrature rules
3. Integration with DOLFINx Form and assembler

Requirements:
- DOLFINx (with updated void* custom_data support)
- CFFI
- basix
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import cffi
import numpy as np

# Import DOLFINx components
try:
    from mpi4py import MPI

    import dolfinx
    from dolfinx import cpp as _cpp
    from dolfinx import fem
    from dolfinx.fem import Form, IntegralType, functionspace
    from dolfinx.mesh import create_unit_square

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False
    print("DOLFINx not available - example for reference only")

import basix
import ffcx
import ufl
from basix.ufl import element
from ufl import grad, inner

from runintgen import (
    CFFI_DEF,
    QuadratureConfig,
    RuntimeDataBuilder,
    build_runtime_element_mapping,
    tabulate_element,
)
from runintgen.codegeneration.C.integrals_template import (
    factory_runtime_kernel,
    runintgen_data_struct,
)
from runintgen.codegeneration.runtime_integrals import RuntimeIntegralGenerator

if TYPE_CHECKING:
    pass


def generate_runtime_kernel():
    """Generate a runtime Laplacian kernel using runintgen."""
    # Create UFL form
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V_el = element("Lagrange", "triangle", 1)
    V = ufl.FunctionSpace(mesh, V_el)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    form = inner(grad(u), grad(v)) * dx

    # Compile with FFCX to get IR
    from ffcx import compiler

    analysis = compiler.analyze_ufl_objects([form], np.float64)
    options = ffcx.get_options()
    ir = compiler.compute_ir(analysis, {0: "a"}, "test", options, visualise=False)

    # Get the integral IR
    integral_ir = ir.integrals[0]

    # Build element mapping
    element_mapping = build_runtime_element_mapping(integral_ir)

    # Generate the kernel body
    rig = RuntimeIntegralGenerator(ir, None)
    kernel_body = rig.generate_runtime(integral_ir, element_mapping)

    return kernel_body, element_mapping


def compile_runtime_kernel(kernel_body: str):
    """JIT compile a runtime kernel with CFFI."""
    # Full C code
    c_code = runintgen_data_struct + factory_runtime_kernel.format(
        factory_name="runtime_laplacian",
        scalar="double",
        geom="double",
        body=kernel_body,
    )

    ffibuilder = cffi.FFI()
    ffibuilder.cdef(CFFI_DEF)
    ffibuilder.cdef(
        """
    void tabulate_tensor_runtime_laplacian(
        double* A,
        const double* w,
        const double* c,
        const double* coordinate_dofs,
        const int* entity_local_index,
        const uint8_t* quadrature_permutation,
        void* custom_data);
    """
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ffibuilder.set_source(
            "runtime_kernel_module",
            c_code,
            extra_compile_args=["-std=c17"],
        )
        ffibuilder.compile(tmpdir=tmpdir, verbose=False)

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
        spec = finder.find_spec("runtime_kernel_module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.lib.tabulate_tensor_runtime_laplacian, module.ffi, module


def example_single_config():
    """Example: All cells use the same quadrature rule."""
    print("=" * 60)
    print("Example: Single-config mode (all cells same quadrature)")
    print("=" * 60)

    # Generate and compile the kernel
    kernel_body, element_mapping = generate_runtime_kernel()
    kernel, ffi, module = compile_runtime_kernel(kernel_body)

    # Create quadrature rule
    qpts, qwts = basix.make_quadrature(basix.CellType.triangle, 2)
    nq = len(qwts)

    # Create P1 element and tabulate
    P1 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)
    tables = P1.tabulate(1, qpts)  # [nderivs, nq, ndofs, ncomps]

    # Build runtime data using the new API
    builder = RuntimeDataBuilder(ffi)

    # Create a quadrature config
    config = QuadratureConfig(
        points=qpts,
        weights=qwts,
    )
    # Add element tables (P1 scalar for both arguments and Jacobian)
    config.elements.append(tabulate_element(P1, qpts, max_deriv_order=1))
    config.elements.append(tabulate_element(P1, qpts, max_deriv_order=1))

    builder.add_config(config)
    data = builder.build()  # Single-config mode

    # Test on a unit triangle
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    coords_flat = np.ascontiguousarray(coords.flatten())

    A = np.zeros((3, 3), dtype=np.float64)
    w = np.zeros(0, dtype=np.float64)
    c = np.zeros(0, dtype=np.float64)
    entity = np.zeros(1, dtype=np.int32)
    perm = np.zeros(1, dtype=np.uint8)

    kernel(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coords_flat.ctypes.data),
        ffi.cast("int*", entity.ctypes.data),
        ffi.cast("uint8_t*", perm.ctypes.data),
        data,
    )

    print("Element matrix A:")
    print(A)
    print()

    # Expected result for unit right triangle Laplacian
    A_expected = np.array([[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]])
    print("Expected:")
    print(A_expected)
    print()
    print(f"Match: {np.allclose(A, A_expected, atol=1e-14)}")


def example_multi_config():
    """Example: Different cells use different quadrature rules."""
    print()
    print("=" * 60)
    print("Example: Multi-config mode (per-cell quadrature)")
    print("=" * 60)

    # Generate and compile the kernel
    kernel_body, element_mapping = generate_runtime_kernel()
    kernel, ffi, module = compile_runtime_kernel(kernel_body)

    # Create two different quadrature rules
    qpts_low, qwts_low = basix.make_quadrature(basix.CellType.triangle, 1)
    qpts_high, qwts_high = basix.make_quadrature(basix.CellType.triangle, 4)

    P1 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

    # Build runtime data with multiple configs
    builder = RuntimeDataBuilder(ffi)

    # Config 0: Low-order quadrature
    config_low = QuadratureConfig(points=qpts_low, weights=qwts_low)
    config_low.elements.append(tabulate_element(P1, qpts_low, max_deriv_order=1))
    config_low.elements.append(tabulate_element(P1, qpts_low, max_deriv_order=1))
    builder.add_config(config_low)

    # Config 1: High-order quadrature
    config_high = QuadratureConfig(points=qpts_high, weights=qwts_high)
    config_high.elements.append(tabulate_element(P1, qpts_high, max_deriv_order=1))
    config_high.elements.append(tabulate_element(P1, qpts_high, max_deriv_order=1))
    builder.add_config(config_high)

    # Simulate 5 cells: cells 0,1,3 use low-order, cells 2,4 use high-order
    cell_config_map = np.array([0, 0, 1, 0, 1], dtype=np.int32)
    data = builder.build(cell_config_map=cell_config_map)

    # Test on unit triangle with different cells
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    coords_flat = np.ascontiguousarray(coords.flatten())

    print(f"Low-order quadrature: {len(qwts_low)} points")
    print(f"High-order quadrature: {len(qwts_high)} points")
    print()

    for cell_idx in [0, 2]:  # Test cell 0 (low) and cell 2 (high)
        A = np.zeros((3, 3), dtype=np.float64)
        w = np.zeros(0, dtype=np.float64)
        c = np.zeros(0, dtype=np.float64)
        entity = np.array([cell_idx], dtype=np.int32)  # Cell index
        perm = np.zeros(1, dtype=np.uint8)

        kernel(
            ffi.cast("double*", A.ctypes.data),
            ffi.cast("double*", w.ctypes.data),
            ffi.cast("double*", c.ctypes.data),
            ffi.cast("double*", coords_flat.ctypes.data),
            ffi.cast("int*", entity.ctypes.data),
            ffi.cast("uint8_t*", perm.ctypes.data),
            data,
        )

        config_type = "low-order" if cell_config_map[cell_idx] == 0 else "high-order"
        print(f"Cell {cell_idx} ({config_type} quadrature):")
        print(A)
        print()


def example_dolfinx_integration():
    """Example: Integration with DOLFINx Form and assembler."""
    if not HAS_DOLFINX:
        print()
        print("=" * 60)
        print("Example: DOLFINx integration (skipped - DOLFINx not available)")
        print("=" * 60)
        return

    print()
    print("=" * 60)
    print("Example: DOLFINx Form integration")
    print("=" * 60)

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=np.float64)
    V = functionspace(mesh, ("Lagrange", 1))

    # Generate and compile the runtime kernel
    kernel_body, element_mapping = generate_runtime_kernel()
    kernel, ffi, module = compile_runtime_kernel(kernel_body)

    # Create quadrature config
    qpts, qwts = basix.make_quadrature(basix.CellType.triangle, 2)
    P1 = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

    builder = RuntimeDataBuilder(ffi)
    config = QuadratureConfig(points=qpts, weights=qwts)
    config.elements.append(tabulate_element(P1, qpts, max_deriv_order=1))
    config.elements.append(tabulate_element(P1, qpts, max_deriv_order=1))
    builder.add_config(config)
    runtime_data = builder.build()

    # Get cells to integrate over
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)

    # Get the kernel function pointer address
    kernel_ptr = int(ffi.cast("intptr_t", kernel))

    # Create Form with custom kernel
    active_coeffs = np.array([], dtype=np.int8)
    integrals = {IntegralType.cell: [(0, kernel_ptr, cells, active_coeffs)]}

    form = Form(
        _cpp.fem.Form_float64(
            [V._cpp_object, V._cpp_object],
            integrals,
            [],
            [],
            False,
            [],
            mesh=mesh._cpp_object,
        )
    )

    # Set custom_data on the form using the new DOLFINx API
    # This passes the runtime data pointer directly to the kernel
    runtime_data_ptr = int(ffi.cast("intptr_t", runtime_data))
    form._cpp_object.set_custom_data(IntegralType.cell, 0, 0, runtime_data_ptr)

    # Assemble the matrix
    A = fem.assemble_matrix(form)
    A.scatter_reverse()

    print(f"Assembled matrix norm: {np.sqrt(A.squared_norm()):.6f}")

    # Verify against standard FFCX assembly
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_ufl = inner(grad(u), grad(v)) * ufl.dx
    a_ffcx = fem.form(a_ufl)
    A_ffcx = fem.assemble_matrix(a_ffcx)
    A_ffcx.scatter_reverse()
    ffcx_norm = np.sqrt(A_ffcx.squared_norm())

    print(f"Standard FFCX matrix norm: {ffcx_norm:.6f}")
    print(f"Match: {np.isclose(np.sqrt(A.squared_norm()), ffcx_norm)}")


if __name__ == "__main__":
    example_single_config()
    example_multi_config()
    example_dolfinx_integration()
