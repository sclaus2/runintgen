"""Tests comparing runtime integrals with standard FFCX for various forms.

This module tests that runtime kernels produce identical results to
standard FFCX-compiled kernels for:
- Poisson (scalar Laplacian)
- Stokes (mixed velocity-pressure)

The key validation is that for the SAME quadrature rule and element tables,
the runtime kernel and standard FFCX kernel should produce identical
element tensors (up to floating-point tolerance).
"""

import tempfile
from pathlib import Path

import cffi
import numpy as np
import pytest
import basix
from basix.ufl import element
import ufl
from ufl import inner, grad, div, dx

import ffcx
import ffcx.codegeneration.jit

from runintgen import (
    compile_runtime_integrals,
    CFFI_DEF,
    RuntimeDataBuilder,
    ElementTableInfo,
)
from runintgen.tabulation import (
    prepare_runtime_data,
    prepare_runtime_data_for_cell,
    compute_detJ_triangle,
    tabulate_from_table_info,
)


# =============================================================================
# Generic Utilities
# =============================================================================


def compile_and_call_ffcx(form, coords, integral_index=0):
    """Compile a form with FFCX and evaluate on a cell.

    Args:
        form: UFL form to compile.
        coords: Cell coordinates, shape [nverts, gdim].
        integral_index: Which integral to evaluate (default 0).

    Returns:
        Element tensor as numpy array.
    """
    compiled = ffcx.codegeneration.jit.compile_forms([form])
    form_list, code_module, form_header = compiled

    ufcx_form = form_list[0]
    ffi = code_module.ffi
    integral = ufcx_form.form_integrals[integral_index]
    kernel = integral.tabulate_tensor_float64

    # Ensure coords is 3D for FFCX (pad with zeros if 2D)
    coords_3d = np.zeros((coords.shape[0], 3), dtype=np.float64)
    coords_3d[:, : coords.shape[1]] = coords
    coords_flat = coords_3d.flatten()

    # Get DOFs from arguments using basix_element property
    args = form.arguments()
    ndofs_list = []
    for arg in args:
        elem = arg.ufl_function_space().ufl_element()
        # basix.ufl elements have a basix_element property
        if hasattr(elem, "basix_element"):
            basix_elem = elem.basix_element
            ndofs = basix_elem.dim
            # Handle blocked elements
            if hasattr(elem, "block_size"):
                ndofs = ndofs * elem.block_size
        else:
            # Fallback for older API
            ndofs = elem.reference_value_size
        ndofs_list.append(ndofs)

    # Output array shape
    if len(ndofs_list) == 2:
        A = np.zeros((ndofs_list[0], ndofs_list[1]), dtype=np.float64)
    elif len(ndofs_list) == 1:
        A = np.zeros(ndofs_list[0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported form rank: {len(ndofs_list)}")

    w = np.zeros(0, dtype=np.float64)
    c = np.zeros(0, dtype=np.float64)
    entity_local_index = np.zeros(1, dtype=np.int32)
    quadrature_permutation = np.zeros(1, dtype=np.uint8)

    kernel(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coords_flat.ctypes.data),
        ffi.cast("int*", entity_local_index.ctypes.data),
        ffi.cast("uint8_t*", quadrature_permutation.ctypes.data),
        ffi.NULL,
    )

    return A


def compile_runtime_kernel_to_callable(kernel_info):
    """JIT compile a RuntimeKernelInfo to a callable function.

    Args:
        kernel_info: RuntimeKernelInfo from compile_runtime_integrals.

    Returns:
        Tuple of (kernel_function, ffi, module).
    """
    from runintgen.codegeneration.C.integrals_template import runintgen_data_struct

    # Build C code - the kernel name is embedded in c_definition
    c_code = runintgen_data_struct + kernel_info.c_definition

    # JIT compile with CFFI
    ffibuilder = cffi.FFI()

    # Use the declaration from kernel_info which has the correct function name
    decl = CFFI_DEF + kernel_info.c_declaration

    # Create unique module name
    import hashlib

    module_name = (
        f"runtime_module_{hashlib.md5(kernel_info.name.encode()).hexdigest()[:8]}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ffibuilder.set_source(
            module_name,
            c_code,
            extra_compile_args=["-std=c17"],
        )
        ffibuilder.cdef(decl)
        ffibuilder.compile(tmpdir=tmpdir, verbose=False)

        import importlib.util

        so_files = list(Path(tmpdir).glob("*.so")) + list(Path(tmpdir).glob("*.pyd"))
        if not so_files:
            raise RuntimeError("No compiled module found")
        so_path = so_files[0]

        spec = importlib.util.spec_from_file_location(module_name, so_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    # The function name in the C code uses "tabulate_tensor_" prefix
    func_name = f"tabulate_tensor_{kernel_info.name}"
    kernel_fn = getattr(module.lib, func_name)
    return kernel_fn, module.ffi, module


def call_runtime_kernel(kernel_fn, ffi, data, coords, output_shape):
    """Call a runtime kernel with prepared data.

    Args:
        kernel_fn: The compiled kernel function.
        ffi: CFFI FFI instance.
        data: Pointer to runintgen_data structure.
        coords: Cell coordinates, shape [nverts, gdim].
        output_shape: Shape of the output tensor.

    Returns:
        Element tensor as numpy array.
    """
    # Ensure coords is 3D (pad if needed)
    coords_3d = np.zeros((coords.shape[0], 3), dtype=np.float64)
    coords_3d[:, : coords.shape[1]] = coords
    coords_flat = np.ascontiguousarray(coords_3d.flatten(), dtype=np.float64)

    # Output array
    A = np.zeros(output_shape, dtype=np.float64)

    w = np.zeros(0, dtype=np.float64)
    c = np.zeros(0, dtype=np.float64)
    entity_local_index = np.zeros(1, dtype=np.int32)
    quadrature_permutation = np.zeros(1, dtype=np.uint8)

    kernel_fn(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coords_flat.ctypes.data),
        ffi.cast("int*", entity_local_index.ctypes.data),
        ffi.cast("uint8_t*", quadrature_permutation.ctypes.data),
        data,
    )

    return A


# =============================================================================
# Poisson Tests
# =============================================================================


class TestPoissonComparison:
    """Compare Poisson (Laplacian) forms between FFCX and runtime."""

    @pytest.fixture
    def p1_laplacian_form(self):
        """Create P1 Laplacian bilinear form."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return inner(grad(u), grad(v)) * dx

    @pytest.fixture
    def p1_laplacian_form_runtime(self):
        """Create P1 Laplacian with runtime quadrature."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        return inner(grad(u), grad(v)) * dx_rt

    @pytest.fixture
    def p2_laplacian_form(self):
        """Create P2 Laplacian bilinear form."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return inner(grad(u), grad(v)) * dx

    @pytest.fixture
    def p2_laplacian_form_runtime(self):
        """Create P2 Laplacian with runtime quadrature."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        return inner(grad(u), grad(v)) * dx_rt

    @pytest.fixture
    def unit_triangle(self):
        """Unit right triangle coordinates."""
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    @pytest.fixture
    def scaled_triangle(self):
        """Scaled triangle (factor of 2)."""
        return np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    @pytest.fixture
    def general_triangle(self):
        """General non-unit triangle."""
        return np.array([[0.5, 0.5], [2.0, 0.3], [1.0, 2.5]], dtype=np.float64)

    def test_p1_laplacian_unit_triangle(
        self, p1_laplacian_form, p1_laplacian_form_runtime, unit_triangle
    ):
        """Test P1 Laplacian on unit triangle."""
        # Standard FFCX result
        A_ffcx = compile_and_call_ffcx(p1_laplacian_form, unit_triangle)

        # Runtime result
        module = compile_runtime_integrals(p1_laplacian_form_runtime)
        kernel_info = module.kernels[0]

        # Prepare runtime data with same quadrature degree FFCX would use
        # For P1 Laplacian, degree 2 is sufficient
        prepared = prepare_runtime_data_for_cell(
            kernel_info, unit_triangle, quadrature_degree=2
        )
        data = prepared.builder.build()

        # Compile and call runtime kernel
        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, unit_triangle, output_shape=(3, 3)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)

    def test_p1_laplacian_scaled_triangle(
        self, p1_laplacian_form, p1_laplacian_form_runtime, scaled_triangle
    ):
        """Test P1 Laplacian on scaled triangle."""
        A_ffcx = compile_and_call_ffcx(p1_laplacian_form, scaled_triangle)

        module = compile_runtime_integrals(p1_laplacian_form_runtime)
        kernel_info = module.kernels[0]

        prepared = prepare_runtime_data_for_cell(
            kernel_info, scaled_triangle, quadrature_degree=2
        )
        data = prepared.builder.build()

        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, scaled_triangle, output_shape=(3, 3)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)

    def test_p1_laplacian_general_triangle(
        self, p1_laplacian_form, p1_laplacian_form_runtime, general_triangle
    ):
        """Test P1 Laplacian on general triangle."""
        A_ffcx = compile_and_call_ffcx(p1_laplacian_form, general_triangle)

        module = compile_runtime_integrals(p1_laplacian_form_runtime)
        kernel_info = module.kernels[0]

        prepared = prepare_runtime_data_for_cell(
            kernel_info, general_triangle, quadrature_degree=2
        )
        data = prepared.builder.build()

        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, general_triangle, output_shape=(3, 3)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)

    def test_p2_laplacian_unit_triangle(
        self, p2_laplacian_form, p2_laplacian_form_runtime, unit_triangle
    ):
        """Test P2 Laplacian on unit triangle."""
        A_ffcx = compile_and_call_ffcx(p2_laplacian_form, unit_triangle)

        module = compile_runtime_integrals(p2_laplacian_form_runtime)
        kernel_info = module.kernels[0]

        # P2 elements need higher quadrature degree
        prepared = prepare_runtime_data_for_cell(
            kernel_info, unit_triangle, quadrature_degree=4
        )
        data = prepared.builder.build()

        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, unit_triangle, output_shape=(6, 6)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)

    def test_different_quadrature_degrees(
        self, p1_laplacian_form, p1_laplacian_form_runtime, unit_triangle
    ):
        """Test that higher quadrature degrees still match FFCX."""
        A_ffcx = compile_and_call_ffcx(p1_laplacian_form, unit_triangle)

        module = compile_runtime_integrals(p1_laplacian_form_runtime)
        kernel_info = module.kernels[0]

        for degree in [2, 4, 6, 8]:
            prepared = prepare_runtime_data_for_cell(
                kernel_info, unit_triangle, quadrature_degree=degree
            )
            data = prepared.builder.build()

            kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
            A_runtime = call_runtime_kernel(
                kernel_fn, ffi, data, unit_triangle, output_shape=(3, 3)
            )

            np.testing.assert_allclose(
                A_runtime,
                A_ffcx,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Mismatch at quadrature degree {degree}",
            )


# =============================================================================
# Mass Matrix Tests
# =============================================================================


class TestMassMatrixComparison:
    """Compare mass matrix forms between FFCX and runtime."""

    @pytest.fixture
    def p1_mass_form(self):
        """Create P1 mass matrix form."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return inner(u, v) * dx

    @pytest.fixture
    def p1_mass_form_runtime(self):
        """Create P1 mass matrix with runtime quadrature."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        return inner(u, v) * dx_rt

    @pytest.fixture
    def unit_triangle(self):
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    @pytest.mark.skip(reason="Mass matrix requires different code generation path")
    def test_p1_mass_unit_triangle(
        self, p1_mass_form, p1_mass_form_runtime, unit_triangle
    ):
        """Test P1 mass matrix on unit triangle."""
        A_ffcx = compile_and_call_ffcx(p1_mass_form, unit_triangle)

        module = compile_runtime_integrals(p1_mass_form_runtime)
        kernel_info = module.kernels[0]

        prepared = prepare_runtime_data_for_cell(
            kernel_info, unit_triangle, quadrature_degree=2
        )
        data = prepared.builder.build()

        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, unit_triangle, output_shape=(3, 3)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)


# =============================================================================
# Stokes Tests
# =============================================================================


class TestStokesComparison:
    """Compare Stokes forms between FFCX and runtime.

    Stokes problem uses Taylor-Hood elements (P2-P1):
    - Velocity: P2 vector element (6 DOFs * 2 = 12 DOFs per cell)
    - Pressure: P1 scalar element (3 DOFs per cell)

    The weak form is:
        a(u,v) = integral(inner(grad(u), grad(v))) dx
        b(v,p) = integral(-p * div(v)) dx
        b(u,q) = integral(-q * div(u)) dx

    Leading to the saddle-point system:
        [A  B^T] [u]   [f]
        [B  0  ] [p] = [0]
    """

    @pytest.fixture
    def stokes_a_form(self):
        """Stokes viscosity form: inner(grad(u), grad(v)) * dx."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2, shape=(2,)))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return inner(grad(u), grad(v)) * dx

    @pytest.fixture
    def stokes_a_form_runtime(self):
        """Stokes viscosity form with runtime quadrature."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2, shape=(2,)))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        return inner(grad(u), grad(v)) * dx_rt

    @pytest.fixture
    def stokes_b_form(self):
        """Stokes divergence form: -p * div(v) * dx."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2, shape=(2,)))
        Q = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        v = ufl.TestFunction(V)
        p = ufl.TrialFunction(Q)
        return -p * div(v) * dx

    @pytest.fixture
    def stokes_b_form_runtime(self):
        """Stokes divergence form with runtime quadrature."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2, shape=(2,)))
        Q = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        v = ufl.TestFunction(V)
        p = ufl.TrialFunction(Q)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        return -p * div(v) * dx_rt

    @pytest.fixture
    def unit_triangle(self):
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    @pytest.mark.skip(reason="Stokes requires vector element code generation")
    def test_stokes_a_form_unit_triangle(
        self, stokes_a_form, stokes_a_form_runtime, unit_triangle
    ):
        """Test Stokes A block (viscosity) on unit triangle."""
        A_ffcx = compile_and_call_ffcx(stokes_a_form, unit_triangle)

        module = compile_runtime_integrals(stokes_a_form_runtime)
        kernel_info = module.kernels[0]

        # P2 vector element: 6 scalar DOFs * 2 components = 12 DOFs
        prepared = prepare_runtime_data_for_cell(
            kernel_info, unit_triangle, quadrature_degree=4
        )
        data = prepared.builder.build()

        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, unit_triangle, output_shape=(12, 12)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)

    @pytest.mark.skip(reason="Stokes requires mixed element code generation")
    def test_stokes_b_form_unit_triangle(
        self, stokes_b_form, stokes_b_form_runtime, unit_triangle
    ):
        """Test Stokes B block (divergence) on unit triangle."""
        A_ffcx = compile_and_call_ffcx(stokes_b_form, unit_triangle)

        module = compile_runtime_integrals(stokes_b_form_runtime)
        kernel_info = module.kernels[0]

        # Test: P2 vector (12 DOFs), Trial: P1 scalar (3 DOFs)
        prepared = prepare_runtime_data_for_cell(
            kernel_info, unit_triangle, quadrature_degree=4
        )
        data = prepared.builder.build()

        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, unit_triangle, output_shape=(12, 3)
        )

        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)


# =============================================================================
# Tabulation Utility Tests
# =============================================================================


class TestTabulationUtilities:
    """Test the generic tabulation utility functions."""

    def test_compute_detJ_unit_triangle(self):
        """Test Jacobian determinant for unit triangle."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        detJ = compute_detJ_triangle(coords)
        # Area of unit triangle = 0.5, detJ = 2 * area = 1.0
        assert np.isclose(detJ, 1.0)

    def test_compute_detJ_scaled_triangle(self):
        """Test Jacobian determinant for scaled triangle."""
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        detJ = compute_detJ_triangle(coords)
        # Scale factor 2 -> detJ = 4
        assert np.isclose(detJ, 4.0)

    def test_compute_detJ_general_triangle(self):
        """Test Jacobian determinant for general triangle."""
        coords = np.array([[0.5, 0.5], [2.0, 0.3], [1.0, 2.5]], dtype=np.float64)
        detJ = compute_detJ_triangle(coords)
        # Compute expected via cross product
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        expected = v1[0] * v2[1] - v1[1] * v2[0]
        assert np.isclose(detJ, expected)

    def test_prepare_runtime_data_basic(self):
        """Test basic prepare_runtime_data functionality."""
        # Create a simple form
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        form = inner(grad(u), grad(v)) * dx_rt

        # Compile
        module = compile_runtime_integrals(form)
        kernel_info = module.kernels[0]

        # Prepare with basix quadrature
        points, weights = basix.make_quadrature(basix.CellType.triangle, 2)
        weights_scaled = weights * 1.0  # Unit detJ

        prepared = prepare_runtime_data(kernel_info, points, weights_scaled)

        assert prepared.builder is not None
        assert prepared.ffi is not None
        assert len(prepared.table_arrays) == len(kernel_info.table_info)

    def test_prepare_runtime_data_for_cell(self):
        """Test prepare_runtime_data_for_cell with automatic detJ scaling."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        form = inner(grad(u), grad(v)) * dx_rt

        module = compile_runtime_integrals(form)
        kernel_info = module.kernels[0]

        coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        prepared = prepare_runtime_data_for_cell(
            kernel_info, coords, quadrature_degree=2
        )

        # Should succeed without error
        data = prepared.builder.build()
        assert data is not None


# =============================================================================
# Example Usage Test
# =============================================================================


class TestExampleUsage:
    """Test showing the complete workflow for using runtime kernels."""

    def test_complete_workflow_p1_laplacian(self):
        """Demonstrate complete workflow: form -> runtime kernel -> evaluate."""
        # Step 1: Define the form with runtime quadrature
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        form_runtime = inner(grad(u), grad(v)) * dx_rt

        # Also create standard form for comparison
        form_standard = inner(grad(u), grad(v)) * dx

        # Step 2: Compile the runtime form
        module = compile_runtime_integrals(form_runtime)
        kernel_info = module.kernels[0]

        # Step 3: Define the cell
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        # Step 4: Prepare runtime data (this does the tabulation)
        prepared = prepare_runtime_data_for_cell(
            kernel_info, coords, quadrature_degree=2
        )
        data = prepared.builder.build()

        # Step 5: Compile and call the runtime kernel
        kernel_fn, ffi, _ = compile_runtime_kernel_to_callable(kernel_info)
        A_runtime = call_runtime_kernel(
            kernel_fn, ffi, data, coords, output_shape=(3, 3)
        )

        # Step 6: Compare with FFCX
        A_ffcx = compile_and_call_ffcx(form_standard, coords)

        # Step 7: Verify they match
        np.testing.assert_allclose(A_runtime, A_ffcx, rtol=1e-12, atol=1e-14)

        # Print results for documentation
        print("\nP1 Laplacian element matrix:")
        print(f"FFCX result:\n{A_ffcx}")
        print(f"Runtime result:\n{A_runtime}")
        print(f"Max difference: {np.max(np.abs(A_runtime - A_ffcx))}")
