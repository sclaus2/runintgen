"""Tests comparing runtime integrals with standard FFCX integrals.

These tests verify that the runtime integral approach produces the same
results as standard FFCX-compiled integrals on single cells.

The key difference is that the runtime approach:
1. Uses JIT-compiled C code (just like FFCX)
2. Receives quadrature points and FE tables at runtime
3. Uses the runintgen_data struct to pass runtime data
4. Uses generate_runtime from RuntimeIntegralGenerator
"""

import tempfile

import cffi
import numpy as np
import pytest
import basix
from basix.ufl import element
import ufl
from ufl import inner, grad
import ffcx
import ffcx.codegeneration.jit

from runintgen.codegeneration.C.integrals_template import (
    runintgen_data_struct,
    factory_runtime_kernel,
)
from runintgen.codegeneration.runtime_integrals import RuntimeIntegralGenerator
from runintgen.runtime_tables import build_runtime_element_mapping
from runintgen.runtime_data import CFFI_DEF


def compute_ffcx_standard(form, coordinate_dofs):
    """Compile and evaluate a form using standard FFCX.

    Args:
        form: UFL form to compile.
        coordinate_dofs: numpy array of shape (num_nodes, 3) with coordinates.

    Returns:
        Element tensor as numpy array.
    """
    compiled = ffcx.codegeneration.jit.compile_forms([form])
    form_list, code_module, form_header = compiled

    ufcx_form = form_list[0]
    ffi = code_module.ffi
    integral0 = ufcx_form.form_integrals[0]
    kernel = integral0.tabulate_tensor_float64

    # Flatten coordinate dofs to match FFCX expected format
    coords_flat = coordinate_dofs.flatten().astype(np.float64)

    # Determine output size from form rank
    ndofs = 3  # P1 triangle has 3 DOFs
    A = np.zeros((ndofs, ndofs), dtype=np.float64)

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


# Cache for compiled runtime kernels
_runtime_kernel_cache = {}


def get_runtime_laplacian_kernel_generated():
    """Get JIT-compiled runtime Laplacian kernel using generate_runtime.

    This uses the actual RuntimeIntegralGenerator.generate_runtime() method
    to produce the C code, then JIT compiles it.

    Returns a tuple of (kernel_function, ffi, module) that can be used
    to evaluate the Laplacian on a cell with runtime quadrature data.
    """
    cache_key = "laplacian_p1_2d_generated"
    if cache_key in _runtime_kernel_cache:
        return _runtime_kernel_cache[cache_key]

    # Create the same Laplacian form as the tests use
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

    # Generate the kernel body using our code generation
    rig = RuntimeIntegralGenerator(ir, None)
    kernel_body = rig.generate_runtime(integral_ir, element_mapping)

    # Format the full C code
    c_code = runintgen_data_struct + factory_runtime_kernel.format(
        factory_name="runtime_laplacian",
        scalar="double",
        geom="double",
        body=kernel_body,
    )

    # JIT compile with CFFI
    ffibuilder = cffi.FFI()

    # Declaration for CFFI - using new multi-config structure
    decl = (
        CFFI_DEF
        + """
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
            "runtime_laplacian_module",
            c_code,
            extra_compile_args=["-std=c17"],
        )
        ffibuilder.cdef(decl)
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
        spec = finder.find_spec("runtime_laplacian_module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel = module.lib.tabulate_tensor_runtime_laplacian
        ffi = module.ffi

        _runtime_kernel_cache[cache_key] = (kernel, ffi, module)
        return kernel, ffi, module


def get_runtime_laplacian_kernel():
    """Get JIT-compiled runtime Laplacian kernel (hardcoded version).

    This is a reference implementation with hardcoded C code that matches
    what generate_runtime produces. Used for comparison testing.

    Returns a tuple of (kernel_function, ffi, module) that can be used
    to evaluate the Laplacian on a cell with runtime quadrature data.
    """
    cache_key = "laplacian_p1_2d"
    if cache_key in _runtime_kernel_cache:
        return _runtime_kernel_cache[cache_key]

    # Hardcoded kernel body that matches generate_runtime output
    kernel_body = """
  // Runtime integral - per-element tables with all derivatives

  // Main quadrature loop
  for (int iq = 0; iq < nq; ++iq)
  {
    const double weight = data->weights[iq];

    // Compute Jacobian at quadrature point
    // J[i][j] = sum_k coord_dofs[k*gdim + i] * dphi_k/dX_j
    double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    {
      const runintgen_element* coord_elem = &data->elements[0];
      const int coord_ndofs = 3;
      const double* coord_table = coord_elem->table;

      for (int k = 0; k < coord_ndofs; ++k)
      {
        // d/dX derivatives at deriv_idx=1, d/dY at deriv_idx=2
        const double dphi_dX = coord_table[1 * nq * coord_ndofs + iq * coord_ndofs + k];
        const double dphi_dY = coord_table[2 * nq * coord_ndofs + iq * coord_ndofs + k];
        J[0][0] += coordinate_dofs[k * 3 + 0] * dphi_dX;
        J[1][0] += coordinate_dofs[k * 3 + 1] * dphi_dX;
        J[0][1] += coordinate_dofs[k * 3 + 0] * dphi_dY;
        J[1][1] += coordinate_dofs[k * 3 + 1] * dphi_dY;
      }
    }

    // Compute determinant and inverse Jacobian
    const double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const double inv_detJ = 1.0 / detJ;
    double Jinv[2][2];
    Jinv[0][0] = J[1][1] * inv_detJ;
    Jinv[0][1] = -J[0][1] * inv_detJ;
    Jinv[1][0] = -J[1][0] * inv_detJ;
    Jinv[1][1] = J[0][0] * inv_detJ;

    // Compute contribution to element tensor
    // For Laplacian: A[i,j] += (grad phi_i . grad phi_j) * |detJ| * w
    const runintgen_element* arg_elem = &data->elements[0];
    const int ndofs = 3;
    const double* arg_table = arg_elem->table;

    const double factor = fabs(detJ) * weight;

    for (int i = 0; i < ndofs; ++i)
    {
      // Reference gradients of test function i
      const double dphi_i_dX = arg_table[1 * nq * ndofs + iq * ndofs + i];
      const double dphi_i_dY = arg_table[2 * nq * ndofs + iq * ndofs + i];
      // Physical gradients: grad = Jinv^T * ref_grad
      const double grad_i_x = Jinv[0][0] * dphi_i_dX + Jinv[1][0] * dphi_i_dY;
      const double grad_i_y = Jinv[0][1] * dphi_i_dX + Jinv[1][1] * dphi_i_dY;

      for (int j = 0; j < ndofs; ++j)
      {
        // Reference gradients of trial function j
        const double dphi_j_dX = arg_table[1 * nq * ndofs + iq * ndofs + j];
        const double dphi_j_dY = arg_table[2 * nq * ndofs + iq * ndofs + j];
        const double grad_j_x = Jinv[0][0] * dphi_j_dX + Jinv[1][0] * dphi_j_dY;
        const double grad_j_y = Jinv[0][1] * dphi_j_dX + Jinv[1][1] * dphi_j_dY;

        // Accumulate: inner product of gradients
        A[i * ndofs + j] += (grad_i_x * grad_j_x + grad_i_y * grad_j_y) * factor;
      }
    }
  }  // end quadrature loop
"""

    # Format the full C code
    c_code = runintgen_data_struct + factory_runtime_kernel.format(
        factory_name="runtime_laplacian",
        scalar="double",
        geom="double",
        body=kernel_body,
    )

    # JIT compile with CFFI
    ffibuilder = cffi.FFI()

    # Declaration for CFFI - using new multi-config structure
    decl = (
        CFFI_DEF
        + """
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
            "runtime_laplacian_module",
            c_code,
            extra_compile_args=["-std=c17"],
        )
        ffibuilder.cdef(decl)
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
        spec = finder.find_spec("runtime_laplacian_module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel = module.lib.tabulate_tensor_runtime_laplacian
        ffi = module.ffi

        _runtime_kernel_cache[cache_key] = (kernel, ffi, module)
        return kernel, ffi, module


def compute_runtime_jit(coordinate_dofs, quadrature_degree=2):
    """Compute Laplacian element matrix using JIT-compiled runtime kernel.

    This uses the actual C code path with runtime tabulation data.

    Args:
        coordinate_dofs: numpy array of shape (num_nodes, 3) with coordinates.
        quadrature_degree: Quadrature degree to use.

    Returns:
        Element tensor as numpy array.
    """
    kernel, ffi, module = get_runtime_laplacian_kernel()

    # Get quadrature rule
    qpts, qwts = basix.make_quadrature(basix.CellType.triangle, quadrature_degree)
    nq = len(qwts)

    # Create P1 Lagrange element and tabulate
    P1_element = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

    # Tabulate with first derivatives: shape [nderivs, nq, ndofs, ncomps]
    # For scalar element, ncomps=1, so shape is [3, nq, 3, 1]
    tables = P1_element.tabulate(1, qpts)

    # Flatten for C: [nderivs, nq, ndofs] - take component 0
    # basix returns [nderivs, nq, ndofs, ncomps]
    table_flat = tables[:, :, :, 0].flatten().astype(np.float64)

    # Ensure arrays are contiguous
    qpts_flat = np.ascontiguousarray(qpts.flatten(), dtype=np.float64)
    qwts_flat = np.ascontiguousarray(qwts, dtype=np.float64)
    coords_flat = np.ascontiguousarray(coordinate_dofs.flatten(), dtype=np.float64)
    table_flat = np.ascontiguousarray(table_flat)

    # Create element info struct
    elem = ffi.new("runintgen_element*")
    elem.ndofs = 3
    elem.nderivs = 3  # (0,0), (1,0), (0,1)
    elem.table = ffi.cast("const double*", table_flat.ctypes.data)

    # Create quadrature config (new multi-config structure)
    config = ffi.new("runintgen_quadrature_config*")
    config.nq = nq
    config.points = ffi.cast("const double*", qpts_flat.ctypes.data)
    config.weights = ffi.cast("const double*", qwts_flat.ctypes.data)
    config.nelements = 1
    config.elements = elem

    # Create main runtime data struct (single-config mode)
    data = ffi.new("runintgen_data*")
    data.num_configs = 1
    data.configs = config
    data.active_config = 0  # Single-config mode
    data.cell_config_map = ffi.NULL

    # Output array
    ndofs = 3
    A = np.zeros((ndofs, ndofs), dtype=np.float64)

    # Empty arrays for unused parameters
    w = np.zeros(0, dtype=np.float64)
    c = np.zeros(0, dtype=np.float64)
    entity_local_index = np.zeros(1, dtype=np.int32)
    quadrature_permutation = np.zeros(1, dtype=np.uint8)

    # Call the kernel
    kernel(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coords_flat.ctypes.data),
        ffi.cast("int*", entity_local_index.ctypes.data),
        ffi.cast("uint8_t*", quadrature_permutation.ctypes.data),
        data,
    )

    return A


class TestRuntimeVsStandard:
    """Compare runtime integral results with standard FFCX."""

    @pytest.fixture
    def mesh_and_space(self):
        """Create mesh and function space for P1 Laplacian."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        return mesh, V

    @pytest.fixture
    def laplacian_form(self, mesh_and_space):
        """Create standard Laplacian form."""
        mesh, V = mesh_and_space
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx = ufl.Measure("dx", domain=mesh)
        return inner(grad(u), grad(v)) * dx

    def test_unit_right_triangle(self, laplacian_form):
        """Test on unit right triangle: (0,0), (1,0), (0,1)."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        A_ffcx = compute_ffcx_standard(laplacian_form, coords)
        A_runtime = compute_runtime_jit(coords)

        # Known analytical result for unit right triangle Laplacian
        A_expected = np.array(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 0.5, 0.0],
                [-0.5, 0.0, 0.5],
            ]
        )

        assert np.allclose(A_ffcx, A_expected, atol=1e-14)
        assert np.allclose(A_runtime, A_expected, atol=1e-14)
        assert np.allclose(A_ffcx, A_runtime, atol=1e-14)

    def test_stretched_triangle(self, laplacian_form):
        """Test on stretched triangle: (0,0), (2,0), (0,1)."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        A_ffcx = compute_ffcx_standard(laplacian_form, coords)
        A_runtime = compute_runtime_jit(coords)

        assert np.allclose(A_ffcx, A_runtime, atol=1e-14)

        # Check symmetry
        assert np.allclose(A_ffcx, A_ffcx.T, atol=1e-14)

        # Check row sums (should be zero for Laplacian)
        assert np.allclose(A_ffcx.sum(axis=1), 0.0, atol=1e-14)

    def test_translated_triangle(self, laplacian_form):
        """Test on translated triangle: (1,1), (2,1), (1,2)."""
        coords = np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )

        A_ffcx = compute_ffcx_standard(laplacian_form, coords)
        A_runtime = compute_runtime_jit(coords)

        assert np.allclose(A_ffcx, A_runtime, atol=1e-14)

        # Should be same as unit right triangle (translation invariant)
        coords_unit = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        A_unit = compute_runtime_jit(coords_unit)
        assert np.allclose(A_ffcx, A_unit, atol=1e-14)

    def test_rotated_triangle(self, laplacian_form):
        """Test on rotated equilateral-ish triangle."""
        # Rotate unit right triangle by 45 degrees
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        coords_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        rotated_2d = coords_2d @ R.T

        coords = np.column_stack([rotated_2d, np.zeros(3)])

        A_ffcx = compute_ffcx_standard(laplacian_form, coords)
        A_runtime = compute_runtime_jit(coords)

        assert np.allclose(A_ffcx, A_runtime, atol=1e-14)

        # Laplacian is rotation invariant, should match unit right triangle
        coords_unit = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        A_unit = compute_runtime_jit(coords_unit)
        assert np.allclose(A_ffcx, A_unit, atol=1e-14)

    def test_different_quadrature_degrees(self, laplacian_form):
        """Test that different quadrature degrees give same result for P1."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        A_ffcx = compute_ffcx_standard(laplacian_form, coords)

        # For P1 Laplacian, any degree >= 0 should work (gradients are constant)
        for degree in [1, 2, 3, 4, 5]:
            A_runtime = compute_runtime_jit(coords, quadrature_degree=degree)
            assert np.allclose(A_ffcx, A_runtime, atol=1e-14), (
                f"Failed for degree {degree}"
            )


class TestRuntimeQuadraturePoints:
    """Test with custom quadrature points using JIT-compiled kernel."""

    def test_single_point_quadrature(self):
        """Test with single centroid quadrature point using JIT kernel."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        # Use degree 1 quadrature (single point at centroid)
        A_runtime = compute_runtime_jit(coords, quadrature_degree=1)

        # Should match standard result (gradients are constant for P1)
        A_expected = np.array(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 0.5, 0.0],
                [-0.5, 0.0, 0.5],
            ]
        )

        assert np.allclose(A_runtime, A_expected, atol=1e-14)


def compute_runtime_jit_generated(coordinate_dofs, quadrature_degree=2):
    """Compute Laplacian using kernel generated by RuntimeIntegralGenerator.

    This uses the actual generate_runtime() method to produce C code.
    The generated code expects 2 elements:
      - Element 0: P1 scalar for argument functions (test/trial)
      - Element 1: P1 vector (blocked) for Jacobian computation
    """
    kernel, ffi, module = get_runtime_laplacian_kernel_generated()

    # Get quadrature rule
    qpts, qwts = basix.make_quadrature(basix.CellType.triangle, quadrature_degree)
    nq = len(qwts)

    # Create P1 Lagrange element and tabulate
    P1_element = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1)

    # Tabulate with first derivatives: shape [nderivs, nq, ndofs, ncomps]
    # For scalar element ncomps=1
    tables = P1_element.tabulate(1, qpts)

    # Element 0: P1 scalar for arguments - shape [nderivs, nq, ndofs]
    # Take component 0 (only component for scalar)
    table0_flat = tables[:, :, :, 0].flatten().astype(np.float64)
    table0_flat = np.ascontiguousarray(table0_flat)

    # Element 1: P1 vector for Jacobian - same tables, but semantically
    # represents the geometry mapping element. For P1 geometry, it's the
    # same as the scalar element.
    # The generated code accesses elements[1] for Jacobian computation.
    table1_flat = tables[:, :, :, 0].flatten().astype(np.float64)
    table1_flat = np.ascontiguousarray(table1_flat)

    # Ensure arrays are contiguous
    qpts_flat = np.ascontiguousarray(qpts.flatten(), dtype=np.float64)
    qwts_flat = np.ascontiguousarray(qwts, dtype=np.float64)
    coords_flat = np.ascontiguousarray(coordinate_dofs.flatten(), dtype=np.float64)

    # Create array of 2 element info structs
    elems = ffi.new("runintgen_element[2]")

    # Element 0: P1 scalar for arguments
    elems[0].ndofs = 3
    elems[0].nderivs = 3  # (0,0), (1,0), (0,1)
    elems[0].table = ffi.cast("const double*", table0_flat.ctypes.data)

    # Element 1: P1 for Jacobian (same shape, geometry uses same basis)
    elems[1].ndofs = 3
    elems[1].nderivs = 3
    elems[1].table = ffi.cast("const double*", table1_flat.ctypes.data)

    # Create quadrature config (new multi-config structure)
    config = ffi.new("runintgen_quadrature_config*")
    config.nq = nq
    config.points = ffi.cast("const double*", qpts_flat.ctypes.data)
    config.weights = ffi.cast("const double*", qwts_flat.ctypes.data)
    config.nelements = 2
    config.elements = elems

    # Create main runtime data struct (single-config mode)
    data = ffi.new("runintgen_data*")
    data.num_configs = 1
    data.configs = config
    data.active_config = 0  # Single-config mode
    data.cell_config_map = ffi.NULL

    # Output array
    ndofs = 3
    A = np.zeros((ndofs, ndofs), dtype=np.float64)

    # Empty arrays for unused parameters
    w = np.zeros(0, dtype=np.float64)
    c = np.zeros(0, dtype=np.float64)
    entity_local_index = np.zeros(1, dtype=np.int32)
    quadrature_permutation = np.zeros(1, dtype=np.uint8)

    # Call the kernel
    kernel(
        ffi.cast("double*", A.ctypes.data),
        ffi.cast("double*", w.ctypes.data),
        ffi.cast("double*", c.ctypes.data),
        ffi.cast("double*", coords_flat.ctypes.data),
        ffi.cast("int*", entity_local_index.ctypes.data),
        ffi.cast("uint8_t*", quadrature_permutation.ctypes.data),
        data,
    )

    return A


class TestGeneratedKernel:
    """Test that kernel generated by RuntimeIntegralGenerator matches FFCX."""

    @pytest.fixture
    def mesh_and_space(self):
        """Create mesh and function space for P1 Laplacian."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        return mesh, V

    @pytest.fixture
    def laplacian_form(self, mesh_and_space):
        """Create standard Laplacian form."""
        mesh, V = mesh_and_space
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx = ufl.Measure("dx", domain=mesh)
        return inner(grad(u), grad(v)) * dx

    def test_generated_matches_ffcx(self, laplacian_form):
        """Test that generated kernel produces same result as FFCX."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        A_ffcx = compute_ffcx_standard(laplacian_form, coords)
        A_generated = compute_runtime_jit_generated(coords)

        assert np.allclose(A_ffcx, A_generated, atol=1e-14)

    def test_generated_matches_hardcoded(self):
        """Test that generated kernel matches our hardcoded reference."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        A_hardcoded = compute_runtime_jit(coords)
        A_generated = compute_runtime_jit_generated(coords)

        assert np.allclose(A_hardcoded, A_generated, atol=1e-14)

    def test_generated_different_triangles(self, laplacian_form):
        """Test generated kernel on various triangle shapes."""
        triangles = [
            # Stretched
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            # Translated
            np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]]),
            # Scaled
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        ]

        for coords in triangles:
            A_ffcx = compute_ffcx_standard(laplacian_form, coords)
            A_generated = compute_runtime_jit_generated(coords)
            assert np.allclose(A_ffcx, A_generated, atol=1e-14)
