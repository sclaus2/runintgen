"""Tests for runtime code generation."""

from __future__ import annotations

import ufl
from basix.ufl import element

from runintgen import (
    compile_runtime_integrals,
    get_runintgen_data_struct,
)


def runtime_dx(subdomain_id: int, domain: ufl.Mesh) -> ufl.Measure:
    """Create a runtime measure using the new API."""
    return ufl.Measure(
        "dx",
        domain=domain,
        subdomain_id=subdomain_id,
        metadata={"quadrature_rule": "runtime"},
    )


class TestCodeGeneration:
    """Test cases for runtime kernel code generation."""

    def test_laplacian_p1_p1(self):
        """Test Laplacian with P1 mesh and P1 solution (shared tables)."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = runtime_dx(subdomain_id=1, domain=mesh)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]

        # Check basic properties
        assert kernel.name == "runint_cell_1_0"
        assert kernel.integral_type == "cell"
        assert kernel.subdomain_id == 1

        # Check table info - P1/P1 should have 2 tables (shared)
        assert len(kernel.table_info) == 2

        # Check C code contains expected patterns
        assert "runintgen_data" in kernel.c_definition
        assert "for (int iq = 0; iq < nq; ++iq)" in kernel.c_definition
        assert "Jacobian" in kernel.c_definition
        assert "data->elements[" in kernel.c_definition
        assert "arg_elem->table" in kernel.c_definition
        assert "ndofs = 3" in kernel.c_definition

    def test_laplacian_p1_p2(self):
        """Test Laplacian with P1 mesh and P2 solution (separate tables)."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 2)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = runtime_dx(subdomain_id=2, domain=mesh)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]

        # Check basic properties
        assert kernel.subdomain_id == 2

        # Check table info - P1/P2 should have 2 unique elements
        # (one for P2 argument, one for P1 coordinate)
        assert len(kernel.table_info) == 2

        # Check elements by their properties
        # Element with is_argument=True should be P2 (6 dofs)
        arg_elements = [t for t in kernel.table_info if t.get("is_argument", False)]
        # Element with is_coordinate=True should be P1 (3 dofs)
        coord_elements = [t for t in kernel.table_info if t.get("is_coordinate", False)]

        assert len(arg_elements) == 1
        assert len(coord_elements) == 1

        # Argument element should have 6 DOFs (P2)
        assert arg_elements[0]["ndofs"] == 6

        # Coordinate element should have 3 DOFs (P1)
        assert coord_elements[0]["ndofs"] == 3

        # Check C code contains expected patterns
        assert "ndofs = 6" in kernel.c_definition
        assert "coord_ndofs = 3" in kernel.c_definition

    def test_runintgen_data_struct(self):
        """Test that runintgen_data struct definition is available."""
        struct_def = get_runintgen_data_struct()

        # Check main struct with multi-config support
        assert "typedef struct" in struct_def
        assert "runintgen_data" in struct_def
        assert "num_configs" in struct_def
        assert "runintgen_quadrature_config* configs" in struct_def
        assert "active_config" in struct_def
        assert "cell_config_map" in struct_def

        # Check quadrature config structure
        assert "runintgen_quadrature_config" in struct_def
        assert "int nq;" in struct_def
        assert "const double* points;" in struct_def
        assert "const double* weights;" in struct_def

        # Check per-element structure
        assert "runintgen_element" in struct_def
        assert "int nelements;" in struct_def
        assert "const runintgen_element* elements;" in struct_def
        assert "int ndofs;" in struct_def
        assert "int nderivs;" in struct_def
        assert "const double* table;" in struct_def

    def test_kernel_declaration(self):
        """Test that kernel declarations are generated correctly."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)
        kernel = module.kernels[0]

        # Check declaration format
        decl = kernel.c_declaration
        assert "void tabulate_tensor_runint_cell_" in decl
        assert "double* A" in decl
        assert "void* custom_data" in decl
        assert decl.strip().endswith(");")

    def test_definition_includes(self):
        """Test that kernel definitions have required includes."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)
        kernel = module.kernels[0]

        # Check includes
        defn = kernel.c_definition
        assert "#include <math.h>" in defn
        assert "#include <stdint.h>" in defn


class TestRuntimeElementMapping:
    """Test cases for the new RuntimeElementMapping class."""

    def test_element_sharing_p1(self):
        """Test that P1 test/trial/coefficient share same element."""
        from runintgen.analysis import build_runtime_info
        from runintgen.fe_tables import extract_integral_metadata
        from runintgen.runtime_tables import build_runtime_element_mapping

        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = ufl.Coefficient(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options={})
        integral_metadata = extract_integral_metadata(runtime_info)
        ir = runtime_info.ir

        for group, meta in integral_metadata.items():
            for integral_ir in ir.integrals:
                if integral_ir.expression.integral_type == group.integral_type:
                    mapping = build_runtime_element_mapping(integral_ir, meta)

                    # Should have 2 unique elements: P1 (shared) and coordinate
                    assert len(mapping.elements) == 2

                    # Find P1 element (ndofs=3, not coordinate)
                    p1_elem = None
                    for elem in mapping.elements:
                        if elem.ndofs == 3 and not elem.is_coordinate:
                            p1_elem = elem
                            break

                    assert p1_elem is not None
                    assert p1_elem.is_argument
                    assert p1_elem.is_coefficient
                    assert p1_elem.max_derivative_order == 1
                    break

    def test_element_separation_different_elements(self):
        """Test that different elements for coefficient are kept separate."""
        from runintgen.analysis import build_runtime_info
        from runintgen.fe_tables import extract_integral_metadata
        from runintgen.runtime_tables import build_runtime_element_mapping

        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 2)  # P2 for test/trial
        W_el = element("DG", "triangle", 0)  # DG0 for coefficient
        V = ufl.FunctionSpace(mesh, V_el)
        W = ufl.FunctionSpace(mesh, W_el)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = ufl.Coefficient(W)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options={})
        integral_metadata = extract_integral_metadata(runtime_info)
        ir = runtime_info.ir

        for group, meta in integral_metadata.items():
            for integral_ir in ir.integrals:
                if integral_ir.expression.integral_type == group.integral_type:
                    mapping = build_runtime_element_mapping(integral_ir, meta)

                    # Should have 3 unique elements: P2, DG0, and coordinate
                    assert len(mapping.elements) == 3

                    # Find each element type
                    p2_elem = dg0_elem = coord_elem = None
                    for elem in mapping.elements:
                        if elem.ndofs == 6:
                            p2_elem = elem
                        elif elem.ndofs == 1:
                            dg0_elem = elem
                        elif elem.is_coordinate:
                            coord_elem = elem

                    assert p2_elem is not None
                    assert p2_elem.is_argument
                    assert not p2_elem.is_coefficient
                    assert p2_elem.max_derivative_order == 1

                    assert dg0_elem is not None
                    assert dg0_elem.is_coefficient
                    assert not dg0_elem.is_argument
                    assert dg0_elem.max_derivative_order == 0  # Only values

                    assert coord_elem is not None
                    assert coord_elem.is_coordinate
                    break

    def test_derivative_to_index_2d(self):
        """Test derivative tuple to basix index mapping for 2D."""
        from runintgen.runtime_tables import DerivativeMapping

        # Order 0: (0,0) -> 0
        assert DerivativeMapping.derivative_to_index_2d((0, 0)) == 0

        # Order 1: (1,0) -> 1, (0,1) -> 2
        assert DerivativeMapping.derivative_to_index_2d((1, 0)) == 1
        assert DerivativeMapping.derivative_to_index_2d((0, 1)) == 2

        # Order 2: (2,0) -> 3, (1,1) -> 4, (0,2) -> 5
        assert DerivativeMapping.derivative_to_index_2d((2, 0)) == 3
        assert DerivativeMapping.derivative_to_index_2d((1, 1)) == 4
        assert DerivativeMapping.derivative_to_index_2d((0, 2)) == 5
