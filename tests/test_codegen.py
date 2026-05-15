"""Tests for runtime code generation."""

from __future__ import annotations

import numpy as np
import ufl
from basix.ufl import element

from runintgen import (
    QuadratureRules,
    RuntimeQuadratureRule,
    compile_runtime_integrals,
    dxq,
    get_runintgen_data_struct,
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

        dx_rt = dxq(subdomain_id=1, domain=mesh)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]

        # Check basic properties
        assert kernel.name.startswith("integral_")
        assert kernel.name.endswith("_triangle")
        assert kernel.integral_type == "cell"
        assert kernel.subdomain_id == 1

        # P1 gradients are piecewise constants, but they still map to a single
        # runtime Basix element tabulation for this integral.
        assert len(kernel.table_info) == 2
        assert {tuple(t["derivative_counts"]) for t in kernel.table_info} == {
            (1, 0),
            (0, 1),
        }
        assert {t["slot"] for t in kernel.table_info} == {0}
        assert {t["element_index"] for t in kernel.table_info} == {0}

        # Check C code contains expected patterns
        assert "runintgen_context" in get_runintgen_data_struct()
        assert "const int local_index = " in kernel.c_definition
        assert "entities->rule_indices[local_index]" in kernel.c_definition
        assert "quadrature->offsets[rule_index]" in kernel.c_definition
        assert "entities->is_cut[local_index]" not in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_runtime" not in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_standard" not in kernel.c_definition
        assert "for (int iq = 0; iq < rt_nq; ++iq)" in kernel.c_definition
        assert "rt_weights" in kernel.c_definition
        assert kernel.c_definition.count(".tabulate(") == 1
        assert "rt_element_0" in kernel.c_definition
        assert "arg_elem->table" not in kernel.c_definition

    def test_laplacian_p1_p2(self):
        """Test Laplacian with P1 mesh and P2 solution (separate tables)."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 2)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = dxq(subdomain_id=2, domain=mesh)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]

        # Check basic properties
        assert kernel.subdomain_id == 2
        assert kernel.scalar_type == "float64"
        assert kernel.geometry_type == "float64"

        # Check runtime table info. Four FFCx table references map to two
        # Basix element tabulations: geometry P1 and argument P2.
        assert len(kernel.table_info) == 4

        assert {tuple(t["derivative_counts"]) for t in kernel.table_info} == {
            (1, 0),
            (0, 1),
        }
        assert {t["slot"] for t in kernel.table_info} == {0, 1}
        assert {t["element_index"] for t in kernel.table_info} == {0, 1}
        assert {t["role"] for t in kernel.table_info} == {"geometry", "trial"}
        assert all(t["element_max_derivative_order"] == 1 for t in kernel.table_info)

        # Check C code contains expected patterns
        assert "rt_element_0" in kernel.c_definition
        assert "rt_element_1" in kernel.c_definition
        assert "static const runintgen_table_request" in kernel.c_definition
        assert kernel.c_definition.count(".tabulate(") == 2
        assert "rt_element_view_0.values" in kernel.c_definition
        assert "(void)" not in kernel.c_definition

    def test_runintgen_data_struct(self):
        """Test that runintgen_data struct definition is available."""
        struct_def = get_runintgen_data_struct()

        # Check main struct (simplified single-config design)
        assert "typedef struct" in struct_def
        assert "runintgen_context" in struct_def
        assert "runintgen_form_context" in struct_def
        assert "runintgen_form_descriptor" in struct_def
        assert "runintgen_form_element_descriptor" in struct_def
        assert "runintgen_quadrature_rule" in struct_def
        assert "runintgen_quadrature_rules" in struct_def
        assert "runintgen_entity_map" in struct_def
        assert "runintgen_basix_element" in struct_def
        assert "int nq;" in struct_def
        assert "const double* points;" in struct_def
        assert "const double* weights;" in struct_def
        assert "const int64_t* offsets;" in struct_def
        assert "const int32_t* entity_indices;" in struct_def
        assert "const int32_t* rule_indices;" in struct_def

        # Check runtime table and callback structures
        assert "runintgen_table_view" in struct_def
        assert "runintgen_table_request" in struct_def
        assert "int derivative_order;" in struct_def
        assert "int is_permuted;" in struct_def
        assert "runintgen_element_tabulate_fn" in struct_def
        assert "const runintgen_basix_element* elements;" in struct_def
        assert "const runintgen_quadrature_rules* quadrature;" in struct_def
        assert "const runintgen_entity_map* entities;" in struct_def
        assert "const runintgen_form_context* form;" in struct_def
        assert "void* scratch;" in struct_def
        assert "runintgen_prepare_fn" not in struct_def
        assert "runintgen_tabulate_fn" not in struct_def
        assert "int derivative_counts[4];" not in struct_def
        assert "int flat_component;" not in struct_def
        assert "int is_uniform;" not in struct_def

        # Verify old multi-config fields are NOT present
        assert "num_configs" not in struct_def
        assert "active_config" not in struct_def
        assert "cell_config_map" not in struct_def
        assert "runintgen_quadrature_config" not in struct_def

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
        assert decl.strip() == f"extern ufcx_integral {kernel.name};"
        assert f"tabulate_tensor_{kernel.name}" in kernel.c_definition
        assert decl.strip().endswith(";")

    def test_default_subdomain_uses_ffcx_integral_name(self):
        """Test default subdomain ids use FFCx integral object names."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = ufl.inner(u, v) * dx_rt

        module = compile_runtime_integrals(a)
        kernel = module.kernels[0]

        assert kernel.subdomain_id == -1
        assert kernel.name.startswith("integral_")
        assert kernel.name.endswith("_triangle")
        assert f"tabulate_tensor_{kernel.name}" in kernel.c_definition
        assert "tabulate_tensor_runint_cell_-1_0" not in kernel.c_definition

    def test_quadrature_subdomain_data_generates_runtime_only_kernel(self):
        """Test quadrature-only subdomain data generates only runtime code."""

        class Quadrature:
            points = [(1.0 / 3.0, 1.0 / 3.0)]
            weights = [0.5]

        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=Quadrature())
        module = compile_runtime_integrals(ufl.inner(u, v) * dx_rt)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]
        assert "entities->rule_indices[local_index]" in kernel.c_definition
        assert "entities->is_cut[local_index]" not in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_runtime" not in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_standard" not in kernel.c_definition

    def test_mixed_subdomain_data_generates_mixed_kernel(self):
        """Test entity plus quadrature subdomain data generates mixed code."""

        class Quadrature:
            points = [(1.0 / 3.0, 1.0 / 3.0)]
            weights = [0.5]

        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=[(0, [0, 2, 4]), (0, Quadrature())],
        )
        module = compile_runtime_integrals(ufl.inner(u, v) * dx_rt)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]
        assert "entities->is_cut[local_index]" in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_runtime" in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_standard" in kernel.c_definition

    def test_direct_mixed_list_with_flat_runtime_rules_generates_mixed_kernel(self):
        """Test [standard_entities, runtime_rules] generates mixed code."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        quadrature = QuadratureRules(
            tdim=2,
            points=np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
            weights=np.array([0.5], dtype=np.float64),
            offsets=np.array([0, 1], dtype=np.int64),
            parent_map=np.array([8], dtype=np.int32),
        )

        standard_entities = np.array([1, 4, 5, 6], dtype=np.int32)
        dx_rt = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_id=0,
            subdomain_data=[standard_entities, quadrature],
        )
        module = compile_runtime_integrals(ufl.inner(u, v) * dx_rt)

        assert module.quadrature_provider[0] is standard_entities
        assert module.quadrature_provider[1] is quadrature
        kernel = module.kernels[0]
        assert "entities->is_cut[local_index]" in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_runtime" in kernel.c_definition
        assert f"tabulate_tensor_{kernel.name}_standard" in kernel.c_definition

    def test_standard_entity_subdomain_data_generates_no_runintgen_kernel(self):
        """Test standard-only entity data stays outside runintgen codegen."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_standard = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=[(0, [0, 2, 4])],
        )
        module = compile_runtime_integrals(ufl.inner(u, v) * dx_standard)

        assert module.kernels == []

    def test_static_and_runtime_integrals_same_type_coexist(self):
        """Test codegen skips standard integrals before runtime integrals."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_standard = ufl.Measure("dx", domain=mesh, subdomain_id=0)
        dx_runtime = dxq(subdomain_id=2, domain=mesh)
        a = ufl.inner(u, v) * dx_standard
        a += ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_runtime

        module = compile_runtime_integrals(a)

        assert len(module.kernels) == 1
        kernel = module.kernels[0]
        assert kernel.subdomain_id == 2
        assert kernel.ir_index == 1
        assert kernel.name.startswith("integral_")
        assert kernel.name.endswith("_triangle")

    def test_write_runtime_code_files(self, tmp_path):
        """Test generated kernels can be written as inspectable files."""
        from runintgen import write_runtime_code

        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 2)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = dxq(subdomain_id=2, domain=mesh)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)
        files = write_runtime_code(module, prefix="laplace-p2", output_dir=tmp_path)

        assert files.header.name == "laplace_p2.h"
        assert files.source.name == "laplace_p2.c"
        assert files.abi_header.name == "runintgen_runtime_abi.h"
        assert files.abi_header.parent.name == "cpp"
        assert not (tmp_path / "laplace_p2.json").exists()
        assert not (tmp_path / "runintgen_runtime_abi.h").exists()

        header_text = files.header.read_text()
        source_text = files.source.read_text()
        abi_text = files.abi_header.read_text()

        assert "extern ufcx_integral" in header_text
        assert "extern const runintgen_form_descriptor" in header_text
        assert "#include <ufcx.h>" in header_text
        assert '#include "runintgen_runtime_abi.h"' in header_text
        assert "runintgen_context" not in header_text
        assert '#include "runintgen_runtime_abi.h"' in source_text
        assert "runintgen_form_elements_laplace_p2" in source_text
        assert "runintgen_form_descriptor_laplace_p2" in source_text
        assert "runintgen_prepare_runtime" not in abi_text
        assert "runintgen_element_tabulate_fn" in abi_text
        assert "runintgen_context" in abi_text
        assert "runintgen_form_element_descriptor" in abi_text

    def test_definition_includes_ufcx_integral_object(self):
        """Test that kernel definitions have the UFCx integral object."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        module = compile_runtime_integrals(a)
        kernel = module.kernels[0]

        defn = kernel.c_definition
        assert f"ufcx_integral {kernel.name}" in defn
        assert ".enabled_coefficients =" in defn
        assert ".coordinate_element_hash = UINT64_C(" in defn

    def test_coordinate_dofs_type_matches_ffcx(self):
        """Test coordinate_dofs uses FFCx's real geometry scalar strategy."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = ufl.inner(u, v) * dx_rt

        complex128_module = compile_runtime_integrals(
            a, options={"scalar_type": np.complex128}
        )
        complex128_kernel = complex128_module.kernels[0]
        assert "double _Complex* restrict A" in complex128_kernel.c_definition
        assert "const double* restrict coordinate_dofs" in (
            complex128_kernel.c_definition
        )
        assert complex128_kernel.geometry_type == "float64"
        assert ".tabulate_tensor_complex128 = tabulate_tensor_" in (
            complex128_kernel.c_definition
        )

        complex64_module = compile_runtime_integrals(
            a, options={"scalar_type": np.complex64}
        )
        complex64_kernel = complex64_module.kernels[0]
        assert "float _Complex* restrict A" in complex64_kernel.c_definition
        assert "const float* restrict coordinate_dofs" in complex64_kernel.c_definition
        assert complex64_kernel.geometry_type == "float32"
        assert ".tabulate_tensor_complex64 = tabulate_tensor_" in (
            complex64_kernel.c_definition
        )

    def test_runtime_local_index_slots(self):
        """Test generated kernels use the DOLFINx extension local-index layout."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        ds_rt = ufl.Measure("ds", domain=mesh, metadata={"quadrature_rule": "runtime"})
        dS_rt = ufl.Measure("dS", domain=mesh, metadata={"quadrature_rule": "runtime"})

        cell_kernel = compile_runtime_integrals(ufl.inner(u, v) * dx_rt).kernels[0]
        exterior_kernel = compile_runtime_integrals(ufl.inner(u, v) * ds_rt).kernels[0]
        interior_kernel = compile_runtime_integrals(
            ufl.inner(ufl.jump(u), ufl.jump(v)) * dS_rt
        ).kernels[0]

        assert "entity_local_index[0]" in cell_kernel.c_definition
        assert "entity_local_index[1]" in exterior_kernel.c_definition
        assert "entity_local_index[2]" in interior_kernel.c_definition

    def test_table_requests_use_form_element_indices(self):
        """Test table requests refer to the form-level element descriptor list."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 2)
        W_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        W = ufl.FunctionSpace(mesh, W_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = ufl.Coefficient(W)

        dx_rt = dxq(subdomain_id=4, domain=mesh)
        module = compile_runtime_integrals(kappa * ufl.inner(u, v) * dx_rt)
        kernel = module.kernels[0]

        elements_by_role = {
            (element.role.name.lower(), element.index): element.form_elem_index
            for element in module.form_metadata.unique_elements
        }
        assert elements_by_role[("test", 0)] == 0
        assert elements_by_role[("coefficient", 0)] == 1

        table_by_role = {table["role"]: table for table in kernel.table_info}
        assert table_by_role["coefficient"]["element_index"] == 1
        assert table_by_role["trial"]["element_index"] == 0
        assert "elements[1].tabulate(" in kernel.c_definition
        assert "elements[0].tabulate(" in kernel.c_definition
        assert ".element_index =" not in kernel.c_definition

    def test_blocked_element_tables_use_basix_component_stride(self):
        """Test blocked element tables use scalar basis values and dof strides."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 2, shape=(2,))
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        dx_rt = dxq(subdomain_id=5, domain=mesh)
        module = compile_runtime_integrals(ufl.inner(u, v) * dx_rt)
        kernel = module.kernels[0]

        assert any(table["flat_component"] == 1 for table in kernel.table_info)
        assert any(table["block_size"] == 2 for table in kernel.table_info)
        assert "num_components" in kernel.c_definition
        assert "A[12 * (2 * (i) + 1) + (2 * (j) + 1)]" in kernel.c_definition
        assert "* rt_element_0_num_components + 1" not in kernel.c_definition

    def test_curved_geometry_jacobian_is_quadrature_varying(self):
        """Test higher-order coordinate Jacobians stay inside the quadrature loop."""
        mesh = ufl.Mesh(element("Lagrange", "triangle", 2, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        runtime_rule = RuntimeQuadratureRule(
            points=np.array(
                [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0]],
                dtype=np.float64,
            ),
            weights=np.array([0.25, 0.25], dtype=np.float64),
        )

        dx_rt = ufl.Measure("dx", domain=mesh, subdomain_data=runtime_rule)
        module = compile_runtime_integrals(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt)
        kernel = module.kernels[0]
        loop_pos = kernel.c_definition.find("for (int iq = 0; iq < rt_nq; ++iq)")
        jacobian_pos = kernel.c_definition.find("// Section: Jacobian")

        assert loop_pos >= 0
        assert jacobian_pos > loop_pos
        assert "for (int ic = 0; ic < 6; ++ic)" in kernel.c_definition
        assert (
            "rt_element_0[((rt_nq + (iq)) * rt_element_0_num_dofs + (ic)) "
            "* rt_element_0_num_components]"
        ) in kernel.c_definition
        assert "2 * (ic)" not in kernel.c_definition


class TestRuntimeElementMapping:
    """Test cases for the new RuntimeElementMapping class."""

    def test_element_sharing_p1(self):
        """Test that P1 test/trial/coefficient share same element."""
        from runintgen.analysis import build_runtime_analysis
        from runintgen.runtime_tables import build_runtime_element_mapping

        mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
        V_el = element("Lagrange", "triangle", 1)
        V = ufl.FunctionSpace(mesh, V_el)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = ufl.Coefficient(V)

        dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        analysis = build_runtime_analysis(a, options={})

        for key, integral_info in analysis.integral_infos.items():
            mapping = build_runtime_element_mapping(integral_info)

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

    def test_element_separation_different_elements(self):
        """Test that different elements for coefficient are kept separate."""
        from runintgen.analysis import build_runtime_analysis
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

        analysis = build_runtime_analysis(a, options={})

        for key, integral_info in analysis.integral_infos.items():
            mapping = build_runtime_element_mapping(integral_info)

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
