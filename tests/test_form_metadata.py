"""Tests for form_metadata module (Plan v2)."""

import pytest
import numpy as np

import basix
import ufl
from basix.ufl import element

from runintgen import compile_runtime_integrals
from runintgen.form_metadata import (
    ElementKey,
    FormElementInfo,
    FormRuntimeMetadata,
    IntegralElementUsage,
    IntegralRuntimeLayout,
    Role,
    build_form_runtime_metadata,
    element_key_from_basix,
    export_metadata_for_cpp,
)


def create_laplacian_form():
    """Create a simple Laplacian form for testing."""
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt
    return a


def create_p2_laplacian_form():
    """Create a P2 Laplacian form for testing."""
    mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_rt = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt
    return a


class TestElementKey:
    """Test ElementKey dataclass."""

    def test_element_key_creation(self):
        """Test creating an ElementKey."""
        key = ElementKey(
            family=1,  # Lagrange
            cell_type=2,  # triangle
            degree=1,
            value_shape=(),
            discontinuous=False,
        )
        assert key.family == 1
        assert key.cell_type == 2
        assert key.degree == 1
        assert key.value_shape == ()
        assert key.discontinuous is False

    def test_element_key_equality(self):
        """Test ElementKey equality."""
        key1 = ElementKey(family=1, cell_type=2, degree=1)
        key2 = ElementKey(family=1, cell_type=2, degree=1)
        key3 = ElementKey(family=1, cell_type=2, degree=2)

        assert key1 == key2
        assert key1 != key3

    def test_element_key_hashable(self):
        """Test ElementKey is hashable (frozen dataclass)."""
        key = ElementKey(family=1, cell_type=2, degree=1)
        # Should be usable as dict key
        d = {key: "test"}
        assert d[key] == "test"

    def test_element_key_to_dict(self):
        """Test ElementKey to_dict conversion."""
        key = ElementKey(
            family=1,
            cell_type=2,
            degree=1,
            value_shape=(2,),
            discontinuous=True,
        )
        d = key.to_dict()
        assert d["family"] == 1
        assert d["cell_type"] == 2
        assert d["degree"] == 1
        assert d["value_shape"] == [2]
        assert d["discontinuous"] is True

    def test_element_key_from_dict(self):
        """Test ElementKey from_dict reconstruction."""
        d = {
            "family": 1,
            "cell_type": 2,
            "degree": 1,
            "value_shape": [2],
            "discontinuous": True,
        }
        key = ElementKey.from_dict(d)
        assert key.family == 1
        assert key.cell_type == 2
        assert key.degree == 1
        assert key.value_shape == (2,)
        assert key.discontinuous is True

    def test_element_key_to_int64(self):
        """Test ElementKey packing to int64."""
        key = ElementKey(family=1, cell_type=2, degree=1)
        packed = key.to_int64()
        assert isinstance(packed, int)
        # Same key should produce same packed value
        key2 = ElementKey(family=1, cell_type=2, degree=1)
        assert key.to_int64() == key2.to_int64()

    def test_element_key_from_basix(self):
        """Test creating ElementKey from basix element."""
        elem = basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.triangle,
            1,
        )
        key = element_key_from_basix(elem)
        assert key.family == int(basix.ElementFamily.P)
        assert key.cell_type == int(basix.CellType.triangle)
        assert key.degree == 1


class TestFormMetadataDataclasses:
    """Test basic dataclass functionality."""

    def test_role_enum(self):
        """Test Role enum values."""
        assert Role.TEST.value == 1
        assert Role.TRIAL.value == 2
        assert Role.COEFFICIENT.value == 3
        assert Role.GEOMETRY.value == 4

    def test_form_element_info(self):
        """Test FormElementInfo dataclass."""
        key = ElementKey(family=1, cell_type=2, degree=1)
        info = FormElementInfo(
            form_elem_index=0,
            element_key=key,
            element=None,
            role=Role.TEST,
            index=0,
            ndofs=3,
            ncomps=1,
        )
        assert info.form_elem_index == 0
        assert info.element_key == key
        assert info.role == Role.TEST
        assert info.index == 0
        assert info.ndofs == 3
        assert info.ncomps == 1

    def test_integral_element_usage(self):
        """Test IntegralElementUsage dataclass."""
        usage = IntegralElementUsage(
            form_elem_index=0,
            max_derivative=1,
            table_slot=0,
        )
        assert usage.form_elem_index == 0
        assert usage.max_derivative == 1
        assert usage.table_slot == 0

    def test_integral_runtime_layout(self):
        """Test IntegralRuntimeLayout dataclass."""
        layout = IntegralRuntimeLayout(
            integral_type="cell",
            ir_index=0,
            subdomain_id=0,
        )
        layout.element_usages.append(
            IntegralElementUsage(form_elem_index=0, max_derivative=1, table_slot=0)
        )
        layout.terminal_to_table_slot[(Role.TEST, 0)] = 0
        layout.terminal_to_table_slot[(Role.TRIAL, 1)] = 0

        assert layout.integral_type == "cell"
        assert len(layout.element_usages) == 1
        assert layout.get_table_slot(Role.TEST, 0) == 0
        assert layout.get_table_slot(Role.TRIAL, 1) == 0


class TestFormRuntimeMetadata:
    """Test FormRuntimeMetadata container."""

    def test_add_unique_element(self):
        """Test adding unique elements."""
        metadata = FormRuntimeMetadata()
        key = ElementKey(family=1, cell_type=2, degree=1)

        info = metadata.add_unique_element(
            element_key=key,
            element=None,
            role=Role.TEST,
            index=0,
            ndofs=3,
            ncomps=1,
        )

        assert info.form_elem_index == 0
        assert len(metadata.unique_elements) == 1
        assert key in metadata.key_to_form_index
        assert metadata.key_to_form_index[key] == 0

    def test_add_duplicate_element(self):
        """Test that adding duplicate elements returns existing."""
        metadata = FormRuntimeMetadata()
        key = ElementKey(family=1, cell_type=2, degree=1)

        info1 = metadata.add_unique_element(
            element_key=key,
            element=None,
            role=Role.TEST,
            index=0,
        )
        info2 = metadata.add_unique_element(
            element_key=key,
            element=None,
            role=Role.TRIAL,
            index=1,
        )

        # Should return same element, not create new one
        assert info1 is info2
        assert len(metadata.unique_elements) == 1

    def test_get_form_element(self):
        """Test retrieving form element by ElementKey."""
        metadata = FormRuntimeMetadata()
        key = ElementKey(family=1, cell_type=2, degree=1)
        metadata.add_unique_element(
            element_key=key,
            element=None,
            role=Role.TEST,
            index=0,
        )

        elem = metadata.get_form_element(key)
        assert elem is not None
        assert elem.element_key == key

        # Non-existent key
        other_key = ElementKey(family=2, cell_type=2, degree=1)
        assert metadata.get_form_element(other_key) is None


class TestBuildFormRuntimeMetadata:
    """Test building metadata from analysis."""

    def test_p1_laplacian_metadata(self):
        """Test metadata for P1 Laplacian."""
        form = create_laplacian_form()
        module = compile_runtime_integrals(form)

        metadata = module.form_metadata
        assert metadata is not None

        # Should have unique elements
        assert len(metadata.unique_elements) > 0

        # Should have integral layout
        assert len(metadata.integral_layouts) > 0

        # Check layout has element usages
        for key, layout in metadata.integral_layouts.items():
            assert len(layout.element_usages) > 0
            assert len(layout.terminal_to_table_slot) > 0

    def test_p2_laplacian_metadata(self):
        """Test metadata for P2 Laplacian."""
        form = create_p2_laplacian_form()
        module = compile_runtime_integrals(form)

        metadata = module.form_metadata
        assert metadata is not None

        # Should have unique elements
        assert len(metadata.unique_elements) > 0

    def test_kernel_has_table_slots(self):
        """Test that kernels have table_slots."""
        form = create_laplacian_form()
        module = compile_runtime_integrals(form)

        for kernel in module.kernels:
            assert kernel.table_slots is not None
            assert len(kernel.table_slots) > 0

    def test_table_slots_match_layout(self):
        """Test that kernel table_slots match layout."""
        form = create_laplacian_form()
        module = compile_runtime_integrals(form)

        metadata = module.form_metadata

        for kernel in module.kernels:
            layout = metadata.get_integral_layout(kernel.integral_type, kernel.ir_index)
            if layout is not None:
                # table_slots in kernel should match layout
                for key, slot in kernel.table_slots.items():
                    # Parse "role_index" format
                    parts = key.rsplit("_", 1)
                    role_str = parts[0]
                    terminal_idx = int(parts[1])

                    # Find matching entry in layout
                    from runintgen.form_metadata import Role

                    role = getattr(Role, role_str.upper())
                    layout_slot = layout.terminal_to_table_slot.get(
                        (role, terminal_idx)
                    )

                    assert layout_slot == slot


class TestExportMetadataForCpp:
    """Test exporting metadata for C++ consumption."""

    def test_export_format(self):
        """Test export format structure."""
        form = create_laplacian_form()
        module = compile_runtime_integrals(form)

        exported = export_metadata_for_cpp(module.form_metadata)

        assert "unique_elements" in exported
        assert "integral_layouts" in exported
        assert isinstance(exported["unique_elements"], list)
        assert isinstance(exported["integral_layouts"], dict)

    def test_export_unique_elements(self):
        """Test exported unique elements."""
        form = create_laplacian_form()
        module = compile_runtime_integrals(form)

        exported = export_metadata_for_cpp(module.form_metadata)

        for elem in exported["unique_elements"]:
            assert "form_elem_index" in elem
            assert "element_key" in elem
            assert "role" in elem
            assert "index" in elem
            assert "ndofs" in elem
            assert "ncomps" in elem
            # Role should be lowercase string
            assert elem["role"] in ["test", "trial", "coefficient", "geometry"]
            # Check element_key structure
            key = elem["element_key"]
            assert "family" in key
            assert "cell_type" in key
            assert "degree" in key

    def test_export_integral_layouts(self):
        """Test exported integral layouts."""
        form = create_laplacian_form()
        module = compile_runtime_integrals(form)

        exported = export_metadata_for_cpp(module.form_metadata)

        for key, layout in exported["integral_layouts"].items():
            assert "integral_type" in layout
            assert "ir_index" in layout
            assert "subdomain_id" in layout
            assert "element_usages" in layout
            assert "terminal_to_table_slot" in layout

            for usage in layout["element_usages"]:
                assert "form_elem_index" in usage
                assert "max_derivative" in usage
                assert "table_slot" in usage
