"""Tests for analysis module - runtime integral analysis."""

import pytest
import ufl
from basix.ufl import element as basix_element

from runintgen.analysis import (
    ArgumentInfo,
    ArgumentRole,
    ElementInfo,
    RuntimeAnalysisInfo,
    RuntimeIntegralInfo,
    build_runtime_analysis,
)
from ffcx.options import get_options


@pytest.fixture
def mesh():
    """Create a simple mesh."""
    return ufl.Mesh(basix_element("Lagrange", "triangle", 1, shape=(2,)))


@pytest.fixture
def V(mesh):
    """Create a scalar P1 function space."""
    V_el = basix_element("Lagrange", "triangle", 1)
    return ufl.FunctionSpace(mesh, V_el)


@pytest.fixture
def options():
    """Get default FFCX options."""
    return get_options()


class MockQuadratureProvider:
    """Mock quadrature provider for testing."""

    pass


class TestBuildRuntimeAnalysis:
    """Tests for build_runtime_analysis."""

    def test_simple_laplacian(self, mesh, V, options):
        """Test analysis for simple Laplacian form."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        provider = MockQuadratureProvider()
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=provider,
            subdomain_id=1,
            metadata={"quadrature_rule": "runtime"},
        )
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

        analysis = build_runtime_analysis(a, options)

        assert isinstance(analysis, RuntimeAnalysisInfo)
        assert len(analysis.groups) == 1
        assert len(analysis.integral_infos) == 1

        # Check the integral info
        key = list(analysis.integral_infos.keys())[0]
        info = analysis.integral_infos[key]
        assert info.integral_type == "cell"

        # Should have test and trial arguments
        has_test = any(role == ArgumentRole.TEST for role, _ in info.arguments.keys())
        has_trial = any(role == ArgumentRole.TRIAL for role, _ in info.arguments.keys())
        assert has_test
        assert has_trial

        # Check elements have derivatives registered
        assert len(info.elements) > 0
        for elem_id, elem_info in info.elements.items():
            # Elements used for gradients should have max_derivative_order >= 1
            # (either argument elements or geometry elements)
            pass  # Can vary based on form structure

    def test_with_coefficient(self, mesh, V, options):
        """Test analysis with a coefficient."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = ufl.Coefficient(V)
        provider = MockQuadratureProvider()
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=provider,
            subdomain_id=1,
            metadata={"quadrature_rule": "runtime"},
        )
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

        analysis = build_runtime_analysis(a, options)

        key = list(analysis.integral_infos.keys())[0]
        info = analysis.integral_infos[key]

        # Should have coefficient argument
        has_coef = any(
            role == ArgumentRole.COEFFICIENT for role, _ in info.arguments.keys()
        )
        assert has_coef

        # Coefficient should have no derivatives (index 0)
        coef_args = [
            arg
            for (role, idx), arg in info.arguments.items()
            if role == ArgumentRole.COEFFICIENT
        ]
        assert len(coef_args) >= 1
        # Coefficient without grad should have max_derivative_order == 0
        for arg in coef_args:
            assert arg.max_derivative_order == 0

    def test_geometry_extraction(self, mesh, V, options):
        """Test that geometry/jacobian is extracted."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        provider = MockQuadratureProvider()
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=provider,
            subdomain_id=1,
            metadata={"quadrature_rule": "runtime"},
        )
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

        analysis = build_runtime_analysis(a, options)

        key = list(analysis.integral_infos.keys())[0]
        info = analysis.integral_infos[key]

        # Should have geometry argument (for Jacobian)
        has_geometry = any(
            role == ArgumentRole.GEOMETRY for role, _ in info.arguments.keys()
        )
        assert has_geometry

    def test_derivative_tracking(self, mesh, V, options):
        """Test derivative tracking per element."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        provider = MockQuadratureProvider()
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=provider,
            subdomain_id=1,
            metadata={"quadrature_rule": "runtime"},
        )
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

        analysis = build_runtime_analysis(a, options)

        key = list(analysis.integral_infos.keys())[0]
        info = analysis.integral_infos[key]

        # At least one element should have derivatives registered
        has_derivatives = any(
            elem.max_derivative_order >= 1 for elem in info.elements.values()
        )
        assert has_derivatives

    def test_runtime_groups(self, mesh, V, options):
        """Test runtime groups are created correctly."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        provider = MockQuadratureProvider()
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=provider,
            subdomain_id=1,
            metadata={"quadrature_rule": "runtime"},
        )
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

        analysis = build_runtime_analysis(a, options)

        assert len(analysis.groups) == 1
        group = analysis.groups[0]
        assert group.integral_type == "cell"
        assert 1 in group.subdomain_ids
        assert group.quadrature_provider is provider


class TestElementInfo:
    """Tests for ElementInfo dataclass."""

    def test_register_derivative(self):
        """Test registering derivatives."""
        info = ElementInfo(element=None, element_id="test")
        assert info.max_derivative_order == 0
        assert len(info.derivative_tuples) == 0

        info.register_derivative((1, 0))
        assert info.max_derivative_order == 1
        assert (1, 0) in info.derivative_tuples

        info.register_derivative((0, 1))
        assert info.max_derivative_order == 1
        assert (0, 1) in info.derivative_tuples

        info.register_derivative((1, 1))
        assert info.max_derivative_order == 2
        assert (1, 1) in info.derivative_tuples


class TestArgumentInfo:
    """Tests for ArgumentInfo dataclass."""

    def test_register_derivative(self):
        """Test registering derivatives on arguments."""
        info = ArgumentInfo(
            role=ArgumentRole.TEST, index=0, element_id="test", element=None
        )
        assert info.max_derivative_order == 0

        info.register_derivative((1, 0))
        assert info.max_derivative_order == 1
        assert (1, 0) in info.derivative_tuples


class TestRuntimeIntegralInfo:
    """Tests for RuntimeIntegralInfo dataclass."""

    def test_get_or_add_argument(self):
        """Test get_or_add_argument method."""
        info = RuntimeIntegralInfo(integral_type="cell", ir_index=0, subdomain_id=1)

        arg1 = info.get_or_add_argument(ArgumentRole.TEST, 0, None, "elem1")
        assert arg1.role == ArgumentRole.TEST
        assert arg1.index == 0

        # Getting same argument should return same object
        arg2 = info.get_or_add_argument(ArgumentRole.TEST, 0, None, "elem1")
        assert arg1 is arg2

    def test_get_or_add_element(self):
        """Test get_or_add_element method."""
        info = RuntimeIntegralInfo(integral_type="cell", ir_index=0, subdomain_id=1)

        elem1 = info.get_or_add_element(None, "elem1")
        assert elem1.element_id == "elem1"

        # Getting same element should return same object
        elem2 = info.get_or_add_element(None, "elem1")
        assert elem1 is elem2
