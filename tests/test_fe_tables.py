"""Tests for FE table metadata extraction."""

import pytest
import ufl
from basix.ufl import element as basix_element

from runintgen import runtime_dx
from runintgen.analysis import build_runtime_info
from runintgen.fe_tables import (
    ComponentRequest,
    IntegralRuntimeMeta,
    extract_integral_metadata,
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


class TestExtractIntegralMetadata:
    """Tests for extract_integral_metadata."""

    def test_simple_laplacian(self, mesh, V, options):
        """Test metadata extraction for simple Laplacian form."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="quadrature")
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options)
        result = extract_integral_metadata(runtime_info)

        assert len(result) == 1
        group, meta = next(iter(result.items()))

        assert group.integral_type == "cell"
        assert 1 in group.subdomain_ids

        # Should have 4 argument requests (test/trial x 2 derivatives each)
        # Plus jacobian requests for geometry
        arg_requests = [r for r in meta.component_requests if r.role == "argument"]
        assert len(arg_requests) == 4

        # Check that we have both test (0) and trial (1)
        arg_indices = {r.index for r in arg_requests}
        assert arg_indices == {0, 1}

        # All should have max_deriv=1 (first derivatives)
        for req in arg_requests:
            assert req.max_deriv == 1

    def test_with_coefficient(self, mesh, V, options):
        """Test metadata extraction with a coefficient."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        kappa = ufl.Coefficient(V)
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="quadrature")
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options)
        result = extract_integral_metadata(runtime_info)

        assert len(result) == 1
        group, meta = next(iter(result.items()))

        # Should have coefficient request
        coef_requests = [r for r in meta.component_requests if r.role == "coefficient"]
        assert len(coef_requests) >= 1

        # Coefficient should have index 0 and no derivatives
        coef_req = coef_requests[0]
        assert coef_req.index == 0
        assert coef_req.max_deriv == 0

    def test_jacobian_extraction(self, mesh, V, options):
        """Test that Jacobian components are extracted."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="quadrature")
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options)
        result = extract_integral_metadata(runtime_info)

        group, meta = next(iter(result.items()))

        # Should have jacobian requests for geometry transforms
        jac_requests = [r for r in meta.component_requests if r.role == "jacobian"]
        assert len(jac_requests) > 0

        # Jacobian should have components 0 and 1 (2D)
        jac_components = {r.component for r in jac_requests}
        assert 0 in jac_components
        assert 1 in jac_components

    def test_max_derivative_per_element(self, mesh, V, options):
        """Test max derivative tracking per element."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="quadrature")
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options)
        result = extract_integral_metadata(runtime_info)

        group, meta = next(iter(result.items()))

        # Should have at least one element with max_deriv >= 1
        assert len(meta.max_derivative_per_element) > 0
        max_derivs = list(meta.max_derivative_per_element.values())
        assert max(max_derivs) >= 1

    def test_components_per_element(self, mesh, V, options):
        """Test component tracking per element."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="quadrature")
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_rt

        runtime_info = build_runtime_info(a, options)
        result = extract_integral_metadata(runtime_info)

        group, meta = next(iter(result.items()))

        # Each element should have at least component 0
        for eh, comps in meta.components_per_element.items():
            assert 0 in comps


class TestComponentRequest:
    """Tests for ComponentRequest dataclass."""

    def test_component_request_creation(self):
        """Test ComponentRequest can be created."""
        el = basix_element("Lagrange", "triangle", 1)
        req = ComponentRequest(
            element=el,
            element_hash=hash(el),
            role="argument",
            index=0,
            component=0,
            max_deriv=1,
            local_derivatives=(1, 0),
        )

        assert req.role == "argument"
        assert req.index == 0
        assert req.component == 0
        assert req.max_deriv == 1
        assert req.local_derivatives == (1, 0)


class TestIntegralRuntimeMeta:
    """Tests for IntegralRuntimeMeta dataclass."""

    def test_empty_metadata(self):
        """Test empty IntegralRuntimeMeta."""
        meta = IntegralRuntimeMeta(
            integral_type="cell",
            subdomain_id=1,
        )

        assert meta.integral_type == "cell"
        assert meta.subdomain_id == 1
        assert len(meta.component_requests) == 0
        assert len(meta.max_derivative_per_element) == 0
        assert len(meta.components_per_element) == 0
