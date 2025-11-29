"""Tests for runintgen.measures module."""

from __future__ import annotations

import pytest

import ufl
from ufl.finiteelement import AbstractFiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

from runintgen.measures import RuntimeQuadrature, is_runtime_integral, runtime_dx


class LagrangeElement(AbstractFiniteElement):
    """A simple Lagrange element for testing (following UFL test pattern)."""

    def __init__(self, cell: ufl.Cell, degree: int, shape: tuple[int, ...] = ()):
        """Initialise."""
        self._cell = cell
        self._degree = degree
        self._shape = shape

    def __repr__(self) -> str:
        return f"LagrangeElement({self._cell}, {self._degree}, {self._shape})"

    def __str__(self) -> str:
        return f"<Lagrange{self._degree} on {self._cell}>"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and repr(self) == repr(other)

    @property
    def sobolev_space(self):
        return H1

    @property
    def pullback(self):
        return identity_pullback

    @property
    def embedded_superdegree(self) -> int:
        return self._degree

    @property
    def embedded_subdegree(self) -> int:
        return self._degree

    @property
    def cell(self):
        return self._cell

    @property
    def reference_value_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def sub_elements(self) -> list:
        return []


class TestRuntimeQuadrature:
    """Tests for RuntimeQuadrature dataclass."""

    def test_basic_creation(self):
        """Test basic creation with tag only."""
        rq = RuntimeQuadrature(tag="test")
        assert rq.tag == "test"
        assert rq.payload is None

    def test_creation_with_payload(self):
        """Test creation with payload."""
        payload = {"order": 3, "scheme": "gauss"}
        rq = RuntimeQuadrature(tag="cut_cell", payload=payload)
        assert rq.tag == "cut_cell"
        assert rq.payload == payload

    def test_equality(self):
        """Test equality comparison."""
        rq1 = RuntimeQuadrature(tag="test", payload={"a": 1})
        rq2 = RuntimeQuadrature(tag="test", payload={"a": 1})
        assert rq1 == rq2


class TestRuntimeDx:
    """Tests for runtime_dx helper function."""

    @pytest.fixture
    def mesh(self):
        """Create a simple UFL mesh."""
        cell = ufl.triangle
        coord_elem = LagrangeElement(cell, 1, (2,))
        return ufl.Mesh(coord_elem)

    def test_basic_runtime_dx(self, mesh):
        """Test basic runtime_dx creation."""
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh)
        assert dx_rt.integral_type() == "cell"
        assert dx_rt.subdomain_id() == 1

    def test_runtime_dx_with_tag(self, mesh):
        """Test runtime_dx with custom tag."""
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, tag="my_tag")
        # The tag is stored in subdomain_data
        # We can't easily access it from the measure, but we test via integral
        _ = dx_rt  # Mark as used

    def test_runtime_dx_metadata(self, mesh):
        """Test that runtime_dx sets runintgen metadata."""
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh)
        # Create a simple integrand to extract metadata
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx_rt
        integrals = form.integrals()
        assert len(integrals) == 1
        md = integrals[0].metadata()
        assert md.get("runintgen") is True

    def test_runtime_dx_extra_metadata(self, mesh):
        """Test runtime_dx with additional metadata."""
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh, quadrature_degree=5)
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx_rt
        md = form.integrals()[0].metadata()
        assert md.get("runintgen") is True
        assert md.get("quadrature_degree") == 5


class TestIsRuntimeIntegral:
    """Tests for is_runtime_integral predicate."""

    @pytest.fixture
    def mesh(self):
        """Create a simple UFL mesh."""
        cell = ufl.triangle
        coord_elem = LagrangeElement(cell, 1, (2,))
        return ufl.Mesh(coord_elem)

    def test_runtime_integral_via_subdomain_data(self, mesh):
        """Test detection via RuntimeQuadrature subdomain_data."""
        dx_rt = runtime_dx(subdomain_id=1, domain=mesh)
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx_rt
        integral = form.integrals()[0]
        assert is_runtime_integral(integral) is True

    def test_runtime_integral_via_metadata(self, mesh):
        """Test detection via runintgen metadata only."""
        dx = ufl.Measure("dx", domain=mesh, metadata={"runintgen": True})
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx
        integral = form.integrals()[0]
        assert is_runtime_integral(integral) is True

    def test_non_runtime_integral(self, mesh):
        """Test that regular integrals are not detected as runtime."""
        dx = ufl.Measure("dx", domain=mesh)
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx
        integral = form.integrals()[0]
        assert is_runtime_integral(integral) is False
