"""Tests for runintgen.measures module."""

from __future__ import annotations

import pytest

import ufl
from ufl.finiteelement import AbstractFiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

from runintgen.measures import (
    RUNTIME_QUADRATURE_RULE,
    get_quadrature_provider,
    is_runtime_integral,
)


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


class TestIsRuntimeIntegral:
    """Tests for is_runtime_integral predicate."""

    @pytest.fixture
    def mesh(self):
        """Create a simple UFL mesh."""
        cell = ufl.triangle
        coord_elem = LagrangeElement(cell, 1, (2,))
        return ufl.Mesh(coord_elem)

    def test_runtime_integral_via_quadrature_rule(self, mesh):
        """Test detection via quadrature_rule='runtime' metadata."""
        dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_rule": "runtime"})
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx
        integral = form.integrals()[0]
        assert is_runtime_integral(integral) is True

    def test_runtime_integral_with_subdomain_data_and_metadata(self, mesh):
        """Test the recommended API with subdomain_data and metadata."""
        # This is the recommended way to create runtime integrals
        fake_quadrature_provider = {"type": "custom_quadrature"}
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=fake_quadrature_provider,
            metadata={"quadrature_rule": "runtime"},
        )
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx
        integral = form.integrals()[0]
        assert is_runtime_integral(integral) is True

        # Check that we can get the provider back
        provider = get_quadrature_provider(integral)
        assert provider == fake_quadrature_provider

    def test_non_runtime_integral(self, mesh):
        """Test that regular integrals are not detected as runtime."""
        dx = ufl.Measure("dx", domain=mesh)
        x = ufl.SpatialCoordinate(mesh)
        form = x[0] * dx
        integral = form.integrals()[0]
        assert is_runtime_integral(integral) is False

    def test_runtime_quadrature_rule_constant(self):
        """Test the RUNTIME_QUADRATURE_RULE constant."""
        assert RUNTIME_QUADRATURE_RULE == "runtime"
