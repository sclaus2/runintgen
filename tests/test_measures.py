"""Tests for runintgen.measures module."""

from __future__ import annotations

import pytest
import ufl
from ufl.finiteelement import AbstractFiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

from runintgen.measures import (
    RUNTIME_QUADRATURE_RULE,
    RuntimeIntegralMode,
    RuntimeMeasure,
    dxq,
    get_quadrature_provider,
    is_runtime_integral,
    runtime_integral_mode,
    runtime_measure,
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

    def test_runtime_integral_via_quadrature_subdomain_data(self, mesh):
        """Test detection via a quadrature rule in subdomain_data."""

        class Quadrature:
            points = [(0.25, 0.25)]
            weights = [0.5]

        quadrature = Quadrature()
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=quadrature)
        x = ufl.SpatialCoordinate(mesh)
        integral = (x[0] * dx).integrals()[0]

        assert is_runtime_integral(integral) is True
        assert runtime_integral_mode(integral) is RuntimeIntegralMode.RUNTIME
        assert get_quadrature_provider(integral) is quadrature

    def test_mixed_subdomain_data_detects_quadrature_payload(self, mesh):
        """Test mixed entity/quadrature subdomain data marks runtime."""

        class Quadrature:
            points = [(0.25, 0.25)]
            weights = [0.5]

        quadrature = Quadrature()
        entities = [0, 2, 4]
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            subdomain_data=[(0, entities), (0, quadrature)],
        )
        x = ufl.SpatialCoordinate(mesh)
        integral = (x[0] * dx).integrals()[0]

        assert is_runtime_integral(integral) is True
        assert runtime_integral_mode(integral) is RuntimeIntegralMode.MIXED
        assert get_quadrature_provider(integral) == [(0, entities), (0, quadrature)]

    def test_direct_mixed_subdomain_data_list_marks_runtime(self, mesh):
        """Test [standard_entities, runtime_rule] marks mixed mode."""

        class Quadrature:
            points = [(0.25, 0.25)]
            weights = [0.5]

        quadrature = Quadrature()
        entities = [1, 4, 5, 6]
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=[entities, quadrature])
        x = ufl.SpatialCoordinate(mesh)
        integral = (x[0] * dx).integrals()[0]

        assert is_runtime_integral(integral) is True
        assert runtime_integral_mode(integral) is RuntimeIntegralMode.MIXED

    def test_entity_only_subdomain_data_is_not_runtime(self, mesh):
        """Test ordinary entity lists do not request runtime codegen."""
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=[(0, [0, 2, 4])])
        x = ufl.SpatialCoordinate(mesh)
        integral = (x[0] * dx).integrals()[0]

        assert is_runtime_integral(integral) is False
        assert runtime_integral_mode(integral) is RuntimeIntegralMode.STANDARD

    def test_dxq_preserves_marker_when_degree_is_set(self, mesh):
        """Test runtime metadata survives UFL measure reconfiguration."""
        quadrature_rule = {"type": "runtime_rule"}
        dx0 = dxq(
            domain=mesh,
            subdomain_id=1,
            subdomain_data=quadrature_rule,
            metadata={"custom": "kept"},
        )
        dx1 = dx0(degree=5)

        assert isinstance(dx1, RuntimeMeasure)
        assert dx1.subdomain_data() is quadrature_rule
        assert dx1.metadata()["quadrature_rule"] == RUNTIME_QUADRATURE_RULE
        assert dx1.metadata()["quadrature_degree"] == 5
        assert dx1.metadata()["custom"] == "kept"

        x = ufl.SpatialCoordinate(mesh)
        integral = (x[0] * dx1).integrals()[0]
        assert is_runtime_integral(integral) is True
        assert get_quadrature_provider(integral) is quadrature_rule

    def test_runtime_measure_rejects_ambiguous_provider_arguments(self, mesh):
        """Test provider aliases cannot both be supplied."""
        with pytest.raises(ValueError, match="either quadrature_provider"):
            runtime_measure(
                "dx",
                domain=mesh,
                quadrature_provider=object(),
                subdomain_data=object(),
            )

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
