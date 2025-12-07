"""Tests for geometry mapping utilities."""

import numpy as np
import pytest

from runintgen.geometry import (
    generate_subcell_quadrature,
    map_quadrature_to_subcell,
    map_subcell_to_parent_reference,
    scale_weights_by_jacobian,
)


class TestQuadratureMapping2D:
    """Tests for 2D quadrature mapping functions."""

    def test_map_quadrature_to_subcell_identity(self):
        """Test mapping with identity transform (reference = physical)."""
        # Reference triangle vertices
        ref_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        # Quadrature points on reference triangle
        points = np.array([[0.25, 0.25], [0.5, 0.25], [0.25, 0.5]], dtype=np.float64)

        # Mapping to identical physical triangle should give same points
        mapped = map_quadrature_to_subcell(points, ref_vertices)
        np.testing.assert_allclose(mapped, points, rtol=1e-14)

    def test_map_quadrature_to_subcell_scaled(self):
        """Test mapping to a scaled triangle."""
        # Scaled triangle (factor of 2)
        vertices = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)

        points = np.array([[0.25, 0.25]], dtype=np.float64)

        # Expect point to be scaled
        mapped = map_quadrature_to_subcell(points, vertices)
        expected = np.array([[0.5, 0.5]])
        np.testing.assert_allclose(mapped, expected, rtol=1e-14)

    def test_map_quadrature_to_subcell_translated(self):
        """Test mapping to a translated triangle."""
        # Translated triangle (offset by [1, 2])
        vertices = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 3.0]], dtype=np.float64)

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        mapped = map_quadrature_to_subcell(points, vertices)
        expected = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 3.0]])
        np.testing.assert_allclose(mapped, expected, rtol=1e-14)

    def test_map_subcell_to_parent_reference_inverse(self):
        """Test that map_subcell_to_parent is inverse of map_quadrature_to_subcell."""
        # Physical subcell vertices
        subcell_vertices = np.array(
            [[0.5, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64
        )

        # Parent cell vertices
        parent_vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
        )

        # Start with reference points
        ref_points = np.array([[0.25, 0.25], [0.5, 0.25]], dtype=np.float64)

        # Map to physical subcell
        phys_points = map_quadrature_to_subcell(ref_points, subcell_vertices)

        # Map back to parent reference
        parent_ref_points = map_subcell_to_parent_reference(
            phys_points, parent_vertices
        )

        # For identity parent (reference = physical), parent_ref should equal phys
        np.testing.assert_allclose(parent_ref_points, phys_points, rtol=1e-14)

    def test_scale_weights_by_jacobian_unit(self):
        """Test weight scaling for unit triangle (det = 1)."""
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

        # Unit reference triangle has Jacobian det = 1
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        scaled = scale_weights_by_jacobian(weights, vertices)
        np.testing.assert_allclose(scaled, weights, rtol=1e-14)

    def test_scale_weights_by_jacobian_scaled(self):
        """Test weight scaling for scaled triangle."""
        weights = np.array([0.5], dtype=np.float64)

        # Triangle scaled by factor 2 has Jacobian det = 4
        vertices = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)

        scaled = scale_weights_by_jacobian(weights, vertices)
        expected = np.array([2.0])  # 0.5 * 4 = 2.0
        np.testing.assert_allclose(scaled, expected, rtol=1e-14)


class TestSubcellQuadrature:
    """Tests for subcell quadrature generation."""

    def test_generate_single_subcell(self):
        """Test quadrature generation for a single subcell."""
        # Reference quadrature points and weights
        ref_points = np.array([[0.25, 0.25]], dtype=np.float64)
        ref_weights = np.array([0.5], dtype=np.float64)

        # Subcell is half of the parent triangle
        subcell_vertices = [
            np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
        ]

        # Parent is unit triangle (reference = physical)
        parent_vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
        )

        mapped_pts, scaled_wts = generate_subcell_quadrature(
            ref_points, ref_weights, subcell_vertices, parent_vertices
        )

        # Check that we got one point
        assert mapped_pts.shape == (1, 2)
        assert scaled_wts.shape == (1,)

        # Weights should be scaled by subcell Jacobian determinant (0.25 for half triangle)
        np.testing.assert_allclose(scaled_wts, [0.125], rtol=1e-14)

    def test_generate_multiple_subcells(self):
        """Test quadrature generation for multiple subcells covering parent."""
        # Two triangular subcells covering the parent triangle
        ref_points = np.array([[1 / 3, 1 / 3]], dtype=np.float64)
        ref_weights = np.array([0.5], dtype=np.float64)

        # Two subcells that together form the parent
        subcell_vertices = [
            np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float64),
            np.array([[0.5, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64),
        ]

        parent_vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
        )

        mapped_pts, scaled_wts = generate_subcell_quadrature(
            ref_points, ref_weights, subcell_vertices, parent_vertices
        )

        # Should have 2 points (one per subcell)
        assert mapped_pts.shape == (2, 2)
        assert scaled_wts.shape == (2,)

    def test_generate_preserves_dtype_float64(self):
        """Test that float64 dtype is preserved."""
        ref_points = np.array([[0.25, 0.25]], dtype=np.float64)
        ref_weights = np.array([0.5], dtype=np.float64)
        subcell_vertices = [
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        ]
        parent_vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
        )

        mapped_pts, scaled_wts = generate_subcell_quadrature(
            ref_points, ref_weights, subcell_vertices, parent_vertices
        )

        assert mapped_pts.dtype == np.float64
        assert scaled_wts.dtype == np.float64

    def test_generate_preserves_dtype_float32(self):
        """Test that float32 dtype is preserved."""
        ref_points = np.array([[0.25, 0.25]], dtype=np.float32)
        ref_weights = np.array([0.5], dtype=np.float32)
        subcell_vertices = [
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        ]
        parent_vertices = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )

        mapped_pts, scaled_wts = generate_subcell_quadrature(
            ref_points, ref_weights, subcell_vertices, parent_vertices
        )

        assert mapped_pts.dtype == np.float32
        assert scaled_wts.dtype == np.float32


class TestQuadratureMapping3D:
    """Tests for 3D quadrature mapping functions."""

    def test_map_quadrature_to_subcell_3d_identity(self):
        """Test 3D mapping with identity transform."""
        # Reference tetrahedron vertices
        ref_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # Centroid point
        points = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)

        mapped = map_quadrature_to_subcell(points, ref_vertices)
        np.testing.assert_allclose(mapped, points, rtol=1e-14)

    def test_map_quadrature_to_subcell_3d_scaled(self):
        """Test 3D mapping with scaled tetrahedron."""
        # Scaled by factor 2
        vertices = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=np.float64,
        )

        points = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)

        mapped = map_quadrature_to_subcell(points, vertices)
        expected = np.array([[0.5, 0.5, 0.5]])
        np.testing.assert_allclose(mapped, expected, rtol=1e-14)

    def test_scale_weights_3d_scaled(self):
        """Test 3D weight scaling for scaled tetrahedron."""
        weights = np.array([1.0], dtype=np.float64)

        # Tetrahedron scaled by 2 has Jacobian det = 8
        vertices = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=np.float64,
        )

        scaled = scale_weights_by_jacobian(weights, vertices)
        expected = np.array([8.0])
        np.testing.assert_allclose(scaled, expected, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
