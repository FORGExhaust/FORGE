"""Tests for ``forge.quadrature`` — quadrature rules over polygons."""

import numpy as np
import pytest
import shapely.geometry

from forge.quadrature import average, polygon_quad, triangle_quad


class TestTriangleQuad:
    """Unit tests for ``triangle_quad``."""

    # A simple right triangle
    TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

    @pytest.mark.parametrize("n", [1, 3, 6])
    def test_returns_correct_number_of_points(self, n):
        pts = triangle_quad(self.TRIANGLE, n=n)
        assert len(pts) == n

    @pytest.mark.parametrize("n", [1, 3, 6])
    def test_weights_sum_to_one(self, n):
        pts = triangle_quad(self.TRIANGLE, n=n)
        total_weight = sum(w for _, _, w in pts)
        np.testing.assert_allclose(total_weight, 1.0, atol=1e-14)

    @pytest.mark.parametrize("n", [1, 3, 6])
    def test_points_lie_inside_triangle(self, n):
        """Quadrature points should be inside the triangle."""
        tri_poly = shapely.geometry.Polygon(self.TRIANGLE)
        pts = triangle_quad(self.TRIANGLE, n=n)
        for r, z, _ in pts:
            point = shapely.geometry.Point(r, z)
            assert tri_poly.contains(point) or tri_poly.touches(point)

    def test_centroid_for_n1(self):
        """n=1 should return the centroid of the triangle."""
        pts = triangle_quad(self.TRIANGLE, n=1)
        r, z, w = pts[0]
        np.testing.assert_allclose(r, 1.0 / 3, atol=1e-14)
        np.testing.assert_allclose(z, 1.0 / 3, atol=1e-14)
        np.testing.assert_allclose(w, 1.0, atol=1e-14)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="Quadrature not available"):
            triangle_quad(self.TRIANGLE, n=5)


class TestPolygonQuad:
    """Unit tests for ``polygon_quad``."""

    def test_square_weights_sum_to_one(self):
        """Weights of a simple unit square should sum to 1."""
        square = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        pts = polygon_quad(square, n=6)
        total_weight = sum(w for _, _, w in pts)
        np.testing.assert_allclose(total_weight, 1.0, atol=1e-12)

    def test_points_inside_polygon(self):
        """All quadrature points should lie inside the polygon."""
        hexagon = shapely.geometry.Point(1.0, 0.0).buffer(0.5, resolution=6)
        pts = polygon_quad(hexagon, n=6)
        for r, z, _ in pts:
            point = shapely.geometry.Point(r, z)
            assert hexagon.contains(point) or hexagon.boundary.distance(point) < 1e-10

    @pytest.mark.parametrize("n", [1, 3, 6])
    def test_constant_function_averages_to_constant(self, n):
        """Average of a constant function f(r,z) = C should be C."""
        square = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        pts = polygon_quad(square, n=n)
        C = 42.0
        result = average(lambda r, z: C, pts)
        np.testing.assert_allclose(result, C, atol=1e-10)

    def test_linear_function_average_on_square(self):
        """Average of f(r,z) = r over a unit square [0,1]x[0,1] should be 0.5."""
        square = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        pts = polygon_quad(square, n=6)
        result = average(lambda r, z: r, pts)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)


class TestAverage:
    """Tests for the ``average`` helper."""

    def test_single_point_quadrature(self):
        """Single-point quad: average equals function at that point."""
        quad = [(2.0, 3.0, 1.0)]
        result = average(lambda r, z: r + z, quad)
        np.testing.assert_allclose(result, 5.0)

    def test_weighted_average(self):
        """Check manual weighted sum."""
        quad = [(1.0, 0.0, 0.5), (2.0, 0.0, 0.5)]
        result = average(lambda r, z: r, quad)
        np.testing.assert_allclose(result, 1.5)
