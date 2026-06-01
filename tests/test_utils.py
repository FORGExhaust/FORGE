"""Tests for ``forge.utils`` — utility / helper functions.

Only the non-interactive (non-GUI) utilities are tested here.
Interactive matplotlib-based tools (shape editor, drawing, etc.) are
excluded because they require user input.
"""

import math

import numpy as np
import pytest
from shapely.geometry import LineString

from forge.utils import (
    calc_winding_number,
    closest_point_along_shape,
    densify_closed_shape,
    estimate_xpoint_location,
    filter_distant_points,
    grid_points_inside_linestring,
    magnitude_scale_factors,
    orthogonalised_convex_hull_from_rects,
    reflect_and_join_shape,
)


# ---------------------------------------------------------------------------
# filter_distant_points
# ---------------------------------------------------------------------------


class TestFilterDistantPoints:

    def test_filters_correctly(self):
        x = np.array([0.0, 1.0, 5.0, 10.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        fx, fy = filter_distant_points(x, y, (0, 0), 2.0)
        # Only points at distance > 2 should remain: (5,0) and (10,0)
        assert len(fx) == 2
        assert 5.0 in fx
        assert 10.0 in fx

    def test_no_points_remain(self):
        x = np.array([0.0, 0.1])
        y = np.array([0.0, 0.0])
        fx, fy = filter_distant_points(x, y, (0, 0), 100.0)
        assert len(fx) == 0

    def test_all_points_remain(self):
        x = np.array([10.0, 20.0])
        y = np.array([0.0, 0.0])
        fx, fy = filter_distant_points(x, y, (0, 0), 1.0)
        assert len(fx) == 2


# ---------------------------------------------------------------------------
# magnitude_scale_factors
# ---------------------------------------------------------------------------


class TestMagnitudeScaleFactors:

    def test_single_value(self):
        assert magnitude_scale_factors(5374) == 1000

    def test_small_value(self):
        assert magnitude_scale_factors(0.0056) == 0.001

    def test_zero_returns_zero(self):
        assert magnitude_scale_factors(0) == 0

    def test_list_input(self):
        result = magnitude_scale_factors([100, 0.05])
        assert result == [100, 0.01]

    def test_numpy_input(self):
        result = magnitude_scale_factors(np.array([100.0, 0.05]))
        np.testing.assert_array_equal(result, np.array([100.0, 0.01]))

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            magnitude_scale_factors("hello")


# ---------------------------------------------------------------------------
# reflect_and_join_shape
# ---------------------------------------------------------------------------


class TestReflectAndJoinShape:

    def test_basic_reflection(self):
        # A simple L-shape below y=0
        points = [(1.0, -1.0), (2.0, -1.0), (2.0, -2.0)]
        result = reflect_and_join_shape(points)
        xs = [p[0] for p in result]
        ys = [p[1] for p in result]
        # Should have points below and reflected points above
        assert any(y > 0 for y in ys)  # reflected part
        assert any(y < 0 for y in ys)  # original part

    def test_points_above_zero_discarded(self):
        """Points at or above y=0 should be dropped before reflection."""
        points = [(1.0, 1.0), (2.0, -1.0), (3.0, -2.0)]
        result = reflect_and_join_shape(points)
        # Original includes only y<0 points
        assert len(result) == 4  # 2 below + 2 reflected


# ---------------------------------------------------------------------------
# densify_closed_shape
# ---------------------------------------------------------------------------


class TestDensifyClosedShape:

    def test_square_densification(self):
        """Densifying a unit square with max_dist=0.5 should add intermediate points."""
        x = [0, 1, 1, 0, 0]
        y = [0, 0, 1, 1, 0]
        xd, yd = densify_closed_shape(x, y, max_dist=0.5)
        # Original has 4 edges of length 1; each split into 2 => 8 + 1 (closed) = 9
        assert len(xd) == len(yd)
        assert len(xd) >= 9
        # Should be closed
        assert xd[0] == xd[-1] and yd[0] == yd[-1]

    def test_already_dense_unchanged(self):
        """If edges are shorter than max_dist, no new points are added."""
        x = [0, 0.1, 0.1, 0, 0]
        y = [0, 0, 0.1, 0.1, 0]
        xd, yd = densify_closed_shape(x, y, max_dist=10.0)
        # 4 edges, each < max_dist => 4 points + 1 closed = 5
        assert len(xd) == 5

    def test_open_output(self):
        x = [0, 1, 1, 0, 0]
        y = [0, 0, 1, 1, 0]
        xd, yd = densify_closed_shape(x, y, max_dist=0.5, return_closed=False)
        # Open: first != last
        assert not (xd[0] == xd[-1] and yd[0] == yd[-1]) or len(xd) > 5

    def test_negative_max_dist_raises(self):
        with pytest.raises(ValueError, match="positive"):
            densify_closed_shape([0, 1, 0], [0, 0, 1], max_dist=-1.0)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="At least 3"):
            densify_closed_shape([0, 1], [0, 0], max_dist=0.5)


# ---------------------------------------------------------------------------
# calc_winding_number
# ---------------------------------------------------------------------------


class TestCalcWindingNumber:

    def test_uniform_rotation_gives_one(self):
        """A field that rotates once around (0,0) → winding number = 1."""
        n = 360
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        Br = np.cos(theta)
        Bz = np.sin(theta)
        wn = calc_winding_number(Br, Bz)
        assert wn == 1

    def test_no_rotation_gives_zero(self):
        """Constant field direction → winding number = 0."""
        n = 100
        Br = np.ones(n)
        Bz = np.zeros(n)
        wn = calc_winding_number(Br, Bz)
        assert wn == 0

    def test_double_rotation_gives_two(self):
        n = 720
        theta = np.linspace(0, 4 * np.pi, n, endpoint=False)
        Br = np.cos(theta)
        Bz = np.sin(theta)
        wn = calc_winding_number(Br, Bz)
        assert wn == 2


# ---------------------------------------------------------------------------
# grid_points_inside_linestring
# ---------------------------------------------------------------------------


class TestGridPointsInsideLinestring:

    def test_square_region(self):
        """Points inside a square region should be identified."""
        coords = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
        ls = LineString(coords)
        X, Y = np.meshgrid(np.arange(-1, 4), np.arange(-1, 4), indexing="ij")
        xs_in, ys_in = grid_points_inside_linestring(X, Y, ls)
        # Points at (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2) = 9
        assert len(xs_in) == 9

    def test_no_points_inside(self):
        coords = [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)]
        ls = LineString(coords)
        X, Y = np.meshgrid([0, 1], [0, 1], indexing="ij")
        xs_in, ys_in = grid_points_inside_linestring(X, Y, ls)
        assert len(xs_in) == 0

    def test_shape_mismatch_raises(self):
        ls = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
        X = np.zeros((2, 3))
        Y = np.zeros((3, 2))
        with pytest.raises(ValueError, match="same shape"):
            grid_points_inside_linestring(X, Y, ls)


# ---------------------------------------------------------------------------
# closest_point_along_shape
# ---------------------------------------------------------------------------


class TestClosestPointAlongShape:

    def test_point_on_boundary_returns_itself(self):
        x_coords = [0, 1, 1, 0, 0]
        y_coords = [0, 0, 1, 1, 0]
        cx, cy = closest_point_along_shape(x_coords, y_coords, 0.5, 0.0)
        np.testing.assert_allclose(cx, 0.5, atol=1e-10)
        np.testing.assert_allclose(cy, 0.0, atol=1e-10)

    def test_external_point_projects_to_edge(self):
        x_coords = [0, 1, 1, 0, 0]
        y_coords = [0, 0, 1, 1, 0]
        cx, cy = closest_point_along_shape(x_coords, y_coords, 0.5, -1.0)
        np.testing.assert_allclose(cx, 0.5, atol=1e-10)
        np.testing.assert_allclose(cy, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# estimate_xpoint_location
# ---------------------------------------------------------------------------


class TestEstimateXpointLocation:

    def test_returns_nan_when_no_xpoint(self):
        """When the Jacobian doesn't indicate a nearby X-point, returns NaN."""
        R = [1.0]
        Z = [0.0]
        Br = [0.1]
        Bz = [0.1]
        jac = [np.eye(2) * 100.0]  # Large det, but displacement won't be small enough
        Rx, Zx, found = estimate_xpoint_location(R, Z, Br, Bz, jac, 0.001, 0.001)
        assert not found
        assert np.isnan(Rx)
        assert np.isnan(Zx)

    def test_finds_xpoint_at_zero_field(self):
        """When Br=Bz=0, the X-point is at the evaluation point itself."""
        R = [1.0]
        Z = [0.0]
        Br = [0.0]
        Bz = [0.0]
        jac = [np.array([[1.0, 0.0], [0.0, -1.0]])]  # Hyperbolic null
        Rx, Zx, found = estimate_xpoint_location(R, Z, Br, Bz, jac, 0.1, 0.1)
        assert found
        np.testing.assert_allclose(Rx, 1.0, atol=1e-10)
        np.testing.assert_allclose(Zx, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# orthogonalised_convex_hull_from_rects
# ---------------------------------------------------------------------------


class TestOrthogonalisedConvexHull:

    def test_single_rectangle(self):
        xs, ys = orthogonalised_convex_hull_from_rects(
            [0.0], [0.0], [2.0], [2.0], closed=True
        )
        # Should be a rectangle (4 corners + closure)
        assert len(xs) == len(ys)
        assert xs[0] == xs[-1] and ys[0] == ys[-1]  # closed
        assert len(xs) == 5  # rectangle: 4 unique + 1 close

    def test_empty_input(self):
        xs, ys = orthogonalised_convex_hull_from_rects([], [], [], [])
        assert xs == []
        assert ys == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            orthogonalised_convex_hull_from_rects([0], [0], [1], [])

    def test_result_is_closed_by_default(self):
        xs, ys = orthogonalised_convex_hull_from_rects(
            [0.0, 2.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]
        )
        assert xs[0] == xs[-1] and ys[0] == ys[-1]
