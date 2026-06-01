"""
Quadrature rules for averaging over polygons.

Based on the quadrature module from FreeGS
(https://github.com/freegs-plasma/freegs), modified for FORGE.

Note: integration weights are set so that the sum of weights is 1, giving
the average of a function over the polygon rather than the integral.

Copyright 2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2025-2026 Chris Marsden

This file is part of FORGE.

FORGE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORGE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FORGE.  If not, see <http://www.gnu.org/licenses/>.
"""
import shapely


def triangle_quad(triangle, n=6):
    """Creates quadrature evaluation points.

    Given a triangle, calculates the evaluation points and weights.
    Coefficients taken from http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf
    Joseph E. Flaherty course notes, Rensselaer Polytechnic Institute

    Parameters
    ----------
    triangle : list
        List of points defining the triangle of the form [(r1,z1), (r2,z2), (r3,z3)].
    n : int
        Number of quadrature points, currently; 1, 3 or 6.

    Returns
    -------
    Evaluation points and weights : list
        A list of evaluation points and their weights of the form [(r,z,weight),...]
    """

    assert len(triangle) >= 3

    r1, z1 = triangle[0]
    r2, z2 = triangle[1]
    r3, z3 = triangle[2]

    if n == 1:
        # One point in the middle of the triangle
        return [((r1 + r2 + r3) / 3, (z1 + z2 + z3) / 3, 1.0)]

    elif n == 3:
        return [
            ((4 * r1 + r2 + r3) / 6, (4 * z1 + z2 + z3) / 6, 1.0 / 3),
            ((r1 + 4 * r2 + r3) / 6, (z1 + 4 * z2 + z3) / 6, 1.0 / 3),
            ((r1 + r2 + 4 * r3) / 6, (z1 + z2 + 4 * z3) / 6, 1.0 / 3),
        ]

    elif n == 6:
        a = 0.816847572980459
        b = 0.5 * (1.0 - a)

        c = 0.108103018168070
        d = 0.5 * (1.0 - c)

        return [
            ((a * r1 + b * r2 + b * r3), (a * z1 + b * z2 + b * z3), 0.109951743655322),
            ((b * r1 + a * r2 + b * r3), (b * z1 + a * z2 + b * z3), 0.109951743655322),
            ((b * r1 + b * r2 + a * r3), (b * z1 + b * z2 + a * z3), 0.109951743655322),
            ((c * r1 + d * r2 + d * r3), (c * z1 + d * z2 + d * z3), 0.223381589678011),
            ((d * r1 + c * r2 + d * r3), (d * z1 + c * z2 + d * z3), 0.223381589678011),
            ((d * r1 + d * r2 + c * r3), (d * z1 + d * z2 + c * z3), 0.223381589678011),
        ]
    else:
        raise ValueError("Quadrature not available for n={}".format(n))


def polygon_quad(polygon, n=6):
    """Calculates quadrature points for an arbitary polygon.

    A polygon is provided, which is meshed into a triangular mesh, with
    the quadrature points and weights for these triangles then calculated.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon object
        A Shapely Polygon to be evaluated.

    Returns
    -------
    quadrature : list
        A list of evaluation points and their weights of the form [(r,z,weight),...]
    """

    # Split polygon into triangles
    tri = shapely.constrained_delaunay_triangles(polygon)

    # List of triangle Polygon objects
    triangles = [tri.geoms[i] for i in range(len(tri.geoms))]

    # Calculate the area of each triangle, to get a relative weighting
    areas = [triangle.area for triangle in triangles]
    total_area = sum(areas)

    quadrature = []  # List of all points
    for triangle, area in zip(triangles, areas):
        triangle_shape = list(triangle.exterior.coords)
        points = triangle_quad(triangle_shape, n=n)  # Quadrature points for this triangle
        quadrature += [
            (r, z, w * area / total_area) for r, z, w in points
        ]  # Modify the weights

    return quadrature


def average(func, quad):
    """Average func(r,z) using given quadrature."""
    return sum(func(r, z) * w for r, z, w in quad)
