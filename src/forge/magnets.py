"""Description of the magnets (PF coils) that make up the tokamak.

Based on the Coil, ShapedCoil, Solenoid and Circuit classes from FreeGS
(https://github.com/freegs-plasma/freegs), with modifications for FORGE
including the addition of the FilamentPointCoil class and JSON serialisation.

Copyright 2016-2022 FreeGS contributors
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

import logging

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from forge import greens
from forge.quadrature import polygon_quad

logger = logging.getLogger(__name__)


class Coil:
    """Represents a poloidal field coil as a point source of current.

    Parameters
    ----------
    R : float
        R coordinate of the coil.
    Z : float
        Z coordinate of the coil.
    name : str
        Name of the coil.
    current : float
        Per-turn current in the coil (A).
    turns : int
        Number of turns. Total coil current is current * turns.
    """

    def __init__(
        self,
        R,
        Z,
        name=None,
        current=0.0,
        turns=1,
    ):
        self.R = R
        self.Z = Z

        self.name = name
        self.current = current
        self.turns = turns

        # Colours used to plot the coil
        self.fill_colour = "orange"
        self.edge_colour = "grey"

    def control_psi(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z) from a unit current through the coil."""
        # Multipy by turns so that total current is current * turns
        return greens.Greens(self.R,self.Z,R,Z) * self.turns

    def control_Br(self, R, Z):
        """Calculates the radial magnetic field at (R,Z) from a unit current through the coil."""
        # Multipy by turns so that total current is current * turns
        return greens.Greens_Br(self.R,self.Z,R,Z) * self.turns

    def control_Bz(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z) from a unit current through the coil."""
        # Multipy by turns so that total current is current * turns
        return greens.Greens_Bz(self.R,self.Z,R,Z) * self.turns

    def control_dBp(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z) from a unit current through the coil.

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """

        if deriv == "dBr_dZ":
            return greens.Greens_dBr_dZ(self.R,self.Z,R,Z) * self.turns

        elif deriv == "dBr_dR":
            return greens.Greens_dBr_dR(self.R,self.Z,R,Z) * self.turns

        elif deriv == "dBz_dZ":
            return greens.Greens_dBz_dZ(self.R,self.Z,R,Z) * self.turns

        elif deriv == "dBz_dR":
            return greens.Greens_dBz_dR(self.R,self.Z,R,Z) * self.turns

        else:
            raise ValueError(f"Unknown derivative '{deriv}'. Expected one of: dBr_dZ, dBr_dR, dBz_dZ, dBz_dR.")

    def control_Bp_jacobian(self,R,Z):
        """Computes the 2x2 Jacobian matrix of poloidal field about the point (R,Z) from a unit current through the coil.

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        a = self.control_dBp(R,Z,deriv="dBr_dR")
        b = self.control_dBp(R,Z,deriv="dBr_dZ")
        c = self.control_dBp(R,Z,deriv="dBz_dR")
        d = self.control_dBp(R,Z,deriv="dBz_dZ")

        return np.array([[a,b],[c,d]])

    def psiRZ(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z)."""
        return self.control_psi(R, Z) * self.current

    def BrRZ(self, R, Z):
        """Calculates the radial magnetic field at (R,Z)."""
        return self.control_Br(R, Z) * self.current

    def BzRZ(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z)."""
        return self.control_Bz(R, Z) * self.current

    def dBpRZ(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z).

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """
        return self.control_dBp(R, Z, deriv) * self.current

    def BpRZ_jacobian(self, R, Z):
        """Computes the 2x2 Jacobian matrix of the poloidal field about the point (R,Z).

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        J = self.control_Bp_jacobian(R,Z)

        return J * self.current

    def set_fill_colour(self, fill_colour):
        """Changes the colour used to fill the coil when plotted."""

        self.fill_colour = fill_colour

    def set_edge_colour(self, edge_colour):
        """Changes the colour of the edge of the coil when plotted."""

        self.edge_colour = edge_colour

    def plot(self, ax=None):
        """Plots the coil.

        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot onto. If None, these will be created.

        Returns
        -------
        ax
            matplotlib axes containing the plot of the machine.
        """

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')

        ax.scatter(
            self.R,
            self.Z,
            marker='o',
            color=self.fill_colour,
            edgecolors=self.edge_colour,
            zorder=2
            )

        return ax

    def to_dict(self):
        """Return a JSON-serialisable dictionary matching the magnets JSON schema."""
        return {
            "type": "point",
            "R": float(self.R),
            "Z": float(self.Z),
            "turns": int(self.turns),
            "current": float(self.current),
        }

class ShapedCoil(Coil):
    """Represents a poloidal field coil with a polygonal shape.

    The coil's shape is represented as a triangular mesh, with Gaussian qadrature
    used to represent the distribution of current throughout the cross section of
    the coil.

    Parameters
    ----------
    shape : list
        Outline of the coil shape as a list of points [(r1,z1), (r2,z2), ...].
        Must have more than two points.
    name : str
        Name of the coil.
    current : float
        Per-turn current in the coil (A).
    turns : int
        Number of turns. Total coil current is current * turns.
    npoints : int
        Number of quadrature points to use.
    """

    def __init__(self,
        shape,
        name=None,
        current=0.0,
        turns=1,
        npoints=6
    ):
        assert len(shape) > 2

        self.current = current
        self.turns = turns
        self.shape = shape
        self.name = name

        self.polygon = Polygon(self.shape)
        self.area = self.polygon.area

        # The quadrature points to be used
        self.npoints_per_triangle = npoints
        self.quad_points = polygon_quad(self.polygon, n=npoints)

        # R,Z centre
        self.R = self.polygon.centroid.x
        self.Z = self.polygon.centroid.y

        # R,Z points
        self.R_points = [point[0] for point in self.shape]
        self.Z_points = [point[1] for point in self.shape]

        # Colours used to plot the coil
        self.fill_colour = "orange"
        self.edge_colour = "grey"

    def control_psi(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil, weight in self.quad_points:
            result += greens.Greens(R_fil, Z_fil, R, Z) * weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_Br(self, R, Z):
        """Calculates the radial magnetic field at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil, weight in self.quad_points:
            result += greens.Greens_Br(R_fil, Z_fil, R, Z) * weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_Bz(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil, weight in self.quad_points:
            result += greens.Greens_Bz(R_fil, Z_fil, R, Z) * weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_dBp(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z) from a unit current through the coil.

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """

        result = 0.0

        if deriv == "dBr_dZ":

            for R_fil, Z_fil, weight in self.quad_points:
                result += greens.Greens_dBr_dZ(R_fil,Z_fil,R,Z) * weight

        elif deriv == "dBr_dR":

            for R_fil, Z_fil, weight in self.quad_points:
                result += greens.Greens_dBr_dR(R_fil,Z_fil,R,Z) * weight

        elif deriv == "dBz_dZ":

            for R_fil, Z_fil, weight in self.quad_points:
                result += greens.Greens_dBz_dZ(R_fil,Z_fil,R,Z) * weight

        elif deriv == "dBz_dR":

            for R_fil, Z_fil, weight in self.quad_points:
                result += greens.Greens_dBz_dR(R_fil,Z_fil,R,Z) * weight

        else:
            raise ValueError(f"Unknown derivative '{deriv}'. Expected one of: dBr_dZ, dBr_dR, dBz_dZ, dBz_dR.")

        return result * self.turns

    def control_Bp_jacobian(self,R,Z):
        """Computes the 2x2 Jacobian matrix of poloidal field about the point (R,Z) from a unit current through the coil.

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        a = self.control_dBp(R,Z,deriv="dBr_dR")
        b = self.control_dBp(R,Z,deriv="dBr_dZ")
        c = self.control_dBp(R,Z,deriv="dBz_dR")
        d = self.control_dBp(R,Z,deriv="dBz_dZ")

        return np.array([[a,b],[c,d]])

    def psiRZ(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z)."""
        return self.control_psi(R, Z) * self.current

    def BrRZ(self, R, Z):
        """Calculates the radial magnetic field at (R,Z)."""
        return self.control_Br(R, Z) * self.current

    def BzRZ(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z)."""
        return self.control_Bz(R, Z) * self.current

    def dBpRZ(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z).

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """
        return self.control_dBp(R, Z, deriv) * self.current

    def BpRZ_jacobian(self, R, Z):
        """Computes the 2x2 Jacobian matrix of the poloidal field about the point (R,Z).

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        J = self.control_Bp_jacobian(R,Z)

        return J * self.current

    def set_fill_colour(self, fill_colour):
        """Changes the colour used to fill the coil when plotted."""

        self.fill_colour = fill_colour

    def set_edge_colour(self, edge_colour):
        """Changes the colour of the edge of the coil when plotted."""

        self.edge_colour = edge_colour

    def plot(self, ax=None):
        """Plots the coil.

        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot onto. If None, these will be created.

        Returns
        -------
        ax
            matplotlib axes containing the plot of the machine.
        """

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')

        R = self.R_points
        Z = self.Z_points

        ax.fill(
            R,
            Z,
            color=self.fill_colour,
            zorder=2
            )

        ax.plot(
            R,
            Z,
            color=self.edge_colour,
            zorder=2,
            linewidth=0.5,
            )

        return ax

    def to_dict(self):
        """Return a JSON-serialisable dictionary matching the magnets JSON schema."""
        return {
            "type": "shaped",
            "R": [float(p[0]) for p in self.shape],
            "Z": [float(p[1]) for p in self.shape],
            "turns": int(self.turns),
            "current": float(self.current),
        }

class Solenoid:
    """Represents a central solenoid.

    The solenoid has no radial thickness, and is represented by a series of point
    sources spread across the vertical extent of the solenoid.

    Parameters
    ----------
    R : float
        R coordinate of the solenoid.
    Z_min : float
        Minimum Z coordinate of the solenoid.
    Z_max : float
        Minimum Z coordinate of the solenoid.
    name : str
        Name of the coil.
    current : float
        Per-turn current in the coil (A).
    turns : int
        Number of turns. Total coil current is current * turns.
    npoints : int
        Number of point sources of current to be spread evenly along
        the vertical extent of the solenoid.
    """

    def __init__(
        self,
        R,
        Z_min,
        Z_max,
        name=None,
        current=0.0,
        turns=1,
        npoints=51,
    ):
        self.R = R
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.npoints = int(npoints)
        self.turns = turns
        self.current = current
        self.name = name

        # Populate the point sources along the length of the solenoid
        self.Z_points = np.linspace(self.Z_min,self.Z_max,self.npoints,endpoint=True).tolist()
        self.R_points = [self.R for i in range(self.npoints)]

        # Weights for the point sources (all have the same weight)
        self.weight = 1 / self.npoints

        # Colours used to plot the coil
        self.fill_colour = "orange"
        self.edge_colour = "grey"

    def control_psi(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil in zip(self.R_points,self.Z_points):
            result += greens.Greens(R_fil, Z_fil, R, Z) * self.weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_Br(self, R, Z):
        """Calculates the radial magnetic field at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil in zip(self.R_points,self.Z_points):
            result += greens.Greens_Br(R_fil, Z_fil, R, Z) * self.weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_Bz(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil in zip(self.R_points,self.Z_points):
            result += greens.Greens_Bz(R_fil, Z_fil, R, Z) * self.weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_dBp(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z) from a unit current through the coil.

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """

        result = 0.0

        if deriv == "dBr_dZ":

            for R_fil, Z_fil in zip(self.R_points,self.Z_points):
                result += greens.Greens_dBr_dZ(R_fil,Z_fil,R,Z) * self.weight

        elif deriv == "dBr_dR":

            for R_fil, Z_fil in zip(self.R_points,self.Z_points):
                result += greens.Greens_dBr_dR(R_fil,Z_fil,R,Z) * self.weight

        elif deriv == "dBz_dZ":

            for R_fil, Z_fil in zip(self.R_points,self.Z_points):
                result += greens.Greens_dBz_dZ(R_fil,Z_fil,R,Z) * self.weight

        elif deriv == "dBz_dR":

            for R_fil, Z_fil in zip(self.R_points,self.Z_points):
                result += greens.Greens_dBz_dR(R_fil,Z_fil,R,Z) * self.weight

        else:
            raise ValueError(f"Unknown derivative '{deriv}'. Expected one of: dBr_dZ, dBr_dR, dBz_dZ, dBz_dR.")

        return result * self.turns

    def control_Bp_jacobian(self,R,Z):
        """Computes the 2x2 Jacobian matrix of poloidal field about the point (R,Z) from a unit current through the coil.

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        a = self.control_dBp(R,Z,deriv="dBr_dR")
        b = self.control_dBp(R,Z,deriv="dBr_dZ")
        c = self.control_dBp(R,Z,deriv="dBz_dR")
        d = self.control_dBp(R,Z,deriv="dBz_dZ")

        return np.array([[a,b],[c,d]])

    def psiRZ(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z)."""
        return self.control_psi(R, Z) * self.current

    def BrRZ(self, R, Z):
        """Calculates the radial magnetic field at (R,Z)."""
        return self.control_Br(R, Z) * self.current

    def BzRZ(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z)."""
        return self.control_Bz(R, Z) * self.current

    def dBpRZ(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z).

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """
        return self.control_dBp(R, Z, deriv) * self.current

    def BpRZ_jacobian(self, R, Z):
        """Computes the 2x2 Jacobian matrix of the poloidal field about the point (R,Z).

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        J = self.control_Bp_jacobian(R,Z)

        return J * self.current

    def set_fill_colour(self, fill_colour):
        """Changes the colour used to fill the coil when plotted."""

        self.fill_colour = fill_colour

    def set_edge_colour(self, edge_colour):
        """Changes the colour of the edge of the coil when plotted."""

        self.edge_colour = edge_colour

    def plot(self, ax=None):
        """Plots the coil.

        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot onto. If None, these will be created.

        Returns
        -------
        ax
            matplotlib axes containing the plot of the machine.
        """

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')

        ax.plot(
            [self.R,self.R],
            [self.Z_min,self.Z_max],
            color=self.edge_colour,
            zorder=2,
            linewidth=0.5,
            )

        return ax

    def to_dict(self):
        """Return a JSON-serialisable dictionary matching the magnets JSON schema."""
        return {
            "type": "solenoid",
            "R": float(self.R),
            "Z_min": float(self.Z_min),
            "Z_max": float(self.Z_max),
            "turns": int(self.turns),
            "current": float(self.current),
        }

class FilamentPointCoil(Coil):
    """Represents a coil containing a set of current filaments.

    Each filament acts as a point source of current. The current sources here are prescribed
    by given R,Z points. Each filament carries a factor of 1/N_filamanets of the
    total coil current, where N_filaments are the number of filaments. A useful case
    for this coil is where filaments are placed at the centre of the real physical turns
    of the coil.

    Parameters
    ----------
    name : str
        Name of the coil.
    current : float
        Per-turn current in the coil (A).
    turns : int
        Number of turns. Total coil current is current * turns.
    R_filaments : list
        R coordinates of the filaments.
    Z_filaments : list
        Z coordinates of the filaments.
    dR : float
        Full radial width of the filaments. If None, they will be plotted as points.
    dZ :float
        Full vertical height of the filaments. If None, they will be plotted as points.
    """

    def __init__(self,
        name=None,
        current=0.0,
        turns=1,
        R_filaments=None,
        Z_filaments=None,
        dR=None,
        dZ=None,
    ):
        self.current = current
        self.turns = turns
        self.name = name

        # The filament points to be used
        self.R_filaments = R_filaments
        self.Z_filaments = Z_filaments
        self.N_filaments = len(self.Z_filaments)
        self.weight = 1.0 / self.N_filaments
        self.filament_points = list(zip(self.R_filaments,self.Z_filaments))

        # Filament sizes for plotting
        self.dR = dR
        self.dZ = dZ

        # Colours used to plot the coil
        self.fill_colour = "orange"
        self.edge_colour = "grey"

    def control_psi(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil in self.filament_points:
            result += greens.Greens(R_fil, Z_fil, R, Z) * self.weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_Br(self, R, Z):
        """Calculates the radial magnetic field at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil in self.filament_points:
            result += greens.Greens_Br(R_fil, Z_fil, R, Z) * self.weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_Bz(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z) from a unit current through the coil."""
        result = 0.0
        for R_fil, Z_fil in self.filament_points:
            result += greens.Greens_Bz(R_fil, Z_fil, R, Z) * self.weight
        # Multipy by turns so that total current is current * turns
        return result * self.turns

    def control_dBp(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z) from a unit current through the coil.

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """

        result = 0.0

        if deriv == "dBr_dZ":

            for R_fil, Z_fil in self.filament_points:
                result += greens.Greens_dBr_dZ(R_fil,Z_fil,R,Z) * self.weight

        elif deriv == "dBr_dR":

            for R_fil, Z_fil in self.filament_points:
                result += greens.Greens_dBr_dR(R_fil,Z_fil,R,Z) * self.weight

        elif deriv == "dBz_dZ":

            for R_fil, Z_fil in self.filament_points:
                result += greens.Greens_dBz_dZ(R_fil,Z_fil,R,Z) * self.weight

        elif deriv == "dBz_dR":

            for R_fil, Z_fil in self.filament_points:
                result += greens.Greens_dBz_dR(R_fil,Z_fil,R,Z) * self.weight

        else:
            raise ValueError(f"Unknown derivative '{deriv}'. Expected one of: dBr_dZ, dBr_dR, dBz_dZ, dBz_dR.")

        return result * self.turns

    def control_Bp_jacobian(self,R,Z):
        """Computes the 2x2 Jacobian matrix of poloidal field about the point (R,Z) from a unit current through the coil.

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        a = self.control_dBp(R,Z,deriv="dBr_dR")
        b = self.control_dBp(R,Z,deriv="dBr_dZ")
        c = self.control_dBp(R,Z,deriv="dBz_dR")
        d = self.control_dBp(R,Z,deriv="dBz_dZ")

        return np.array([[a,b],[c,d]])

    def psiRZ(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z)."""
        return self.control_psi(R, Z) * self.current

    def BrRZ(self, R, Z):
        """Calculates the radial magnetic field at (R,Z)."""
        return self.control_Br(R, Z) * self.current

    def BzRZ(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z)."""
        return self.control_Bz(R, Z) * self.current

    def dBpRZ(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z).

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """
        return self.control_dBp(R, Z, deriv) * self.current

    def BpRZ_jacobian(self, R, Z):
        """Computes the 2x2 Jacobian matrix of the poloidal field about the point (R,Z).

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        J = self.control_Bp_jacobian(R,Z)

        return J * self.current

    def set_fill_colour(self, fill_colour):
        """Changes the colour used to fill the coil when plotted."""

        self.fill_colour = fill_colour

    def set_edge_colour(self, edge_colour):
        """Changes the colour of the edge of the coil when plotted."""

        self.edge_colour = edge_colour

    def plot(self, ax=None):
        """Plots the coil.

        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot onto. If None, these will be created.

        Returns
        -------
        ax
            matplotlib axes containing the plot of the machine.
        """

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')

        if self.dR is None and self.dZ is None:

            # Plot filament centres
            ax.scatter(
                self.R_filaments,
                self.Z_filaments,
                color=self.edge_colour,
                marker='x',
                zorder=2
                )

        else:

            # Plot filaments
            for Rfil, Zfil in zip(self.R_filaments,self.Z_filaments):

                r1 = Rfil - 0.5 * self.dR
                r2 = Rfil + 0.5 * self.dR
                z1 = Zfil - 0.5 * self.dZ
                z2 = Zfil + 0.5 * self.dZ

                R = [r1,r2,r2,r1,r1]
                Z = [z1,z1,z2,z2,z1]

                ax.fill(R,
                        Z,
                        color=self.fill_colour
                        )

                ax.plot(R,
                        Z,
                        color=self.edge_colour,
                        linewidth=0.5,
                        )

        return ax

    def to_dict(self):
        """Return a JSON-serialisable dictionary matching the magnets JSON schema."""
        return {
            "type": "filament",
            "R": [float(r) for r in self.R_filaments],
            "Z": [float(z) for z in self.Z_filaments],
            "turns": int(self.turns),
            "current": float(self.current),
            "dR": float(self.dR),
            "dZ": float(self.dZ),
        }

class Circuit:
    """Represents a collection of coils connected together in the same circuit.

    Parameters
    ----------
    magnets : list
        List of PF coil objects - [Coils, ShapedCoils, Solenoids] etc.
    multipliers : list
        List of circuit current multipliers for the current in each coil. E.g,
        if a circuit of 2 coils, ["PF1U","PF1L"], a set of multipliers [1, -1] would correspond
        to "PF1U" recieving the circuit current and "PF1L" recieving the negative of the circuit
        current.
    name: str
        Name of the circuit.
    circuit_current : float
        Current in the circuit (A).
    """

    def __init__(
        self,
        magnets = None,
        multipliers = None,
        name = None,
        circuit_current = None,
    ):
        # Set the name of the circuit
        self.name = name

        # Set multipliers - a list of multipliers of the circuit current that determines the current
        # supplied by the circuit to each coil. If this is None, it is assumed all coils in
        # the circuit will recieve the circuit current.
        if multipliers is None:

            multipliers = [1 for magnet in magnets]

        self.multipliers = multipliers

        # The PF coils in this circuit will be stored in a dictionary called "coilset".
        # The dictionary will have the following example structure:
        #
        # coilset = {
        #    "PF1U": {
        #        "magnet": forge.magnets.Coil object,
        #        "current_multiplier": 1.0,
        #    },
        #    "PF1L": {
        #        "magnet": forge.magnets.Coil object,
        #        "current_multiplier": -1.0,
        #    }
        # }

        coilset = {}

        # Track the currents of the coils that make up the circuit coilset
        currents = []

        for magnet, multiplier in list(zip(magnets,self.multipliers)):

            name = magnet.name
            current = magnet.current

            coilset[name] = {
                "magnet": magnet,
                "current_multiplier": multiplier
            }

            currents.append(current)

        self.coilset = coilset

        # If circuit_current was None, i.e. the user did not specify the current in the circuit,
        # then the currents of the coils in the circuit coilset will be used to estimate the
        # circuit current.
        if circuit_current is None:

            estimated_circuit_current_vals = []

            # For each Coil in the circuit, estimate the Circuit current from the Coil's current
            for current, multiplier in zip(currents,self.multipliers):

                estimated_circuit_current = current / multiplier
                estimated_circuit_current_vals.append(estimated_circuit_current)

            self.current = np.mean(estimated_circuit_current_vals)

        else:
            self.current = circuit_current

    def control_psi(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z) from a unit current through the circuit."""

        result = 0.0

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]
            current_multiplier = coil_dict["current_multiplier"]

            result += coil.control_psi(R, Z) * current_multiplier

        return result

    def control_Br(self, R, Z):
        """Calculates the radial magnetic field at (R,Z) from a unit current through the circuit."""

        result = 0.0

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]
            current_multiplier = coil_dict["current_multiplier"]

            result += coil.control_Br(R, Z) * current_multiplier

        return result

    def control_Bz(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z) from a unit current through the circuit."""

        result = 0.0

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]
            current_multiplier = coil_dict["current_multiplier"]

            result += coil.control_Bz(R, Z) * current_multiplier

        return result

    def control_dBp(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z) from a unit current through the circuit.

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """

        result = 0.0

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]
            current_multiplier = coil_dict["current_multiplier"]

            result += coil.control_dBp(R, Z, deriv) * current_multiplier

        return result

    def control_Bp_jacobian(self,R,Z):
        """Computes the 2x2 Jacobian matrix of poloidal field about the point (R,Z) from a unit circuit current.

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        a = self.control_dBp(R,Z,deriv="dBr_dR")
        b = self.control_dBp(R,Z,deriv="dBr_dZ")
        c = self.control_dBp(R,Z,deriv="dBz_dR")
        d = self.control_dBp(R,Z,deriv="dBz_dZ")

        return np.array([[a,b],[c,d]])

    def psiRZ(self, R, Z):
        """Calculates the poloidal magnetic flux at (R,Z)."""

        return self.control_psi(R, Z) * self.current

    def BrRZ(self, R, Z):
        """Calculates the radial magnetic field at (R,Z)."""

        return self.control_Br(R, Z) * self.current

    def BzRZ(self, R, Z):
        """Calculates the vertical magnetic field at (R,Z)."""

        return self.control_Bz(R, Z) * self.current

    def dBpRZ(self, R, Z, deriv=None):
        """Calculates the selected poloidal field derivative at (R,Z).

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """
        return self.control_dBp(R, Z, deriv) * self.current

    def BpRZ_jacobian(self, R, Z):
        """Computes the 2x2 Jacobian matrix of the poloidal field about the point (R,Z).

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        J = self.control_Bp_jacobian(R,Z)

        return J * self.current

    def set_fill_colour(self, fill_colour):
        """Changes the colour used to fill the coils in the Circuit when plotted."""

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]

            coil.set_fill_colour(fill_colour)

    def set_edge_colour(self, edge_colour):
        """Changes the colour of the edge of the coils in the Circuit when plotted."""

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]

            coil.set_edge_colour(edge_colour)

    def plot(self, ax=None):
        """Plots the coils in the circuit.

        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot onto. If None, these will be created.

        Returns
        -------
        ax
            matplotlib axes containing the plot of the machine.
        """

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')

        for coil_dict in self.coilset.values():

            coil = coil_dict["magnet"]
            ax = coil.plot(ax=ax)

        return ax
