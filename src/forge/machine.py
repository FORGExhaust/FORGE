"""
Defines the Machine class representing a tokamak's PF coil set.

Inspired by the Machine class in machine.py from FreeGS
(https://github.com/freegs-plasma/freegs), but substantially rewritten
for FORGE.

Copyright 2016-2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
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
from shapely.geometry import LineString, Polygon
from shapely.plotting import plot_polygon

from forge.magnets import Circuit, Coil, FilamentPointCoil, ShapedCoil, Solenoid

logger = logging.getLogger(__name__)


class Machine:
    """Object representing a Tokamak.

    Represents the machine (Tokamak) including the PF coils, wall, and additional structures.
    Coils may be powered independently, or wired in Circuits with one another.

    Parameters
    ----------
    magnets_data : dict
        A dictionary of data containing information on the PF coils.
    wall_R : list
        R coordinates of the machine wall.
    wall_Z : list
        Z coordinates of the machine wall.
    circuits : dict
        Dictionary of list of names of coils in Circuits and associated current multipliers - each entry
        represents a Circuit. Can be None if there are no Circuits.

        Example::

            {
                "P1": {
                    "coil_names": ['P1U', 'P1L'],
                    "current_multipliers": [1.0, 1.0]
                },
                "P3": {
                    "coil_names": ['P3U', 'P3L'],
                    "current_multipliers": [1.0, 1.0]
                },
            }

    other_structures: dict
        A dictionary of other structures (e.g. a TF coil) that can be used in plotting
        of the device. Either lines or filled polygons can be plotted.

        Example::

            {
                "TF": {
                    "exterior": {
                        "R": [R_1, R_2, ... R_N],
                        "Z": [Z_1, Z_2, ... Z_N],
                    },
                    "interior": {
                        "R": [R_1, R_2, ... R_M],
                        "Z": [Z_1, Z_2, ... Z_M],
                    },
                    "colour": "k",
                    "fill_colour": "orange"
                }
            }

        The exterior and interior of a shape are specified, along with a colour for the outline and the fill colour.
        If a filled shape without an interior is desired, interior can be None. Likewise, if a simple line is
        desired, then both interior and fill_colour can be None.
    """

    def __init__(
            self,
            magnets_data = None,
            wall_R = None,
            wall_Z = None,
            circuits = None,
            other_structures = None,
        ):

        self.magnets_data = magnets_data
        self.wall_R = wall_R
        self.wall_Z = wall_Z
        self.circuits = circuits
        self.other_structures = other_structures

        # Record the max/min R/Z of the wall. This can be used later in plotting routines.
        self.wall_R_min = np.amin(self.wall_R)
        self.wall_R_max = np.amax(self.wall_R)
        self.wall_Z_min = np.amin(self.wall_Z)
        self.wall_Z_max = np.amax(self.wall_Z)

        # Create a Shapely LineString object of the wall
        self.wall = LineString(list(zip(self.wall_R,self.wall_Z)))

        # Create the coil objects
        self.create_coilset()

    def create_coilset(self):
        """Creates the PF coil set.

        Creates PF coil objects from the input magnets data. If Circuits
        are present, coils will be wired in Circuits accordingly.
        """

        coils = []

        # Iterate through the PF coil data and first create a list of coil objects.
        # This will include all coils, both those that will be circuits and those that
        # will be independently powered.
        for name, data in self.magnets_data.items():

            # Get the type of coil
            coil_type = self.magnets_data[name]["type"]

            # Create the relevant PF coil object
            if coil_type == "point":

                coils.append(
                    Coil(
                    R = self.magnets_data[name]["R"],
                    Z = self.magnets_data[name]["Z"],
                    turns = self.magnets_data[name]["turns"],
                    current = self.magnets_data[name]["current"],
                    name = name,
                    )
                )

            elif coil_type == "shaped":

                R = self.magnets_data[name]["R"]
                Z = self.magnets_data[name]["Z"]
                shape = list(zip(R,Z))

                coils.append(
                    ShapedCoil(
                    shape = shape,
                    turns = self.magnets_data[name]["turns"],
                    current = self.magnets_data[name]["current"],
                    name = name,
                    )
                )

            elif coil_type == "solenoid":

                coils.append(
                    Solenoid(
                    R = self.magnets_data[name]["R"],
                    Z_min = self.magnets_data[name]["Z_min"],
                    Z_max = self.magnets_data[name]["Z_max"],
                    turns = self.magnets_data[name]["turns"],
                    current = self.magnets_data[name]["current"],
                    name = name,
                    )
                )

            elif coil_type == "filament":

                coils.append(
                    FilamentPointCoil(
                        R_filaments = self.magnets_data[name]["R"],
                        Z_filaments = self.magnets_data[name]["Z"],
                        turns = self.magnets_data[name]["turns"],
                        current = self.magnets_data[name]["current"],
                        name = name,
                        dR = self.magnets_data[name]["dR"],
                        dZ = self.magnets_data[name]["dZ"],
                    )
                )

            else:

                logger.warning('Coil %s has an unknown type - %s', name, coil_type)

        # We have now made all of our coil objects.

        # Next, we will create a list of coils that appear in Circuits
        coils_in_circuits = []

        if self.circuits is not None:

            for circuit_name in self.circuits:

                coil_names = self.circuits[circuit_name]["coil_names"]
                coils_in_circuits += coil_names

        # Now that we have made all of our coil objects, we will add all of the independently powered coils
        # to the coilset.
        coilset = {}

        for coil in coils:

            name = coil.name

            # Check if this coil exists in a circuit or not. Only perform this check if circuits
            # have actually been defined.
            if self.circuits is not None:

                # Check if this coil is NOT a part of a Circuit
                if name not in coils_in_circuits:

                    # Coil does not exist in a circuit - add to the coil set
                    coilset[name] = coil

            else:

                # No circuits exist

                coilset[name] = coil

        # Now that all of the independently powered coils have been added to the coilset,
        # we will create and add any remaining Circuits to the coilset.
        if self.circuits is not None:

            for circuit_name, circuit in self.circuits.items():

                # Extract the list of names of coils in this Circuit
                circuit_coil_names = circuit["coil_names"]

                # Extract a list of coil current multipliers. These may
                # not be defined. In this case, all coils in the circuit are
                # assumed to have the circuit current.
                try:
                    circuit_current_multipliers = circuit["current_multipliers"]
                except KeyError:
                    circuit_current_multipliers = [1 for name in circuit_coil_names]

                # Populate a list with coil objects for those coils in this circuit
                circuit_coils = []

                # Iterate over the coils, and append those in this circuit to the list
                for coil in coils:

                    name = coil.name

                    if name in circuit_coil_names:

                        circuit_coils.append(coil)

                # Create the Circuit
                circuit =  Circuit(
                    magnets = circuit_coils,
                    multipliers = circuit_current_multipliers,
                    name = circuit_name,
                )

                # Add the circuit to the coil set
                coilset[circuit_name] = circuit

        self.coilset = coilset

        # Record the number of coils in the coilset. Note, herein when we refer to the number of
        # coils in the coilset, we are implicitly stating that this includes circuits - really
        # we should refer to the "actuators", to not muddle between coils and circuits.
        self.N_coils = len(self.coilset)

    def to_dict(self):
        """Return a JSON-serialisable dictionary matching the magnets JSON schema.

        Reads current values from the live coilset objects, so the output
        reflects any post-optimisation current changes.

        Returns
        -------
        dict
            ``{"coils": {...}, "circuits": {...}}`` suitable for
            :func:`forge.io.save_fancy_json`.
        """
        coils = {}
        circuits_dict = {}

        for name, obj in self.coilset.items():
            if isinstance(obj, Circuit):
                circuits_dict[name] = {
                    "coil_names": list(obj.coilset.keys()),
                    "current_multipliers": [float(m) for m in obj.multipliers],
                }
                for cname, cdata in obj.coilset.items():
                    d = cdata["magnet"].to_dict()
                    d["current"] = float(obj.current * cdata["current_multiplier"])
                    coils[cname] = d
            else:
                coils[name] = obj.to_dict()

        result = {"coils": coils}
        if circuits_dict:
            result["circuits"] = circuits_dict
        return result

    def psiRZ(self,R,Z):
        """Calculates the poloidal magnetic flux at (R,Z) from the coilset (not including the plasma)."""
        psi_coils = 0.0
        for _, coil in self.coilset.items():
            psi_coils += coil.psiRZ(R,Z)

        return psi_coils

    def BrRZ(self,R,Z):
        """Calculates the radial magnetic field at (R,Z) from the coilset (not including the plasma)."""
        br_coils = 0.0
        for _, coil in self.coilset.items():
            br_coils += coil.BrRZ(R,Z)

        return br_coils

    def BzRZ(self,R,Z):
        """Calculates the vertical magnetic field at (R,Z) from the coilset (not including the plasma)."""
        bz_coils = 0.0
        for _, coil in self.coilset.items():
            bz_coils += coil.BzRZ(R,Z)

        return bz_coils

    def control_psi(self,R,Z):
        """Calculates the poloidal magnetic flux at (R,Z) from a unit current through each of the PF coils.

        This will return result of size N_coils x shape(R,Z).
        """
        result = []
        for _, coil in self.coilset.items():
            result.append(coil.control_psi(R,Z))

        return result

    def control_Br(self,R,Z):
        """Calculates the radial magnetic field at (R,Z) from a unit current through each of the PF coils.

        This will return result of size N_coils x shape(R,Z).
        """
        result = []
        for coil in self.coilset.values():
            result.append(coil.control_Br(R,Z))

        return result

    def control_Bz(self,R,Z):
        """Calculates the vertical magnetic field at (R,Z) from a unit current through each of the PF coils.

        This will return result of size N_coils x shape(R,Z).
        """
        result = []
        for coil in self.coilset.values():
            result.append(coil.control_Bz(R,Z))

        return result

    def control_Bp_jacobians(self,R,Z):
        """Calculates the poloidal magnetic field Jacobians at (R,Z) for a unit current through each of the PF coils.

        This will return result of size N_coils x shape(R,Z).
        """
        result = []
        for coil in self.coilset.values():
            result.append(coil.control_Bp_jacobian(R,Z))

        return result

    def get_coil_names(self):
        """Returns a list of names of the coils in the PF coilset."""

        names = list(self.coilset.keys())
        return names

    def get_currents(self):
        """Returns a list of currents in the PF coilset."""

        currents = np.asarray([coil.current for coil in list(self.coilset.values())])
        return currents

    def update_currents(self,new_currents):
        """Updates the currents in the PF coilset, taking a list of new currents as input."""

        for index, new_current in enumerate(new_currents):
            list(self.coilset.values())[index].current = new_current

    def plot(
            self,
            ax = None,
            plot_wall = True,
            plot_other_structures = True,
            show = False
            ):
        """Plots the tokamak.

        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot onto. If None, these will be created.
        plot_wall : bool
            Flag for plotting the wall.
        plot_other_structures : bool
            Flag for plotting additional structures that may exist.
        show : bool
            Flag for whether the plot should be displayed or not. If False,
            the axes will be returned.

        Returns
        -------
        ax
            matplotlib axes containing the plot of the machine. Only returned if
            show is False.

        """

        if ax is None:

            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')

        for _, coil in self.coilset.items():

            ax = coil.plot(ax=ax)

        if self.wall_R is not None and self.wall_Z is not None and plot_wall:

            ax.plot(self.wall_R,self.wall_Z,color='k')

        if plot_other_structures and self.other_structures is not None:

            for structure_data in self.other_structures.values():

                if structure_data["interior"] is not None:

                    # Interior data has been provided, hence a region filled
                    # between the exterior and interior shapes will be created.

                    R_exterior = structure_data["exterior"]["R"]
                    Z_exterior = structure_data["exterior"]["Z"]

                    R_interior = structure_data["interior"]["R"]
                    Z_interior = structure_data["interior"]["Z"]

                    structure = Polygon(zip(R_exterior,Z_exterior),holes=[list(zip(R_interior,Z_interior))])

                    plot_polygon(
                        structure,
                        ax=ax,
                        facecolor=structure_data["fill_colour"],
                        edgecolor=structure_data["colour"],
                        add_points=False,
                        )

                else:

                    # Interior data was not provided.

                    if structure_data["fill_colour"] is not None:

                        # A fill colour has been provided for this shape without
                        # an interior. Hence a simple filled region will be produced.

                        R = structure_data["exterior"]["R"]
                        Z = structure_data["exterior"]["Z"]

                        ax.fill(R, Z, color=structure_data["fill_colour"])
                        ax.plot(R, Z, color=structure_data["colour"])

                    else:

                        # A fill colour was not provided, hence a simple line will
                        # be produced.

                        R = structure_data["exterior"]["R"]
                        Z = structure_data["exterior"]["Z"]

                        ax.plot(R, Z, color=structure_data["colour"])

        if show:

            plt.show()

        else:

            return ax

