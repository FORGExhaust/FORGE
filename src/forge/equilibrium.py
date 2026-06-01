"""Contains the Equilibrium Class used to define the equilibrium state of the plasma being optimised.

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
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline, interp1d

from forge import critical, greens
from forge import plotting as forge_plotting
from forge.utils import interactive_shape_editor

logger = logging.getLogger(__name__)

MU0 = 4.0 * np.pi * 1.0e-07
ONE_2PI = 1.0 / (2.0 * np.pi)

class Equilibrium:
    """Object representing an equilibrium state of the plasma. This is a single time instance.

    Parameters
    ----------
    eq_data : dict
        A dictionary of equilibrium data used to initialise the Equilibrium object.
    tokamak : floatforge.magnets.Machine object
        Machine object containing the PF coil set, wall, and additional structures.
    calculate_flux_from_coils : bool
        Flag for whether or not to calculate the PF coil set flux using their initial
        coil currents. If False, the flux from the plasma's internal current profiles
        will instead be calculated first, with the PF coil set flux calculated as the
        difference between the full flux and the plasma flux.
    """

    def __init__(
        self,
        eq_data = None,
        tokamak = None,
        calculate_flux_from_coils = False,
        ):

        if not isinstance(eq_data, dict):
            raise TypeError("Error. eq_data was not of the required type - dict. Type passed was " + str(type(eq_data)))

        self.eq_data = eq_data

        # Initialise with data from eq_data dict.
        self.R_min = self.eq_data["R_min"]
        self.R_max = self.eq_data["R_max"]
        self.nR = self.eq_data["nR"]
        self.R_1D = self.eq_data["R_1D"]
        self.dR = self.eq_data["dR"]
        self.R_2D = self.eq_data["R_2D"]

        self.Z_min = self.eq_data["Z_min"]
        self.Z_max = self.eq_data["Z_max"]
        self.nZ = self.eq_data["nZ"]
        self.Z_1D = self.eq_data["Z_1D"]
        self.dZ = self.eq_data["dZ"]
        self.Z_2D = self.eq_data["Z_2D"]

        self.tokamak = tokamak
        self.calculate_flux_from_coils = calculate_flux_from_coils

        self.wall_R = self.tokamak.wall_R
        self.wall_Z = self.tokamak.wall_Z

        self.psi_2D = self.eq_data["psi_2D"]

        self.psin_data = self.eq_data["psin_data"]
        self.pprime_data = self.eq_data["pprime_data"]
        self.ffprime_data = self.eq_data["ffprime_data"]
        self.q_data = self.eq_data["q_data"]
        self.pressure_data = self.eq_data["pressure_data"]
        self.fpol_data = self.eq_data["fpol_data"]

        self.plasma_current = self.eq_data["plasma_current"]

        self.fvac = self.eq_data["fvac"]

        # Create 1D interpolators for pprime, ffprime, q = func(psiN)
        self.pprime_func = interp1d(self.psin_data,self.pprime_data,bounds_error=False,fill_value=0.0)
        self.ffprime_func = interp1d(self.psin_data,self.ffprime_data,bounds_error=False,fill_value=0.0)
        self.q_func = interp1d(self.psin_data,self.q_data,bounds_error=False,fill_value=0.0)
        self.pressure_func = interp1d(self.psin_data,self.pressure_data,bounds_error=False,fill_value=0.0)
        self.fpol_func = interp1d(self.psin_data,self.fpol_data,bounds_error=False,fill_value=self.fvac)

        # Create a 2D (R,Z) interpolator for psi - poloidal magnetic flux
        self.psi_func = RectBivariateSpline(self.R_1D, self.Z_1D, self.psi_2D)

        # Analyse the equilibrium, finding O- and X-points, as well as the separatrix
        self.check_geometry()

        # Detect and correct COCOS convention.
        # FORGE internally uses COCOS 11-18: psi is full poloidal flux (Wb), and
        # B_R = -(1/2piR) dpsi/dZ,  B_Z = (1/2piR) dpsi/dR.
        # COCOS 1-8 stores psi/(2pi). If detected, correct psi, pprime, and ffprime.
        self._detect_and_correct_cocos()

        # Calculate the flux from the plasma, psi_plas_2D, and the flux from the machine, psi_mach_2D
        self.calc_fluxes()

        # Initialise an interpolator for (R,Z) as a function of normalised (0,1) distance along the separatrix
        self.generate_separatrix_interpolator()

    def _detect_and_correct_cocos(self):
        """Detect if the equilibrium uses COCOS 1-8 and convert to COCOS 11-18.

        FORGE expects psi to be the full poloidal flux (Wb), i.e. COCOS 11-18.
        COCOS 1-8 stores psi/(2pi). Detection is performed by computing Bp at
        the outboard midplane from dpsi/dR (assuming COCOS 11-18) and comparing
        with the expected Bp from Ampere's law. If they disagree by approximately
        2pi, the input is COCOS 1-8 and a correction is applied.

        Corrected quantities:
            psi_2D      *= 2pi
            psi_axis    *= 2pi
            psi_lcfs    *= 2pi
            pprime_data /= 2pi
            ffprime_data /= 2pi
        """
        TWO_PI = 2.0 * np.pi

        # Evaluate dpsi/dR at the midplane, halfway between the magnetic axis and the LCFS.
        # At the midplane, Br ~ 0 (dpsi/dZ ~ 0 by symmetry), so Bp ~ |Bz| = |(1/2piR) dpsi/dR|.
        R_eval = 0.5 * (self.R_mag + self.R_OMP)
        Z_eval = self.Z_mag  # midplane at the magnetic axis height

        # dpsi/dR from the spline interpolator
        dpsi_dR = float(self.psi_func.ev(R_eval, Z_eval, dx=1))

        # Bp assuming COCOS 11-18: Bz = (1/2piR) * dpsi/dR
        Bp_from_psi = abs(dpsi_dR) / (TWO_PI * R_eval)

        # Expected Bp from Ampere's law: mu_0 * Ip / (2pi * R)
        # This is an approximation at the midplane
        Ip = abs(self.plasma_current)
        Bp_ampere = MU0 * Ip / (TWO_PI * R_eval)

        if Bp_ampere == 0 or Bp_from_psi == 0:
            self.cocos_input = "11-18"
            return

        ratio = Bp_from_psi / Bp_ampere

        # COCOS 11-18: ratio ~ 1 (order of magnitude)
        # COCOS 1-8:   ratio ~ 1/(2pi) ~ 0.16
        # Use a threshold of 0.5 to distinguish
        if ratio < 0.5:
            self.cocos_input = "1-8"
            logging.getLogger("forge").info(
                "COCOS 1-8 detected (Wb per radian). Converting to COCOS 11-18 (full Wb)."
            )

            # Correct psi
            self.psi_2D = self.psi_2D * TWO_PI
            self.psi_axis = self.psi_axis * TWO_PI
            self.psi_lcfs = self.psi_lcfs * TWO_PI

            # Correct derivatives w.r.t. psi (they become smaller since psi is larger)
            self.pprime_data = self.pprime_data / TWO_PI
            self.ffprime_data = self.ffprime_data / TWO_PI

            # Rebuild interpolators with corrected data
            self.psi_func = RectBivariateSpline(self.R_1D, self.Z_1D, self.psi_2D)
            self.pprime_func = interp1d(self.psin_data, self.pprime_data, bounds_error=False, fill_value=0.0)
            self.ffprime_func = interp1d(self.psin_data, self.ffprime_data, bounds_error=False, fill_value=0.0)

            # Recompute normalised psi
            self.psin_2D = (self.psi_2D - self.psi_axis) / (self.psi_lcfs - self.psi_axis)
        else:
            self.cocos_input = "11-18"

    def calc_fluxes(self):
        """Calculates the poloidal magnetic flux from the plasma and from the machine."""

        # Create a mask around the core
        self.mask = critical.core_mask(self.R_2D, self.Z_2D, self.psi_2D, self.opt, self.xpt, self.psi_lcfs)

        # Adjust the mask by checking for points outside of the bounding box around the core
        # as sometimes the mask can slip past the X-points a little.
        outside_bbox = (
            (self.R_2D > self.R_OMP) |
            (self.R_2D < self.R_IMP) |
            (self.Z_2D < self.Z_vertical_lower) |
            (self.Z_2D > self.Z_vertical_upper)
        )
        self.mask[outside_bbox & (self.mask == 1.0)] = 0

        # Calculate pprime and ffprime in 2D, and mask outside the core
        self.pprime_2D = self.pprime(self.psin_2D) * self.mask
        self.ffprime_2D = self.ffprime(self.psin_2D) * self.mask

        # Calculate the toroidal current density
        self.jtor_2D = 2.0 * np.pi * ((self.R_2D * self.pprime_2D) + (self.ffprime_2D / (self.R_2D * MU0)))

        # Calculate the toroidal current
        self.itor_2D = self.jtor_2D * self.dR * self.dZ

        # Check the plasma current
        self.plasma_current_check = simpson(simpson(self.itor_2D,axis=1),axis=0)

        logger.info('Plasma current (input) (kA): %s', self.plasma_current / 1e3)
        logger.info('Plasma current (check) (kA): %s', self.plasma_current_check / 1e3)

        # We have 2 modes for calculating the flux from the plasma, psi_plas_2D, and the flux
        # from the machine (coils), psi_mach_2D. In the first mode, we start by using the coils
        # and their known initial currents to calculate the flux from the coils. Afterwhich
        # psi_plas = psi - psi_coils. In the second mode we instead first calculate the flux
        # from the plasma from the toroidal current distribution, afterwhich
        # psi_mach = psi - psi_plas. If the coil currents are not known, the second mode
        # is appropriate. If using the first mode it is assumed that the flux map that the
        # coils will produce will turn out to be close to the real flux map that they produced
        # in whatever equilibrium code was used. Differences in the representation of the coils
        # in FORGE vs the original equilibrium code can introduce discrepencies here. This first
        # approach is however much faster, and particularly suited to large grid sizes.

        if self.calculate_flux_from_coils:

            # Calculate the poloidal magnetic flux from the coils
            self.psi_mach_2D = np.zeros(np.shape(self.psi_2D))

            # Iterate over the coils in the coilset and get the flux on the grid.
            for coil in self.tokamak.coilset.values():

                self.psi_mach_2D += coil.psiRZ(self.R_2D,self.Z_2D)

            # Calculate the polodial magnetic flux from the plasma on the 2D grid.
            self.psi_plas_2D = self.psi_2D - self.psi_mach_2D

        else:

            # Calculate the poloidal magnetic flux from the plasma current sources on the full 2D grid.
            # We will do this by moving over points in the grid, and, if there is current at a point, calculating
            # the flux on the entire grid from this current source, before summing over all current sources.
            self.psi_plas_2D = np.zeros(np.shape(self.psi_2D))

            for i in range(self.nR):

                for j in range(self.nZ):

                    current = self.itor_2D[i][j]
                    r = self.R_2D[i][j]
                    z = self.Z_2D[i][j]

                    if current != 0.0:

                        # Calculate the flux (Wb) from this current source on the grid
                        self.psi_plas_2D += greens.Greens(r, z, self.R_2D, self.Z_2D) * current

            # Calculate the poloidal magnetic flux from the machine on the full 2D grid
            self.psi_mach_2D = self.psi_2D - self.psi_plas_2D

        # Create 2D (R,Z) interpolators for the plasma and machine fluxes
        self.psi_plas_func = RectBivariateSpline(self.R_1D, self.Z_1D, self.psi_plas_2D)
        self.psi_mach_func = RectBivariateSpline(self.R_1D, self.Z_1D, self.psi_mach_2D)

    def pprime(self,psin):
        """Returns pprime as a function of normalised poloidal magnetic flux."""

        return self.pprime_func(psin)

    def ffprime(self,psin):
        """Returns ffprime as a function of normalised poloidal magnetic flux."""

        return self.ffprime_func(psin)

    def psiRZ(self,R,Z):
        """Returns the poloidal magnetic flux at given (R,Z) location(s)."""

        return self.psi_func(R,Z,grid=False)

    def psi_plasRZ(self,R,Z):
        """Returns the poloidal magnetic flux from the plasma at given (R,Z) location(s)."""

        return self.psi_plas_func(R,Z,grid=False)

    def psi_machRZ(self,R,Z):
        """Returns the poloidal magnetic flux from the machine at given (R,Z) location(s)."""

        return self.psi_mach_func(R,Z,grid=False)

    def dpsi_dR_RZ(self,R,Z):
        """Returns dpsi/dR at given (R,Z) location(s)."""

        return self.psi_func.ev(R,Z,dx=1)

    def dpsi_plas_dR_RZ(self,R,Z):
        """Returns dpsi/dR from the plasma at given (R,Z) location(s)."""

        return self.psi_plas_func.ev(R,Z,dx=1)

    def dpsi_mach_dR_RZ(self,R,Z):
        """Returns dpsi/dR from the machine at given (R,Z) location(s)."""

        return self.psi_mach_func.ev(R,Z,dx=1)

    def dpsi_dZ_RZ(self,R,Z):
        """Returns dpsi/dZ at given (R,Z) location(s)."""

        return self.psi_func.ev(R,Z,dy=1)

    def dpsi_plas_dZ_RZ(self,R,Z):
        """Returns dpsi/dZ from the plasma at given (R,Z) location(s)."""

        return self.psi_plas_func.ev(R,Z,dy=1)

    def dpsi_mach_dZ_RZ(self,R,Z):
        """Returns dpsi/dZ from the machine at given (R,Z) location(s)."""

        return self.psi_mach_func.ev(R,Z,dy=1)

    def BrRZ(self,R,Z):
        """Returns the radial magnetic field at given (R,Z) location(s)."""

        return -(ONE_2PI / R) * self.dpsi_dZ_RZ(R,Z)

    def Br_plasRZ(self,R,Z):
        """Returns the radial magnetic field from the plasma at given (R,Z) location(s)."""

        return -(ONE_2PI / R) * self.dpsi_plas_dZ_RZ(R,Z)

    def Br_machRZ(self,R,Z):
        """Returns the radial magnetic field from the machine at given (R,Z) location(s)."""

        return -(ONE_2PI / R) * self.dpsi_mach_dZ_RZ(R,Z)

    def BzRZ(self,R,Z):
        """Returns the vertical magnetic field at given (R,Z) location(s)."""

        return (ONE_2PI / R) * self.dpsi_dR_RZ(R,Z)

    def Bz_plasRZ(self,R,Z):
        """Returns the vertical magnetic field from the plasma at given (R,Z) location(s)."""

        return (ONE_2PI / R) * self.dpsi_plas_dR_RZ(R,Z)

    def Bz_machRZ(self,R,Z):
        """Returns the vertical magnetic field from the machine at given (R,Z) location(s)."""

        return (ONE_2PI / R) * self.dpsi_mach_dR_RZ(R,Z)

    def BpolRZ(self,R,Z):
        """Returns the poloidal magnetic field at given (R,Z) location(s)."""

        Br = self.BrRZ(R,Z)
        Bz = self.BzRZ(R,Z)

        return np.sqrt(Br * Br + Bz * Bz)

    def BtorRZ(self,R,Z):
        """Returns the toroidal magnetic field at given (R,Z) location(s) outside the plasma."""

        return self.fvac / R

    def BtotRZ(self,R,Z):
        """Returns the total magnetic field at given (R,Z) location(s) outside the plasma."""

        Bpol = self.BpolRZ(R,Z)
        Btor = self.BtorRZ(R,Z)

        return np.sqrt(Bpol * Bpol + Btor * Btor)

    def dBp_plasRZ(self, R, Z, deriv=None):
        """Calculates the relevant derivative of the poloidal field from the plasma at (R,Z).

        Options for deriv:
        ------------------
        "dBr_dZ" - d(Br)/dZ - vertical derivative of the radial field
        "dBr_dR" - d(Br)/dR - radial derivative of the radial field
        "dBz_dZ" - d(Bz)/dZ - vertical derivative of the vertical field
        "dBz_dR" - d(Bz)/dR - radial derivative of the vertical field
        """

        result = 0.0

        if deriv == "dBr_dZ":

            # Calculate d(Br)/dZ
            # d(Br)/dZ = (1/2pi) * (-1/R * ( d^{2}(psi)/dZ^{2} ) )
            result = (1.0 / (2.0 * np.pi)) * ((-1.0 / R) * ( self.psi_plas_func.ev(R,Z,dy=2) ) )

        elif deriv == "dBr_dR":

            # Calculate d(Br)/dR
            # d(Br)/dR = (1/2pi) * ( ( (1/R^{2}) * (dpsi/dZ) ) - ( (1/R) * (d^{2}(psi)/dRdZ) ) )
            result = (1.0 / (2.0 * np.pi)) * ( ( (1.0 / (R * R)) * self.psi_plas_func.ev(R,Z,dy=1) ) - \
                                              ( (1.0 / R) * self.psi_plas_func.ev(R,Z,dx=1,dy=1) ) )

        elif deriv == "dBz_dZ":

            # Calculate d(Bz)/dZ
            # d(Bz)/dZ = (1/2pi) * (1/R * ( d^{2}(psi)/dZdR ) ) = (1/2pi) * (1/R * ( d^{2}(psi)/dRdZ ) )
            result = (1.0 / (2.0 * np.pi)) * (1.0 / R) * self.psi_plas_func.ev(R,Z,dx=1,dy=1)

        elif deriv == "dBz_dR":

            # Calculate d(Br)/dR
            # d(Bz)/dR = (1/2pi) * ( ( (-1/R^{2}) * (dpsi/dR) ) + ( (1/R) * (d^{2}(psi)/dR^{2}) ) )
            result = (1.0 / (2.0 * np.pi)) * ( ( (-1.0 / (R * R)) * self.psi_plas_func.ev(R,Z,dx=1) ) + \
                                              ( (1.0 / R) * self.psi_plas_func.ev(R,Z,dx=2) ) )

        else:
            print("Failed to provide a relevant derivative.")

        return result

    def Bpol_jacobian_plasRZ(self,R,Z):
        """Computes the 2x2 Jacobian matrix of the poloidal field from the plasma about the point (R,Z).

        J = [[dBr/dR,dBr/dZ],
            [dBz/dR,dBz/dZ]]
        """

        a = self.dBp_plasRZ(R,Z,deriv="dBr_dR")
        b = self.dBp_plasRZ(R,Z,deriv="dBr_dZ")
        c = self.dBp_plasRZ(R,Z,deriv="dBz_dR")
        d = self.dBp_plasRZ(R,Z,deriv="dBz_dZ")

        return np.array([[a,b],[c,d]])

    def plot_profiles(self):
        """Plots the pprime and ffprime profiles."""

        forge_plotting.plot_profiles(
            self.psin_data, self.pprime_data, self.ffprime_data,
            self.pprime_func, self.ffprime_func, MU0,
        )

    def plot_equilibrium(self, axis=None):
        """Plots the equilibrium."""

        return forge_plotting.plot_equilibrium(
            self.R_2D, self.Z_2D, self.psi_2D, self.psi_lcfs,
            self.wall_R, self.wall_Z, axis=axis,
        )

    def plot_fluxes(self):
        """Plots the total, plasma and machine fluxes."""

        forge_plotting.plot_fluxes(
            self.R_2D, self.Z_2D, self.psi_2D, self.psi_plas_2D,
            self.psi_mach_2D, self.psi_lcfs, self.wall_R, self.wall_Z,
            self.R_lcfs, self.Z_lcfs,
        )

    def plot_fields(self):
        """Plots the total, plasma and machine field components, Br and Bz."""

        forge_plotting.plot_fields(
            self.R_2D, self.Z_2D,
            self.wall_R, self.wall_Z,
            self.R_lcfs, self.Z_lcfs,
            self.BrRZ(self.R_2D, self.Z_2D),
            self.Br_plasRZ(self.R_2D, self.Z_2D),
            self.Br_machRZ(self.R_2D, self.Z_2D),
            self.BzRZ(self.R_2D, self.Z_2D),
            self.Bz_plasRZ(self.R_2D, self.Z_2D),
            self.Bz_machRZ(self.R_2D, self.Z_2D),
        )

    def check_geometry(self):
        """Analyses the geometry of the core plasma.
        
        Analysis routine that identifies key geometrical and topological features
        of the core plasma, such as locating X-points and O-points, identifying whether
        the plasma is in a single/double/disconnected-double null, find key points such
        as the maximum/minimum radial and vertical extrema points along the separatrix
        as well as identifying the location of points along the separatrix.
        """

        # Helper function that will (if required) insert the X-point(s) into the
        # list of points that constitute the LCFS
        def insert_point_to_LCFS(Rp,Zp):

            distances = []

            # Calculate the distance (squared) between each point on the LCFS and the new point
            for R_sep, Z_sep in zip(self.R_lcfs,self.Z_lcfs):

                s2 = (R_sep - Rp)**2. + (Z_sep - Zp)**2.
                distances.append(s2)

            indeces_sorted = np.argsort(distances)
            index_to_insert = max(indeces_sorted[0],indeces_sorted[1])

            self.R_lcfs = np.insert(self.R_lcfs,index_to_insert,Rp)
            self.Z_lcfs = np.insert(self.Z_lcfs,index_to_insert,Zp)

        # Get the O-points and X-points
        self.opt, self.xpt = critical.find_critical(self.R_2D, self.Z_2D, self.psi_2D)

        # O-point
        self.R_mag = self.opt[0][0]
        self.Z_mag = self.opt[0][1]
        self.psi_axis = self.opt[0][2]

        # First X-point
        self.R_xpt_lcfs = self.xpt[0][0]
        self.Z_xpt_lcfs = self.xpt[0][1]
        self.psi_lcfs = self.xpt[0][2]

        # Second X-point
        self.R_xpt_lcfs2 = self.xpt[1][0]
        self.Z_xpt_lcfs2 = self.xpt[1][1]
        self.psi_lcfs2 = self.xpt[1][2]

        # Normalise psi_2D
        self.psin_2D = (self.psi_2D - self.psi_axis) / (self.psi_lcfs - self.psi_axis)

        # Examine the primary X-point
        if self.Z_xpt_lcfs < self.Z_xpt_lcfs2:

            # Lower X-point is primary X-point
            self.lower_xpoint_primary = True
            self.upper_xpoint_primary = False

            self.R_xpt_lower = self.R_xpt_lcfs
            self.Z_xpt_lower = self.Z_xpt_lcfs
            self.psi_xpt_lower = self.psi_lcfs

            self.R_xpt_upper = self.R_xpt_lcfs2
            self.Z_xpt_upper = self.Z_xpt_lcfs2
            self.psi_xpt_upper = self.psi_lcfs2

            self.USND = False
            self.LSND = True
            self.DND = False
            self.mag_con = 'LSND'

        else:

            # Upper X-point is primary X-point
            self.lower_xpoint_primary = False
            self.upper_xpoint_primary = True

            self.R_xpt_lower = self.R_xpt_lcfs2
            self.Z_xpt_lower = self.Z_xpt_lcfs2
            self.psi_xpt_lower = self.psi_lcfs2

            self.R_xpt_upper = self.R_xpt_lcfs
            self.Z_xpt_upper = self.Z_xpt_lcfs
            self.psi_xpt_upper = self.psi_lcfs

            self.USND = True
            self.LSND = False
            self.DND = False
            self.mag_con = 'USND'

        # Identify the LCFS (R,Z) coordinates
        self.R_lcfs, self.Z_lcfs = critical.find_separatrix(
            self,
            self.opt,
            self.xpt,
            ntheta=1000,
            psi=self.psi_2D
        )

        # Insert primary X-point into the LCFS
        insert_point_to_LCFS(self.R_xpt_lcfs,self.Z_xpt_lcfs)

        # Check if DND
        if (abs(self.psi_lcfs - self.psi_lcfs2) / abs(self.psi_lcfs)) < 1.0e-03:

            self.USND = False
            self.LSND = False
            self.DND = True
            self.mag_con = 'DND'

            # Insert secondary X-point into the LCFS
            insert_point_to_LCFS(self.R_xpt_lcfs2,self.Z_xpt_lcfs2)

        # Log the magnetic configuration
        psi_rel_diff = abs(self.psi_lcfs - self.psi_lcfs2) / abs(self.psi_lcfs) * 100.0
        nominal_primary = "lower" if self.lower_xpoint_primary else "upper"
        if self.DND:
            logger.info(
                "Magnetic configuration: Double Null Diverted (DND). "
                "Nominal primary X-point: %s. "
                "Relative psi difference between X-points: %.4f%%. "
                "Lower X-point: (R=%.4f, Z=%.4f), Upper X-point: (R=%.4f, Z=%.4f).",
                nominal_primary, psi_rel_diff,
                self.R_xpt_lower, self.Z_xpt_lower, self.R_xpt_upper, self.Z_xpt_upper,
            )
        elif self.LSND:
            logger.info(
                "Magnetic configuration: Lower Single Null Diverted (LSND). "
                "Primary X-point (lower): (R=%.4f, Z=%.4f).",
                self.R_xpt_lower, self.Z_xpt_lower,
            )
        elif self.USND:
            logger.info(
                "Magnetic configuration: Upper Single Null Diverted (USND). "
                "Primary X-point (upper): (R=%.4f, Z=%.4f).",
                self.R_xpt_upper, self.Z_xpt_upper,
            )

        # Get the (R,Z) bounding box around the separatrix from the radial and vertical extrema points
        index_R_min = np.argmin(self.R_lcfs)
        index_R_max = np.argmax(self.R_lcfs)
        index_Z_min = np.argmin(self.Z_lcfs)
        index_Z_max = np.argmax(self.Z_lcfs)

        # OMP
        self.R_OMP = self.R_lcfs[index_R_max]
        self.Z_OMP = self.Z_lcfs[index_R_max]

        # IMP
        self.R_IMP = self.R_lcfs[index_R_min]
        self.Z_IMP = self.Z_lcfs[index_R_min]

        # Vertical extrema upper point
        self.R_vertical_upper = self.R_lcfs[index_Z_max]
        self.Z_vertical_upper = self.Z_lcfs[index_Z_max]

        # Vertical extrema lower point
        self.R_vertical_lower = self.R_lcfs[index_Z_min]
        self.Z_vertical_lower = self.Z_lcfs[index_Z_min]

        # Get the minor radius - this is a useful length scale to know.
        # The way we calculate this implicitly assumes triangularity sits between (-1,1),
        # which is a reasonable assumption.
        self.minor_radius = 0.5 * (self.R_OMP - self.R_IMP)

        # Record the number of active X-points
        if self.DND:
            self.N_active_xpoints = 2
        else:
            self.N_active_xpoints = 1

        # Finally, re-arrange the points along the LCFS so that the first point is the
        # outer midplane point, and that the points move clockwise.

        # Check if CW or CCW.
        index_omp = np.argmax(self.R_lcfs)
        Z_omp = self.Z_lcfs[index_omp]
        Z_next = self.Z_lcfs[index_omp + 1]

        if Z_next < Z_omp:

            # Points ordered CW - need revesing
            self.R_lcfs = self.R_lcfs[::-1]
            self.Z_lcfs = self.Z_lcfs[::-1]

            # The ordering has changed, hence the index of the OMP has changed
            index_omp = np.argmax(self.R_lcfs)

        # Shift the data so that the OMP is the first point
        R_lcfs = np.concatenate((self.R_lcfs[index_omp::],self.R_lcfs[0:index_omp]))
        Z_lcfs = np.concatenate((self.Z_lcfs[index_omp::],self.Z_lcfs[0:index_omp]))
        self.R_lcfs = R_lcfs
        self.Z_lcfs = Z_lcfs

    def generate_separatrix_interpolator(self):
        """Separatrix (R,Z) points interpolator.
        
        Creates an interpolator for points (R,Z) as a function of
        normalised distance along the separatrix (0,1).
        """

        # Combine (R,Z) along the separatrix into points and compute distances between them
        points = np.column_stack((self.R_lcfs, self.Z_lcfs))
        dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))

        # Cumulative distance
        cum_dists = np.insert(np.cumsum(dists), 0, 0)

        self.lcfs_poloidal_length = cum_dists[-1]

        # Normalise distances (0,1)
        norm_dists = cum_dists / self.lcfs_poloidal_length

        # Create an interpolator for (R, Z)
        self.separatrix_interpolator = interp1d(norm_dists, points, axis=0)

        # In addition to creating the interpolator, we will also record the distance (incl. normalised distance)
        # along the separatrix that the X-point(s) are at, as this will be useful later.

        # Primary X-point
        self.index_xpt_lcfs = next(i for i, (r, z) in enumerate(zip(self.R_lcfs, self.Z_lcfs)) \
                                   if (r, z) == (self.R_xpt_lcfs,self.Z_xpt_lcfs))
        self.separatrix_dist_xpt_lcfs = cum_dists[self.index_xpt_lcfs]
        self.separatrix_dist_norm_xpt_lcfs = norm_dists[self.index_xpt_lcfs]

        # Secondary X-point (if DND)
        if self.DND:
            self.index_xpt_lcfs2 = next(i for i, (r, z) in enumerate(zip(self.R_lcfs, self.Z_lcfs)) \
                                        if (r, z) == (self.R_xpt_lcfs2,self.Z_xpt_lcfs2))
            self.separatrix_dist_xpt_lcfs2 = cum_dists[self.index_xpt_lcfs2]
            self.separatrix_dist_norm_xpt_lcfs2 = norm_dists[self.index_xpt_lcfs2]

        # We will also record the distance (incl. normalised distance) along the separatrix that the inboard midplane (IMP)
        # and the maximum/minimum vertical extrema points. The OMP is always at 0 distance(and index), by way of the ordering
        # of the points.

        # IMP
        self.index_imp_lcfs = next(i for i, (r, z) in enumerate(zip(self.R_lcfs, self.Z_lcfs)) \
                                   if (r, z) == (self.R_IMP,self.Z_IMP))
        self.separatrix_dist_imp_lcfs = cum_dists[self.index_imp_lcfs]
        self.separatrix_dist_norm_imp_lcfs = norm_dists[self.index_imp_lcfs]

        # Vertical extrema upper point
        self.index_vertical_upper_lcfs = next(i for i, (r, z) in enumerate(zip(self.R_lcfs, self.Z_lcfs)) \
                                              if (r, z) == (self.R_vertical_upper,self.Z_vertical_upper))
        self.separatrix_dist_vertical_upper_lcfs = cum_dists[self.index_vertical_upper_lcfs]
        self.separatrix_dist_norm_vertical_upper_lcfs = norm_dists[self.index_vertical_upper_lcfs]

        # Vertical extrema lower point
        self.index_vertical_lower_lcfs = next(i for i, (r, z) in enumerate(zip(self.R_lcfs, self.Z_lcfs)) \
                                              if (r, z) == (self.R_vertical_lower,self.Z_vertical_lower))
        self.separatrix_dist_vertical_lower_lcfs = cum_dists[self.index_vertical_lower_lcfs]
        self.separatrix_dist_norm_vertical_lower_lcfs = norm_dists[self.index_vertical_lower_lcfs]

    def get_points_along_separatrix(self,npoints=360):
        """Locates points along the separatrix.

        Returns the (R,Z) coordinates of npoints evenly spaced points along the
        separatrix around the core plasma.
        """

        # Get the normalised distances of the npoints
        dist_norm = np.linspace(0.0,1.0,npoints,endpoint=False)

        # Get the (R,Z) coordinates of the points
        points = self.separatrix_interpolator(dist_norm)

        R = [r for (r,z) in points]
        Z = [z for (r,z) in points]

        return R,Z

    def modify_wall(
            self,
            R_min = None,
            R_max = None,
            Z_min = None,
            Z_max = None,
    ):
        """Interactive editor for the tokamak wall structure.

        Takes as input the minimum/maximum (R,Z) coordinates for plotting. Prints the (R,Z)
        coordinates of the modified wall.
        """

        if R_min < 0.0:
            R_min = 0.0

        fig, ax = plt.subplots()
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.contour(self.R_2D,self.Z_2D,self.psi_2D,levels=100,alpha=0.25)
        ax.contour(self.R_2D,self.Z_2D,self.psi_2D,levels=[self.psi_lcfs],colors='r')
        ax.plot(self.wall_R,self.wall_Z,color='k',alpha=0.25)

        self.tokamak.plot(
            ax = ax,
            plot_wall = False,
            show = False
            )

        new_wall_R, new_wall_Z = interactive_shape_editor(
            self.tokamak.wall_R,
            self.tokamak.wall_Z,
            x_min = R_min,
            x_max = R_max,
            y_min = Z_min,
            y_max = Z_max,
            ax = ax,
            )

        print('new_wall_R: ',new_wall_R)
        print('new_wall_Z: ',new_wall_Z)

    def create_eq_with_new_grid(
            self,
            new_grid_R_min,
            new_grid_R_max,
            new_grid_Z_min,
            new_grid_Z_max,
            new_grid_nR,
            new_grid_nZ,
            ):
        """Creates a copy of Equilibrium on a new (R,Z) grid.

        Creates and returns a new forge.equilibrium Equilibrium object that
        is a copy of the exisiting equilibrium on an (R,Z) grid different to the
        original one. This is useful if for instance you want to extend the grid to encompass
        a new divertor structure.
        """

        # Create the new grid
        new_grid_R_1D = np.linspace(new_grid_R_min,new_grid_R_max,new_grid_nR,endpoint=True)
        new_grid_Z_1D = np.linspace(new_grid_Z_min,new_grid_Z_max,new_grid_nZ,endpoint=True)
        new_grid_R_2D, new_grid_Z_2D = np.meshgrid(new_grid_R_1D,new_grid_Z_1D,indexing="ij")

        # Unfortunately, we must calculate the flux from the plasma on the new
        # grid. This can be a very slow process at present.

        # Calculate the poloidal magnetic flux from the plasma current sources on the full 2D grid.
        # We will do this by moving over points in the existing grid, and, if there is current at a point, calculating
        # the flux on the entire new grid from this current source, before summing over all current sources.
        new_grid_psi_plas_2D = np.zeros(np.shape(new_grid_R_2D))

        for i in range(self.nR):
            print(str(i) + '/' + str(self.nR))
            for j in range(self.nZ):

                current = self.itor_2D[i][j]
                r = self.R_2D[i][j]
                z = self.Z_2D[i][j]

                if current != 0.0:

                    # Calculate the flux (Wb) from this current source on the new grid
                    new_grid_psi_plas_2D += greens.Greens(r, z, new_grid_R_2D, new_grid_Z_2D) * current

        # Next, we will calculate the flux from the coils on the new grid
        new_grid_psi_mach_2D = np.zeros(np.shape(new_grid_R_2D))

        # Iterate over the coils, calculating the flux from the coil on the new grid
        for coil in self.tokamak.coilset.values():

            new_grid_psi_mach_2D += coil.psiRZ(new_grid_R_2D, new_grid_Z_2D)

        # The total flux on the new grid is the sum of the plasma and coil fluxes
        new_grid_psi_2D = new_grid_psi_plas_2D + new_grid_psi_mach_2D

        # Machine flux plot
        fig, ax = plt.subplots(1,2)

        ax[0].set_xlabel('R (m)')
        ax[0].set_ylabel('Z (m)')
        ax[0].set_aspect('equal')
        ax[0].set_xlim((new_grid_R_min,new_grid_R_max))
        ax[0].set_ylim((new_grid_Z_min,new_grid_Z_max))
        ax[0].contour(self.R_2D,self.Z_2D,self.psi_mach_2D,levels=100)
        ax[0].title.set_text(r"$\rm\psi_{machine}$" + " initial")

        ax[1].set_xlabel('R (m)')
        ax[1].set_ylabel('Z (m)')
        ax[1].set_aspect('equal')
        ax[1].set_xlim((new_grid_R_min,new_grid_R_max))
        ax[1].set_ylim((new_grid_Z_min,new_grid_Z_max))
        ax[1].contour(new_grid_R_2D,new_grid_Z_2D,new_grid_psi_mach_2D,levels=100)
        ax[1].title.set_text(r"$\rm\psi_{machine}$" + " new")

        plt.show()

        # Plasma flux plot
        fig, ax = plt.subplots(1,2)

        ax[0].set_xlabel('R (m)')
        ax[0].set_ylabel('Z (m)')
        ax[0].set_aspect('equal')
        ax[0].set_xlim((new_grid_R_min,new_grid_R_max))
        ax[0].set_ylim((new_grid_Z_min,new_grid_Z_max))
        ax[0].contour(self.R_2D,self.Z_2D,self.psi_plas_2D,levels=100)
        ax[0].plot(self.wall_R,self.wall_Z,color='k')
        ax[0].title.set_text(r"$\rm\psi_{plasma}$" + " initial")

        ax[1].set_xlabel('R (m)')
        ax[1].set_ylabel('Z (m)')
        ax[1].set_aspect('equal')
        ax[1].set_xlim((new_grid_R_min,new_grid_R_max))
        ax[1].set_ylim((new_grid_Z_min,new_grid_Z_max))
        ax[1].contour(new_grid_R_2D,new_grid_Z_2D,new_grid_psi_plas_2D,levels=100)
        ax[1].title.set_text(r"$\rm\psi_{plasma}$" + " new")

        plt.show()

        # Total flux plot
        fig, ax = plt.subplots(1,2)

        ax[0].set_xlabel('R (m)')
        ax[0].set_ylabel('Z (m)')
        ax[0].set_aspect('equal')
        ax[0].set_xlim((new_grid_R_min,new_grid_R_max))
        ax[0].set_ylim((new_grid_Z_min,new_grid_Z_max))
        ax[0].contour(self.R_2D,self.Z_2D,self.psi_2D,levels=100)
        ax[0].contour(self.R_2D,self.Z_2D,self.psi_2D,levels=[self.psi_lcfs],colors='r')
        ax[0].plot(self.wall_R,self.wall_Z,color='k')
        ax[0].title.set_text(r"$\rm\psi$" + " initial")

        ax[1].set_xlabel('R (m)')
        ax[1].set_ylabel('Z (m)')
        ax[1].set_aspect('equal')
        ax[1].set_xlim((new_grid_R_min,new_grid_R_max))
        ax[1].set_ylim((new_grid_Z_min,new_grid_Z_max))
        ax[1].contour(new_grid_R_2D,new_grid_Z_2D,new_grid_psi_2D,levels=100)
        ax[1].contour(new_grid_R_2D,new_grid_Z_2D,new_grid_psi_2D,levels=[self.psi_lcfs],colors='r')
        ax[1].plot(self.wall_R,self.wall_Z,color='k')
        ax[1].title.set_text(r"$\rm\psi$" + " new")

        plt.show()

        # Now that we have the flux on the new grid, we need to calculate
        # the new set of equilibrium data that will be used to initialise
        # the new Equilibrium. This is the dictionary passed in as eq_data.
        # We will need to generate the 1D profile data again, as this is
        # sampled at a number of points equal to the radial resolution of
        # the grid.

        # Define the new points in psiN [0,1]
        new_grid_psin_data = np.linspace(0.0,1.0,new_grid_nR)

        # Get pprime, ffprime, q, pressure and fpol at these points
        new_grid_pprime_data = self.pprime_func(new_grid_psin_data)
        new_grid_ffprime_data = self.ffprime_func(new_grid_psin_data)
        new_grid_q_data = self.q_func(new_grid_psin_data)
        new_grid_pressure_data = self.pressure_func(new_grid_psin_data)
        new_grid_fpol_data = self.fpol_func(new_grid_psin_data)

        self.new_eq_data = {
            "R_min": new_grid_R_min,
            "R_max": new_grid_R_max,
            "nR": new_grid_nR,
            "R_1D": new_grid_R_1D,
            "dR": new_grid_R_1D[1] - new_grid_R_1D[0],
            "R_2D": new_grid_R_2D,

            "Z_min": new_grid_Z_min,
            "Z_max": new_grid_Z_max,
            "nZ": new_grid_nZ,
            "Z_1D": new_grid_Z_1D,
            "dZ": new_grid_Z_1D[1] - new_grid_Z_1D[0],
            "Z_2D": new_grid_Z_2D,

            "wall_R": self.wall_R,
            "wall_Z": self.wall_Z,

            "psi_2D": new_grid_psi_2D,
            "psi_lcfs": self.psi_lcfs,
            "psi_axis": self.psi_axis,

            "psin_data": new_grid_psin_data,
            "pprime_data": new_grid_pprime_data,
            "ffprime_data": new_grid_ffprime_data,
            "q_data": new_grid_q_data,
            "pressure_data": new_grid_pressure_data,
            "fpol_data": new_grid_fpol_data,

            "plasma_current": self.plasma_current,

            "fvac": self.fvac
        }

        new_eq = Equilibrium(
            eq_data = self.new_eq_data,
            tokamak = self.tokamak,
            calculate_flux_from_coils = True,
        )

        return new_eq
