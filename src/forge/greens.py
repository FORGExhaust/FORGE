"""
Green's functions for the Grad-Shafranov equation.

The ``Greens`` function is based on the equivalent function in
gradshafranov.py from FreeGS (https://github.com/freegs-plasma/freegs).
Modified for FORGE to produce Wb instead of Wb/rad. All other functions
in this module are original to FORGE.

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
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

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4.0 * np.pi * 1.0e-07

# In the functions defined below, the following intermediate variables
# are used for legibility, wherein the quantity being defined at the point
# R,Z is subject to a source at the point Rc,Zc:

#    h = Z - Zc
#    u^2 = (R + Rc)^2 + h^2
#    k^2 = 4 * R * Rc / u^2
#    d^2 = (Rc - R)^2 + h^2
#    v^2 = Rc^2 + R^2 + h^2
#    w^2 = Rc^2 - R^2 - h^2
#    p = h^2 * d^2 * v^2
#    q = d^2 * u^2 * v^2 - d^2 * u^2 * ( R^2 - Rc^2 ) - h^2 * ( d^4 + u^4 )


def Greens(Rc, Zc, R, Z):
    """Poloidal magnetic flux due to a unit current.
    
    Calculates the poloidal magnetic flux (Wb) at (R,Z) due to a unit current at (Rc,Zc) using Green's function.
    """

    # Calculate k^2
    k2 = np.multiply(np.divide(np.multiply(R, Rc), (np.power((np.add(R, Rc)), 2) + np.power((np.subtract(Z, Zc)), 2))), 4.0)

    # Clip to between 0 and 1 to avoid nans e.g. when the coil is on a grid point
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)
    k = np.sqrt(k2)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
    result = np.multiply(
        MU0,
        np.multiply(
            np.sqrt(np.multiply(R, Rc)),
            np.divide(
                np.subtract(
                    np.multiply(np.subtract(2.0, k2), ellipk(k2)),
                    np.multiply(ellipe(k2), 2.0)
                    ),
                k
                )
            )
        )

    # If (R,Z) == (Rc, Zc) then ellipk diverges - in this case the result will be NaN and we will return 0 flux
    return np.nan_to_num(result,nan=0.0)

def Greens_dpsi_dR(Rc, Zc, R, Z):
    """First radial derivative of poloidal magnetic flux due to a unit current.

    Calculates the radial first derivative of the poloidal magnetic flux (Wb) at (R,Z) due to a unit current
    at (Rc,Zc) using Green's function.

    d(psi)/dR = ( (mu0 * R) / u ) * ( ( (w^2/d^2)*E(k^2) ) + K(k^2) )

    All intermediate variables are defined above.
    """

    # Calculate k^2
    k2 = np.multiply(np.divide(np.multiply(R, Rc), (np.power((np.add(R, Rc)), 2) + np.power((np.subtract(Z, Zc)), 2))), 4.0)

    # Clip to between 0 and 1 to avoid nans e.g. when the coil is on a grid point
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)

    # Calculate h^2
    h = Z - Zc
    h2 = h * h

    # Calculate u
    u2 = (R + Rc)**2.0 + h2
    u = np.sqrt(u2)

    # Calculate d^2
    d2 = (Rc - R)**2.0 + h2

    # Calculate w^2
    w2 = Rc**2.0 - R**2.0 - h2

    result = ( (MU0 * R) / u ) * ( ( (w2 / d2) * ellipe(k2) ) + ellipk(k2) )

    return np.nan_to_num(result,nan=0.0)

def Greens_dpsi_dZ(Rc, Zc, R, Z):
    """First vertical derivative of poloidal magnetic flux due to a unit current.

    Calculates the vertical first derivative of the poloidal magnetic flux (Wb) at (R,Z) due to a unit current
    at (Rc,Zc) using Green's function.

    d(psi)/dZ = ( (mu0 * h) / u ) * ( K(k^2) - ( (v^2/d^2) * E(k^2) ) )

    All intermediate variables are defined above.
    """

    # Calculate k^2
    k2 = np.multiply(np.divide(np.multiply(R, Rc), (np.power((np.add(R, Rc)), 2) + np.power((np.subtract(Z, Zc)), 2))), 4.0)

    # Clip to between 0 and 1 to avoid nans e.g. when the coil is on a grid point
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)

    # Calculate h2
    h = Z - Zc
    h2 = h * h

    # Calculate u
    u2 = (R + Rc)**2.0 + h2
    u = np.sqrt(u2)

    # Calculate d^2
    d2 = (Rc - R)**2.0 + h2

    # Calculate v^2
    v2 = Rc**2.0 + R**2.0 + h2

    result = ( (MU0 * h) / u ) * ( ellipk(k2) - ( (v2 / d2) * ellipe(k2) ) )

    return np.nan_to_num(result,nan=0.0)


def Greens_d2psi_dR2(Rc, Zc, R, Z):
    """Second radial derivative of poloidal magnetic flux due to a unit current.

    Calculates the radial second derivative of the poloidal magnetic flux (Wb) at (R,Z) due to a unit current
    at (Rc,Zc) using Green's function.

    d^{2}(psi)/dR^{2} = ( mu0 / ( d^{4} * u^{3} ) ) * ( ( x_1 * K(k^2) ) + ( x_2 * E(k^2) ) )

    All intermediate variables are defined above.
    """

    # Calculate k^2
    k2 = np.multiply(np.divide(np.multiply(R, Rc), (np.power((np.add(R, Rc)), 2) + np.power((np.subtract(Z, Zc)), 2))), 4.0)

    # Clip to between 0 and 1 to avoid nans e.g. when the coil is on a grid point
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)

    # Calculate h^2
    h = Z - Zc
    h2 = h * h

    # Calculate u
    u2 = (R + Rc)**2.0 + h2
    u = np.sqrt(u2)

    # Calculate d^2
    d2 = (Rc - R)**2.0 + h2

    # Calculate v^2
    v2 = Rc**2.0 + R**2.0 + h2

    # Calculate p
    p = h2 * d2 * v2

    # Calculate q
    q = d2 * u2 * v2 - d2 * u2 * ( R**2.0 - Rc**2.0 ) - h2 * ( d2**2.0 + u2**2.0 )

    result = ( MU0 / ( d2**2.0 * u2 * u ) ) * ( ( p * ellipk(k2) ) + ( q * ellipe(k2) ) )

    return np.nan_to_num(result,nan=0.0)

def Greens_d2psi_dZ2(Rc, Zc, R, Z):
    """Second vertical derivative of poloidal magnetic flux due to a unit current.

    Calculates the vertical second derivative of the poloidal magnetic flux (Wb) at (R,Z) due to a unit current
    at (Rc,Zc) using Green's function.

    d^{2}(psi)/dZ^{2} = ( -mu0 / ( d^2 * u^3 ) ) * ( ( v^2 * u^2 - h^2 * ( d^2 + ( ( k^2 * (u^2)^2 ) / d^2 ) ) )
                                                                * E(k^2) + ( h^2 * v^2 - u^2 * d^2 ) * K(k^2) )

    All intermediate variables are defined above.
    """

    # Calculate k^2
    k2 = np.multiply(np.divide(np.multiply(R, Rc), (np.power((np.add(R, Rc)), 2) + np.power((np.subtract(Z, Zc)), 2))), 4.0)

    # Clip to between 0 and 1 to avoid nans e.g. when the coil is on a grid point
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)

    # Calculate h^2
    h = Z - Zc
    h2 = h * h

    # Calculate u
    u2 = (R + Rc)**2.0 + h2
    u = np.sqrt(u2)

    # Calculate d^2
    d2 = (Rc - R)**2.0 + h2

    # Calculate v^2
    v2 = Rc**2.0 + R**2.0 + h2

    result = ( (-MU0) / (d2 * u2 * u) ) * ( (v2 * u2 - h2 * ( d2 + ( (k2 * (u2)**2.0) / d2 ) ) ) * ellipe(k2) + \
                                           ( h2 * v2 - u2 * d2 ) * ellipk(k2) )

    return np.nan_to_num(result,nan=0.0)

def Greens_d2psi_dR_dZ(Rc, Zc, R, Z):
    """Second mixed radial-vertical derivative of poloidal magnetic flux due to a unit current.

    Calculates the mixed second derivative of the poloidal magnetic flux (Wb) at (R,Z) due to a unit current
    at (Rc,Zc) using Green's function

    d^{2}(psi)/(dRdZ) = ( ( mu0 * h * R ) / ( d^2 * u^3 ) ) * ( ( -( ( 3.0 * u^2 ) + ( ( 4.0 * v^2 * w^2 ) / d^2 ) )
                                                                * E(k^2) ) + ( w^2 * K(k^2) ) )

    All intermediate variables are defined above.
    """

    # Calculate k^2
    k2 = np.multiply(np.divide(np.multiply(R, Rc), (np.power((np.add(R, Rc)), 2) + np.power((np.subtract(Z, Zc)), 2))), 4.0)

    # Clip to between 0 and 1 to avoid nans e.g. when the coil is on a grid point
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)

    # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)

    # Calculate h^2
    h = Z - Zc
    h2 = h * h

    # Calculate u
    u2 = (R + Rc)**2.0 + h2
    u = np.sqrt(u2)

    # Calculate d^2
    d2 = (Rc - R)**2.0 + h2

    # Calculate v^2
    v2 = Rc**2.0 + R**2.0 + h2

    # Calculate w^2
    w2 = Rc**2.0 - R**2.0 - h2

    result = ( ( MU0 * h * R ) / ( d2 * u2 * u ) ) * ( ( -( ( 3.0 * u2 ) + ( ( 4.0 * v2 * w2 ) / d2 ) ) \
                                                        * ellipe(k2) ) + ( w2 * ellipk(k2) ) )

    return np.nan_to_num(result,nan=0.0)

def Greens_Br(Rc, Zc, R, Z):
    """Calculates the radial magnetic field at (R,Z) due to a unit current at (Rc, Zc).

    Br = -(1/2piR) * d(psi)/dZ
    """

    return -(1.0 / (2.0  * np.pi * R)) * Greens_dpsi_dZ(Rc, Zc, R, Z)

def Greens_Bz(Rc, Zc, R, Z):
    """Calculates the vertical magnetic field at (R,Z) due to a unit current at (Rc, Zc).

    Bz = (1/2piR) * d(psi)/dR
    """

    return (1.0 / (2.0  * np.pi * R)) * Greens_dpsi_dR(Rc, Zc, R, Z)

def Greens_dBr_dR(Rc, Zc, R, Z):
    """Calculates the radial derivative of the radial magnetic field at (R,Z) due to a unit current at (Rc,Zc).

    d(Br)/dR = (1/2pi) * ( ( (1/R^{2}) * (dpsi/dZ) ) - ( (1/R) * (d^{2}(psi)/dRdZ) ) )
    """

    return (1.0 / (2.0 * np.pi)) * ( ( (1.0 / (R * R)) * Greens_dpsi_dZ(Rc, Zc, R, Z) ) \
                                    - ( (1.0 / R) * Greens_d2psi_dR_dZ(Rc, Zc, R, Z) ) )

def Greens_dBr_dZ(Rc, Zc, R, Z):
    """Calculates the vertical derivative of the radial magnetic field at (R,Z) due to a unit current at (Rc,Zc).

    d(Br)/dZ = (1/2pi) * (-1/R * ( d^{2}(psi)/dZ^{2} ) )
    """

    return (1.0 / (2.0 * np.pi)) * (-1.0 / R) * Greens_d2psi_dZ2(Rc, Zc, R, Z)

def Greens_dBz_dR(Rc, Zc, R, Z):
    """Calculates the radial derivative of the vertical magnetic field at (R,Z) due to a unit current at (Rc,Zc).

    d(Bz)/dR = (1/2pi) * ( ( (-1/R^{2}) * (dpsi/dR) ) + ( (1/R) * (d^{2}(psi)/dR^{2}) ) )
    """

    return (1.0 / (2.0 * np.pi)) * ( ( (-1.0 / (R * R)) * Greens_dpsi_dR(Rc, Zc, R, Z) ) \
                                    + ( (1.0 / R) * Greens_d2psi_dR2(Rc, Zc, R, Z) ) )

def Greens_dBz_dZ(Rc, Zc, R, Z):
    """Calculates the vertical derivative of the vertical magnetic field at (R,Z) due to a unit current at (Rc,Zc).

    d(Bz)/dZ = (1/2pi) * (1/R * ( d^{2}(psi)/dZdR ) ) = (1/2pi) * (1/R * ( d^{2}(psi)/dRdZ ) )
    """

    return (1.0 / (2.0 * np.pi)) * (1.0 / R) * Greens_d2psi_dR_dZ(Rc, Zc, R, Z)
