"""FORGE - FORGE Optimises Reactor Geometries to improve Exhaust.

FORGE is a tool for optimising the magnetic geometry of the divertors in
diverted tokamak plasmas. It takes a standard GEQDSK equilibrium file and
a description of the PF coils, and tunes the currents in the coils to
optimise the magnetic geometry of the divertors. A simulated annealing
approach is used for the optimisation, with the user able to flexibly
tune the optimisation cost functions to their needs.

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

# Set up a library-level logger following Python best practice.
# By default only WARNING and above are shown. Users can configure
# the level or add handlers to the "forge" logger to see more detail:
#
#   import logging
#   logging.getLogger("forge").setLevel(logging.INFO)
#
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
