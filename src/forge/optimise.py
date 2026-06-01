"""Contains the principal FORGE optimisation procedure.

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
import math
import threading
from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from shapely import LineString, Point
from shapely.prepared import prep

from forge.critical import find_critical
from forge.equilibrium import Equilibrium
from forge.utils import (
    densify_closed_shape,
    estimate_xpoint_location,
    grid_points_inside_linestring,
)

logger = logging.getLogger(__name__)

ONE_2PI = 1.0 / (2.0 * np.pi)

# Shared singleton returned by check_field_line_intersection when there
# is no intersection.  Avoids allocating a new dict on every call (the
# common case — the vast majority of RK4 steps do not hit anything).
_NO_INTERSECTION = {"intersects": False}


def _fast_bilinear(grid, fi, fj, nR_m1, nZ_m1):
    """Bilinear interpolation on a regular grid from pre-computed fractional indices.

    Avoids the ~50-100 us Python overhead of RectBivariateSpline.ev() per scalar
    call by doing the interpolation directly.  On a regular grid this reduces to
    simple index arithmetic plus four multiply-adds.

    Parameters
    ----------
    grid : 2D numpy array (nR, nZ)
        The data field to interpolate.
    fi : float
        Fractional index in the R direction: (R - R_min) / dR.
    fj : float
        Fractional index in the Z direction: (Z - Z_min) / dZ.
    nR_m1 : int
        nR - 1 (maximum valid integer index in the R direction).
    nZ_m1 : int
        nZ - 1 (maximum valid integer index in the Z direction).

    Returns
    -------
    float
        Interpolated value.
    """
    # Clamp integer indices to valid range
    i = int(fi)
    j = int(fj)
    if i < 0:
        i = 0
    elif i > nR_m1 - 1:
        i = nR_m1 - 1
    if j < 0:
        j = 0
    elif j > nZ_m1 - 1:
        j = nZ_m1 - 1

    # Fractional parts
    dr = fi - i
    dz = fj - j

    # Bilinear interpolation (4 grid lookups, 4 multiplies)
    return (grid[i, j]     * (1.0 - dr) * (1.0 - dz) +
            grid[i + 1, j] * dr          * (1.0 - dz) +
            grid[i, j + 1] * (1.0 - dr) * dz +
            grid[i + 1, j + 1] * dr      * dz)


def _fast_bilinear_vec(grid, fi_arr, fj_arr, nR_m1, nZ_m1):
    """Vectorised bilinear interpolation on a regular grid.

    Same as ``_fast_bilinear`` but operates on 1-D arrays of fractional
    indices, returning a 1-D array of interpolated values.  This is used
    for the connection-length calculation where Br/Bz are needed at points
    along a traced field line.
    """
    i = np.clip(fi_arr.astype(np.intp), 0, nR_m1 - 1)
    j = np.clip(fj_arr.astype(np.intp), 0, nZ_m1 - 1)
    dr = fi_arr - i
    dz = fj_arr - j
    return (grid[i, j]         * (1.0 - dr) * (1.0 - dz) +
            grid[i + 1, j]     * dr          * (1.0 - dz) +
            grid[i, j + 1]     * (1.0 - dr) * dz          +
            grid[i + 1, j + 1] * dr          * dz)


class Optimiser:
    """Object representing the simulated annealing optimiser.

    This Optimiser will carry out the simulated annealing-based optimisation of
    of the magnetic geometry of the divertor(s).
    
    Parameters
    ----------
    eq : forge.equilibrium Equilibrium object
        The equilibrium that will be optimised.
    divertor_data : dict
        A dictionary of data pertaining to the divertor region(s) to be optimised.
    tokamak_initial : forge.machine Machine object
        Tokamak associated with the initial starting equilibrium. This machine
        should contain the PF coil set used to make the the initial equilibrium.
    tokamak_opt : forge.machine Machine object
        Tokamak that the user wishes to perform the optimisation on. This machine
        does not need to have the same PF coil set as tokamak_initial. If this is None,
        then tokamak_initial - the tokamak used to make the initial equilibrium - will
        be used. If tokamak_opt is not None then an estimation of an initial set of
        coil currents will be made.
    max_evals : int
        The maximum number of cost function evaluations to carry out (not including the initial evaluation).
    current_step_size_factor : float
        The step size taken when moving in current space is given by this factor of a typical coil current size
        taken from the set of initial coil currents.
    initial_temperature : float
        The initial temperature of the system. This pseudo-temperature is what is lowered during the
        annealing process - analogous of a hot metal being cooled.
    threshold_acceptance_rate_decay : float
        The annealing acceptance rate, over a specified window, must fall below a threshold value before cooling is
        allowed. This threshold acceptance rate is temperature dependent, and evolves as R = R0 * exp(-λ*(1-T/T0))
        where R0 is the initial threshold value, T is the current temperature, T0 is the initial temperature and λ
        is this parameter, the threshold_acceptance_rate_decay constant. If λ is larger, the threshold drops
        more aggresively. If the optimiser should be more explorative, λ should be lower.
    initial_threshold_acceptance_rate : float
        The initial threshold value for the acceptance rate, R0.
    n_window : int
        The size of the window over which the rolling acceptance rate is calculated.
    constraints : dict
        A dictionary of constraints to be used. See the "generate_default_constraints" method.
    field_line_trace_step_size : float
        The poloidal step size (in metres) used by the RK4 field-line tracer.
    field_line_trace_max_steps : int
        The maximum number of steps the field-line tracer will take before
        abandoning a trace (e.g. if it never reaches the wall).
    field_line_trace_psi_tollerance : float
        Target maximum value of the relative error on the poloidal magnetic flux of the traced field line(s).
    buffer_intersection_penalty_factor : float
        A multiplier on the connection length cost term to be applied if the divertor leg intersects a buffer on
        the tokamak's wall.
    initial_total_connection_length_cost : float
        The initial total combined connection length cost across all divertor regions.
    initial_total_strike_point_distance_cost : float
        The initial total combined strike point distance cost across all divertor regions.
    initial_coil_currents_cost : float
        The initial cost due to the coil currents.
    self.initial_xpoint_regions_cost : float
        The initial cost due to regions in which additional X-points are ncouraged to be formed.
    use_xpoint_regions : bool
        Flag for using regions in which secondary divertor X-point(s) will be encouraged to form.
    xpoint_regions : dict
        A dictionary mapping divertor region names to X-point region definitions.
        Required when ``use_xpoint_regions`` is True.
        Each key is a divertor region name (e.g. ``"lower_outer"``) and each
        value is a dictionary containing the R and Z coordinates of the region
        boundary: ``{"lower_outer": {"R": [...], "Z": [...]}, ...}``.
    max_magnetic_disconnection_factor : float
        Maximum relative difference in poloidal magnetic flux between a secondary divertor X-point
        and the separatrix, i.e. |ψ_xpt − ψ_lcfs| / |ψ_lcfs|. This sets the scale at which the
        magnetic disconnection cost saturates when forming an X-point target (XPT) geometry.
    use_buffers : bool
        Flag for adding buffer regions around parts of the wall.
    buffers : dict
        Buffer region definitions keyed by divertor region name. Required when
        ``use_buffers`` is True. Each value is a list of dictionaries containing
        the R and Z coordinates of the wall segment endpoints and the buffer
        distance::

            {"lower_outer": [{"R": [R1, R2], "Z": [Z1, Z2], "distance": d}, ...], ...}
    initial_alpha : float
        The initial value of the alpha parameter used in the Metropolis-Hastings
        acceptance criterion. Alpha controls the sensitivity of the acceptance
        probability to the energy change. FORGE will automatically tune alpha
        to meet the first threshold acceptance rate, after which it is frozen.
    alpha_update_factor : float
        The factor by which the parameter alpha is updated in order to induce a first cooling event.
    max_cooling_factor : float
        The maximum allowed temperature cooling multiplication factor, β_max.
    min_cooling_factor : float
        The minimum allowed temperature cooling multiplication factor, β_min.
    detailed_logging : bool
        Flag for carrying out a detailed logging of data.
    cost_termination_fraction : float
        If the incumbent cost drops below this fraction of the initial cost, the
        annealing will terminate early. For example, 0.01 means the optimisation
        stops once the cost reaches 1% of its initial value.
    min_temperature : float
        The minimum temperature below which the annealing will stop.
    """

    def __init__(
        self,
        eq,
        tokamak_initial,
        tokamak_opt = None,
        divertor_data = None,
        max_evals = 3000,
        current_step_size_factor = 0.05,
        estimate_initial_currents = False,
        initial_temperature = 10.0,
        threshold_acceptance_rate_decay = 2.0,
        initial_threshold_acceptance_rate = 0.75,
        n_window = 50,
        constraints = None,
        field_line_trace_step_size = 0.15,
        field_line_trace_max_steps = 1000,
        field_line_trace_psi_tollerance = 0.01,
        buffer_intersection_penalty_factor = 1.05,
        initial_total_connection_length_cost = 1.0,
        initial_total_strike_point_distance_cost = 1.0,
        initial_coil_currents_cost = 1.0,
        initial_xpoint_regions_cost = 1.0,
        use_xpoint_regions = False,
        xpoint_regions = None,
        max_magnetic_disconnection_factor = 0.5,
        use_buffers = False,
        buffers = None,
        initial_alpha = 50.0,
        alpha_update_factor = 1.05,
        max_cooling_factor = 0.99,
        min_cooling_factor = 0.9,
        detailed_logging = False,
        cost_termination_fraction = 0.01,
        min_temperature = 1.0,
        on_iteration = None,
        stop_event = None,
    ):
        self.eq = eq
        self.divertor_data = divertor_data
        self.tokamak_initial = deepcopy(tokamak_initial)

        # Iteration callback: called after each iteration with the optimiser as argument.
        # If None (default), no callback is invoked, preserving the original scripting behaviour.
        self.on_iteration = on_iteration

        # Cancellation mechanism: if a threading.Event is supplied, the optimise loop will
        # check it every iteration and stop gracefully when set.
        if stop_event is None:
            self._stop_event = threading.Event()
        else:
            self._stop_event = stop_event

        # The user can choose to use the same tokamak as was used to
        # produce the initial equilibrum, or provide a separate one.
        if tokamak_opt is None:

            self.tokamak_opt = deepcopy(tokamak_initial)

        else:

            self.tokamak_opt = tokamak_opt

        self.max_evals = max_evals
        self.current_step_size_factor = current_step_size_factor
        self.estimate_initial_currents = estimate_initial_currents
        self.threshold_acceptance_rate_decay = threshold_acceptance_rate_decay
        self.initial_temperature = initial_temperature
        self.initial_threshold_acceptance_rate = initial_threshold_acceptance_rate
        self.n_window = n_window
        self.field_line_trace_step_size = field_line_trace_step_size
        self.field_line_trace_max_steps = field_line_trace_max_steps
        self.field_line_psi_tollerance = field_line_trace_psi_tollerance
        self.buffer_intersection_penalty_factor = buffer_intersection_penalty_factor
        self.initial_total_connection_length_cost = initial_total_connection_length_cost
        self.initial_total_strike_point_distance_cost = initial_total_strike_point_distance_cost
        self.initial_coil_currents_cost = initial_coil_currents_cost
        self.initial_xpoint_regions_cost = initial_xpoint_regions_cost
        self.use_xpoint_regions = use_xpoint_regions
        self.xpoint_regions = xpoint_regions
        self.max_magnetic_disconnection_factor = max_magnetic_disconnection_factor
        self.use_buffers = use_buffers
        self.buffers_input = buffers
        self.initial_alpha = initial_alpha
        self.alpha_update_factor = alpha_update_factor
        self.max_cooling_factor = max_cooling_factor
        self.min_cooling_factor = min_cooling_factor
        self.detailed_logging = detailed_logging
        self.cost_termination_fraction = cost_termination_fraction
        self.min_temperature = min_temperature

        # Pre-compute the Green's functions on the 2D grid for each coil in the coilset. This will
        # later be used to quickly calculate the new machine flux as the optimiser changes the
        # coil currents.
        self.compute_coilset_greens_on_grid()

        # Pre-compute the background plasma dpsi/dR and dpsi/dZ grids.  The plasma
        # is treated as a fixed background throughout the optimisation, so these
        # are evaluated once and cached.
        self._psi_plas_dpsi_dR_grid = self.eq.dpsi_plas_dR_RZ(self.eq.R_2D, self.eq.Z_2D)
        self._psi_plas_dpsi_dZ_grid = self.eq.dpsi_plas_dZ_RZ(self.eq.R_2D, self.eq.Z_2D)

        # Cache the grid metadata that the fast tracer / bilinear interpolators need.
        # These never change (the equilibrium grid is fixed).
        self._grid_R_min  = self.eq.R_min
        self._grid_Z_min  = self.eq.Z_min
        self._grid_inv_dR = 1.0 / self.eq.dR
        self._grid_inv_dZ = 1.0 / self.eq.dZ
        self._grid_nR_m1  = self.eq.nR - 1
        self._grid_nZ_m1  = self.eq.nZ - 1

        # Build a prepared (spatially indexed) version of the wall and buffer geometries.
        # This dramatically accelerates the per-step intersection tests inside the field
        # line tracer, which is the innermost hot loop of the optimisation.
        self._prepared_wall = prep(self.tokamak_opt.wall)

        # Set those settings related to the use of constraints in the optimisation.
        if constraints is None:

            # The user did not specify the constraints settings. Use default values.
            self.constraints = self.generate_default_constraints()

        else:

            self.constraints = constraints

        # Generate the constraints using the above settings.
        self.generate_constraints()

        # Set the initial coil currents and 2D map of poloidal magnetic flux:
        if self.estimate_initial_currents:

            # Estimate initial currents in the PF coils
            self.estimate_currents()
            self.initial_coil_currents = self.initial_currents_estimate

            # Initialise the 2D map of poloidal magnetic flux that will be evolved during the optimisation.
            self.psi_2D = self.initial_psi_2D_estimate

        else:

            # Use the currents that come with the tokamak
            self.initial_coil_currents = deepcopy(self.tokamak_opt.get_currents())

            # Initialise the 2D map of poloidal magnetic flux that will be evolved during the optimisation.
            self.psi_2D = deepcopy(self.eq.psi_2D)

        # Extract the divertor regions that the user has selected for optimisation, from the
        # divertor_data provided.
        self.divertor_regions = list(self.divertor_data.keys())

        # Determine the starting location for field line traces of the divertor region(s) aswell
        # as the field line trace direction.
        self.init_divertor_data()

        # Initialise data for regions in which additional X-points will be encouraged to form, if
        # such regions are present within the Machine.
        if self.use_xpoint_regions:

            self.init_xpoint_regions()

        else:

            self.N_xpoint_regions = 0

        # Create buffer regions around parts of the wall.
        if self.use_buffers:

            self.create_buffers()

            # Build prepared (spatially indexed) versions of each buffer geometry
            # for the same reason as the wall: O(log N) pre-filtering in the
            # field-line tracing hot loop.  Keyed by divertor region.
            self._prepared_buffers = {
                region: [prep(b) for b in geoms]
                for region, geoms in self.buffers.items()
            }

        else:

            self.buffers = None
            self._prepared_buffers = None

    def optimise(self):
        """Performs the simulated annealing-based optimisation of the divertor(s)' magnetic geometry."""

        # Keep track of the number of cost function evaluations
        self.N_evals = 0

        # Get the coil current step size from the initial coil currents.
        currents = self.initial_coil_currents
        self.init_current_step_size(currents)

        # Calculate the initial cost function, which will set the initial value for the cost function
        # from the previous step, as we are yet to actually start iterating. Note that the Equilibrium object
        # will not itself be updated until the annealing is complete, hence any new flux maps generated during
        # the annealing must be passed directly to the cost function. The output of calculate_cost is a dictionary
        # containing all the key data defining the state and its associated cost function.
        previous_state_data = self.calculate_cost(
            currents = currents,
            flux_map = self.psi_2D,
        )

        # As this is the initial state, these initial currents will also serve as the first set of accepted currents.
        accepted_currents = deepcopy(currents)

        # Record the initial cost for use in early termination checks.
        self.initial_cost = previous_state_data["cost"]

        # As this is the first cost function evaluation it is by definition also the best state currently explored.
        # Hence, the incumbent state will be this one, for now.
        self.incumbent_data = previous_state_data
        self.incumbent_data["iteration_num"] = 0

        # Now we can actually begin the simulated annealing algorithm. To briefly quote Google:
        # "Simulated annealing is a probabilistic optimisation technique inspired by the physical process of
        # annealing metals, which gradually cools a material to achieve a stable, low-energy state. Its core principle
        # is to start at a high "temperature" allowing many "uphill" or worse moves to escape local optima and explore a
        # broad solution space. As the "temperature" gradually decreases over time (following a cooling schedule), the
        # algorithm becomes less likely to accept worse solutions, eventually settling into a near-optimal solution,
        # similar to how a metal crystalizes into its lowest energy, most ordered state".
        #
        # Our temperature here is not associated with a real-world counterpart. The measure by which we will assess
        # the goodness of each state is the cost function.

        # Initialise the number of cost function evaluations carried out (not including the first one).
        # The optimisation will hard-stop if this exceeds the defined maximum number of evaluations.
        self.num_evals = 0

        # Initialise list(s) that will be used to track key quantities
        # List of state data
        self.tracking_temperature = []
        self.tracking_energy_change = []
        self.tracking_acceptance = []
        self.tracking_cost = []
        self.tracking_cost_strike_point_distance = []
        self.tracking_cost_connection_length = []
        self.tracking_cost_coil_currents = []
        self.tracking_cost_xpoint_regions = []
        self.tracking_acceptance_rate = []
        self.tracking_acceptance_prob = []
        self.tracking_alpha = []

        # Additional tracking data, which is tied to specific divertor regions.
        self.tracking_connection_length = {}
        self.tracking_strike_point_R = {}
        self.tracking_strike_point_Z = {}
        self.tracking_field_lines_R = {}
        self.tracking_field_lines_Z = {}

        for divertor_region in self.divertor_regions:
            self.tracking_connection_length[divertor_region] = []
            self.tracking_strike_point_R[divertor_region] = []
            self.tracking_strike_point_Z[divertor_region] = []

        # If more detailed logging is desired, initialise lists that will track other key data
        if self.detailed_logging:

            # In the event that detailed logging is desired, the following will be recorded
            # at every iteration:
            #
            # - coil currents
            # - 2D flux map (can also be recalculated from coil currents, but recorded for convenience)
            # - incumbent state
            # - cooling rate factor
            # - threshold acceptance rate

            self.tracking_coil_currents = []
            self.tracking_psi_2D = []
            self.tracking_incumbent_state = []
            self.tracking_cooling_factor = []
            self.tracking_threshold_acceptance_rate = []

        # Seed the tracking lists with the initial (unperturbed) state so that
        # iteration 0 on the plots corresponds to the true starting point.
        self.tracking_temperature.append(self.initial_temperature)
        self.tracking_energy_change.append(0.0)
        self.tracking_cost.append(previous_state_data["cost"])
        self.tracking_cost_strike_point_distance.append(previous_state_data["cost_strike_point_distance"])
        self.tracking_cost_connection_length.append(previous_state_data["cost_connection_length"])
        self.tracking_cost_coil_currents.append(previous_state_data["cost_coil_currents"])
        self.tracking_cost_xpoint_regions.append(previous_state_data["cost_xpoint_regions"])
        self.tracking_acceptance_rate.append(np.nan)
        self.tracking_acceptance_prob.append(np.nan)
        self.tracking_alpha.append(self.initial_alpha)

        for divertor_region in self.divertor_regions:
            self.tracking_connection_length[divertor_region].append(
                previous_state_data["divertors"][divertor_region]["connection_length"]
            )
            self.tracking_strike_point_R[divertor_region].append(
                previous_state_data["divertors"][divertor_region]["strike_point_R"]
            )
            self.tracking_strike_point_Z[divertor_region].append(
                previous_state_data["divertors"][divertor_region]["strike_point_Z"]
            )

        if self.detailed_logging:
            self.tracking_coil_currents.append(previous_state_data["currents"])
            self.tracking_psi_2D.append(previous_state_data["psi_2D"])
            self.tracking_incumbent_state.append(previous_state_data)
            self.tracking_cooling_factor.append(np.nan)
            self.tracking_threshold_acceptance_rate.append(np.nan)

        # Set an initial value for alpha - this will be adjusted automatically, in the event that
        # the resultant acceptance rate is (initially) higher than the (first) value of the
        # threshold acceptance rate, which is used to signal cooling. It can be challenging to
        # pick a value of alpha that results in the desired acceptance rate, hence FORGE will
        # automatically tune it. Note, that this tuning only occurs to help the FIRST threshold
        # accpetance rate value be met - after this first cooling event, alpha is frozen (if we
        # kept adjusting it, we would just forcefully quench the system down to the final temperature).
        self.alpha = self.initial_alpha
        self.update_alpha = True

        # The temperature will change based on a dynamic cooling factor. The associated cooling
        # factor is initialy NaN and will be set at the first cooling event.
        self.cooling_factor = np.nan

        # Set the current temperature
        # (plotting is handled externally, e.g. by the GUI)

        # Set the current temperature of the system equal to the initial temperature
        self.temperature = self.initial_temperature

        while self.num_evals < self.max_evals and self.temperature > self.min_temperature \
              and self.incumbent_data["cost"] > self.cost_termination_fraction * self.initial_cost \
              and not self._stop_event.is_set():

            # Perform simulated annealing

            # Initialise the number of cost function evaluations (not including the first one)
            # performed at the current temperature value, excluding those evaluations where
            # there was a negative energy change - only counting those states that were
            # accepted/rejected probabalistically. These states, where the cost function increased,
            # are referred to as having moved "uphill" in the cost function landscape.
            self.num_evals_at_temp_uphill = 0

            # Determine the threshold acceptance rate over the specified window. If the acceptance
            # rate over the window drops below the threshold value, then it is time to cool the
            # system in accordance with the cooling schedule. Note that the window will only include
            # states that result from an uphill move in the cost function landscape.
            self.threshold_acceptance_rate = self.initial_threshold_acceptance_rate * \
                np.exp(-self.threshold_acceptance_rate_decay * (1.0 - (self.temperature / self.initial_temperature)))

            # Create the buffer of length n_window that tracks the rolling acceptance rate. This is reset at each (new)
            # temperature.
            acceptance_buffer = deque([1.0] * self.n_window, maxlen=self.n_window)

            # Start annealing
            keep_iterating = True
            ready_to_cool = False

            while keep_iterating:

                # Generate a new neighbouring state. This is done by randomly perturbing the coil
                # currents used to generate the state, with this perturbation about the most recently
                # accepted set of coil currents.
                neighbour = self.generate_new_neighbour_in_current_space(
                                currents = accepted_currents,
                                step_size = self.current_step_size,
                )

                # Evaluate the new neighbour candidate point
                new_state_data = self.evaluate_neighbour(
                    neighbour = neighbour,
                    previous_accepted_cost = previous_state_data["cost"],
                )

                # Check if the new candidate neighbour point is to be accepted or not
                if new_state_data["acceptance"]:

                    # Acceptance successful
                    self.psi_2D = new_state_data["psi_2D"]
                    currents = new_state_data["currents"]
                    accepted_currents = currents.copy()

                    # This newly accepted state will become the previously accepted state for the
                    # next state that will be produced in the next iteration.
                    previous_state_data = new_state_data

                    # Update the global acceptance tracker
                    self.tracking_acceptance.append(1.0)

                    # Check if the newly accepted state is the incumbent.
                    # Sometimes we allow worse solutions (particularly at higher temperature)
                    # hence this new accepted state wont necessarily be the incumbent.
                    if previous_state_data["cost"] < self.incumbent_data["cost"]:

                        self.incumbent_data = previous_state_data
                        self.incumbent_data["iteration_num"] = self.num_evals

                else:

                    # Update the global acceptance tracker
                    self.tracking_acceptance.append(0.0)

                # Next we check if the state was accepted/rejected probabilistically or not.
                # If the acceptence probability was eactly equal to 1 - this corresponds to
                # there having been a downhill movement in the cost function landscape, which
                # is deterministically accepted. Hence, probabilistic evaluations, which correspond
                # to uphill movements in the cost function landscape, will have occured when the acceptance
                # probability was not exactly equal to 1.
                # We need to record the acceptence rate for probabilistic accpetances only.
                if new_state_data["acceptance_prob"] != 1.0:

                    # We will also increment the counter that tracks the number of probabilistic
                    # evaluations.
                    self.num_evals_at_temp_uphill += 1

                    # State was evaluated probabilistically.
                    # The deque has a fixed maxlen so appending automatically
                    # discards the oldest entry — no manual pop(0) needed.
                    if new_state_data["acceptance"]:

                        acceptance_buffer.append(1.0)

                    else:

                        acceptance_buffer.append(0.0)

                # Increase counters
                self.num_evals += 1

                # Update tracking data
                self.tracking_temperature.append(self.temperature)
                self.tracking_energy_change.append(new_state_data["energy_change"])
                self.tracking_cost.append(new_state_data["cost"])
                self.tracking_cost_strike_point_distance.append(new_state_data["cost_strike_point_distance"])
                self.tracking_cost_connection_length.append(new_state_data["cost_connection_length"])
                self.tracking_cost_coil_currents.append(new_state_data["cost_coil_currents"])
                self.tracking_cost_xpoint_regions.append(new_state_data["cost_xpoint_regions"])
                self.tracking_acceptance_prob.append(new_state_data["acceptance_prob"])
                self.tracking_alpha.append(self.alpha)

                # Update tracking data for the divertor regions
                for divertor_region in self.divertor_regions:

                    # Update connection length
                    self.tracking_connection_length[divertor_region].append(
                        new_state_data["divertors"][divertor_region]["connection_length"]
                        )

                    # Update strike point R coordinate
                    self.tracking_strike_point_R[divertor_region].append(
                        new_state_data["divertors"][divertor_region]["strike_point_R"]
                        )

                    # Update strike point Z coodinate
                    self.tracking_strike_point_Z[divertor_region].append(
                        new_state_data["divertors"][divertor_region]["strike_point_Z"]
                        )

                    # Update traced field line R coordinates
                    self.tracking_field_lines_R[divertor_region] = new_state_data["divertors"][divertor_region]["field_line_R"]

                    # Update traced field line Z coordinates
                    self.tracking_field_lines_Z[divertor_region] = new_state_data["divertors"][divertor_region]["field_line_Z"]

                if self.detailed_logging:

                    self.tracking_coil_currents.append(new_state_data["currents"])
                    self.tracking_psi_2D.append(new_state_data["psi_2D"])
                    self.tracking_incumbent_state.append(self.incumbent_data)
                    self.tracking_cooling_factor.append(self.cooling_factor)
                    self.tracking_threshold_acceptance_rate.append(self.threshold_acceptance_rate)

                # Check whether or not the system is ready to cool.
                # We must fill up the buffer before evaluating.
                if self.num_evals_at_temp_uphill >= self.n_window:

                    # Calculate the rolling acceptance rate over the window
                    acceptance_rate = np.mean(acceptance_buffer)

                    # Record the acceptance rate
                    self.tracking_acceptance_rate.append(acceptance_rate)

                    # Check if this falls below the threshold acceptance rate.
                    if acceptance_rate <= self.threshold_acceptance_rate:

                        ready_to_cool = True

                        # Get the number of iterations required to reach the threshold acceptance rate
                        # not including the initial filling of the buffer
                        self.iterations_to_acceptance = max(self.num_evals_at_temp_uphill - self.n_window, 0)

                        # The first time we match the threshold acceptance rate, we can say that
                        # the value of alpha is appropriate, and hence no longer needs updating.
                        self.update_alpha = False

                    else:

                        # Update the (initial) value of alpha (if it hasn't been done so already), in order
                        # to adjust the accpetance rate (if required). The update to alpha only occurs when the number
                        # # of probabilistic cost evaluations at the current temperature is an integer multiple of the
                        # window size. The reason for this is that there is an inertia to the acceptance rate - it takes
                        # time for the effects of the updated value of alpha to (noticebly) affect the acceptance rate over
                        # the window. Hence we should wait until the another window's worth of probabilistic cost evaluations
                        # have passed, before changing alpha again. If we don't do this then we can quickly over-correct alpha,
                        # leading to a sudden quench of the system. It is also worth highlighting that the optimiser may go on
                        # a streak of downhill (in the cost function landscape) improvements. Hence, we need to ensure that
                        # those downhill iterations are not included in the evaluations counter, as otherwise alpha would just
                        # skyrocket during a later period of improvement, which would lead to a sudden quench once the streak
                        # ends.
                        if self.update_alpha and self.num_evals_at_temp_uphill%self.n_window == 0:

                            self.alpha *= self.alpha_update_factor

                else:

                    # The system cannot cool unless the acceptance rate of the window drops below some
                    # threshold value. The acceptance rate over the window cannot be evaluated until
                    # the number of iterations has reached the window size.
                    ready_to_cool = False

                    # As the acceptance rate cannot be evaluated, record the result as a NaN.
                    self.tracking_acceptance_rate.append(np.nan)

                # Fire the iteration callback (if one has been provided)
                if self.on_iteration is not None:
                    self.on_iteration(self)

                # Check for external stop request
                if self._stop_event.is_set():
                    logger.info('Optimisation stopped by external request.')
                    keep_iterating = False

                # Check for early termination if the incumbent cost has dropped below the threshold
                cost_below_threshold = self.incumbent_data["cost"] <= self.cost_termination_fraction * self.initial_cost

                if (self.num_evals >= self.max_evals) or ready_to_cool or cost_below_threshold:

                    if cost_below_threshold:
                        logger.info(
                            'Early termination: incumbent cost (%.6e) fell below %.1f%% of initial cost (%.6e).',
                            self.incumbent_data["cost"],
                            self.cost_termination_fraction * 100.0,
                            self.initial_cost,
                        )

                    keep_iterating = False

            # Either we have are ready to cool, or we have met our maximum number of iterations
            if ready_to_cool:

                # Ready to lower the temperature

                # Lower the temperature according to the cooling shedule.
                # Calculate the cooling factor based on how many iterations were
                # required to reach the threshold acceptance rate.
                self.update_cooling_factor()
                self.temperature *= self.cooling_factor

        # Log the reason the optimisation stopped
        if self._stop_event.is_set():
            logger.info('Optimisation stopped by user request.')
        elif self.num_evals >= self.max_evals:
            logger.info('Optimisation stopped: max iterations (%d) reached.', self.max_evals)
        elif self.temperature <= self.min_temperature:
            logger.info('Optimisation stopped: min temperature (%.4e) reached.', self.min_temperature)
        elif self.incumbent_data["cost"] <= self.cost_termination_fraction * self.initial_cost:
            logger.info('Optimisation stopped: cost target reached (incumbent cost %.6e <= %.1f%% of initial %.6e).',
                        self.incumbent_data["cost"], self.cost_termination_fraction * 100.0, self.initial_cost)

        # Optimisation completed. Next we generate new Equilibrium and Machine objects
        # for the optimised geometry and corresponding coil currents.
        self.generate_optimised_eq_machine()

    def evaluate_neighbour(
        self,
        neighbour,
        previous_accepted_cost,
    ):
        """Calculates the cost of a neighbouring state and accepts/rejects it accordingly.

        Evaluates whether or not a new neighbouring state is going to be kept or not.
        In doing so, the cost function for this candidate state is evaluated.

        Parameters
        ----------
        neighbour : dict
            The new candidate neighbour state to be evaluated.
        previous_accepted_cost : float
            The cost of the previous state that was deemed acceptable [dict]

        Returns
        -------
        state_data : dict
            Dictionary of data of the neighbouring state. Includes updated data
            related to the cost of the state.
        """

        state_data = self.calculate_cost(
            currents = neighbour["currents"],
            flux_map = neighbour["psi_2D"],
        )

        # Evaluate the change in "energy" between the new candidate neighbour state
        # and the previous state. This is given by the change in cost function.
        energy_change = state_data["cost"] - previous_accepted_cost

        state_data["energy_change"] = energy_change

        # Evaluate whether or not we will accept this new candidate state using the
        # Metropolis-Hastings criterion. If the energy change is negative, immediately
        # accepted the new state, otherwise the probability of acceptance is evaluated.

        if energy_change < 0:

            # Always accept states with a lower cost
            state_data["acceptance"] = True

            # Record the probability of acceptance
            state_data["acceptance_prob"] = 1.0

        else:

            # State has a higher cost. Calculate the probability of acceptance
            probability_of_acceptance = np.exp(-self.alpha * energy_change / self.temperature)

            # Record the probability of acceptance
            state_data["acceptance_prob"] = probability_of_acceptance

            # Check if the acceptance is successful or not
            if np.random.rand() < probability_of_acceptance:


                # Accept the new candidate neighbour state
                state_data["acceptance"] = True

            else:

                # Reject the new candidate neighbour state
                state_data["acceptance"] = False

        # Return the state data for the evaluated candidate neighbour state
        return state_data

    def generate_new_neighbour_in_current_space(
            self,
            currents,
            step_size,
    ):
        """Generates a new neighbouring state.

        Generates a new position in coil current space for evaluation.
        Samples a random unit vector from the null space already generated and
        multiplies it by a given step size. This new perturbation vector is then
        added to the current vector to make a new set of coil currents. A new flux map
        is then generated for this configuration.

        Parameters
        ----------
        currents : 1D numpy array.
            The exisiting coil currents (A).
        step_size : float
            A factor to scale the step size taken in coil current space.

        Returns
        -------
        new state dictionary
            A dictionary of data representing the new state.
        """

        # Start by generating a vector of random numbers between 0-1 of length of the number
        # of columns of the null space matrix (as all of our constraints constraints are linearly
        # independent, this will be equivalent to the the number of coils minus the number of
        # constraints.

        unit_vector = np.random.rand(self.tokamak_opt.N_coils - self.annealing_N_constraints)

        # Currents should be able to both increase and decrease - shift [0,1] elements to be [-0.5,0.5]
        unit_vector = np.subtract(unit_vector,0.5)

        # Normalise the vector
        unit_vector /= np.linalg.norm(unit_vector)

        # Calculate the coil current perturbation
        current_perturbation = step_size * (unit_vector @ self.annealing_greens_matrix_nullspace_transpose)

        # Calculate the new resultant coil currents by adding the perturbed currents to the existing currents
        new_currents = np.add(currents,current_perturbation)

        # Generate the new equilibrium psi on the 2D grid, for these new coil currents
        return self.generate_eq_from_currents(
            new_currents = new_currents
        )

    def generate_eq_from_currents(
            self,
            new_currents,
    ):
        """Generates equilibrium data from a set of coil currents.
        
        Parameters
        ----------
        new_currents : 1D numpy array
            The set of coil currents.

        Returns
        -------
        dictionary of equilibrium data.
            Contains the 2D grid of poloidal magnetic flux, as well as the machine and plasma
            poloidal magnetic fluxes, and the coil currents.
        """

        # At this stage we have our new coil currents. As the total poloidal magnetic
        # flux is simply the flux from the plasma (which we take as a known background flux)
        # plus the flux from the machine, we now need to calculate the new machine (coilset) flux.
        # As we have already pre-computed the Green's functions on our 2D grid for every coil
        # we simply need to multiply the new coil currents by the corresponding Green's grids
        # and superimpose them ontop of the background plasma flux to calculate the new total flux.

        # Calculate the new machine flux on the 2D grid by contracting the coil
        # currents with the pre-computed Green's function matrices.
        new_psi_mach_2D = np.einsum('i,ijk->jk', new_currents, self.coilset_greens_on_grid)

        # Add the new flux from the machine to the existing background plasma flux
        # to get the new total poloidal magnetic flux on the 2D grid.
        new_psi_2D = self.eq.psi_plas_2D + new_psi_mach_2D

        return {
            "currents":new_currents,
            "psi_2D":new_psi_2D,
            "psi_mach_2D":new_psi_mach_2D,
            "psi_plas_2D":self.eq.psi_plas_2D,
            }

    def calculate_cost(
            self,
            currents,
            flux_map,
    ):
        """Evaluates the cost of a state.
        
        Calculates the cost of an equilibrium state. This includes all cost components of all
        divertor regions being optimised, as well as cost terms not tied to specific divertor
        regions, such as the coil currents cost term. In calculating the cost, field lines are
        traced in the divertor regions as appropriate. If X-point regions are used, then any
        secondary divertor X-points in these regions are located.

        Parameters
        ----------
        currents : 1D numpy array
            The set of coil currents.
        flux_map : 2D numpy array
            2D array of the state's poloidal magnetic flux.

        Returns
        -------
        Dictionary of cost data and divertor data
            A dictionary containing detailed cost data as well as divertor data (such as the strike
            point, connection length, traced field line etc.) for each divertor region.
        """

        # We have moved in current space, hence the 2D map of poloidal magnetic flux has changed,
        # and so a new connection length for our new divertor geometry must be calculated.

        # Compute the dpsi/dR and dpsi/dZ grids analytically from the pre-computed
        # coil Green's function derivative matrices.  This replaces the previous
        # approach of constructing a RectBivariateSpline and numerically differentiating
        # it — the einsum is a simple matrix-vector multiply over the coil dimension
        # and is both faster (~10x) and more accurate (analytical Green's function
        # derivatives rather than spline numerical derivatives).
        _dpsi_dR_grid = self._psi_plas_dpsi_dR_grid + np.einsum(
            'i,ijk->jk', currents, self.coilset_dpsi_dR_greens_on_grid
        )
        _dpsi_dZ_grid = self._psi_plas_dpsi_dZ_grid + np.einsum(
            'i,ijk->jk', currents, self.coilset_dpsi_dZ_greens_on_grid
        )

        # Bundle the grid metadata that the fast tracer needs.  The grid
        # dimensions are cached on self and never change; only the derivative
        # grids and flux_map vary per iteration.
        _tracer_grid_data = {
            "psi_grid": flux_map,
            "dpsi_dR_grid": _dpsi_dR_grid,
            "dpsi_dZ_grid": _dpsi_dZ_grid,
            "R_min": self._grid_R_min,
            "Z_min": self._grid_Z_min,
            "inv_dR": self._grid_inv_dR,
            "inv_dZ": self._grid_inv_dZ,
            "nR_m1": self._grid_nR_m1,
            "nZ_m1": self._grid_nZ_m1,
        }

        # Create an empty dictionary that will hold pertinent data for each divertor region.
        divertor_results = {}

        # Iterate over the divertor region(s) to be optimised.
        for divertor_region in self.divertor_regions:

            # Perform the relevant field line trace

            # Get the starting location of the field line trace and the trace direction
            trace_starting_position_R = self.divertor_data[divertor_region]["trace_starting_position_R"]
            trace_starting_position_Z = self.divertor_data[divertor_region]["trace_starting_position_Z"]
            trace_direction = self.divertor_data[divertor_region]["trace_direction"]

            # Field line trace starting location
            trace_starting_position = [trace_starting_position_R,trace_starting_position_Z]

            # Trace a field line from the starting position
            field_line_data = self.trace_field_line(
                starting_position = trace_starting_position,
                step_size = self.field_line_trace_step_size,
                max_steps = self.field_line_trace_max_steps,
                direction = trace_direction,
                grid_data = _tracer_grid_data,
                divertor_region = divertor_region,
            )

            field_line_R = field_line_data["field_line_R"]
            field_line_Z = field_line_data["field_line_Z"]
            intersection_with_buffer = field_line_data["intersection_with_buffer"]

            # We have finished the field line tracing. We will now calculate the distance between the end
            # point and the intended strike geometry. This distance will be used to penalise geometries
            # where the strike point is far from the intended location.

            # Extract the end point - the strike point
            strike_point_R = field_line_R[-1]
            strike_point_Z = field_line_Z[-1]

            # Create a Shapely Point object of this strike point
            strike_point = Point(strike_point_R,strike_point_Z)

            # Target strike geometry
            strike_geometry = self.divertor_data[divertor_region]["strike_geometry"]

            # Get the distance between the strike point and the intended strike geometry
            strike_point_distance = strike_point.distance(strike_geometry)

            # Calculate the strike point distance cost

            # The initial strike point distance cost for this divertor region
            initial_strike_point_distance_cost = self.divertor_data[divertor_region]["initial_strike_point_distance_cost"]

            # If this is the first cost function evaluation, we will record the initial strike point distance.
            if self.N_evals == 0:

                self.divertor_data[divertor_region]["initial_strike_point_distance"] = strike_point_distance

            # One issue that may occur is if the strike point is initially on the target strike geometry, and later
            # ends up also being on the target strike geometry. In this case both the strike point distance
            # and the initial strike point distance are vanishingly small, but due to floating point precision
            # limitations can still end up having a large ratio, giving the false impression that the strike point
            # if far off the target strike geometry. Hence, if the strike point distance is small, the strike
            # point distance cost is just set to zero.

            # Calculate the strike point distance multiplication factor for this divertor
            if strike_point_distance < 1.0e-04:

                strike_point_distance_multiplication_factor = 0.0

            else:

                initial_strike_point_distance = self.divertor_data[divertor_region]["initial_strike_point_distance"]
                strike_point_distance_multiplication_factor = strike_point_distance / initial_strike_point_distance

            # The strike point distance cost multiplication factor is the same as the distance multiplication factor itself

            # Calculate the strike point distance cost
            cost_strike_point_distance = initial_strike_point_distance_cost * strike_point_distance_multiplication_factor

            # We have traced our field line of interest in the poloidal plane, however, connection length here is
            # the full parallel connection length, not just the poloidal one. We will now calculate the parallel
            # connection length by integrating along the field line.

            # field_line_R, field_line_Z

            # Calculate the cumulative poloidal connection length from the start
            # to the end of the field line (so 0 at the start).
            points = np.column_stack((field_line_R, field_line_Z))
            s_pol_dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            s_pol_dists = np.insert(np.cumsum(s_pol_dists), 0, 0)

            # Calculate the parallel connection length

            # Get Br, Bz and Btor at the points along the field line.
            # Use the vectorised bilinear interpolator on the analytically-derived
            # derivative grids — avoids the need for a RectBivariateSpline entirely.
            _fi_fl = (field_line_R - self._grid_R_min) * self._grid_inv_dR
            _fj_fl = (field_line_Z - self._grid_Z_min) * self._grid_inv_dZ
            _dpsi_dZ_fl = _fast_bilinear_vec(_dpsi_dZ_grid, _fi_fl, _fj_fl, self._grid_nR_m1, self._grid_nZ_m1)
            _dpsi_dR_fl = _fast_bilinear_vec(_dpsi_dR_grid, _fi_fl, _fj_fl, self._grid_nR_m1, self._grid_nZ_m1)
            Br = -(ONE_2PI / field_line_R) * _dpsi_dZ_fl
            Bz =  (ONE_2PI / field_line_R) * _dpsi_dR_fl
            Btor = self.eq.fvac / field_line_R

            # Calculate the connection length integrand
            integrand = np.sqrt(1.0 + ((Btor * Btor) / (Br * Br + Bz * Bz)))

            # Perform numerical integration
            connection_length = np.trapz(integrand,s_pol_dists)

            # Connection_length is now the full parallel connection length for this configuration

            # Calculate the connection length cost for this divertor

            # The initial connection length cost for this divertor region
            initial_connection_length_cost = self.divertor_data[divertor_region]["initial_connection_length_cost"]

            # If this is the first cost function evaluation, we will record the initial connection
            # length.
            if self.N_evals == 0:
                self.divertor_data[divertor_region]["initial_connection_length"] = connection_length

            # Calculate the connection length multiplication factor for this divertor
            initial_connection_length = self.divertor_data[divertor_region]["initial_connection_length"]
            connection_length_multiplication_factor = connection_length / initial_connection_length

            # Calculate the connection length cost multiplication factor for this divertor

            # The connection length multiplication factor at which the cost multiplication factor will zero-out.
            # E.g. at 1.5 - if the connection length has increased by 50%, the cost will be zero.
            connection_length_multiplication_factor_zero = (
                self.divertor_data[divertor_region]["connection_length_multiplication_factor_zero"]
            )

            connection_length_cost_multiplication_factor = 1.0 + (-1.0 / (connection_length_multiplication_factor_zero - 1.0))\
                                                                        * (connection_length_multiplication_factor - 1.0)

            connection_length_cost_multiplication_factor = max(0.0,connection_length_cost_multiplication_factor)

            # Calculate the connection length cost
            cost_connection_length = initial_connection_length_cost * connection_length_cost_multiplication_factor

            # If the field line trace struck a buffer region, apply a penalty factor to the connection length cost
            if intersection_with_buffer:
                cost_connection_length *= self.buffer_intersection_penalty_factor

            # Next, we will check if any X-point regions associated with this divertor are present and
            # perform the required cost calculations accordingly.
            xpoint_region_present = self.divertor_data[divertor_region]["xpoint_region_present"]

            if xpoint_region_present:

                # An X-point region associated with this divertor region is present

                # Check if the separatrix is in the region by testing if the traced field line crosses
                # the xpoint region boundary.

                # Retrieve the pre-computed region boundary
                region_boundary = self.divertor_data[divertor_region]["xpoint_region"]["region_boundary"]

                # Retrieve the traced field line
                field_line = field_line_data["field_line"]

                # The separatrix is in the region if the traced field line intersects the region boundary
                separatrix_in_region = field_line.intersects(region_boundary)

                # If the separatrix is within this region, proceed.
                if separatrix_in_region:

                    # Extract the relevant pre-calculated Green's function matrices for Br and Bz
                    xpoint_region_greens_br_matrix_transpose = (
                        self.divertor_data[divertor_region]["xpoint_region"]["greens_br_matrix_transpose"]
                    )

                    xpoint_region_greens_bz_matrix_transpose = (
                        self.divertor_data[divertor_region]["xpoint_region"]["greens_bz_matrix_transpose"]
                    )

                    # Extract the background poloidal field from the plasma at points in the region
                    Br_plas = self.divertor_data[divertor_region]["xpoint_region"]["background_Br_points"]
                    Bz_plas = self.divertor_data[divertor_region]["xpoint_region"]["background_Bz_points"]

                    # Calculate Br and Bz from the coils at points in the region
                    Br_coils = currents @ xpoint_region_greens_br_matrix_transpose
                    Bz_coils = currents @ xpoint_region_greens_bz_matrix_transpose

                    # Total poloidal field
                    Br = Br_coils + Br_plas
                    Bz = Bz_coils + Bz_plas

                    # Calculate the square of the poloidal field at points in the region
                    Bp2 = Br * Br + Bz * Bz

                    # Extract the minimumn value
                    Bp2_min = np.amin(Bp2)

                    # Could this be done more elegantly?
                    try:

                        if self.divertor_data[divertor_region]["xpoint_region"]["first_pass"]:

                            initial_Bp2_min = self.divertor_data[divertor_region]["xpoint_region"]["initial_Bp2_min"]

                    except KeyError:

                        initial_Bp2_min = Bp2_min

                        self.divertor_data[divertor_region]["xpoint_region"]["first_pass"] = True
                        self.divertor_data[divertor_region]["xpoint_region"]["initial_Bp2_min"] = initial_Bp2_min

                    # Now we will calculate the part of the cost associated with this poloidal field.
                    # The cost multiplier from the poloidal field will be the ratio of the achieved square
                    # of the poloidal field to the initial square of the poloidal field. The other parts of the
                    # X-point region cost will come from whether or not there exists an X-point within
                    # the X-point region. We will also extract the initial X-point region cost for this divertor region.

                    # The initial X-point region cost for this divertor region
                    initial_xpoint_region_cost = self.divertor_data[divertor_region]["initial_xpoint_region_cost"]

                    Bp2_cost_multplication_factor = Bp2_min / initial_Bp2_min
                    cost_Bp2 = (1. / 3.) * initial_xpoint_region_cost * Bp2_cost_multplication_factor

                    # Next we will check to see whether or not an X-point is present within the region.
                    # First, we will extract data for theis region that will be used.

                    # Extract the matrix of 2x2 Jacobian matrices for the control poloidal field from the coils
                    xpoint_region_coils_Bp_jacobians_matrix_transpose = (
                        self.divertor_data[divertor_region]["xpoint_region"]["coils_Bp_jacobians_matrix_transpose"]
                    )

                    # Extract the (row) vector of 2x2 Jacobian matrices for the background poloidal field from the coils
                    xpoint_region_background_Bp_jacobian_points = (
                        self.divertor_data[divertor_region]["xpoint_region"]["xpoint_region_background_Bp_jacobian_points"]
                    )

                    # Calculate the (row) vector of 2x2 Jacobian matrices for the full poloidal field from the coils
                    xpoint_region_coils_Bp_jacobian_points = np.einsum(
                        'i,ijkl->jkl', currents, xpoint_region_coils_Bp_jacobians_matrix_transpose
                    )

                    # Calculate the (row) vector of 2x2 Jacobian matrices for the full poloidal field (background + coils)
                    xpoint_region_Bp_jacobian_points = xpoint_region_background_Bp_jacobian_points + \
                                                            xpoint_region_coils_Bp_jacobian_points

                    # Extract the R,Z locations of these points inside the X-point region
                    R_xpoint_region = self.divertor_data[divertor_region]["xpoint_region"]["R_points"]
                    Z_xpoint_region = self.divertor_data[divertor_region]["xpoint_region"]["Z_points"]

                    self.R_xpoint_estimate, self.Z_xpoint_estimate, xpoint_present = estimate_xpoint_location(
                        R_xpoint_region,
                        Z_xpoint_region,
                        Br,
                        Bz,
                        xpoint_region_Bp_jacobian_points,
                        self.eq.dR,
                        self.eq.dZ
                        )

                    # If an X-point is present, we don't need to include the Bp^2 cost term, as this is really only
                    # there to reward lowering the poloidal field on the way to making an X-point. As the poloidal field
                    # is only evaluated on a finite number of grid points, the exact X-point position, if one is present, will
                    # not be on a grid point, hence the lowest Bp value will not strictly be zero. As such, we will, if an
                    # X-point if present, just set this cost component to zero, otherwise this remains as is.
                    
                    # The final component of the total X-point cost for this region will be a cost associated with the level of
                    # magnetic disconnection between the flux surface that the X-point sits on (if one is present) and the
                    # separatrix. The idea here is to reward having the secondary X-point on/close to the separatrix. If no
                    # secondary X-point is present the associated multiplication factor for this part of the X-point cost is
                    # 1.0 . This cost multiplication factor is dependent on the degree of magentic disconnection, with the cost
                    # scaled about a set maximium level of disconnection. The idea here is that even a small level of magnetic
                    # disconnection is unnaceptable, hence the cost multiplication factor needs to be sensitive to the level of
                    # disconnetion.

                    if xpoint_present:

                        # At least one X-point is present inside the region. Part of the total X-point region cost comes
                        # from whether or not an X-point is present within the region. If an X-point is present, this
                        # part of the total cost has a multiplication factor of zero, otherwise the factor is 1 if
                        # no X-point is present inside the region.
                        xpoint_present_cost_multplication_factor = 0.0

                        # Calculate the cost multiplication factor associated with the level of magnetic disconnection of
                        # the secondary X-point.

                        # Get the value of flux at the secondary X-point
                        _fi_xpt = (self.R_xpoint_estimate - self._grid_R_min) * self._grid_inv_dR
                        _fj_xpt = (self.Z_xpoint_estimate - self._grid_Z_min) * self._grid_inv_dZ
                        psi_secondary_divertor_xpoint = _fast_bilinear(
                            flux_map, _fi_xpt, _fj_xpt, self._grid_nR_m1, self._grid_nZ_m1
                        )

                        magnetic_disconnection_factor = np.abs( (psi_secondary_divertor_xpoint - self.eq.psi_lcfs) \
                                                               / self.eq.psi_lcfs )

                        xpoint_magnetic_connection_cost_multiplication_factor = (1.0 / self.max_magnetic_disconnection_factor)\
                            * magnetic_disconnection_factor

                        xpoint_magnetic_connection_cost_multiplication_factor = min(
                            1.0,
                            xpoint_magnetic_connection_cost_multiplication_factor
                        )

                    else:

                        # No X-points are present inside the region
                        xpoint_present_cost_multplication_factor = 1.0

                        # As no X-point is present within the region, the cost multiplication factor associated with the
                        # magnetic disconnection of the secondary X-point is maximal - set to an upper limit of 1.
                        xpoint_magnetic_connection_cost_multiplication_factor = 1.0

                    # Calculate the part of the X-point region cost from this binary check for the presence of an X-point
                    cost_xpoint_present = (1. / 3.) * initial_xpoint_region_cost * xpoint_present_cost_multplication_factor

                    # Adjust the part of the X-point cost coming from Bp^2
                    cost_Bp2 *= xpoint_present_cost_multplication_factor

                    # Calculate the part of the X-point region cost from the magnetic disconnection of the secondary X-point
                    cost_xpoint_magnetic_connection = (1. / 3.) * initial_xpoint_region_cost \
                                                            * xpoint_magnetic_connection_cost_multiplication_factor

                    # Total cost for the X-point region
                    cost_xpoint_region = cost_Bp2 + cost_xpoint_present + cost_xpoint_magnetic_connection

                else:

                    # The separatrix is not in the divertor region
                    # Is this consistent with the stated method in the paper?

                    # The initial X-point region cost for this divertor region
                    initial_xpoint_region_cost = self.divertor_data[divertor_region]["initial_xpoint_region_cost"]

                    cost_xpoint_region = initial_xpoint_region_cost

                    self.R_xpoint_estimate = np.nan
                    self.Z_xpoint_estimate = np.nan

            else:

                # No X-point region associated with this divertor region is present
                cost_xpoint_region = 0.0

            # Caclulate the cost associated with this divertor.

            # Record the connection length, strike point distance and location for this divertor, along with the
            # associated costs. The R,Z of the field line trace is also included.
            divertor_results[divertor_region] = {
                "connection_length": connection_length,
                "strike_point_distance": strike_point_distance,
                "cost_connection_length": cost_connection_length,
                "cost_strike_point_distance": cost_strike_point_distance,
                "cost_xpoint_region": cost_xpoint_region,
                "field_line_R": field_line_R,
                "field_line_Z": field_line_Z,
                "strike_point_R": strike_point_R,
                "strike_point_Z": strike_point_Z,
            }

        # All divertor regions have now had their connection length and strike point distances
        # calculated, along with relevant checks on a potential X-point region, with all associated
        # costs now calculated.

        # Calculate the total connection length, strike point distance and X-point region costs across all divertor regions.
        cost_connection_length = 0
        cost_strike_point_distance = 0
        cost_xpoint_regions = 0

        for divertor_region in self.divertor_regions:

            cost_connection_length += divertor_results[divertor_region]["cost_connection_length"]
            cost_strike_point_distance += divertor_results[divertor_region]["cost_strike_point_distance"]
            cost_xpoint_regions += divertor_results[divertor_region]["cost_xpoint_region"]

        # Calculate the sum square of the coil currents - this will contribute to the cost
        sum_square_coil_currents = np.dot(currents,currents)

        # The cost from the coil currents is as follows. The cost is nominally the initial cost value multiplied
        # by a cost multiplication factor, given by the sum square of the coil currents normalised to the initial
        # sum square of the coil currents. However, we establish a null-zone, wherein if the sum square lies between
        # the initial sum square value and sum square value representing the coils all being "reasonably energised",
        # the cost multiplication factor remains at a value of one. If the sum square is below the initial sum square,
        # those data are used in the normalisation, else if the sum square is below the energised sum square value, those
        # data are used in the normalisation.
        if self.initial_sum_square_coil_currents <= sum_square_coil_currents <= self.energised_sum_square_coil_currents:
            coil_currents_multiplication_factor = 1.0

        elif sum_square_coil_currents < self.initial_sum_square_coil_currents:
            coil_currents_multiplication_factor = sum_square_coil_currents / self.initial_sum_square_coil_currents

        else:
            coil_currents_multiplication_factor = sum_square_coil_currents / self.energised_sum_square_coil_currents

        #if self.N_evals % self.n_window == 0:
        #    print('sum square currents achieved: ',sum_square_coil_currents)

        # Contribution from the coil currents - these are regularised so as to penalise states
        # with prohibitively large coil currents. The sum square of the coil currents is used
        # as the square of the currents is more physicaly meaningful (e.g. forces between coils
        # goes as the square of the currents).
        cost_coil_currents = self.initial_coil_currents_cost * coil_currents_multiplication_factor

        # Finally, we will compute the total cost for this configuration.

        # Calculate the total cost
        cost = cost_strike_point_distance
        cost += cost_connection_length
        cost += cost_coil_currents
        cost += cost_xpoint_regions

        # Update the tracker for number of cost function evaluations
        self.N_evals += 1

        # Return a dictionary of key information about the state and its cost
        # This is later reffered to as the state_data dictionary.
        return {
            "cost": cost,
            "cost_strike_point_distance": cost_strike_point_distance,
            "cost_connection_length": cost_connection_length,
            "cost_coil_currents": cost_coil_currents,
            "cost_xpoint_regions": cost_xpoint_regions,
            "divertors": divertor_results,
            "psi_2D": flux_map,
            "currents": currents,
         }

    def trace_field_line(
            self,
            starting_position,
            step_size,
            max_steps = 1000,
            direction = 1.0,
            grid_data = None,
            divertor_region = None,
    ):
        """Trace a magnetic field line until it hits a wall segment.

        Trace a field line from the specified starting location. Stop tracing
        either when the field line intersects the machine's wall, or a maximum
        number of steps along the field line has been taken.

        Uses RK4 integration with a **fast bilinear interpolator** on pre-computed
        derivative grids and a **gradient-projection correction** after every step
        to keep the trace pinned to the correct flux surface.

        Performance notes
        -----------------
        The original bottleneck was ``psi_func.ev()`` called as a scalar inside a
        Python loop (~30-100 us overhead per call).  By pre-computing dpsi/dR and
        dpsi/dZ on the regular grid once (in ``calculate_cost``) and interpolating
        them bilinearly here (~1 us per call), the per-step cost drops by **30-50x**.
        Combined with RK4's larger feasible step size this gives an overall speed-up
        of roughly **10-20x** on the tracer.

        The gradient-projection step ensures that the accumulated psi drift after
        each RK4 step is corrected to machine precision, so the trace stays on
        the target flux surface regardless of step size.

        Parameters
        ----------
        starting_position : list
            [R,Z] of the starting position for the trace.
        step_size : float
            The size of each poloidal step along the field line.
        max_steps : int
            Maximum number of steps to take along the field line.
        direction : float
            Direction to trace the field line. +1/-1 corresponds to the poloidal helicity of the trace.
        grid_data : dict
            Pre-computed grids and metadata from ``calculate_cost``.  Keys:
            ``psi_grid``, ``dpsi_dR_grid``, ``dpsi_dZ_grid``, ``R_min``,
            ``Z_min``, ``inv_dR``, ``inv_dZ``, ``nR_m1``, ``nZ_m1``.
        divertor_region : str, optional
            Name of the divertor region being traced (e.g. ``"lower_outer"``).
            When buffers are keyed per region, only the buffers for this region
            are checked for intersections.
        """

        # Unpack grid data for the fast bilinear evaluator.
        psi_grid     = grid_data["psi_grid"]
        dpsi_dR_grid = grid_data["dpsi_dR_grid"]
        dpsi_dZ_grid = grid_data["dpsi_dZ_grid"]
        gR_min  = grid_data["R_min"]
        gZ_min  = grid_data["Z_min"]
        g_inv_dR = grid_data["inv_dR"]
        g_inv_dZ = grid_data["inv_dZ"]
        g_nR_m1  = grid_data["nR_m1"]
        g_nZ_m1  = grid_data["nZ_m1"]

        # Local references to the fast interpolator (avoid repeated global/dict lookups)
        _interp = _fast_bilinear

        # Pre-allocate output arrays.
        locations_R = np.empty(max_steps + 1)
        locations_Z = np.empty(max_steps + 1)
        locations_R[0] = starting_position[0]
        locations_Z[0] = starting_position[1]
        n_points = 1

        intersection_with_buffer = False

        # Target psi value — the flux surface the trace must stay on.
        fi0 = (starting_position[0] - gR_min) * g_inv_dR
        fj0 = (starting_position[1] - gZ_min) * g_inv_dZ
        psi_target = _interp(psi_grid, fi0, fj0, g_nR_m1, g_nZ_m1)

        for i in range(max_steps):

            R0 = locations_R[n_points - 1]
            Z0 = locations_Z[n_points - 1]

            # Fractional grid indices for the starting point (reused by stage 1)
            fi = (R0 - gR_min) * g_inv_dR
            fj = (Z0 - gZ_min) * g_inv_dZ

            # ---- RK4 Stage 1 ----
            dR_val = _interp(dpsi_dR_grid, fi, fj, g_nR_m1, g_nZ_m1)
            dZ_val = _interp(dpsi_dZ_grid, fi, fj, g_nR_m1, g_nZ_m1)
            Br1 = -(ONE_2PI / R0) * dZ_val
            Bz1 =  (ONE_2PI / R0) * dR_val
            Bp1_sq = Br1 * Br1 + Bz1 * Bz1
            if Bp1_sq < 1e-30:
                break
            inv_Bp1 = direction / math.sqrt(Bp1_sq)
            k1R = step_size * Br1 * inv_Bp1
            k1Z = step_size * Bz1 * inv_Bp1

            # ---- RK4 Stage 2 ----
            Rm = R0 + 0.5 * k1R
            Zm = Z0 + 0.5 * k1Z
            fi = (Rm - gR_min) * g_inv_dR
            fj = (Zm - gZ_min) * g_inv_dZ
            dR_val = _interp(dpsi_dR_grid, fi, fj, g_nR_m1, g_nZ_m1)
            dZ_val = _interp(dpsi_dZ_grid, fi, fj, g_nR_m1, g_nZ_m1)
            Br2 = -(ONE_2PI / Rm) * dZ_val
            Bz2 =  (ONE_2PI / Rm) * dR_val
            Bp2_sq = Br2 * Br2 + Bz2 * Bz2
            if Bp2_sq < 1e-30:
                break
            inv_Bp2 = direction / math.sqrt(Bp2_sq)
            k2R = step_size * Br2 * inv_Bp2
            k2Z = step_size * Bz2 * inv_Bp2

            # ---- RK4 Stage 3 ----
            Rm = R0 + 0.5 * k2R
            Zm = Z0 + 0.5 * k2Z
            fi = (Rm - gR_min) * g_inv_dR
            fj = (Zm - gZ_min) * g_inv_dZ
            dR_val = _interp(dpsi_dR_grid, fi, fj, g_nR_m1, g_nZ_m1)
            dZ_val = _interp(dpsi_dZ_grid, fi, fj, g_nR_m1, g_nZ_m1)
            Br3 = -(ONE_2PI / Rm) * dZ_val
            Bz3 =  (ONE_2PI / Rm) * dR_val
            Bp3_sq = Br3 * Br3 + Bz3 * Bz3
            if Bp3_sq < 1e-30:
                break
            inv_Bp3 = direction / math.sqrt(Bp3_sq)
            k3R = step_size * Br3 * inv_Bp3
            k3Z = step_size * Bz3 * inv_Bp3

            # ---- RK4 Stage 4 ----
            Rm = R0 + k3R
            Zm = Z0 + k3Z
            fi = (Rm - gR_min) * g_inv_dR
            fj = (Zm - gZ_min) * g_inv_dZ
            dR_val = _interp(dpsi_dR_grid, fi, fj, g_nR_m1, g_nZ_m1)
            dZ_val = _interp(dpsi_dZ_grid, fi, fj, g_nR_m1, g_nZ_m1)
            Br4 = -(ONE_2PI / Rm) * dZ_val
            Bz4 =  (ONE_2PI / Rm) * dR_val
            Bp4_sq = Br4 * Br4 + Bz4 * Bz4
            if Bp4_sq < 1e-30:
                break
            inv_Bp4 = direction / math.sqrt(Bp4_sq)
            k4R = step_size * Br4 * inv_Bp4
            k4Z = step_size * Bz4 * inv_Bp4

            # Combine the four stages (standard RK4 weights)
            R_end = R0 + (k1R + 2.0 * k2R + 2.0 * k3R + k4R) / 6.0
            Z_end = Z0 + (k1Z + 2.0 * k2Z + 2.0 * k3Z + k4Z) / 6.0

            # ---- Gradient-projection correction ----
            # Project the point back onto the target flux surface.
            # This is a single Newton step:
            #   x_corrected = x - (psi(x) - psi_target) / |grad(psi)|^2 * grad(psi)
            # It eliminates accumulated psi drift to first order, keeping
            # the trace on its flux surface to machine precision.
            fi_end = (R_end - gR_min) * g_inv_dR
            fj_end = (Z_end - gZ_min) * g_inv_dZ
            psi_here = _interp(psi_grid, fi_end, fj_end, g_nR_m1, g_nZ_m1)
            psi_err  = psi_here - psi_target
            grad_R   = _interp(dpsi_dR_grid, fi_end, fj_end, g_nR_m1, g_nZ_m1)
            grad_Z   = _interp(dpsi_dZ_grid, fi_end, fj_end, g_nR_m1, g_nZ_m1)
            grad_sq  = grad_R * grad_R + grad_Z * grad_Z
            if grad_sq > 1e-30:
                correction = psi_err / grad_sq
                R_end -= correction * grad_R
                Z_end -= correction * grad_Z

            # Check for wall / buffer intersection
            intersection_data = self.check_field_line_intersection(
                field_line_R_start = R0,
                field_line_Z_start = Z0,
                field_line_R_end = R_end,
                field_line_Z_end = Z_end,
                divertor_region = divertor_region,
            )

            if intersection_data["intersects"]:
                locations_R[n_points] = intersection_data["strike_point_R"]
                locations_Z[n_points] = intersection_data["strike_point_Z"]
                n_points += 1
                intersection_with_buffer = intersection_data.get("intersection_with_buffer", False)
                break
            else:
                locations_R[n_points] = R_end
                locations_Z[n_points] = Z_end
                n_points += 1

        # Trim the pre-allocated arrays to the actual number of traced points.
        field_line_R = locations_R[:n_points].copy()
        field_line_Z = locations_Z[:n_points].copy()

        field_line_data = {
            "field_line_R": field_line_R,
            "field_line_Z": field_line_Z,
            "field_line": LineString(np.column_stack((field_line_R, field_line_Z))),
            "intersection_with_buffer": intersection_with_buffer,
        }

        return field_line_data

    def _extract_intersection_point(self, intersection):
        """Extracts (R, Z) from a Shapely intersection result (Point or MultiPoint)."""
        if intersection.geom_type == "Point":
            return intersection.x, intersection.y
        elif intersection.geom_type == "MultiPoint":
            return intersection.geoms[0].x, intersection.geoms[0].y
        return None

    def check_field_line_intersection(
            self,
            field_line_R_start,
            field_line_Z_start,
            field_line_R_end,
            field_line_Z_end,
            divertor_region=None,
    ):
        """Locates intersections between a traced field line segment and wall/buffer structures.

        Checks for an intersection between the provided field line segment and the tokamak's wall. If
        buffer regions are present, intersections with these are also examined. If multiple intersections
        are present, the point closest to the field line's starting point is returned. Note that this is
        for only a single segment of a field line, i.e. between two traced points.

        Uses the prepared (spatially-indexed) wall geometry created during __init__
        for a fast boolean pre-check before computing the full intersection geometry.
        This avoids the expensive intersection computation on steps that are entirely
        inside the domain (the vast majority).

        Parameters
        ----------
        field_line_R_start : float
            R coordinate of the starting location of the field line segment.
        field_line_Z_start : float
            Z coordinate of the starting location of the field line segment.
        field_line_R_end : float
            R coordinate of the end location of the field line segment.
        field_line_Z_end : float
            Z coordinate of the end location of the field line segment.
        divertor_region : str, optional
            Name of the divertor region being traced. When provided and buffers
            are keyed per region, only the buffers for this region are checked.

        Returns
        -------
        intersection_data : dict
            Dictionary of intersection data containing a flag for whether or not there was
            an intersection, the strike point (R,Z) coordinates (if there is an intersection)
            and an additional flag for if an intersection occurs with a buffer.
        """

        # Build the segment LineString once
        field_line_segment = LineString([(field_line_R_start, field_line_Z_start),
                                         (field_line_R_end, field_line_Z_end)])

        # Fast boolean pre-check using the prepared (spatially indexed) wall.
        # For the majority of steps, where the segment is well inside the
        # domain, this returns False in O(log N) time without computing the
        # full intersection geometry.
        wall_may_intersect = self._prepared_wall.intersects(field_line_segment)

        # Determine the buffer geometries relevant to this trace.
        # Buffers are keyed per divertor region for efficiency.
        region_buffers = None
        region_prep_buffers = None
        if self.buffers is not None and divertor_region is not None:
            region_buffers = self.buffers.get(divertor_region)
            if region_buffers is not None:
                region_prep_buffers = self._prepared_buffers.get(divertor_region)
        buffers_present = region_buffers is not None and len(region_buffers) > 0

        if not wall_may_intersect and not buffers_present:
            return _NO_INTERSECTION

        # --- Collect candidate intersection points ---
        intersection_points_R = []
        intersection_points_Z = []
        intersection_with_buffer = []

        # Wall intersection (only computed when the fast check said yes)
        if wall_may_intersect:
            intersection = field_line_segment.intersection(self.tokamak_opt.wall)
            if not intersection.is_empty:
                pt = self._extract_intersection_point(intersection)
                if pt is not None:
                    intersection_points_R.append(pt[0])
                    intersection_points_Z.append(pt[1])
                    intersection_with_buffer.append(False)

        # Buffer intersections (only for the active divertor region)
        if buffers_present:
            for prep_buf, buffer in zip(region_prep_buffers, region_buffers):
                if not prep_buf.intersects(field_line_segment):
                    continue
                intersection = field_line_segment.intersection(buffer)
                if not intersection.is_empty:
                    pt = self._extract_intersection_point(intersection)
                    if pt is not None:
                        intersection_points_R.append(pt[0])
                        intersection_points_Z.append(pt[1])
                        intersection_with_buffer.append(True)

        if len(intersection_points_R) == 0:
            return _NO_INTERSECTION

        # Find the intersection point closest to the segment start
        if len(intersection_points_R) == 1:
            # Only one candidate — skip the distance calculation
            return {
                "intersects": True,
                "strike_point_R": intersection_points_R[0],
                "strike_point_Z": intersection_points_Z[0],
                "intersection_with_buffer": intersection_with_buffer[0],
            }

        d2 = [(R - field_line_R_start)**2 + (Z - field_line_Z_start)**2
              for R, Z in zip(intersection_points_R, intersection_points_Z)]
        idx = min(range(len(d2)), key=lambda i: d2[i])

        return {
            "intersects": True,
            "strike_point_R": intersection_points_R[idx],
            "strike_point_Z": intersection_points_Z[idx],
            "intersection_with_buffer": intersection_with_buffer[idx],
        }

    def compute_coilset_greens_on_grid(self):
        """Computes the coils' Green's functions on the 2D equilibrium grid.

        Pre-calculates, for each coil in the coilset, the Green's function
        matrices for:

        * Poloidal magnetic flux, psi — via ``coil.control_psi``.
        * The R-derivative of psi, dpsi/dR — derived from the coil's
          analytical ``control_Bz`` response using the identity
          dpsi/dR = 2 pi R Bz.
        * The Z-derivative of psi, dpsi/dZ — derived from the coil's
          analytical ``control_Br`` response using the identity
          dpsi/dZ = -2 pi R Br.

        These derivative matrices are used inside ``calculate_cost`` to
        construct the full dpsi/dR and dpsi/dZ grids via a single
        ``np.einsum`` call (a currents-vector times the precomputed
        matrices), which is *much* faster than fitting a
        ``RectBivariateSpline`` and numerically differentiating it
        every iteration.  The analytical Green's function derivatives
        (via ``greens.Greens_dpsi_dR`` / ``greens.Greens_dpsi_dZ``,
        which underpin ``control_Bz`` / ``control_Br``) are also
        more accurate than spline-based numerical differentiation.
        """

        R_2D = self.eq.R_2D
        Z_2D = self.eq.Z_2D
        TWO_PI = 2.0 * np.pi
        TWO_PI_R = TWO_PI * R_2D

        coilset_greens_on_grid = []
        coilset_dpsi_dR_greens_on_grid = []
        coilset_dpsi_dZ_greens_on_grid = []

        for coil in self.tokamak_opt.coilset.values():

            # Flux Green's function (same as before)
            coil_greens_on_grid = coil.control_psi(R_2D, Z_2D)
            coilset_greens_on_grid.append(coil_greens_on_grid)

            # Derivative Green's functions from the analytical field responses.
            # dpsi/dR = 2*pi*R * Bz   and   dpsi/dZ = -2*pi*R * Br
            coil_Bz_on_grid = coil.control_Bz(R_2D, Z_2D)
            coil_Br_on_grid = coil.control_Br(R_2D, Z_2D)
            coilset_dpsi_dR_greens_on_grid.append(TWO_PI_R * coil_Bz_on_grid)
            coilset_dpsi_dZ_greens_on_grid.append(-TWO_PI_R * coil_Br_on_grid)

        self.coilset_greens_on_grid = np.asarray(coilset_greens_on_grid)
        self.coilset_dpsi_dR_greens_on_grid = np.asarray(coilset_dpsi_dR_greens_on_grid)
        self.coilset_dpsi_dZ_greens_on_grid = np.asarray(coilset_dpsi_dZ_greens_on_grid)

    def generate_constraints(self):
        """Generates constraint matrices/vector for simulated annealing and Tikhonov regularisation (if used).

        Generates various constraint matrices/vectors. In general, there will
        exist a matrix of Green's functions related to the various constraints, as well as a
        corresponding vector of the constraints themselves. These constraints will be things
        such as the value of magnetic field or flux from the coils at a point.

        There will exist up to two sets of these constraints. One set of constraints are used
        during the simulated annealing. The other set is used in the determination of the
        initial coil currents. Determining the initial coil currents is itself optional, but
        is particularly needed if the user decides to perform an optimisation using a set
        of PF coils that were different to those that were used to make the initial equilibrium.
        If the user is using the same PF set as was used to make the initial equilibrium, then
        this determination of the initial coil currents is not required. In such a case the user
        can still choose to estimate the initial coil currents. They may want to do this
        in such a way as to incroporate new divertor constraints, such as those describing
        the position of the divertor legs, as such a resultant equilibrium may provide a more
        useful initial starting point for their particular optimisation problem.

        During the annealing a null-space approach is utilised, wherein there is a requirement
        that number of coils - number of constraints >= 1. It may not be possible for the user
        to include lots of additional divetor geometry constraints in the annealing, if the number of
        coils they have supplied is insufficient. However, if the user chooses to estimate the
        initial coil currents using the Tikhonov regulariastion approach, then during this process
        they are not subject to the strict requirement on the number of constraints, and can
        try to include more divertor constraints. In such a case, a larger set of cosntraints
        can be used, with a smaller set of constraints then later used during the annealing process.
        In this way, the user may be able to bias the initial equilibrium into having favourable
        features in the divertor, which whilst not constrained during the annealing, may be more
        likely to be produced naturally, having started from an initial equilibrium with such features.

        The set of constraints used during the annealing are reffered to as the annealing constraints,
        with the set used during the Tikhonov regularisation reffered to as the Tikhonov constraints.
        """

        # Annealing constraints
        #######################

        annealing_greens_matrix = []
        annealing_constraints = []
        annealing_constraint_points_R = []
        annealing_constraint_points_Z = []

        # X-point constraint - determine which X-point to use.
        # By default ("primary"), the primary X-point is used. In DND configurations,
        # the user may instead specify "lower" or "upper" to select that X-point.
        annealing_xpt_choice = self.constraints["annealing"].get("xpoint_constraint", "primary")
        R_xpt_annealing, Z_xpt_annealing = self._resolve_xpoint_constraint(annealing_xpt_choice, "annealing")

        # Machine total Br at X-point
        annealing_constraints.append(self.eq.Br_machRZ(R_xpt_annealing, Z_xpt_annealing))

        # Machine total Bz at X-point
        annealing_constraints.append(self.eq.Bz_machRZ(R_xpt_annealing, Z_xpt_annealing))

        # Machine flux at the X-point
        annealing_constraints.append(self.eq.psi_machRZ(R_xpt_annealing, Z_xpt_annealing))

        # Control Br at X-point
        annealing_greens_matrix.append(self.tokamak_opt.control_Br(R_xpt_annealing, Z_xpt_annealing))

        # Control Bz at X-point
        annealing_greens_matrix.append(self.tokamak_opt.control_Bz(R_xpt_annealing, Z_xpt_annealing))

        # Control psi at the X-point
        annealing_greens_matrix.append(self.tokamak_opt.control_psi(R_xpt_annealing, Z_xpt_annealing))

        annealing_constraint_points_R.append(R_xpt_annealing)
        annealing_constraint_points_Z.append(Z_xpt_annealing)

        # Secondary X-point - removed pending further consideration

        # Upper right quadrant
        if self.constraints["annealing"]["constrain_upper_right_quadrant"]:

            # Extract the number of constraint points to use
            N_points = self.constraints["annealing"]["N_constraints_upper_right_quadrant"]

            # Calculate the normalised distance along the separatrix of these points
            norm_distances_points = np.linspace(0.0,self.eq.separatrix_dist_norm_vertical_upper_lcfs,N_points + 2)[1:-1]

            # Extract the R,Z location of these points
            points = self.eq.separatrix_interpolator(norm_distances_points)
            R_points = [r for (r,z) in points]
            Z_points = [z for (r,z) in points]

            # Iterate over the constraint points
            for R_point, Z_point in list(zip(R_points,Z_points)):

                # Machine flux at the constraint point
                annealing_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                # Control psi at the constraint point
                annealing_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point))

                annealing_constraint_points_R.append(R_point)
                annealing_constraint_points_Z.append(Z_point)

        # Upper left quadrant
        if self.constraints["annealing"]["constrain_upper_left_quadrant"]:

            # Extract the number of constraint points to use
            N_points = self.constraints["annealing"]["N_constraints_upper_left_quadrant"]

            # Calculate the normalised distance along the separatrix of these points
            norm_distances_points = np.linspace(
                self.eq.separatrix_dist_norm_vertical_upper_lcfs,
                self.eq.separatrix_dist_norm_imp_lcfs,N_points + 2
            )[1:-1]

            # Extract the R,Z location of these points
            points = self.eq.separatrix_interpolator(norm_distances_points)
            R_points = [r for (r,z) in points]
            Z_points = [z for (r,z) in points]

            # Iterate over the constraint points
            for R_point, Z_point in list(zip(R_points,Z_points)):

                # Machine flux at the constraint point
                annealing_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                # Control psi at the constraint point
                annealing_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point))

                annealing_constraint_points_R.append(R_point)
                annealing_constraint_points_Z.append(Z_point)

        # Lower left quadrant
        if self.constraints["annealing"]["constrain_lower_left_quadrant"]:

            # Extract the number of constraint points to use
            N_points = self.constraints["annealing"]["N_constraints_lower_left_quadrant"]

            # Calculate the normalised distance along the separatrix of these points
            norm_distances_points = np.linspace(
                self.eq.separatrix_dist_norm_imp_lcfs,
                self.eq.separatrix_dist_norm_vertical_lower_lcfs,
                N_points + 2
            )[1:-1]

            # Extract the R,Z location of these points
            points = self.eq.separatrix_interpolator(norm_distances_points)
            R_points = [r for (r,z) in points]
            Z_points = [z for (r,z) in points]

            # Iterate over the constraint points
            for R_point, Z_point in list(zip(R_points,Z_points)):

                # Machine flux at the constraint point
                annealing_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                # Control psi at the constraint point
                annealing_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point))

                annealing_constraint_points_R.append(R_point)
                annealing_constraint_points_Z.append(Z_point)

        # Lower right quadrant
        if self.constraints["annealing"]["constrain_lower_right_quadrant"]:

            # Extract the number of constraint points to use
            N_points = self.constraints["annealing"]["N_constraints_lower_right_quadrant"]

            # Calculate the normalised distance along the separatrix of these points
            norm_distances_points = np.linspace(self.eq.separatrix_dist_norm_vertical_lower_lcfs,1,N_points + 2)[1:-1]

            # Extract the R,Z location of these points
            points = self.eq.separatrix_interpolator(norm_distances_points)
            R_points = [r for (r,z) in points]
            Z_points = [z for (r,z) in points]

            # Iterate over the constraint points
            for R_point, Z_point in list(zip(R_points,Z_points)):

                # Machine flux at the constraint point
                annealing_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                # Control psi at the constraint point
                annealing_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point))

                annealing_constraint_points_R.append(R_point)
                annealing_constraint_points_Z.append(Z_point)

        # Outer midplane - OMP
        if self.constraints["annealing"]["constrain_omp"]:

            # Machine flux at the OMP
            annealing_constraints.append(self.eq.psi_machRZ(self.eq.R_OMP,self.eq.Z_OMP))

            # Control psi at the OMP
            annealing_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_OMP,self.eq.Z_OMP))

            annealing_constraint_points_R.append(self.eq.R_OMP)
            annealing_constraint_points_Z.append(self.eq.Z_OMP)

        # Inner midplane - IMP
        if self.constraints["annealing"]["constrain_imp"]:

            # Machine flux at the IMP
            annealing_constraints.append(self.eq.psi_machRZ(self.eq.R_IMP,self.eq.Z_IMP))

            # Control psi at the IMP
            annealing_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_IMP,self.eq.Z_IMP))

            annealing_constraint_points_R.append(self.eq.R_IMP)
            annealing_constraint_points_Z.append(self.eq.Z_IMP)

        # Upper-most point
        if self.constraints["annealing"]["constrain_upper_point"]:

            # Machine flux at the upper-most point
            annealing_constraints.append(self.eq.psi_machRZ(self.eq.R_vertical_upper,self.eq.Z_vertical_upper))

            # Control psi at the upper-most point
            annealing_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_vertical_upper,self.eq.Z_vertical_upper))

            annealing_constraint_points_R.append(self.eq.R_vertical_upper)
            annealing_constraint_points_Z.append(self.eq.Z_vertical_upper)

        # Lower-most point
        if self.constraints["annealing"]["constrain_lower_point"]:

            # Machine flux at the upper-most point
            annealing_constraints.append(self.eq.psi_machRZ(self.eq.R_vertical_lower,self.eq.Z_vertical_lower))

            # Control psi at the upper-most point
            annealing_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_vertical_lower,self.eq.Z_vertical_lower))

            annealing_constraint_points_R.append(self.eq.R_vertical_lower)
            annealing_constraint_points_Z.append(self.eq.Z_vertical_lower)

        # Divertor constraint points - these are points that should lie on the divertor
        # legs, i.e. the separatrix. The points may or may not currently be on the separatrix.
        if self.constraints["annealing"]["additional_divertor_constraint_points"] is not None:

            # Iterate over the points
            for point in self.constraints["annealing"]["additional_divertor_constraint_points"]:

                # Extract the R,Z of the point
                R_point = point[0]
                Z_point = point[1]

                # The coils must produce a flux equal to the difference between the desired
                # total flux (that of the separatrix) and the background flux from the plasma,
                # at this point
                required_machine_flux = self.eq.psi_lcfs - self.eq.psi_plasRZ(R_point,Z_point)

                # Machine flux at the constraint point
                annealing_constraints.append(required_machine_flux)

                # Control psi at the constraint point
                annealing_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point))

                annealing_constraint_points_R.append(R_point)
                annealing_constraint_points_Z.append(Z_point)

        # Additional divertor X-points - these are X-points in addition to the primary X-point(s).
        # These may be used in advanced divertor configurations such as the X-point target,
        # snowflake, X and super-X divertor configurations.
        if self.constraints["annealing"]["additional_divertor_xpoints"] is not None:

            # Iterate over the points
            for point in self.constraints["annealing"]["additional_divertor_xpoints"]:

                # Extract the R,Z of the point
                R_point = point[0]
                Z_point = point[1]

                # To make a new X-point, the machine must provide both a vertical and radial field
                # that exactly cancels out the vertical and radial field from the background plasma.
                background_Br = self.eq.Br_plasRZ(R_point,Z_point)
                background_Bz = self.eq.Bz_plasRZ(R_point,Z_point)

                # Machine total Br at X-point
                annealing_constraints.append(-background_Br)

                # Machine total Bz at X-point
                annealing_constraints.append(-background_Bz)

                # Control Br at X-point
                annealing_greens_matrix.append(self.tokamak_opt.control_Br(R_point,Z_point))

                # Control Bz at X-point
                annealing_greens_matrix.append(self.tokamak_opt.control_Bz(R_point,Z_point))

                annealing_constraint_points_R.append(R_point)
                annealing_constraint_points_Z.append(Z_point)

        self.annealing_constraint_points_R = annealing_constraint_points_R
        self.annealing_constraint_points_Z = annealing_constraint_points_Z

        # Cast to numpy arrays
        self.annealing_constraints = np.asarray(annealing_constraints)
        self.annealing_greens_matrix = np.asarray(annealing_greens_matrix)

        # Calculate the null space of the greens matrix
        self.annealing_greens_matrix_nullspace = null_space(self.annealing_greens_matrix)

        # Calculate the transpose of the nullspace matrix
        self.annealing_greens_matrix_nullspace_transpose = np.transpose(self.annealing_greens_matrix_nullspace)

        # Record the number of constraints
        self.annealing_N_constraints = len(self.annealing_constraints)

        logger.info('N_coils: %d', self.tokamak_opt.N_coils)
        logger.info('N_constraints (annealing): %d', self.annealing_N_constraints)

        # Tikhonov constraints - these will only exist if an initial estimate of the coil
        # currents is required to be calculated.
        #######################

        if self.estimate_initial_currents:

            tikhonov_greens_matrix = []
            tikhonov_constraints = []
            tikhonov_constraint_points_R = []
            tikhonov_constraint_points_Z = []

            # The user can elect to exclude/turn off certain coils during this estimation
            # of the initial coil currents. To enable this, the Green's functions for such coils
            # are set to 0, which will lead to their currents being returned as 0.

            # We now create a "mask" array of 1's, of length N_coils, wherein enetries corresponding
            # to excluded coils are set to 0.
            self.tikhonov_mask = np.ones(self.tokamak_opt.N_coils)

            if self.constraints["tikhonov"]["exclude_coils"] is not None:

                # Get the names of the coils in the coilset
                coil_names = self.tokamak_opt.get_coil_names()

                # Get a list of indeces of the excluded coils as they appear in the above list
                name_indeces = [coil_names.index(name) for name in self.constraints["tikhonov"]["exclude_coils"]]

                # Adjust the mask, setting the excluded coils entries' to 0
                for index in name_indeces:
                    self.tikhonov_mask[index] = 0

            # X-point constraint - determine which X-point to use.
            tikhonov_xpt_choice = self.constraints["tikhonov"].get("xpoint_constraint", "primary")
            R_xpt_tikhonov, Z_xpt_tikhonov = self._resolve_xpoint_constraint(tikhonov_xpt_choice, "tikhonov")

            # Machine total Br at X-point
            tikhonov_constraints.append(self.eq.Br_machRZ(R_xpt_tikhonov, Z_xpt_tikhonov))

            # Machine total Bz at X-point
            tikhonov_constraints.append(self.eq.Bz_machRZ(R_xpt_tikhonov, Z_xpt_tikhonov))

            # Machine flux at the X-point
            tikhonov_constraints.append(self.eq.psi_machRZ(R_xpt_tikhonov, Z_xpt_tikhonov))

            # Control Br at X-point
            tikhonov_greens_matrix.append(
                self.tokamak_opt.control_Br(R_xpt_tikhonov, Z_xpt_tikhonov) * self.tikhonov_mask
            )

            # Control Bz at X-point
            tikhonov_greens_matrix.append(
                self.tokamak_opt.control_Bz(R_xpt_tikhonov, Z_xpt_tikhonov) * self.tikhonov_mask
            )

            # Control psi at the X-point
            tikhonov_greens_matrix.append(
                self.tokamak_opt.control_psi(R_xpt_tikhonov, Z_xpt_tikhonov) * self.tikhonov_mask
            )

            tikhonov_constraint_points_R.append(R_xpt_tikhonov)
            tikhonov_constraint_points_Z.append(Z_xpt_tikhonov)

            # Secondary X-point - removed pending further consideration

            # Upper right quadrant
            if self.constraints["tikhonov"]["constrain_upper_right_quadrant"]:

                # Extract the number of constraint points to use
                N_points = self.constraints["tikhonov"]["N_constraints_upper_right_quadrant"]

                # Calculate the normalised distance along the separatrix of these points
                norm_distances_points = np.linspace(0.0,self.eq.separatrix_dist_norm_vertical_upper_lcfs,N_points + 2)[1:-1]

                # Extract the R,Z location of these points
                points = self.eq.separatrix_interpolator(norm_distances_points)
                R_points = [r for (r,z) in points]
                Z_points = [z for (r,z) in points]

                # Iterate over the constraint points
                for R_point, Z_point in list(zip(R_points,Z_points)):

                    # Machine flux at the constraint point
                    tikhonov_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                    # Control psi at the constraint point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point) * self.tikhonov_mask)

                    tikhonov_constraint_points_R.append(R_point)
                    tikhonov_constraint_points_Z.append(Z_point)

            # Upper left quadrant
            if self.constraints["tikhonov"]["constrain_upper_left_quadrant"]:

                # Extract the number of constraint points to use
                N_points = self.constraints["tikhonov"]["N_constraints_upper_left_quadrant"]

                # Calculate the normalised distance along the separatrix of these points
                norm_distances_points = np.linspace(
                    self.eq.separatrix_dist_norm_vertical_upper_lcfs,
                    self.eq.separatrix_dist_norm_imp_lcfs,
                    N_points + 2
                )[1:-1]

                # Extract the R,Z location of these points
                points = self.eq.separatrix_interpolator(norm_distances_points)
                R_points = [r for (r,z) in points]
                Z_points = [z for (r,z) in points]

                # Iterate over the constraint points
                for R_point, Z_point in list(zip(R_points,Z_points)):

                    # Machine flux at the constraint point
                    tikhonov_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                    # Control psi at the constraint point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point) * self.tikhonov_mask)

                    tikhonov_constraint_points_R.append(R_point)
                    tikhonov_constraint_points_Z.append(Z_point)

            # Lower left quadrant
            if self.constraints["tikhonov"]["constrain_lower_left_quadrant"]:

                # Extract the number of constraint points to use
                N_points = self.constraints["tikhonov"]["N_constraints_lower_left_quadrant"]

                # Calculate the normalised distance along the separatrix of these points
                norm_distances_points = np.linspace(
                    self.eq.separatrix_dist_norm_imp_lcfs,
                    self.eq.separatrix_dist_norm_vertical_lower_lcfs,
                    N_points + 2
                )[1:-1]

                # Extract the R,Z location of these points
                points = self.eq.separatrix_interpolator(norm_distances_points)
                R_points = [r for (r,z) in points]
                Z_points = [z for (r,z) in points]

                # Iterate over the constraint points
                for R_point, Z_point in list(zip(R_points,Z_points)):

                    # Machine flux at the constraint point
                    tikhonov_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                    # Control psi at the constraint point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point) * self.tikhonov_mask)

                    tikhonov_constraint_points_R.append(R_point)
                    tikhonov_constraint_points_Z.append(Z_point)

            # Lower right quadrant
            if self.constraints["tikhonov"]["constrain_lower_right_quadrant"]:

                # Extract the number of constraint points to use
                N_points = self.constraints["tikhonov"]["N_constraints_lower_right_quadrant"]

                # Calculate the normalised distance along the separatrix of these points
                norm_distances_points = np.linspace(self.eq.separatrix_dist_norm_vertical_lower_lcfs,1,N_points + 2)[1:-1]

                # Extract the R,Z location of these points
                points = self.eq.separatrix_interpolator(norm_distances_points)
                R_points = [r for (r,z) in points]
                Z_points = [z for (r,z) in points]

                # Iterate over the constraint points
                for R_point, Z_point in list(zip(R_points,Z_points)):

                    # Machine flux at the constraint point
                    tikhonov_constraints.append(self.eq.psi_machRZ(R_point,Z_point))

                    # Control psi at the constraint point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point) * self.tikhonov_mask)

                    tikhonov_constraint_points_R.append(R_point)
                    tikhonov_constraint_points_Z.append(Z_point)

            # Outer midplane - OMP
            if self.constraints["tikhonov"]["constrain_omp"]:

                # Machine flux at the OMP
                tikhonov_constraints.append(self.eq.psi_machRZ(self.eq.R_OMP,self.eq.Z_OMP))

                # Control psi at the OMP
                tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_OMP,self.eq.Z_OMP) * self.tikhonov_mask)

                tikhonov_constraint_points_R.append(self.eq.R_OMP)
                tikhonov_constraint_points_Z.append(self.eq.Z_OMP)

            # Inner midplane - IMP
            if self.constraints["tikhonov"]["constrain_imp"]:

                # Machine flux at the IMP
                tikhonov_constraints.append(self.eq.psi_machRZ(self.eq.R_IMP,self.eq.Z_IMP))

                # Control psi at the IMP
                tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_IMP,self.eq.Z_IMP) * self.tikhonov_mask)

                tikhonov_constraint_points_R.append(self.eq.R_IMP)
                tikhonov_constraint_points_Z.append(self.eq.Z_IMP)

            # Upper-most point
            if self.constraints["tikhonov"]["constrain_upper_point"]:

                # Machine flux at the upper-most point
                tikhonov_constraints.append(self.eq.psi_machRZ(self.eq.R_vertical_upper,self.eq.Z_vertical_upper))

                # Control psi at the upper-most point
                tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_vertical_upper,self.eq.Z_vertical_upper))

                tikhonov_constraint_points_R.append(self.eq.R_vertical_upper)
                tikhonov_constraint_points_Z.append(self.eq.Z_vertical_upper)

            # Lower-most point
            if self.constraints["tikhonov"]["constrain_lower_point"]:

                # Machine flux at the upper-most point
                tikhonov_constraints.append(self.eq.psi_machRZ(self.eq.R_vertical_lower,self.eq.Z_vertical_lower))

                # Control psi at the upper-most point
                tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(self.eq.R_vertical_lower,self.eq.Z_vertical_lower))

                tikhonov_constraint_points_R.append(self.eq.R_vertical_lower)
                tikhonov_constraint_points_Z.append(self.eq.Z_vertical_lower)

            # Divertor constraint points - these are points that should lie on the divertor
            # legs, i.e. the separatrix. The points may or may not currently be on the separatrix.
            if self.constraints["tikhonov"]["additional_divertor_constraint_points"] is not None:

                # Iterate over the points
                for point in self.constraints["tikhonov"]["additional_divertor_constraint_points"]:

                    # Extract the R,Z of the point
                    R_point = point[0]
                    Z_point = point[1]

                    # The coils must produce a flux equal to the difference between the desired
                    # total flux (that of the separatrix) and the background flux from the plasma,
                    # at this point
                    required_machine_flux = self.eq.psi_lcfs - self.eq.psi_plasRZ(R_point,Z_point)

                    # Machine flux at the constraint point
                    tikhonov_constraints.append(required_machine_flux)

                    # Control psi at the constraint point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_psi(R_point,Z_point) * self.tikhonov_mask)

                    tikhonov_constraint_points_R.append(R_point)
                    tikhonov_constraint_points_Z.append(Z_point)

            # Additional divertor X-points - these are X-points in addition to the primary X-point(s).
            # These may be used in advanced divertor configurations such as the X-point target,
            # snowflake, X and super-X divertor configurations.
            if self.constraints["tikhonov"]["additional_divertor_xpoints"] is not None:

                # Iterate over the points
                for point in self.constraints["tikhonov"]["additional_divertor_xpoints"]:

                    # Extract the R,Z of the point
                    R_point = point[0]
                    Z_point = point[1]

                    # To make a new X-point, the machine must provide both a vertical and radial field
                    # that exactly cancels out the vertical and radial field from the background plasma.
                    background_Br = self.eq.Br_plasRZ(R_point,Z_point)
                    background_Bz = self.eq.Bz_plasRZ(R_point,Z_point)

                    # Machine total Br at X-point
                    tikhonov_constraints.append(-background_Br)

                    # Machine total Bz at X-point
                    tikhonov_constraints.append(-background_Bz)

                    # Control Br at X-point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_Br(R_point,Z_point) * self.tikhonov_mask)

                    # Control Bz at X-point
                    tikhonov_greens_matrix.append(self.tokamak_opt.control_Bz(R_point,Z_point) * self.tikhonov_mask)

                    tikhonov_constraint_points_R.append(R_point)
                    tikhonov_constraint_points_Z.append(Z_point)

            self.tikhonov_constraint_points_R = tikhonov_constraint_points_R
            self.tikhonov_constraint_points_Z = tikhonov_constraint_points_Z

            # Cast to numpy arrays
            self.tikhonov_constraints = np.asarray(tikhonov_constraints)
            self.tikhonov_greens_matrix = np.asarray(tikhonov_greens_matrix)

            # Record the number of constraints
            self.tikhonov_N_constraints = len(self.tikhonov_constraints)

    def create_buffers(self):
        """Creates buffer regions around the wall of the machine.

        The user can choose to create buffer regions around parts of the wall. These are useful
        if the user wants the divertor leg(s) to not come within a certain distance of parts of
        the wall. An example of this might be regions such as the divertor nose in tightly
        baffled geometries, which may not be equipped to handle heat fluxes as high as the main
        divertor targets.

        Buffer definitions are keyed by divertor region name so that, at runtime,
        only the buffers belonging to the region currently being traced are checked
        for intersections.  This improves efficiency and avoids false-positive
        buffer hits from unrelated regions.

        Raises
        ------
        ValueError
            If ``use_buffers`` is True but no buffer data has been supplied.
        TypeError
            If ``buffers`` is not a dict keyed by region name.
        """

        if self.buffers_input is None:
            raise ValueError(
                "use_buffers=True but no buffer data was supplied. "
                "Pass buffer definitions via the 'buffers' parameter, e.g. "
                "buffers={'lower_outer': [{'R': [R1, R2], 'Z': [Z1, Z2], 'distance': d}, ...], ...}, "
                "or use the FORGE GUI to define buffers interactively."
            )

        if not isinstance(self.buffers_input, dict):
            raise TypeError(
                "The 'buffers' parameter must be a dict keyed by divertor region "
                "name, e.g. {'lower_outer': [{'R': [R1, R2], 'Z': [Z1, Z2], 'distance': d}, ...]}. "
                f"Got {type(self.buffers_input).__name__}."
            )

        # Build per-region Shapely geometries and definition records.
        # self.buffers_input = {"region": [{"R":[R1,R2],"Z":[Z1,Z2],"distance":d}, ...], ...}

        self.buffers = {}      # {region: [boundary_geom, ...]}
        self.buffers_data = {} # {region: [{"R":..., "Z":..., "distance":...}, ...]}

        for region, defs_list in self.buffers_input.items():

            region_geoms = []
            region_data = []

            for buffer_def in defs_list:

                R = buffer_def["R"]
                Z = buffer_def["Z"]
                distance = buffer_def["distance"]

                # Reconstruct the Shapely buffer geometry
                line_geom = LineString(list(zip(R, Z)))
                buffer_geom = line_geom.buffer(distance)
                region_geoms.append(buffer_geom.boundary)

                # Record the definition data for potential saving
                region_data.append({
                    "R": list(R),
                    "Z": list(Z),
                    "distance": float(distance),
                })

            if region_geoms:
                self.buffers[region] = region_geoms
                self.buffers_data[region] = region_data

    def create_xpoint_regions(self):
        """Defines regions for additional divertor X-points.

        The user can choose to define regions within which X-points are encouraged to be
        formed under optimisation. The regions are recorded as a dictionary mapping
        divertor region names to Shapely LineString objects.

        Raises
        ------
        ValueError
            If ``use_xpoint_regions`` is True but no region data has been supplied.
        TypeError
            If ``xpoint_regions`` is not a dictionary keyed by divertor region name.
        """

        if self.xpoint_regions is None:
            raise ValueError(
                "use_xpoint_regions=True but no X-point region data was supplied. "
                "Pass region definitions via the 'xpoint_regions' parameter as a "
                "dictionary mapping divertor region names to {R, Z} definitions, e.g. "
                'xpoint_regions={"lower_outer": {"R": [...], "Z": [...]}}'
            )

        if not isinstance(self.xpoint_regions, dict):
            raise TypeError(
                "xpoint_regions must be a dictionary mapping divertor region names "
                "to {R, Z} definitions, e.g. "
                'xpoint_regions={"lower_outer": {"R": [...], "Z": [...]}}. '
                "Received type: " + type(self.xpoint_regions).__name__
            )

        # The user has supplied the X-point region locations keyed by divertor region.
        # self.xpoint_regions = {"lower_outer": {"R":[], "Z":[]}, ...}
        xpoint_regions = {}

        for region_name, region_data in self.xpoint_regions.items():

            if region_name not in self.divertor_regions:
                logger.warning(
                    "X-point region defined for '%s' but that divertor region is not "
                    "being optimised — skipping.", region_name
                )
                continue

            # Get the region
            R_region = region_data["R"]
            Z_region = region_data["Z"]

            # Densify the shape. Here, we insert more points along the
            # edges of the shape.
            R_region, Z_region = densify_closed_shape(R_region, Z_region, 0.01)

            # Make the region into a Shapely LineString
            xpoint_region = LineString(list(zip(R_region, Z_region)))

            xpoint_regions[region_name] = xpoint_region

        self.xpoint_regions = xpoint_regions

    def init_xpoint_regions(self):
        """Initialises secondary divertor X-point regions.

        Initialises regions in which the optimiser will encourage additional X-points to form. This is particularly
        useful if X-points target or snowflake divertor configurations are desired.

        A list of regions in the form of Shapely LineStrings will have been pre-defined as an attribute of the
        tokamak being used in the optimisation. For each of these regions, those points on the 2D R,Z equilibrium
        grid that lie within the region(s) are identified and recorded. There will be N_points number of points
        that lie within the X-points region(s).

        For each of these points a set of 2x2 Jacobian matrices for the poloidal field from each of the coils in the
        coilset is created, along with a corresponding 2x2 Jacobian matrix for the background poloidal field from
        the plasma.

        The coil Jacobian matrices are themselves stored in a N_points x N_coils matrix, with the background field
        Jacobian matrices stored in an N_points vector.

        These data will later be used to identify the location of additional secondary divertor X-points that
        may form within these regions.

        We also record the background poloidal field components (Br,Bz) themselves, along with coil Br/Bz control
        responses, as a means of later quickly calculating the poloidal field strength inside the regions.
        """

        # The user can define the X-point region(s) interactively or by supplying the points defining the region(s)

        # First, the user defines the X-point regions
        self.create_xpoint_regions()

        # Record the number of X-point regions
        self.N_xpoint_regions = len(self.xpoint_regions)

        for divertor_region, xpoint_region in self.xpoint_regions.items():

            # Get the R,Z of the grid points inside this region
            R_points, Z_points = grid_points_inside_linestring(
                self.eq.R_2D, self.eq.Z_2D, xpoint_region, include_boundary=True
            )

            # Get the control Br/Bz at these points. This is an N_points x N_coils matrix
            xpoint_region_greens_br_matrix = []
            xpoint_region_greens_bz_matrix = []

            # Get the Br/Bz background field from the plasma at these points. These are list of length N_points.
            xpoint_region_background_Br_points = []
            xpoint_region_background_Bz_points = []

            # Get the coils poloidal field Jacobians matrices. This is an N_points x N_coils matrix of matrices.
            xpoint_region_coils_Bp_jacobians_matrix = []

            # Get the background poloidal field Jacobian from the plasma at these points. This is a list of length N_points.
            xpoint_region_background_Bp_jacobian_points = []

            for R_point,Z_point in zip(R_points,Z_points):

                # Get the control Br/Bz from the coils
                xpoint_region_greens_br_matrix.append(self.tokamak_opt.control_Br(R_point,Z_point))
                xpoint_region_greens_bz_matrix.append(self.tokamak_opt.control_Bz(R_point,Z_point))

                # Get the background Br/Bz from the plasma
                xpoint_region_background_Br_points.append(self.eq.Br_plasRZ(R_point,Z_point))
                xpoint_region_background_Bz_points.append(self.eq.Bz_plasRZ(R_point,Z_point))

                # Get the control poloidal field Jacobian matrices
                xpoint_region_coils_Bp_jacobians_matrix.append(self.tokamak_opt.control_Bp_jacobians(R_point,Z_point))

                # Get the backhround poloidal field Jacobian matrix
                xpoint_region_background_Bp_jacobian_points.append(self.eq.Bpol_jacobian_plasRZ(R_point,Z_point))

            # Convert to numpy arrays
            R_points = np.asarray(R_points)
            Z_points = np.asarray(Z_points)

            xpoint_region_greens_br_matrix = np.asarray(xpoint_region_greens_br_matrix)
            xpoint_region_greens_bz_matrix = np.asarray(xpoint_region_greens_bz_matrix)

            xpoint_region_background_Br_points = np.asarray(xpoint_region_background_Br_points)
            xpoint_region_background_Bz_points = np.asarray(xpoint_region_background_Bz_points)

            xpoint_region_coils_Bp_jacobians_matrix = np.asarray(xpoint_region_coils_Bp_jacobians_matrix)

            xpoint_region_background_Bp_jacobian_points = np.asarray(xpoint_region_background_Bp_jacobian_points)

            # Calculate the transpose of these matrices. We do this because it is more
            # convenient to work with row vectors to represent 1D quantities.
            xpoint_region_greens_br_matrix_transpose = np.transpose(xpoint_region_greens_br_matrix)
            xpoint_region_greens_bz_matrix_transpose = np.transpose(xpoint_region_greens_bz_matrix)
            xpoint_region_coils_Bp_jacobians_matrix_transpose = np.swapaxes(xpoint_region_coils_Bp_jacobians_matrix,0,1)

            # The divertor region for this X-point region is explicitly specified
            # by the user via the xpoint_regions dictionary key.

            # Using these matrices, calculate the initial poloidal field square in the region

            Br_coils_initial = self.initial_coil_currents @ xpoint_region_greens_br_matrix_transpose
            Bz_coils_initial = self.initial_coil_currents @ xpoint_region_greens_bz_matrix_transpose

            Br_initial = Br_coils_initial + xpoint_region_background_Br_points
            Bz_initial = Bz_coils_initial + xpoint_region_background_Bz_points

            Bp2_initial = Br_initial * Br_initial + Bz_initial * Bz_initial

            # Extract the minimum value and record
            initial_Bp2_min = np.amin(Bp2_initial)

            # Extract the R,Z of the X-point region boundary, as it will be convenient to store this data
            # in the divertor region subdict
            R_region, Z_region = xpoint_region.xy

            # Note that an X-point region exists within this divertor region and initialise the subdict
            self.divertor_data[divertor_region]["xpoint_region_present"] = True

            self.divertor_data[divertor_region]["xpoint_region"] = {
                "R_points": R_points,
                "Z_points": Z_points,
                "R_region": R_region,
                "Z_region": Z_region,
                "region_boundary": xpoint_region,
                "greens_br_matrix": xpoint_region_greens_br_matrix, # N_points x N_coils
                "greens_bz_matrix": xpoint_region_greens_bz_matrix, # N_points x N_coils
                "greens_br_matrix_transpose": xpoint_region_greens_br_matrix_transpose,
                "greens_bz_matrix_transpose": xpoint_region_greens_bz_matrix_transpose,
                "background_Br_points": xpoint_region_background_Br_points,
                "background_Bz_points": xpoint_region_background_Bz_points,
                "coils_Bp_jacobians_matrix": xpoint_region_coils_Bp_jacobians_matrix, # N_points x N_coils
                "coils_Bp_jacobians_matrix_transpose": xpoint_region_coils_Bp_jacobians_matrix_transpose, # N_points x N_coils
                "xpoint_region_background_Bp_jacobian_points": xpoint_region_background_Bp_jacobian_points,
                "initial_Bp2_min": initial_Bp2_min,
            }

        # Set any divertor regions without an X-point region to have an X-point region cost of 0.
        for divertor_region in self.divertor_regions:

            if not self.divertor_data[divertor_region]["xpoint_region_present"]:

                self.divertor_data[divertor_region]["initial_xpoint_region_cost"] = 0.0

    def init_current_step_size(self,currents):
        """Sets the typical step size for perturbing the coil currents.
        
        Parameters
        ----------
        currents : 1D numpy array.
            The coil currents (A).
        """

        # Filter the currents to remove small currents
        eps = 0.1 * 1.0e03
        currents_filtered = [current for current in currents if abs(current) >= eps]

        # Get the mean current from the filtered set
        self.typical_current = np.mean(np.abs(currents_filtered))

        # Set the typical current step size
        self.current_step_size = self.current_step_size_factor * self.typical_current

        # Whilst here, we will calculate the initial sum square of the coil currents, which will
        # later be used in the optimisation's cost function.
        self.initial_sum_square_coil_currents = np.dot(currents,currents)

        # We will also need to consider what would happen if coils that are initially off (or have a realtively
        # low current) find themselves being made use of as a result of the optimisation, i.e. they obtain currents
        # comparable to those in the other PF coils that are initally energised. We can imagine a similar
        # sum square of the coil currents in such an instance.

        # We will now create an equivalent list of coil currents, with all coils reasonably energised.
        energised_currents = np.abs(deepcopy(currents))
        energised_currents[energised_currents < 0.05 * (self.typical_current)] = 0.5 * self.typical_current

        # Calculate the corresponding sum square of the coil currents.
        self.energised_sum_square_coil_currents = np.dot(energised_currents,energised_currents)
        logger.info('typical current (kA): %.3f', self.typical_current * 1.0e-03)
        logger.info('initial currents (kA): %s', currents * 1.0e-03)
        logger.info('energised currents (kA): %s', energised_currents * 1.0e-03)
        logger.info('sum square: %.6e', self.initial_sum_square_coil_currents)
        logger.info('energised sum square: %.6e', self.energised_sum_square_coil_currents)

    def estimate_currents(self):
        """Estimates the initial currents in the coils using Tikhonov regularisation.

        Performs Tikhonov reguralisation to estimate initial coil currents, automatically tuning
        the regularisation parameter, alpha, to meet the equilibrium constraints within a specified
        threshold. This is optionally used, and is particularly useful when using a coil set that
        differs from the coil set used to produce the initial equilibrium. The simulated annealing
        will move us around in the subspace of coil currents that satisfy our constraints - this
        only works by perturbing an initial set of coil currents that themselves satisfy the constraints.
        Hence, if you have a new coil set, and don't know what the initial currents are,
        you can use this to estimate such currents.
        """

        self.initial_currents_estimate = self.tikhonov_min_residual(
            G = self.tikhonov_greens_matrix,
            b = self.tikhonov_constraints,
            threshold = 0.01,
            alpha_start = 1e-10,
            alpha_min = 1e-40,
            decay_factor = 0.8,
            max_iter = 1000,
        )

        np.set_printoptions(suppress=True)
        logger.info('initial_currents_estimate (kA): %s', self.initial_currents_estimate * 1.0e-03)
        logger.info('initial_currents_estimate (MA): %s', self.initial_currents_estimate * 1.0e-06)

        # Generate the new 2D grid of poloidal magnetic flux from these coil currents
        data = self.generate_eq_from_currents(
            new_currents=self.initial_currents_estimate
            )

        # Record the 2D map of poloidal magnetic flux from this estimate of the
        # initial coil currents
        self.initial_psi_2D_estimate = data["psi_2D"]

        # Visualise the resultant equilibrium

        # Locate the X-point
        _, xpt = find_critical(self.eq.R_2D,self.eq.Z_2D,data["psi_2D"])

        # Plot - total flux
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')

        ax.contour(self.eq.R_2D,self.eq.Z_2D,data["psi_2D"],levels=100,alpha=0.4,colors='k')
        ax.contour(self.eq.R_2D,self.eq.Z_2D,self.eq.psi_2D,levels=[self.eq.psi_lcfs],colors='r')
        ax.contour(self.eq.R_2D,self.eq.Z_2D,data["psi_2D"],levels=[xpt[0][2]],colors='tab:orange')
        ax.plot(self.tokamak_opt.wall_R,self.tokamak_opt.wall_Z,color='k')
        ax.scatter(self.tikhonov_constraint_points_R,self.tikhonov_constraint_points_Z,color='r',marker='s',zorder=2)
        ax = self.tokamak_opt.plot(ax=ax,show=False)
        ax.title.set_text('Pseudo-inverse')
        plt.show()

    def tikhonov_min_residual(
        self,
        G = None,
        b = None,
        threshold = 0.01,
        alpha_start = 1e-10,
        alpha_min = 1e-35,
        decay_factor = 0.8,
        max_iter = 1000,
    ):
        """Performs Tikhonov regularisation on the coil curents.

        Performs Tikhonov regularisation to solve G I = b for I, where I is a vector of coil
        currents, G is a matrix of Green's functions related to a set of equilibrium constraints
        on field/flux, denoted by the vector b. The regularisation parameter, alpha, is automatically
        tuned (becoming smaller each iteration), such that the residuals are minimised until the maximum
        error value drops below some threshold value set by the user.

        Parameters
        ----------
        G : 2D numpy array
            2D matrix of Green's functions/coil responses.
        b : 1D numpy array
            Vector of constraints.
        threshold : float
            Value that the maximum relative error must drop below.
        alpha_start : float
            The initial value of alpha.
        alpha_min : float
            The minimum allowed value of alpha.
        decay_factor : float
            Value by which alpha is multiplied during each tuning.
        max_iter : int
            Maximum number of interations/tunings of alpha to take.

        Returns
        -------
        I : 1D numpy array
            Vector of coil currents.
        """

        alpha = alpha_start

        # These quantities depend only on G and b, which are constant
        # across iterations — compute once outside the loop.
        GTG = G.T @ G
        GTb = G.T @ b
        identity = np.eye(G.shape[1])

        for _ in range(max_iter):

            regularised = GTG + alpha * identity
            I = np.linalg.solve(regularised, GTb)
            b_achieved = G @ I
            relative_errors = np.abs((b_achieved - b) / b)
            max_relative_error = np.amax(relative_errors)

            if max_relative_error < threshold:
                return I

            alpha *= decay_factor

            if alpha < alpha_min:
                break

    def update_cooling_factor(self):
        """Updates the cooling factor, β.
        
        Sets the cooling factor, β, for use in the cooling schedule (T_new = β * T_old).
        If many iterations at the current temperature were required to reach the threshold
        acceptance rate, then this indicates that the optimiser is still acting in an exploratory
        manner, hence β will remain high (the temperature will not decrease much). This aims to
        prevent premature cooling. Likewise, if not that many iterations were required, this
        indicates the optimiser is not being as explorative, hence β will remain low (the
        temperature will decrease more rapidly). β is bounded between β_min and β_max.
        """

        baseline_iterations = int(0.5 * self.n_window)

        # Factor is capped at 1 to prevent the cooling becoming too slow.
        factor = min(self.iterations_to_acceptance / baseline_iterations,1.0)
        self.cooling_factor = self.max_cooling_factor - \
            (self.max_cooling_factor - self.min_cooling_factor) * (1.0 - factor)

    def _resolve_xpoint_constraint(self, choice, label):
        """Resolves the X-point (R, Z) coordinates to use for constraints.

        Parameters
        ----------
        choice : str
            One of ``"primary"``, ``"lower"``, or ``"upper"``.
        label : str
            A label for logging purposes (e.g. ``"annealing"`` or ``"tikhonov"``).

        Returns
        -------
        R_xpt : float
            Major radius of the selected X-point.
        Z_xpt : float
            Vertical position of the selected X-point.
        """

        if choice == "primary":
            return self.eq.R_xpt_lcfs, self.eq.Z_xpt_lcfs

        if choice not in ("lower", "upper"):
            raise ValueError(
                f"xpoint_constraint must be 'primary', 'lower', or 'upper'. Got '{choice}'."
            )

        if not self.eq.DND:
            logger.warning(
                "xpoint_constraint='%s' was specified for %s constraints, but the equilibrium "
                "is not a double null. Falling back to the primary X-point.",
                choice, label,
            )
            return self.eq.R_xpt_lcfs, self.eq.Z_xpt_lcfs

        if choice == "lower":
            logger.info(
                "Using the lower X-point for %s constraints (R=%.4f, Z=%.4f).",
                label, self.eq.R_xpt_lower, self.eq.Z_xpt_lower,
            )
            return self.eq.R_xpt_lower, self.eq.Z_xpt_lower

        # choice == "upper"
        logger.info(
            "Using the upper X-point for %s constraints (R=%.4f, Z=%.4f).",
            label, self.eq.R_xpt_upper, self.eq.Z_xpt_upper,
        )
        return self.eq.R_xpt_upper, self.eq.Z_xpt_upper

    def generate_default_constraints(self):
        """If the user does not specify any constraints, a dictionary of default constraints will be generated and returned."""

        constraints = {

            "annealing":{

                "constrain_omp": True,
                "constrain_imp": True,
                "constrain_upper_point": False,
                "constrain_lower_point": False,
                "constrain_upper_right_quadrant": True,
                "N_constraints_upper_right_quadrant": 1,
                "constrain_upper_left_quadrant": True,
                "N_constraints_upper_left_quadrant": 1,
                "constrain_lower_left_quadrant": True,
                "N_constraints_lower_left_quadrant": 1,
                "constrain_lower_right_quadrant": True,
                "N_constraints_lower_right_quadrant": 1,
                "additional_divertor_constraint_points": None,
                "additional_divertor_xpoints": None,
                "xpoint_constraint": "primary",

            },

            "tikhonov":{

                "constrain_omp": True,
                "constrain_imp": True,
                "constrain_upper_right_quadrant": True,
                "N_constraints_upper_right_quadrant": 1,
                "constrain_upper_left_quadrant": True,
                "N_constraints_upper_left_quadrant": 1,
                "constrain_lower_left_quadrant": True,
                "N_constraints_lower_left_quadrant": 1,
                "constrain_lower_right_quadrant": True,
                "N_constraints_lower_right_quadrant": 1,
                "additional_divertor_constraint_points": None,
                "additional_divertor_xpoints": None,
                "xpoint_constraint": "primary",
                "exclude_coils": None,

            }
        }

        return constraints

    def init_divertor_data(self):
        """Initialises key pieces of data related to the divertor regions.
        
        The starting position of the field line trace for each divertor region is determined, as is the direction
        of the field line trace. Divertor-related cost terms are also initialised.
        """

        # We will now identify the (R,Z) starting location for the field line trace along
        # the divertor leg, which will start near the relevant X-point. In addition, the direction of the
        # field line trace from X-point->downstream is determined, along with additional data
        # for plotting routines, such as short-hand labels and colours. Cost terms associated with
        # each divertor region are also initialised.

        # We will also determine the min/max Z plotting limits whilst examining each region.

        # Set default plotting values, which may be modified by the presence of divertor regions.
        self.plotting_R_max = self.tokamak_opt.wall_R_max
        self.plotting_R_min = self.tokamak_opt.wall_R_min
        self.plotting_Z_max = self.eq.Z_mag
        self.plotting_Z_min = self.eq.Z_mag

        for divertor_region in self.divertor_regions:

            if divertor_region == "lower_outer":

                # R,Z starting position of the field line trace
                trace_starting_position_R = self.eq.R_xpt_lower + 2.0 * self.eq.dR
                trace_starting_position_Z = self.eq.Z_xpt_lower

                # Short-hand label for the region
                short_label = "LO"

                # Plotting colour
                colour = "#1F77B4"

                # Z limit for plotting
                self.plotting_Z_min = self.tokamak_opt.wall_Z_min

            elif divertor_region == "lower_inner":

                # R,Z starting position of the field line trace
                trace_starting_position_R = self.eq.R_xpt_lower - 2.0 * self.eq.dR
                trace_starting_position_Z = self.eq.Z_xpt_lower

                # Short-hand label for the region
                short_label = "LI"

                # Plotting colour
                colour = "#FF7F0E"

                # Z limit for plotting
                self.plotting_Z_min = self.tokamak_opt.wall_Z_min

            elif divertor_region == "upper_outer":

                # R,Z starting position of the field line trace
                trace_starting_position_R = self.eq.R_xpt_upper + 2.0 * self.eq.dR
                trace_starting_position_Z = self.eq.Z_xpt_upper

                # Short-hand label for the region
                short_label = "UO"

                # Plotting colour
                colour = "#D62728"

                # Z limit for plotting
                self.plotting_Z_max = self.tokamak_opt.wall_Z_max

            elif divertor_region == "upper_inner":

                # R,Z starting position of the field line trace
                trace_starting_position_R = self.eq.R_xpt_upper - 2.0 * self.eq.dR
                trace_starting_position_Z = self.eq.Z_xpt_upper

                # Short-hand label for the region
                short_label = "UI"

                # Plotting colour
                colour = "#2CA02C"

                # Z limit for plotting
                self.plotting_Z_max = self.tokamak_opt.wall_Z_max

            else:

                raise ValueError(str(divertor_region) + " is not a valid divertor region.")

            # Determine the trace direction from the local poloidal field.
            # The trace must go from near the X-point downstream toward the
            # divertor target: downward (Z decreasing) for lower divertors,
            # upward (Z increasing) for upper divertors. The Z-component of
            # the RK4 step is proportional to direction * Bz, so we pick the
            # sign of direction that gives the correct vertical step.
            Bz_start = float(self.eq.BzRZ(trace_starting_position_R, trace_starting_position_Z))
            if "lower" in divertor_region:
                # Need the trace to move downward (dZ < 0) → direction * Bz < 0
                trace_direction = -1.0 if Bz_start > 0 else 1.0
            else:
                # Need the trace to move upward (dZ > 0) → direction * Bz > 0
                trace_direction = 1.0 if Bz_start > 0 else -1.0

            # Check if the asociated cost function weights for this divertor region have been initialised.

            # If a weight has not been initialised, set it to (1/N_divertors)
            N_divertors = len(self.divertor_regions)
            backup_weight = 1.0 / N_divertors

            # Connection length
            try:

                weight_connection_length = self.divertor_data[divertor_region]["weight_connection_length"]

            except KeyError:

                weight_connection_length = backup_weight
                self.divertor_data[divertor_region]["weight_connection_length"] = weight_connection_length

            # Strike point distance
            try:

                weight_strike_point_distance = self.divertor_data[divertor_region]["weight_strike_point_distance"]

            except KeyError:

                weight_strike_point_distance = backup_weight
                self.divertor_data[divertor_region]["weight_strike_point_distance"] = weight_strike_point_distance

            # Xpoint region
            try:

                weight_xpoint_region = self.divertor_data[divertor_region]["weight_xpoint_region"]

            except KeyError:

                weight_xpoint_region = backup_weight
                self.divertor_data[divertor_region]["weight_xpoint_region"] = weight_xpoint_region

            # Calculate the initial costs for this divertor region

            # Connection length
            initial_connection_length_cost = self.initial_total_connection_length_cost * weight_connection_length

            # Strike point distance
            initial_strike_point_distance_cost = self.initial_total_strike_point_distance_cost * weight_strike_point_distance

            # X-point region - at this stage we have not checked if an X-point region associated with this divertor is present.
            # If it later transpires that one is not present, this cost will be altered to zero. For now, we assume there may
            # be one.
            initial_xpoint_region_cost = self.initial_xpoint_regions_cost * weight_xpoint_region

            # Check if a connection length multiplication factor zero point has been defined for this divertor region
            try:

                connection_length_multiplication_factor_zero = (
                    self.divertor_data[divertor_region]["connection_length_multiplication_factor_zero"]
                )

            except KeyError:

                self.divertor_data[divertor_region]["connection_length_multiplication_factor_zero"] = 1.1

            # Lastly, the relevant Shapely object for the intended strike geometry is created. If a single
            # R,Z point pair is provided, then a strike point is assumed (Shapely Point created), and .if more
            # than one point pair is provided, a strike surface is assumed (Shapely LineString created)
            strike_R = self.divertor_data[divertor_region]["strike_R"]
            strike_Z = self.divertor_data[divertor_region]["strike_Z"]

            if (
                (isinstance(strike_R,int) or isinstance(strike_R,float)) and
                (isinstance(strike_Z,int) or isinstance(strike_Z,float))
                ):

                # Strike point
                strike_object = Point(strike_R,strike_Z)

            elif (isinstance(strike_R,list)) and (isinstance(strike_Z,list)):

                # Strike surface
                strike_object = LineString(list(zip(strike_R,strike_Z)))

            else:

                logger.warning('Unable to determine provided strike geometry for the divertor region: %s', divertor_region)

            # Record these data
            self.divertor_data[divertor_region]["trace_starting_position_R"] = trace_starting_position_R
            self.divertor_data[divertor_region]["trace_starting_position_Z"] = trace_starting_position_Z
            self.divertor_data[divertor_region]["trace_direction"] = trace_direction
            self.divertor_data[divertor_region]["short_label"] = short_label
            self.divertor_data[divertor_region]["colour"] = colour
            self.divertor_data[divertor_region]["initial_connection_length_cost"] = initial_connection_length_cost
            self.divertor_data[divertor_region]["initial_strike_point_distance_cost"] = initial_strike_point_distance_cost
            self.divertor_data[divertor_region]["initial_xpoint_region_cost"] = initial_xpoint_region_cost
            self.divertor_data[divertor_region]["strike_geometry"] = strike_object

            # Adjust the min/max R/Z plotting limits to have a small offset
            dR = self.plotting_R_max - self.plotting_R_min
            dZ = self.plotting_Z_max - self.plotting_Z_min
            offset = 0.025
            self.plotting_R_min -= offset * dR
            self.plotting_R_max += offset * dR
            self.plotting_Z_min -= offset * dZ
            self.plotting_Z_max += offset * dZ

            # Each divetor region may or may not also feature a region in which an secondary divertor X-point is encouraged
            # to form. If such regions exists, they will be initialised later.
            self.divertor_data[divertor_region]["xpoint_region_present"] = False

    def request_stop(self):
        """Request the optimisation loop to stop gracefully.

        The loop will terminate at the start of the next iteration.
        This is safe to call from any thread.
        """
        self._stop_event.set()

    def get_tracking_snapshot(self):
        """Return a snapshot dict of all tracking data for external consumers (e.g. GUI).

        Returns
        -------
        dict
            A dictionary containing copies of the current tracking lists and
            key state needed by :func:`forge.plotting.update_tracking_plots`.
        """
        return {
            "temperature": list(self.tracking_temperature),
            "energy_change": list(self.tracking_energy_change),
            "cost": list(self.tracking_cost),
            "cost_strike_point_distance": list(self.tracking_cost_strike_point_distance),
            "cost_connection_length": list(self.tracking_cost_connection_length),
            "cost_coil_currents": list(self.tracking_cost_coil_currents),
            "cost_xpoint_regions": list(self.tracking_cost_xpoint_regions),
            "acceptance_rate": list(self.tracking_acceptance_rate),
            "acceptance_prob": list(self.tracking_acceptance_prob),
            "alpha": list(self.tracking_alpha),
            "connection_length": {r: list(v) for r, v in self.tracking_connection_length.items()},
            "field_lines_R": dict(self.tracking_field_lines_R),
            "field_lines_Z": dict(self.tracking_field_lines_Z),
            "incumbent_data": self.incumbent_data,
            "flux_map": self.psi_2D,
            "num_evals": self.num_evals,
            "constraint_points_R": self.annealing_constraint_points_R,
            "constraint_points_Z": self.annealing_constraint_points_Z,
            "buffers": self.buffers,  # dict keyed by region, or None
            "threshold_acceptance_rate": getattr(self, "threshold_acceptance_rate", 0.0),
            "plotting_lims": {
                "R_min": self.plotting_R_min,
                "R_max": self.plotting_R_max,
                "Z_min": self.plotting_Z_min,
                "Z_max": self.plotting_Z_max,
            },
        }

    def generate_optimised_eq_machine(
            self,
    ):
        """Creates new Equilibrium and Machine objects of the final state.
        
        Creates a new forge.equilibrium.Equilibrium object for the optimised equilibrium.
        Also creates a new forge.machine.Machine object for the tokamak with the updated
        coil currents.
        """

        # Create a copy of the exisiting machine
        self.optimised_tokamak = deepcopy(self.tokamak_opt)

        # Update the coil currents
        updated_coil_currents = self.incumbent_data["currents"]

        self.optimised_tokamak.update_currents(
            new_currents = updated_coil_currents,
        )

        # Update the equilibrium data
        # First, get the data used to initialise the starting equilibrium
        updated_eq_data = deepcopy(self.eq.eq_data)

        # Modify the (total) poloidal magnetic flux
        updated_eq_data["psi_2D"] = self.incumbent_data["psi_2D"]

        # Create the new Equilibrium object
        self.optimised_eq = Equilibrium(
            eq_data = updated_eq_data,
            tokamak = self.optimised_tokamak,
            calculate_flux_from_coils = True,
        )

    def separatrix_coil_flux_change(self):
        """Quantify the change in coil flux along the separatrix after optimisation.

        Evaluates the poloidal magnetic flux from the coils (psi_mach) along the
        initial separatrix for both the initial and optimised equilibria. If the
        coil flux is unchanged on the separatrix, the separatrix position is
        preserved and the background plasma equilibrium remains self-consistent.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``R_lcfs`` : ndarray — R coordinates of the separatrix evaluation points.
            - ``Z_lcfs`` : ndarray — Z coordinates of the separatrix evaluation points.
            - ``psi_mach_initial`` : ndarray — Coil flux along the separatrix (initial).
            - ``psi_mach_optimised`` : ndarray — Coil flux along the separatrix (optimised).
            - ``delta_psi_mach`` : ndarray — Absolute change (optimised - initial).
            - ``delta_psi_mach_rel`` : ndarray — Relative change (optimised - initial) / initial.
            - ``max_abs_change`` : float — Maximum absolute change along the separatrix.
            - ``max_rel_change`` : float — Maximum relative change (dimensionless).
            - ``mean_abs_change`` : float — Mean absolute change along the separatrix.
            - ``mean_rel_change`` : float — Mean relative change (dimensionless).
        """
        R_sep = self.eq.R_lcfs
        Z_sep = self.eq.Z_lcfs

        # Evaluate coil flux along the initial separatrix
        psi_mach_init = self.eq.psi_machRZ(R_sep, Z_sep)
        psi_mach_opt = self.optimised_eq.psi_machRZ(R_sep, Z_sep)

        delta = psi_mach_opt - psi_mach_init

        # Relative change normalised by the initial coil flux at each point
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_rel = np.where(psi_mach_init != 0, delta / psi_mach_init, 0.0)

        result = {
            "R_lcfs": R_sep,
            "Z_lcfs": Z_sep,
            "psi_mach_initial": psi_mach_init,
            "psi_mach_optimised": psi_mach_opt,
            "delta_psi_mach": delta,
            "delta_psi_mach_rel": delta_rel,
            "max_abs_change": float(np.max(np.abs(delta))),
            "max_rel_change": float(np.max(np.abs(delta_rel))),
            "mean_abs_change": float(np.mean(np.abs(delta))),
            "mean_rel_change": float(np.mean(np.abs(delta_rel))),
        }

        logger.info(
            "Separatrix coil flux change: "
            "max |Δψ_mach/ψ_mach| = %.4f%%, mean |Δψ_mach/ψ_mach| = %.4f%%.",
            result["max_rel_change"] * 100,
            result["mean_rel_change"] * 100,
        )

        return result

