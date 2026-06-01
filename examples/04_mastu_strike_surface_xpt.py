"""Example 04: Strike surface optimisation with an X-point target configuration on MAST-U.

This example builds on Example 02 by enabling an X-point target divertor geometry. When
X-point regions are active, the optimiser rewards geometries that promote the
formation of a secondary X-point in a specified region of the divertor,
encouraging an X-Point Target (XPT) topology.

Wall buffers are not used in this example. The strike surface and base
constraints are identical to Example 02, with the addition of a divertor
constraint point.

The input data files are bundled with the FORGE package under
forge/data/geqdsk/ and forge/data/json/.
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from forge.io import read_geqdsk, read_magnets
from forge.machine import Machine
from forge.equilibrium import Equilibrium
from forge.optimise import Optimiser
from forge.utils import closest_point_along_shape
from forge.plotting import plot_optimisation_summary
from forge._paths import data_dir_path

# ============================================================================
# 1. Load input data
# ============================================================================

# Paths to the equilibrium (GEQDSK) and magnet / coil definition (JSON) files,
# bundled with the FORGE package.
data_dir = data_dir_path()
path_to_geqdsk = os.path.join(data_dir, "geqdsk", "examples", "04-mastu.geqdsk")
path_to_magnets = os.path.join(data_dir, "json", "examples", "04-mastu.json")

eq_data = read_geqdsk(path_to_geqdsk)
magnets_data, circuits, suggested_circuits = read_magnets(path_to_magnets)

# ============================================================================
# 2. Build the tokamak machine description
# ============================================================================

# The Machine object holds the PF coil set, the first-wall geometry, and
# (optionally) the circuit wiring that groups coils together.
tokamak = Machine(
    magnets_data=magnets_data,
    wall_R=eq_data["wall_R"],
    wall_Z=eq_data["wall_Z"],
    circuits=circuits,
)

# ============================================================================
# 3. Create the equilibrium
# ============================================================================

# Setting calculate_flux_from_coils=True tells FORGE to recompute the flux
# contribution of each PF coil via Green's functions, which is required for
# the optimiser to adjust coil currents.
eq = Equilibrium(
    eq_data=eq_data,
    tokamak=tokamak,
    calculate_flux_from_coils=True,
)

eq.plot_fluxes()

# ============================================================================
# 4. Define constraints
# ============================================================================

# Constraints control which points on the plasma boundary are held fixed
# during optimisation. Two sets can be provided:
#   - "annealing": constraints enforced during the simulated annealing phase.
#     These are always required.
#   - "tikhonov":  optional constraints for the Tikhonov regularisation step
#     that estimates coil currents from the optimised flux. Tikhonov constraints
#     are only needed if (a) the machine used for optimisation differs from the
#     one that produced the initial equilibrium, or (b) you want to re-estimate
#     the initial currents with new constraints (e.g. divertor constraint
#     points, secondary X-point targets, or excluded coils).

constraints = {
    "annealing": {
        "constrain_omp": True,
        "constrain_imp": True,
        "constrain_upper_point": True,
        "constrain_lower_point": False,
        "constrain_upper_right_quadrant": False,
        "N_constraints_upper_right_quadrant": 1,
        "constrain_upper_left_quadrant": False,
        "N_constraints_upper_left_quadrant": 1,
        "constrain_lower_left_quadrant": True,
        "N_constraints_lower_left_quadrant": 1,
        "constrain_lower_right_quadrant": True,
        "N_constraints_lower_right_quadrant": 1,
        "additional_divertor_constraint_points": None,
        "additional_divertor_xpoints": None,
        "xpoint_constraint": "lower",
    },
    "tikhonov": {
        "constrain_omp": True,
        "constrain_imp": True,
        "constrain_upper_point": True,
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
        "xpoint_constraint": "lower",
        "exclude_coils": None,
    },
}

# ============================================================================
# 5. Define the target strike surface
# ============================================================================

# Define the strike surface as a multi-segment surface on the lower
# outer divertor. Six approximate (R, Z) coordinates are snapped
# onto the wall polygon using closest_point_along_shape, forming 5
# connected segments that cover the divertor target region.

surface_points_approx = [
    (0.964, -1.595),
    (1.564, -1.569),
    (1.730, -1.681),
    (1.348, -2.065),
    (1.088, -2.066),
    (0.979, -1.959),
]

strike_R, strike_Z = zip(
    *[
        closest_point_along_shape(tokamak.wall_R, tokamak.wall_Z, R, Z)
        for R, Z in surface_points_approx
    ]
)

divertor_data = {
    "lower_outer": {
        "strike_R": list(strike_R),
        "strike_Z": list(strike_Z),
        "connection_length_multiplication_factor_zero": 2.0,
    },
}

# ============================================================================
# 6. Configure and run the optimiser
# ============================================================================

# Compared to Example 02, this example enables X-point region targeting
# (use_xpoint_regions=True) with a non-zero cost weight. The optimiser will
# reward geometries that form a secondary X-point in the divertor region.
#
# The xpoint_regions dictionary maps each divertor region name to a polygon
# (R, Z boundary) inside which a secondary X-point is encouraged to form.

xpoint_regions = {
    "lower_outer": {
        "R": [1.120, 1.300, 1.415, 1.215],
        "Z": [-1.915, -1.910, -1.725, -1.730],
    },
}

opt = Optimiser(
    eq=eq,
    tokamak_initial=tokamak,
    divertor_data=divertor_data,
    constraints=constraints,
    # --- Annealing schedule ---
    max_evals=100_000,                         # Maximum cost-function evaluations
    initial_temperature=10.0,                 # Starting temperature
    max_cooling_factor=0.995,                 # Slowest cooling rate
    min_cooling_factor=0.99,                  # Fastest cooling rate
    threshold_acceptance_rate_decay=2.0,      # Controls adaptive cooling
    initial_threshold_acceptance_rate=0.95,    # Target acceptance rate at start
    n_window=50,                              # Window size for acceptance rate tracking
    # --- Step size ---
    current_step_size_factor=0.005,            # Scales random current perturbations
    # --- Cost function weights ---
    initial_total_connection_length_cost=0.0,  # Weight for connection length
    initial_total_strike_point_distance_cost=1.0,  # Weight for strike point proximity
    initial_xpoint_regions_cost=1.0,          # Weight for secondary X-point formation
    initial_coil_currents_cost=0.0,           # Weight penalising large coil currents
    # --- Field line tracing ---
    field_line_trace_step_size=0.05,          # Poloidal step size (m) for the RK4 tracer
    field_line_trace_max_steps=1000,          # Max steps before abandoning a trace
    field_line_trace_psi_tollerance=0.001,    # Tolerance for field line tracer
    # --- Geometry options ---
    use_buffers=False,                        # No wall buffer penalties
    use_xpoint_regions=True,                  # Reward secondary X-point formation
    xpoint_regions=xpoint_regions,             # Region(s) for secondary X-point
    # --- Logging ---
    detailed_logging=True,                    # Enable detailed per-step logging
)

opt.optimise()

# ============================================================================
# Post-optimisation summary
# ============================================================================
# Show equilibrium comparison, cost history, coil currents, and flux
# decomposition for the optimised result.
plot_optimisation_summary(opt)
