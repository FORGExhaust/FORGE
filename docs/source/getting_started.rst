Getting Started
===============

This page provides an introduction to using FORGE via Python scripts,
from the input data formats to a fully featured worked example.

.. note::

   FORGE also offers a graphical user interface (GUI) that provides an
   interactive way to set up and run optimisations without writing
   scripts. See :doc:`gui` for installation and usage instructions.


Overview
--------

A typical FORGE workflow consists of:

1. **Load an equilibrium** — read a GEQDSK file and construct an
   ``Equilibrium`` object that holds the poloidal flux, magnetic fields,
   and critical-point information.

2. **Define the machine** — create a ``Machine`` object describing the
   PF coils (positions, sizes, circuits, and current limits).

3. **Set up the optimiser** — instantiate an ``Optimiser`` with the
   equilibrium, machine, and a dictionary of constraints and cost
   functions (strike point/surface, connection length, X-point targets,
   wall buffers, etc.).

4. **Run the optimisation** — call the optimiser, which performs
   simulated annealing over the coil currents and returns an optimised
   equilibrium.

5. **Inspect the results** — use the built-in plotting utilities to
   compare initial and optimised geometries.


Input Data
----------

FORGE requires two input files: an equilibrium file (GEQDSK format) and
a machine/coil definition file (JSON format).


The GEQDSK file
^^^^^^^^^^^^^^^

A **GEQDSK** file is the
standard exchange format for tokamak magnetic equilibria. It is produced
by equilibrium reconstruction codes such as EFIT and by free-boundary
solvers such as FreeGS.

A GEQDSK file contains:

* The **poloidal magnetic flux** :math:`\psi(R,Z)` on a regular
  :math:`(R,Z)` grid.
* **Plasma profiles** as functions of normalised flux:
  :math:`p'(\psi_N)` (pressure gradient), :math:`FF'(\psi_N)`
  (the poloidal current function multiplied by its derivative), :math:`q(\psi_N)` (safety
  factor), :math:`p(\psi_N)` (pressure), and :math:`f_{\text{pol}}(\psi_N)`
  (poloidal current function).
* The **plasma boundary** and **limiter / first-wall** coordinates.
* Scalar quantities: plasma current :math:`I_p`, flux at the magnetic
  axis :math:`\psi_{\text{axis}}`, flux at the LCFS
  :math:`\psi_{\text{lcfs}}`, and the vacuum toroidal field parameter
  :math:`F_{\text{vac}} = R_0 B_0`.

FORGE reads GEQDSK files via :func:`forge.io.read_geqdsk`, which
returns a dictionary with all of the above data unpacked onto NumPy
arrays.

.. note:: **COCOS convention**

   FORGE internally uses the `COCOS <https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf>`_
   11–18 convention, where
   :math:`\psi` is the **full poloidal magnetic flux** (in Wb) and the
   magnetic field components are:

   .. math::

      B_R = -\frac{1}{2\pi R}\frac{\partial\psi}{\partial Z}, \qquad
      B_Z = +\frac{1}{2\pi R}\frac{\partial\psi}{\partial R}

   Some equilibrium codes (e.g. certain EFIT configurations) output
   GEQDSK files using COCOS 1–8, where :math:`\psi` is divided by
   :math:`2\pi` (i.e. flux per radian). FORGE automatically detects
   this on loading and applies the necessary corrections—no user
   intervention is required.


The machine JSON file
^^^^^^^^^^^^^^^^^^^^^

The machine definition is a JSON file containing the PF coil set and,
optionally, circuit wiring. An example looks like this:

.. code-block:: json

   {
     "coils": {
       "P1U": {
         "type": "filament",
         "R": [0.92, 0.92, 0.95, 0.95],
         "Z": [1.10, 1.15, 1.15, 1.10],
         "turns": 42,
         "current": 3469.63,
         "dR": 0.011,
         "dZ": 0.018
       },
       "P1L": {
         "type": "filament",
         "R": [0.92, 0.92, 0.95, 0.95],
         "Z": [-1.10, -1.15, -1.15, -1.10],
         "turns": 42,
         "current": 3469.63,
         "dR": 0.011,
         "dZ": 0.018
       },
       "D5": {
         "type": "filament",
         "R": [0.35, 0.35, 0.38, 0.38],
         "Z": [-1.80, -1.85, -1.85, -1.80],
         "turns": 20,
         "current": 1200.0,
         "dR": 0.010,
         "dZ": 0.015
       }
     },
     "circuits": {
       "P1": {
         "coils": ["P1U", "P1L"],
         "multipliers": [1.0, 1.0]
       }
     }
   }

In this example, coils ``P1U`` and ``P1L`` are wired together in a
circuit called ``P1`` with equal multipliers — when the optimiser
changes the circuit current, both coils receive the same adjustment.
Coil ``D5`` does not appear in any circuit, so the optimiser treats it
as independently controllable.

The ``"circuits"`` section is **optional**. If it is omitted (or not
present in the JSON file), every coil is treated as independently
powered — equivalent to each coil being its own single-coil circuit.
Circuits only need to be defined when coils are physically wired
together and must carry related currents.

Each coil entry contains:

* ``"type"`` — the coil model (see `Coil types`_ below).
* ``"R"`` and ``"Z"`` — position data whose meaning depends on the coil
  type.
* ``"turns"`` — number of turns.
* ``"current"`` — the initial current per turn (Amps).
* Additional fields depending on the coil type (e.g. ``"dR"``,
  ``"dZ"`` for filament coils).

FORGE reads machine JSON files via :func:`forge.io.read_magnets`, which
returns the coil data dictionary, the circuits dictionary (or ``None``),
and a suggested-circuits dictionary inferred from coil naming
conventions.


Coil types
^^^^^^^^^^

FORGE supports four coil representations. The ``"type"`` field in the
JSON selects which model is used.

**"point"** — a single point source of current. The simplest model:
the entire coil is a delta-function current source at one
:math:`(R,Z)` location.

.. code-block:: json

   {
     "type": "point",
     "R": 0.93,
     "Z": 1.12,
     "turns": 42,
     "current": 3469.63
   }

* ``"R"``, ``"Z"`` — scalar coordinates of the coil centre.

**"filament"** — a set of current filaments, each acting as a point
source. Each filament carries :math:`1/N` of the total coil current,
where :math:`N` is the number of filaments. This is the most commonly
used type, and is well suited to cases where filament positions are
placed at the centres of the real physical turns.

.. code-block:: json

   {
     "type": "filament",
     "R": [0.92, 0.92, 0.95, 0.95],
     "Z": [1.10, 1.15, 1.15, 1.10],
     "turns": 42,
     "current": 3469.63,
     "dR": 0.011,
     "dZ": 0.018
   }

* ``"R"``, ``"Z"`` — lists of :math:`(R,Z)` coordinates for each
  filament.
* ``"dR"``, ``"dZ"`` — the radial width and vertical height of each
  filament (used for plotting).

**"shaped"** — a coil with a polygonal cross-section. The cross-section
is triangulated and Gaussian quadrature is used to distribute the
current density across the area, providing a more accurate
representation for coils with complex shapes.

.. code-block:: json

   {
     "type": "shaped",
     "R": [0.90, 0.96, 0.96, 0.90],
     "Z": [1.08, 1.08, 1.16, 1.16],
     "turns": 42,
     "current": 3469.63
   }

* ``"R"``, ``"Z"`` — lists of vertex coordinates defining the polygon
  outline. Must have more than two points.

**"solenoid"** — a central solenoid with no radial thickness,
represented by a series of point sources spread evenly along the
vertical extent.

.. code-block:: json

   {
     "type": "solenoid",
     "R": 0.175,
     "Z_min": -1.50,
     "Z_max": 1.50,
     "turns": 648,
     "current": 0.0
   }

* ``"R"`` — scalar radial position of the solenoid.
* ``"Z_min"``, ``"Z_max"`` — the bottom and top of the solenoid.


Worked Example
--------------

The example below is based on **Example 05** from the FORGE examples
directory. It demonstrates all the major FORGE features on a MAST-U
equilibrium:

* A **multi-segment strike surface** spanning the lower outer divertor.
* **Wall buffers** that penalise field lines grazing the first wall.
* **X-Point Target (XPT) divertor configuration** — encourages
  formation of a secondary X-point, promoting an XPT topology.

.. code-block:: python

   import logging
   import os

   logging.basicConfig(
       level=logging.INFO,
       format="%(name)s - %(levelname)s - %(message)s",
   )

   from forge.io import read_geqdsk, read_magnets
   from forge.machine import Machine
   from forge.equilibrium import Equilibrium
   from forge.optimise import Optimiser
   from forge.utils import closest_point_along_shape
   from forge._paths import data_dir_path

   # --- 1. Load input data ---
   data_dir = data_dir_path()
   path_to_geqdsk = os.path.join(data_dir, "geqdsk", "examples", "05-mastu.geqdsk")
   path_to_magnets = os.path.join(data_dir, "json", "examples", "05-mastu.json")

   eq_data = read_geqdsk(path_to_geqdsk)
   magnets_data, circuits, suggested_circuits = read_magnets(path_to_magnets)

   # --- 2. Build the machine ---
   tokamak = Machine(
       magnets_data=magnets_data,
       wall_R=eq_data["wall_R"],
       wall_Z=eq_data["wall_Z"],
       circuits=circuits,
   )

   # --- 3. Create the equilibrium ---
   eq = Equilibrium(
       eq_data=eq_data,
       tokamak=tokamak,
       calculate_flux_from_coils=True,
   )

   eq.plot_fluxes()

   # --- 4. Define constraints ---
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

   # --- 5. Define the strike surface ---
   surface_points_approx = [
       (1.027, -1.573),
       (1.563, -1.573),
       (1.733, -1.687),
       (1.349, -2.068),
       (1.086, -2.073),
       (0.907, -1.889),
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

   # --- 6. Configure and run the optimiser ---
   opt = Optimiser(
       eq=eq,
       tokamak_initial=tokamak,
       divertor_data=divertor_data,
       constraints=constraints,
       max_evals=100_000,
       initial_temperature=10.0,
       max_cooling_factor=0.995,
       min_cooling_factor=0.99,
       threshold_acceptance_rate_decay=2.0,
       initial_threshold_acceptance_rate=0.95,
       n_window=50,
       current_step_size_factor=0.005,
       initial_total_connection_length_cost=0.0,
       initial_total_strike_point_distance_cost=2.0,
       initial_xpoint_regions_cost=1.0,
       initial_coil_currents_cost=0.0,
       field_line_trace_step_size=0.05,
       field_line_trace_max_steps=1000,
       field_line_trace_psi_tollerance=0.001,
       use_buffers=True,
       buffers={"lower_outer": [{"R": [...], "Z": [...], "distance": 0.03}]},
       use_xpoint_regions=True,
       xpoint_regions={"lower_outer": {"R": [...], "Z": [...]}},
       detailed_logging=True,
   )

   opt.optimise()

The rest of this page walks through each section of the script.


Step 1: Loading Input Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   data_dir = data_dir_path()
   path_to_geqdsk = os.path.join(data_dir, "geqdsk", "examples", "05-mastu.geqdsk")
   path_to_magnets = os.path.join(data_dir, "json", "examples", "05-mastu.json")

   eq_data = read_geqdsk(path_to_geqdsk)
   magnets_data, circuits, suggested_circuits = read_magnets(path_to_magnets)

:func:`~forge._paths.data_dir_path` returns the path to FORGE's bundled
data directory, which includes example GEQDSK and JSON files for MAST-U.

:func:`~forge.io.read_geqdsk` reads the GEQDSK file and returns a
dictionary (``eq_data``) containing the 2D poloidal flux array, plasma
profiles, wall coordinates, scalar quantities, and grid parameters.

:func:`~forge.io.read_magnets` reads the machine JSON file and returns
three values:

* ``magnets_data`` — a dictionary of PF coil definitions (positions,
  turns, currents, filament sizes).
* ``circuits`` — a dictionary of circuit definitions (coil groupings
  and current multipliers), or ``None`` if the JSON file does not define
  circuits.
* ``suggested_circuits`` — a dictionary of automatically inferred
  circuit groupings based on coil naming conventions (e.g. ``"P1U"``
  and ``"P1L"`` are grouped into a ``"P1"`` circuit).


Step 2: Building the Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   tokamak = Machine(
       magnets_data=magnets_data,
       wall_R=eq_data["wall_R"],
       wall_Z=eq_data["wall_Z"],
       circuits=circuits,
   )

The :class:`~forge.machine.Machine` constructor takes:

* ``magnets_data`` — the PF coil dictionary from
  :func:`~forge.io.read_magnets`.
* ``wall_R``, ``wall_Z`` — the first-wall polygon coordinates. These
  are taken from the GEQDSK data since the wall is defined there.
* ``circuits`` — the circuit wiring dictionary. If ``None``, each coil
  is treated as independently controllable. If circuits are provided,
  coils within the same circuit are adjusted together according to their
  multipliers.


Step 3: Creating the Equilibrium
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   eq = Equilibrium(
       eq_data=eq_data,
       tokamak=tokamak,
       calculate_flux_from_coils=True,
   )

   eq.plot_fluxes()

The :class:`~forge.equilibrium.Equilibrium` constructor takes:

* ``eq_data`` — the dictionary returned by
  :func:`~forge.io.read_geqdsk`.
* ``tokamak`` — the :class:`~forge.machine.Machine` object.
* ``calculate_flux_from_coils`` — controls how the equilibrium
  decomposition is performed:

  - ``True`` — compute :math:`\psi_{\text{machine}}` from the coil
    currents and Green's functions, then derive
    :math:`\psi_{\text{plasma}} = \psi_{\text{total}} - \psi_{\text{machine}}`.
    Use this when accurate coil currents are available.
  - ``False`` — compute :math:`\psi_{\text{plasma}}` from the toroidal
    current density on the grid, then derive
    :math:`\psi_{\text{machine}} = \psi_{\text{total}} - \psi_{\text{plasma}}`.
    Use this when coil currents are unknown or unreliable.

The constructor automatically pre-computes Green's functions (and their
first and second derivatives) for every coil at every grid point,
locates the critical points, and performs the flux decomposition.

``eq.plot_fluxes()`` displays a diagnostic plot of the total, plasma,
and machine flux contributions, which is useful for verifying the
decomposition.


Step 4: Defining Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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
       "tikhonov": { ... },
   }

The ``constraints`` dictionary controls how the core plasma geometry is
preserved. It has two keys:

**"annealing"** — constraints enforced during the simulated annealing
optimisation. These define the null-space within which coil currents are
perturbed (see :doc:`how_it_works`). The available options are:

* ``constrain_omp`` / ``constrain_imp`` — pin the machine flux at the
  outboard and inboard midplane separatrix crossings.
* ``constrain_upper_point`` / ``constrain_lower_point`` — pin the
  machine flux at the highest/lowest point of the separatrix.
* ``constrain_<quadrant>`` and ``N_constraints_<quadrant>`` — pin the
  machine flux at evenly spaced points along the separatrix in the
  specified quadrant (upper-right, upper-left, lower-left,
  lower-right). The ``N_constraints`` value controls how many points
  per quadrant.
* ``additional_divertor_constraint_points`` — a list of extra
  :math:`(R,Z)` points in the divertor where flux should be
  constrained. These are useful for shaping specific divertor features.
* ``additional_divertor_xpoints`` — a list of target X-point
  :math:`(R,Z)` locations, adding :math:`B_R = 0` and :math:`B_Z = 0`
  constraints at those locations.
* ``xpoint_constraint`` — which X-point is being constrained:
  ``"lower"``, ``"upper"``, or ``"both"``.

**"tikhonov"** — constraints for the optional Tikhonov regularisation
step that estimates coil currents. This accepts the same options as
``"annealing"`` plus an ``"exclude_coils"`` key (a list of coil names
to exclude from the Tikhonov solve). Tikhonov constraints are only
needed if the machine used for optimisation differs from the one that
produced the initial equilibrium, or if you want to re-estimate the
initial currents.

In this example, the annealing constraints fix the outboard midplane,
inboard midplane, upper point, and one separatrix point in each of the
lower-left and lower-right quadrants. The upper quadrants are left
unconstrained because the divertor being optimised is in the lower half
of the machine. The lower point is also unconstrained to allow the lower
separatrix shape to change with the divertor geometry.


Step 5: Defining the Strike Surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   surface_points_approx = [
       (1.027, -1.573),
       (1.563, -1.573),
       (1.733, -1.687),
       (1.349, -2.068),
       (1.086, -2.073),
       (0.907, -1.889),
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

The **strike surface** defines where on the divertor target plate the
optimiser should try to place the strike point.

First, approximate :math:`(R,Z)` coordinates are defined for key points
along the desired target region.
:func:`~forge.utils.closest_point_along_shape` snaps each point onto
the nearest location on the wall polygon, ensuring the strike surface
lies exactly on the wall.

Six points produce five connected line segments spanning the lower outer
divertor. If the traced
field line lands anywhere on this surface, the strike-point cost is
zero. If it misses, the cost is proportional to the distance to the
nearest point on the surface.

The ``divertor_data`` dictionary is keyed by divertor region. The
available regions are ``"lower_outer"``, ``"lower_inner"``,
``"upper_outer"``, and ``"upper_inner"``. Multiple regions can be
optimised simultaneously. Each region entry contains:

* ``strike_R``, ``strike_Z`` — the target strike geometry. A single
  :math:`(R,Z)` value defines a **strike point**; a list of values
  defines a **strike surface**.
* ``connection_length_multiplication_factor_zero`` — the multiplication
  factor above the initial connection length at which the connection
  length cost reaches zero. A value of ``2.0`` means the optimiser aims
  to at least double the connection length.


Step 6: Configuring and Running the Optimiser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   opt = Optimiser(
       eq=eq,
       tokamak_initial=tokamak,
       divertor_data=divertor_data,
       constraints=constraints,
       max_evals=100_000,
       initial_temperature=10.0,
       max_cooling_factor=0.995,
       min_cooling_factor=0.99,
       threshold_acceptance_rate_decay=2.0,
       initial_threshold_acceptance_rate=0.95,
       n_window=50,
       current_step_size_factor=0.005,
       initial_total_connection_length_cost=0.0,
       initial_total_strike_point_distance_cost=2.0,
       initial_xpoint_regions_cost=1.0,
       initial_coil_currents_cost=0.0,
       field_line_trace_step_size=0.05,
       field_line_trace_max_steps=1000,
       field_line_trace_psi_tollerance=0.001,
       use_buffers=True,
       buffers={"lower_outer": [{"R": [...], "Z": [...], "distance": 0.03}]},
       use_xpoint_regions=True,
       xpoint_regions={"lower_outer": {"R": [...], "Z": [...]}},
       detailed_logging=True,
   )

   opt.optimise()

The :class:`~forge.optimise.Optimiser` constructor takes the
equilibrium, machine, divertor data, and constraints, along with a
range of configuration parameters grouped into the following
categories:

**Annealing schedule**

* ``max_evals`` — maximum number of cost-function evaluations before
  termination.
* ``initial_temperature`` — starting pseudo-temperature :math:`T`.
  Higher values allow more exploration early on.
* ``max_cooling_factor`` / ``min_cooling_factor`` — bounds on the
  dynamic cooling factor :math:`f_{\text{cool}}`. Values close to 1.0
  cool slowly (more exploration); values further from 1.0 cool
  aggressively.
* ``threshold_acceptance_rate_decay`` — :math:`\beta`, the decay
  constant for the rolling acceptance-rate threshold. See the
  :ref:`temperature scheduling <temperature-scheduling>` section of
  :doc:`how_it_works` for the full formula.
* ``initial_threshold_acceptance_rate`` — :math:`R_0`, the initial
  threshold acceptance rate. At the start of the run cooling is
  suppressed until the rolling acceptance rate drops below this value.
* ``n_window`` — the window size used to compute the rolling acceptance
  rate.

**Step size**

* ``current_step_size_factor`` — scales the magnitude of random
  perturbations to coil currents at each iteration, expressed as a
  fraction of a typical coil current. Smaller values give finer
  exploration; larger values allow bigger jumps.

**Cost function weights**

* ``initial_total_connection_length_cost`` — weight for the connection
  length cost component. Set to ``0.0`` to disable connection-length
  optimisation entirely, or increase to prioritise longer connection
  lengths.
* ``initial_total_strike_point_distance_cost`` — weight for the
  strike-point/surface proximity cost. In this example it is set to
  ``2.0``, making it the dominant cost component.
* ``initial_xpoint_regions_cost`` — weight for the X-Point Target (XPT)
  cost component. Set to ``1.0`` to enable it, or ``0.0`` to disable.
* ``initial_coil_currents_cost`` — weight for the coil-current
  regularisation cost. Set to ``0.0`` in this example to allow the
  optimiser to explore freely.

**Field line tracing**

* ``field_line_trace_step_size`` — the poloidal step size (in metres)
  used by the RK4 field-line tracer. Smaller values increase accuracy
  but slow down each iteration. The default is ``0.05``.
* ``field_line_trace_max_steps`` — the maximum number of integration
  steps the tracer will take before abandoning a trace (e.g. if the
  field line never reaches the wall). The default is ``1000``.
* ``field_line_trace_psi_tollerance`` — the Newton-correction tolerance
  used during RK4 field line tracing to keep the trace on the target
  flux surface.

**Feature toggles**

* ``use_buffers`` — enables wall-buffer penalties. Buffers penalise
  solutions whose field lines graze wall segments that are not designed
  for high heat flux. When ``use_buffers=True``, buffer definitions
  **must** be supplied via the ``buffers`` parameter — a dictionary
  keyed by divertor region name, where each value is a list of buffer
  definition dictionaries containing the ``"R"`` and ``"Z"`` coordinates
  of the wall segment endpoints and a ``"distance"`` value:

  .. code-block:: python

     buffers = {
         "lower_outer": [
             {"R": [R1, R2], "Z": [Z1, Z2], "distance": 0.05},
             {"R": [R3, R4], "Z": [Z3, Z4], "distance": 0.03},
         ],
     }

  By keying buffers per region, the optimiser only checks buffer
  intersections relevant to the divertor region currently being traced,
  which is both more efficient and avoids false-positive hits from
  unrelated regions.

  See **Example 03** in the examples directory for a complete
  demonstration. Buffers can also be defined interactively using the
  :doc:`GUI <gui>`, or loaded from JSON with ``forge.io.load_buffers``.
* ``use_xpoint_regions`` — enables optimisation towards an X-Point
  Target (XPT) divertor configuration. When enabled, the
  ``xpoint_regions`` parameter **must** be supplied as a dictionary
  mapping each divertor region name to an ``{R, Z}`` polygon definition:

  .. code-block:: python

     xpoint_regions = {
         "lower_outer": {
             "R": [1.08, 1.34, 1.34, 1.08, 1.08],
             "Z": [-1.84, -1.84, -1.76, -1.76, -1.84],
         },
     }

  X-point regions can also be defined interactively using the
  :doc:`GUI <gui>`, where each polygon drawn on the canvas is
  automatically assigned to the currently selected divertor region.

**Logging**

* ``detailed_logging`` — enables per-iteration logging of cost
  components, acceptance rate, and temperature.

After construction, ``opt.optimise()`` runs the simulated annealing
loop. The terminal displays progress information throughout.
The best (incumbent) coil current set and its associated
cost are retained throughout.


After the Optimisation
^^^^^^^^^^^^^^^^^^^^^^

Once the optimisation completes, FORGE automatically creates an
optimised ``Equilibrium`` and ``Machine`` with the best coil currents
found. These are stored as attributes on the ``Optimiser``:

.. code-block:: python

   opt_eq = opt.optimised_eq            # forge.equilibrium.Equilibrium
   opt_tokamak = opt.optimised_tokamak  # forge.machine.Machine

To inspect the new coil currents programmatically (for example to log
them or pass them to another tool), use:

.. code-block:: python

   currents = opt_tokamak.get_currents()      # NumPy array of currents
   names    = opt_tokamak.get_coil_names()     # corresponding coil names

To save the optimised machine (coils + circuits with updated currents)
to a JSON file that can be loaded back with ``read_magnets()``:

.. code-block:: python

   from forge.io import write_magnets

   write_magnets(opt_tokamak, "optimised_machine.json")

To save the optimised equilibrium to a new GEQDSK file:

.. code-block:: python

   from forge.io import write_geqdsk

   write_geqdsk(opt_eq, "optimised_equilibrium.geqdsk")

The output GEQDSK can then be used as input for further analysis, for
other codes, or as the starting point for a subsequent FORGE
optimisation.


Checking Separatrix Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After optimisation, the coil currents have changed. If the coil
(machine) flux along the original separatrix has shifted significantly,
the plasma equilibrium would no longer be self-consistent in the core.
The :meth:`~forge.optimise.Optimiser.separatrix_coil_flux_change` method
quantifies this by evaluating the coil flux :math:`\psi_{\text{mach}}`
along the initial separatrix for both the original and optimised
equilibria:

.. code-block:: python

   flux_diag = opt.separatrix_coil_flux_change()

   print(f"Max |Δψ_mach|:  {flux_diag['max_abs_change']:.4e} Wb "
         f"({flux_diag['max_rel_change'] * 100:.4f}%)")
   print(f"Mean |Δψ_mach|: {flux_diag['mean_abs_change']:.4e} Wb "
         f"({flux_diag['mean_rel_change'] * 100:.4f}%)")

The returned dictionary contains:

* ``R_lcfs``, ``Z_lcfs`` — coordinates of the separatrix evaluation
  points.
* ``psi_mach_initial``, ``psi_mach_optimised`` — coil flux values at
  each point (initial and optimised).
* ``delta_psi_mach`` — absolute change (optimised − initial).
* ``delta_psi_mach_rel`` — relative change, normalised by the initial
  coil flux at each point.
* ``max_abs_change``, ``mean_abs_change`` — scalar summaries of
  absolute change (Wb).
* ``max_rel_change``, ``mean_rel_change`` — scalar summaries of
  relative change (dimensionless; multiply by 100 for %).

Small values confirm that the optimisation has only modified the flux
in the divertor region while preserving the core separatrix shape. If
the relative change is large, consider adding more constraints (see
Step 4) to better pin the separatrix.

.. note::

   The GUI Analysis tab displays this diagnostic automatically as a
   colour-mapped separatrix plot and as text in the incumbent summary
   pane (see :doc:`gui`).


.. note::

   The MAST-U example data files used above are bundled with FORGE
   under ``forge/data/``. See the :doc:`examples` page for the full set
   of example scripts covering progressively more features.
