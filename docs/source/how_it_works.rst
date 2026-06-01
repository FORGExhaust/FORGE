How It Works
============

This page describes the principles and approach behind FORGE. It focuses
on the concepts rather than the code, so that users can understand *what*
FORGE is doing and *why* before diving into the API.


The Goal
--------

In a tokamak with a diverted configuration, the **divertor** is the region where open
magnetic field lines guide escaping plasma from the core onto
solid surfaces (the divertor targets). The geometry of the magnetic
field in this region is critical: it determines where the plasma
strikes the target, how long the field lines are between the upstream
plasma and the target, and how effectively the plasma can radiate away
its energy before reaching the surface.

Longer **parallel connection lengths** give the plasma more opportunity
to radiate energy through volumetric losses (radiation, charge exchange,
recombination) before reaching the target, promoting a regime called
**plasma detachment** in which the heat flux to the target is
dramatically reduced. Achieving detachment is essential for protecting
the divertor components from damage in a reactor.

FORGE's purpose is to **optimise the magnetic geometry of the divertor**
to promote these favourable exhaust conditions — increasing connection
length and steering the strike point to a desired location — all by
adjusting the currents in the machine's poloidal field (PF) coil set.

To do this efficiently, FORGE uses a decomposition approach that
separates the magnetic flux into plasma and machine contributions. By
holding the plasma contribution fixed and varying only the machine
(coil) contribution, FORGE can reshape the divertor magnetic geometry
while **preserving the core plasma equilibrium** — eliminating the need
to re-run a free-boundary equilibrium solver at every iteration. This
makes it practical to employ a guided optimisation strategy (simulated
annealing) in which thousands of candidate coil current sets are
evaluated against a configurable cost function that encodes the desired
divertor properties.


Starting Point: The Equilibrium
-------------------------------

FORGE begins with a **free-boundary equilibrium** — a solved
Grad-Shafranov equilibrium typically produced by a code such as FreeGS
or EFIT. This is supplied as a standard GEQDSK file and contains the
poloidal magnetic flux :math:`\psi(R,Z)` on a regular :math:`(R,Z)`
grid, along with the plasma profiles (:math:`p'`, :math:`FF'`,
pressure, :math:`q`, etc.).

Alongside the equilibrium, the user provides a **Machine definition**: a
description of the PF coil set (positions, sizes, types, turns, initial
currents) and the first wall geometry. The Machine may also define
**Circuits** — groups of coils wired together with configurable current
multipliers, mirroring the physical wiring of real tokamak PF systems.

From these two inputs, FORGE reconstructs the magnetic field and
identifies the key topological features: the magnetic axis, the
X-points, and the separatrix.


Critical Points and the Separatrix
-----------------------------------

FORGE automatically locates the **critical points** of the equilibrium:

* **O-points** — local extrema of :math:`\psi` where the poloidal
  magnetic field vanishes (:math:`B_{\text{pol}} = 0`) and the Hessian
  indicates a local minimum or maximum. The primary O-point is the
  **magnetic axis**.
* **X-points** — saddle points where :math:`B_{\text{pol}} = 0` and
  the Hessian indicates a saddle. The primary X-point defines the
  **last closed flux surface (LCFS)**, also known as the
  **separatrix**.

The separatrix divides closed (core) flux surfaces from open field
lines in the **scrape-off layer (SOL)**. It is the boundary between the
confined core plasma and the divertor, and its shape and position are
central to everything FORGE does.

FORGE classifies the magnetic configuration as **Lower Single Null
Diverted (LSND)**, **Upper Single Null Diverted (USND)**, or **Double
Null Diverted (DND)** based on which X-point is primary and the relative
flux at each X-point.


Equilibrium Decomposition
--------------------------

With the critical points identified, FORGE decomposes the total
poloidal magnetic flux into two independent contributions:

.. math::

   \psi_{\text{total}}(R,Z) \;=\; \psi_{\text{plasma}}(R,Z)
                                  \;+\; \psi_{\text{machine}}(R,Z)

* :math:`\psi_{\text{plasma}}` — the flux produced by the plasma's
  internal toroidal current distribution, derived from the :math:`p'`
  and :math:`FF'` profiles. This is treated as a **fixed background**
  throughout the optimisation: the core plasma pressure, safety
  factor, and current distribution are kept unaltered.
* :math:`\psi_{\text{machine}}` — the flux produced by the external
  PF coil set. This is the part FORGE adjusts.

Two modes are available for performing this decomposition:

1. **Calculate flux from coils first** — use the known coil currents
   and Green's functions (see below) to compute
   :math:`\psi_{\text{machine}}` at every grid point, then obtain
   :math:`\psi_{\text{plasma}} = \psi_{\text{total}} - \psi_{\text{machine}}`.
2. **Calculate flux from plasma first** — reconstruct
   :math:`\psi_{\text{plasma}}` from the toroidal current density at
   each grid point using Green's functions, then obtain
   :math:`\psi_{\text{machine}} = \psi_{\text{total}} - \psi_{\text{plasma}}`.
   This is more robust when accurate coil currents are not available.

The decomposition alone is not sufficient. Changing the coil currents
alters :math:`\psi_{\text{machine}}` everywhere — including the core
plasma region, where it would distort the separatrix, shift the
X-points, and change the magnetic axis. FORGE therefore also employs a
**constrained optimisation approach** that keeps
:math:`\psi_{\text{machine}}` fixed in the core plasma region while
allowing it to change freely in the divertor region (see
`Navigating Coil Current Space`_ below). Together, the decomposition
and the core constraints allow FORGE to reshape the divertor magnetic
geometry without re-running a full equilibrium solver and without
altering the core plasma geometry.


Green's Functions
-----------------

The connection between coil currents and the magnetic flux they produce
is provided by **Green's functions**. A Green's function gives the
poloidal magnetic flux at any point :math:`(R,Z)` produced by a unit
current filament at a coil location :math:`(R_c, Z_c)`. They are
expressed in terms of complete elliptic integrals :math:`K(k^2)` and
:math:`E(k^2)`.

The key property is **linearity**: the flux scales directly with the
current. If a coil carries current :math:`I`, its flux contribution is
simply :math:`G \times I`. FORGE pre-computes Green's function values
for every coil at every point on the :math:`(R,Z)` grid, so that
updating the flux for a new set of coil currents reduces to a single
matrix–vector multiplication:

.. math::

   \psi_{\text{machine}}(R,Z) \;=\; \sum_i I_i \cdot G_i(R,Z)

This is what makes FORGE fast: rather than re-solving the equilibrium
for each trial set of coil currents, the new machine flux is obtained
by a simple weighted sum of pre-computed arrays.

Analytical first and second derivatives of the Green's functions with
respect to :math:`R` and :math:`Z` are also pre-computed. These provide
the magnetic field components (:math:`B_R`, :math:`B_Z`) and their
derivatives, which are used in the field line tracer and in the
detection of critical points.


Divertor Regions
----------------

FORGE can optimise up to four divertor regions simultaneously: **lower
outer**, **lower inner**, **upper outer**, and **upper inner**. Each
region is defined relative to its nearest X-point.

For each region, the user specifies what "good" looks like:

* A **target strike geometry** — where the field line should hit the
  wall (a single strike point, or a strike surface along the target
  plate).
* A **connection length target** — how much longer the connection
  length should be relative to its starting value.
* **Cost weights** — how much each cost component matters relative to
  the others for this region.
* Optionally, an **XPT region** — an area in which the optimiser should try to form a secondary X-point, promoting an X-Point Target (XPT) divertor configuration.

With the goal and setup defined, FORGE then applies a simulated
annealing optimisation procedure to find the coil currents that best satisfy
these targets.


Simulated Annealing
-------------------

FORGE uses `simulated annealing <https://en.wikipedia.org/wiki/Simulated_annealing>`_ — a probabilistic optimisation
technique inspired by the physical process of annealing metals.
The idea is intuitive:

1. Start at a high pseudo-temperature :math:`T`, allowing many
   "uphill" moves (worse solutions accepted) to explore the solution
   space broadly and avoid getting trapped in poor local minima.
2. Gradually cool :math:`T` according to a schedule, making uphill
   moves increasingly unlikely.
3. Eventually settle near an optimum, analogous to a metal
   crystallising into its lowest-energy state.

Acceptance criterion (Metropolis-Hastings)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At each iteration, FORGE perturbs the coil currents about the **most
recently accepted state** and evaluates the resulting cost. It then
decides whether to accept or reject the new state:

* If the cost decreased (:math:`\Delta C < 0`): **always accept**.
* If the cost increased (:math:`\Delta C \geq 0`): accept with
  probability

  .. math::

     P = \exp\!\left(-\alpha \cdot \frac{\Delta C}{T}\right)

  where :math:`\alpha` is a sensitivity parameter. FORGE automatically
  tunes :math:`\alpha` at the start of a run to achieve a target
  initial acceptance rate, then freezes it once cooling begins.

In addition to the current accepted state, FORGE tracks a **running
best (incumbent) state** — the lowest-cost solution found at any point
during the run. The optimisation returns this incumbent as the final
result, so even if the annealer moves away from the best solution
during exploration, it is never lost.

.. _temperature-scheduling:

Temperature scheduling
^^^^^^^^^^^^^^^^^^^^^^

Rather than following a fixed cooling schedule, FORGE uses **dynamic
temperature scheduling**. The system monitors the rolling acceptance
rate over a window of uphill evaluations:

* When the acceptance rate drops below a threshold, the system cools:
  :math:`T \leftarrow T \times \beta`.
* The cooling factor :math:`\beta` is itself dynamic — if
  many iterations were needed before the threshold was reached (the
  system is still exploring), cooling is slow; if few iterations were
  needed, cooling is more aggressive.
* The **threshold acceptance rate** itself decays as the temperature
  falls, ensuring progressively tighter convergence. It evolves as:

  .. math::

     R_{\text{threshold}} = R_0 \, \exp\!\left(
       -\lambda \left(1 - \frac{T}{T_0}\right)
     \right)

  where :math:`R_0` is the initial threshold acceptance rate,
  :math:`T_0` is the initial temperature, :math:`T` is the current
  temperature, and :math:`\lambda` is a decay constant that controls
  how aggressively the threshold drops. A larger :math:`\lambda` causes
  the threshold to drop more quickly, making the optimiser more
  exploitative earlier; a smaller :math:`\lambda` keeps the threshold
  high for longer, encouraging more exploration.

  At the start of the run (:math:`T = T_0`), the exponent is zero and
  :math:`R_{\text{threshold}} = R_0`. As :math:`T` cools towards
  zero, the threshold decays towards zero as well, so that eventually
  almost no uphill moves are accepted.

The optimisation terminates when the maximum number of evaluations is
reached, the temperature drops below a minimum, or the cost falls below
a fraction of the initial cost.


Navigating Coil Current Space
-----------------------------

A naive approach would perturb coil currents arbitrarily. This would
be inefficient, because most random perturbations would destroy the
core plasma topology — moving the X-point, distorting the separatrix,
or changing the magnetic axis position.

Instead, FORGE constructs a **constraint matrix** that encodes the
conditions the core topology must satisfy. These constraints are imposed
on the magnetic field and flux at specific locations:

* **X-point constraints** — :math:`B_R`, :math:`B_Z`, and
  :math:`\psi` from the machine at the X-point must remain fixed
  (preserving its position and the separatrix flux value).
* **Separatrix shape constraints** — the machine flux at points along
  the separatrix in each quadrant, as well as at the outboard and
  inboard midplane (preserving the shape of the LCFS).

The **null space** of this constraint matrix is then computed. Random
perturbations are generated *within this null space*, which guarantees
that every proposed coil current change **exactly satisfies all
constraints** by construction. The optimiser is free to explore
divertor geometry changes without ever disturbing the core.


What Gets Evaluated: The Cost Function
---------------------------------------

Each proposed set of coil currents produces a new equilibrium, from
which FORGE traces field lines in each divertor region and evaluates
a total cost made up of several components:

.. math::

   C = C_{\text{strike}} + C_{L_\parallel}
     + C_{\text{coils}} + C_{\text{xpt-region}}

Strike point / strike surface cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Penalises the distance between where the field line actually hits
the wall (the **strike point**) and the user's desired **strike
geometry**:

* **Strike point** — a single :math:`(R,Z)` target location. The
  cost is proportional to the distance between the actual and target
  strike points.
* **Strike surface** — a line segment along the divertor target
  plate. The cost is the distance to the nearest point on this line;
  if the field line lands on the surface, the cost is zero.

Connection length cost
^^^^^^^^^^^^^^^^^^^^^^

Rewards **increasing the parallel connection length**
:math:`L_\parallel` in the divertor. As described above, longer
connection lengths promote detachment by giving the plasma more
distance over which to radiate energy before reaching the target.

The cost decreases linearly as connection length increases, reaching
zero at a user-specified multiplication factor above the initial value.

.. note::

   In a future release, FORGE will support optimising for more
   comprehensive terms related to the magnetic geometry of the divertor,
   beyond connection length alone. One such term is the **detachment
   threshold** from the DLS model, which depends on connection length,
   total flux expansion :math:`B_X / B_t`, and the divertor-averaged
   magnetic field :math:`\langle B / B_X \rangle` (see equation 11 of
   `Cowley et al. 2022 <https://doi.org/10.1088/1741-4326/ac7a4c>`_).
   Incorporating such terms would allow FORGE to optimise directly for
   detachment access rather than using connection length as a proxy.

Coil current cost
^^^^^^^^^^^^^^^^^

Regularises coil currents to prevent unphysical solutions. The cost
is based on the sum of squared currents :math:`\sum I_i^2`, which is
physically meaningful since inter-coil forces scale as :math:`I^2`.
A "null zone" exists around the initial current level within which no
penalty is applied, so the optimiser is not penalised for modest
adjustments.

X-Point Target region cost
^^^^^^^^^^^^^^^^^^^^^^^^^^

Encourages the formation of **secondary divertor X-points** (for
X-Point Target topologies). This cost has three
sub-components:

1. **Poloidal field minimisation** — rewards lowering the minimum
   :math:`B_p^2` inside the target region (a necessary condition for
   X-point formation).
2. **X-point presence** — a binary reward for an X-point existing
   within the region.
3. **Magnetic connection** — rewards proximity of the secondary
   X-point's flux surface to the separatrix, penalising
   "magnetically disconnected" X-points.


Field Line Tracing
------------------

To evaluate the strike point and connection length, FORGE must trace
field lines from the X-point downstream to the wall. This is done in
the poloidal :math:`(R,Z)` plane using **4th-order Runge-Kutta (RK4)**
integration. At each point, the poloidal field direction is computed
from the flux derivatives:

.. math::

   B_R = -\frac{1}{2\pi R}\frac{\partial\psi}{\partial Z}, \qquad
   B_Z = \frac{1}{2\pi R}\frac{\partial\psi}{\partial R}

and the trace steps along the unit poloidal-field direction.

After each RK4 step, a **gradient-projection correction** (a single
Newton step) projects the point back onto the target flux surface:

.. math::

   \mathbf{x}_{\text{corrected}} = \mathbf{x}
     - \frac{\psi(\mathbf{x}) - \psi_{\text{target}}}{|\nabla\psi|^2}
       \,\nabla\psi

This greatly reduces accumulated numerical drift, keeping the trace
accurately on its flux surface regardless of step size.

The trace terminates when the field line intersects the wall (or a
buffer boundary — see below). The poloidal step size and maximum
number of steps are controlled by the ``field_line_trace_step_size``
and ``field_line_trace_max_steps`` parameters, respectively.


Parallel Connection Length
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a poloidal field line has been traced, the full three-dimensional
parallel connection length is computed:

.. math::

   L_\parallel = \int \sqrt{1 + \frac{B_\phi^2}{B_R^2 + B_Z^2}}
                 \; ds_{\text{pol}}

where :math:`B_\phi = F_{\text{vac}} / R` is the toroidal field and
:math:`ds_{\text{pol}}` is the elemental poloidal arc length. The
factor under the square root accounts for the helical winding of the
field line around the torus — the actual 3D path is longer than its
poloidal projection.


Buffer Regions
--------------

In addition to targeting a specific strike point, it is often important
to keep the SOL field lines away from sensitive parts of the wall that
are not designed for high heat flux.

**Buffers** are offset regions around user-selected segments of the
wall, created by buffering those segments outward by a specified
distance. If a field line intersects a buffer before reaching the main
target, the connection length cost is multiplied by a penalty factor,
nudging the optimiser away from solutions that direct plasma toward
vulnerable surfaces.

Buffers can be defined in Python scripts via JSON data (see
:doc:`getting_started`) or interactively using the :doc:`GUI <gui>`.


Tikhonov Regularisation
-----------------------

Sometimes the PF coil set available for optimisation is different from
the one that produced the initial equilibrium, or the initial coil
currents are simply not known. In these cases, FORGE can estimate
suitable starting currents using **Tikhonov regularisation**:

.. math::

   \mathbf{I} = \left(
     \mathbf{G}^T\mathbf{G} + \lambda\,\mathbf{I}_{\text{identity}}
   \right)^{-1} \mathbf{G}^T \mathbf{b}

where :math:`\mathbf{G}` is the Green's function constraint matrix,
:math:`\mathbf{b}` is the constraint vector, and :math:`\lambda` is a
regularisation parameter that prevents ill-conditioning. FORGE
auto-tunes :math:`\lambda` by starting large and decaying it until
the constraint residuals fall below a threshold.

The Tikhonov constraint set can be larger than the annealing constraint
set (the null-space approach requires
:math:`N_{\text{coils}} - N_{\text{constraints}} \geq 2`, but Tikhonov
has no such restriction).
