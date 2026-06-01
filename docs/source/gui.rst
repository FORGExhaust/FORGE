GUI
===

FORGE includes an optional graphical user interface (GUI) built with
`Panel <https://panel.holoviz.org/>`_ and
`Bokeh <https://bokeh.org/>`_. The GUI provides an interactive way to
load data, edit geometry, run optimisations, and analyse results —
without writing Python scripts.

Installation
------------

The GUI dependencies are optional. Install them with:

.. code-block:: bash

   pip install ".[gui]"

or, for an editable install:

.. code-block:: bash

   pip install -e ".[gui]"


Launching the GUI
-----------------

There are several ways to start the GUI:

.. code-block:: bash

   # Using the console entry point (recommended)
   forge-gui

   # Using the Python module
   python -m forge.gui

   # Using Panel serve directly (must be run from the repository root)
   cd /path/to/forge
   panel serve src/forge/gui/app.py --show

By default the GUI binds to ``localhost:5006``. To allow access from
other machines, set the ``FORGE_GUI_ADDRESS`` environment variable:

.. code-block:: bash

   FORGE_GUI_ADDRESS=0.0.0.0 forge-gui

The port can also be changed via ``FORGE_GUI_PORT``.

The GUI can also be embedded in a **Jupyter notebook**. First install the
notebook dependencies:

.. code-block:: bash

   pip install ".[notebook]"

Then, in a notebook cell:

.. code-block:: python

   import panel as pn
   pn.extension()

   from forge.gui.app import create_app
   create_app().servable()


Overview
--------

The GUI is organised into four tabs that follow the typical FORGE
workflow:

1. **Setup** — load input files and build the equilibrium.
2. **Geometry** — interactively define strike surfaces, wall buffers,
   and X-Point Target (XPT) regions.
3. **Optimise** — configure and run the optimisation with real-time
   monitoring.
4. **Analysis** — inspect results, compare coil currents, and export
   data.


Setup Tab
^^^^^^^^^

The Setup tab is the starting point. It provides:

* **File upload widgets** for the GEQDSK equilibrium file and the
  magnets JSON file.
* A **Load & Build** button that constructs the ``Machine`` and
  ``Equilibrium`` objects from the uploaded files.
* An **equilibrium information panel** showing key parameters
  (configuration type, X-point locations, plasma current, etc.).
* **Flux decomposition plots** showing the total, plasma, and coil
  contributions to the poloidal flux.
* **1D profile plots** — four small line plots showing the plasma
  profiles read from the GEQDSK file:

  - :math:`p'(\psi_N)` — pressure gradient (kPa/Wb).
  - :math:`FF'/\mu_0(\psi_N)` — poloidal current function derivative
    (kA/m).
  - :math:`p(\psi_N)` — pressure (kPa).
  - :math:`F(\psi_N)` — poloidal current function (T·m).

  These provide a quick visual check that the equilibrium profiles are
  physically reasonable before proceeding to geometry definition.
* A **coils table** listing all PF coils with their positions, turns,
  and initial currents.

Once the data is loaded, the equilibrium and machine are stored in a
shared state dictionary and passed to the other tabs.


Geometry Tab
^^^^^^^^^^^^

The Geometry tab provides an interactive Bokeh canvas for defining the
geometric inputs to the optimisation.  Select a **divertor region** from
the dropdown, then define strike geometry, buffers, and XPT regions for
that region. Switch regions to define geometry for each one
independently — other regions' geometry is shown faded on the canvas.

.. |icon_point_draw| image:: _static/icon-point-draw.png
   :height: 18px
   :class: no-scaled-link

.. |icon_poly_draw| image:: _static/icon-poly-draw.png
   :height: 18px
   :class: no-scaled-link

.. |icon_poly_edit| image:: _static/icon-poly-edit.png
   :height: 18px
   :class: no-scaled-link

**Strike Geometry** (|icon_point_draw| *Draw strike geometry* tool)
   The target location(s) on the wall where the divertor field line
   should intersect. This can be a **single strike point** (one
   :math:`(R,Z)` coordinate) or a **strike surface** (a polyline of
   two or more points defining a target region).

   To draw, select the |icon_point_draw| **Draw strike geometry** tool
   in the Bokeh toolbar and click on the canvas to place points.  A line
   connects them in order.  Points can be dragged to adjust their
   position.  Enable the *Snap to wall* checkbox to constrain points to
   the wall contour.

**Wall Buffers** (sidebar controls)
   Buffer regions penalise field lines that graze sensitive wall surfaces
   near the divertor.  Buffers are created by selecting wall segments and
   offsetting them outward by a specified distance.

   To add a buffer, tick the **Select segment** checkbox in the sidebar,
   hover over the wall to highlight segments (yellow), then click to
   select one (cyan).  Set the buffer distance and press
   **Add Buffer**.

**X-point Regions** (|icon_poly_draw| *Draw X-point region*, |icon_poly_edit| *Edit X-point region vertices*)
   Polygonal regions in which the optimiser should try to form secondary
   X-points, promoting an X-Point Target (XPT) divertor configuration.

   * |icon_poly_draw| **Draw X-point region:** Select the
     *Draw X-point region* tool, then **click and hold** to place the
     first vertex. **Click** to add more vertices. Press **Esc** or
     **click and hold** to finish.
   * |icon_poly_edit| **Edit X-point region vertices:** Select the
     *Edit X-point region vertices* tool, then **click and hold** on
     the shaded polygon area (not on a vertex) — red vertices will
     appear. Drag a vertex to move it; **click and hold** on a vertex
     to insert a new one; press **Backspace** to delete a selected
     vertex.


Optimise Tab
^^^^^^^^^^^^

The Optimise tab is the main control panel for running the
optimisation. The sidebar exposes all ``Optimiser`` parameters,
organised into sections.

**Annealing Schedule**
   Controls how the simulated annealing search explores and converges:

   * *Max evaluations* — hard limit on cost-function evaluations.
   * *Current step size factor* — fraction of current magnitude used
     for random perturbations.
   * *Initial / Min temperature* — start and floor of the SA
     temperature.
   * *Window size (n_window)* — rolling window for acceptance-rate
     calculation.
   * *Threshold acceptance decay, λ* — rate at which the acceptance
     threshold tightens.
   * *Initial threshold acceptance rate* — first target acceptance
     rate that triggers cooling.
   * *Cost termination fraction* — early-stop if incumbent cost drops
     below this fraction of the initial cost.
   * *Max / Min cooling factor, β* — bounds on the adaptive cooling
     multiplier.

**Cost Weights**
   Relative weighting of each cost-function component:

   * *Strike distance weight*
   * *Connection length weight*
   * *Coil currents weight*
   * *XPT regions weight*

**Regularisation**
   * *Initial alpha* — starting Tikhonov regularisation parameter.
   * *Alpha update factor* — multiplicative adjustment applied to
     alpha when tuning it at higher temperatures.

**Field-line Tracing**
   * *Trace step size* — the poloidal step size (in metres) used by
     the RK4 field-line tracer. Smaller values increase accuracy but
     slow down each iteration (default 0.05 m).
   * *Trace max steps* — the maximum number of integration steps the
     tracer will take before abandoning a trace. Increase this for
     very long divertor legs or small step sizes (default 1000).
   * *ψ trace tolerance* — normalised-flux tolerance for the
     field-line integrator.
   * *Buffer penalty factor* — multiplicative penalty applied when a
     field line intersects a buffer region.

**X-point Regions**
   * *Max magnetic disconnection factor* — maximum relative difference in
     ψ between a secondary divertor X-point and the separatrix,
     i.e. \|ψ\ :sub:`xpt` − ψ\ :sub:`lcfs`\| / \|ψ\ :sub:`lcfs`\|,
     at which the associated cost saturates. Relevant when forming
     X-point Target (XPT) geometries.

**Feature Toggles**
   * *Use buffers* / *Use X-point regions* — enable geometry features
     defined in the Geometry tab.
   * *Detailed logging* — record extra per-iteration data (currents,
     full flux maps, cooling factor, etc.).

**Run / Stop** and progress bar sit below the parameters. The
optimisation runs in a background thread so the GUI stays responsive;
press **Stop** at any time to trigger a graceful shutdown via the
optimiser's ``request_stop()`` method.

**Live Equilibrium Plot**
   A Bokeh figure on the right-hand side of the tab shows the current
   equilibrium, styled identically to the Setup-tab plot: grey
   background, white wall interior, grey flux contour lines, a red
   separatrix at :math:`\psi_{\mathrm{lcfs}}`, black wall outline,
   coil glyphs with orange mask overlays, and per-region traced field
   lines colour-coded to the divertor regions. The plot updates every
   1 second so the user can watch the divertor geometry evolve as coil
   currents change during the run.


Tracking Charts
^^^^^^^^^^^^^^^

Below the parameter sidebar, a set of live Bokeh line plots update via
periodic Panel callbacks (every 1 s) throughout the optimisation:

* **Cost** — total cost and the four individual components (strike
  distance, connection length, coil currents, XPT regions). Clicking a
  legend entry toggles that trace.
* **Connection Length** — per-divertor-region connection length,
  colour-coded to match the region colours used elsewhere.
* **Temperature** — SA temperature vs. iteration.
* **Acceptance Rate** — rolling probabilistic acceptance rate.
* **Alpha** — Tikhonov regularisation parameter over time.
* **Log** — a scrollable pane showing the most recent ``forge`` log
  messages.

These charts give immediate feedback on whether the optimisation is
converging, still exploring, or stuck — enabling the user to stop early
and adjust parameters before re-running.


Analysis Tab
^^^^^^^^^^^^

The Analysis tab provides post-optimisation tools:

* **Equilibrium comparison plots** — a row of four panels:

  - *Initial Equilibrium* — the starting equilibrium before
    optimisation.
  - *Incumbent Equilibrium* — the best (lowest-cost) equilibrium found
    during the optimisation.
  - *Separatrix Comparison* — initial and optimised separatrices
    overlaid to visualise shape changes.
  - *Relative Separatrix Coil Flux Change (%)* — the initial
    separatrix coloured by the magnitude of the relative change in
    coil (machine) flux, :math:`|\Delta\psi_{\text{mach}}/\psi_{\text{mach}}|\times 100`.
    A colour bar shows the percentage scale. Small values (uniform
    low colour) indicate the core equilibrium is well preserved;
    large localised values indicate regions where constraints should
    be tightened.

* **Incumbent summary pane** — key diagnostics including per-region
  connection lengths and the separatrix coil flux change metrics
  (max and mean :math:`|\Delta\psi_{\text{mach}}|` in Wb and %).
* **Cost breakdown charts** showing the final contribution of each cost
  component.
* **Connection length history** plots.
* **Coil current comparison** bar chart showing initial vs. optimised
  currents.
* **Save / Load** buttons:

  - Save the full optimiser state to a pickle file for later
    inspection or continued optimisation. The pickle is serialised
    in-memory and streamed to the browser via HTTP (with a progress
    bar), so no temporary file is written to the server's disk.
  - Load a previously saved optimiser pickle. Two options are
    provided:

    * **Load (server path)** — enter the absolute path to a pickle
      file accessible from the machine running the FORGE server.
      Python loads it directly from disk.
    * **Upload from browser** — select a pickle file from the machine
      where your browser is running. The file is uploaded via a
      streaming HTTP POST to the server, bypassing the Panel/Bokeh
      websocket to avoid browser memory limitations. Upload progress
      is printed to the server terminal. No disk space is required
      on the server — the file is held in RAM, unpickled, and then
      freed.

  - Export the optimised equilibrium as a GEQDSK file.
  - Export the optimised machine (coils and circuits with updated
    currents) as a magnets JSON file.
