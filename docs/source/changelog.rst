Changelog
=========

v1.1.0
------
* **Wall editor tab** in the GUI — interactively edit the machine wall
  outline on a temporary copy, with load/save of wall JSON and GEQDSK,
  before committing to the machine.  Strike geometry and buffers on the
  Geometry tab are cleared when a new wall is committed.
* New I/O helpers: ``read_wall``, ``write_wall``, ``wall_from_dict``, and
  ``geqdsk_dict`` (with optional wall override in ``write_geqdsk``).
* ``Machine.set_wall()`` for updating wall coordinates and derived bounds /
  Shapely geometry consistently.
* Shared GUI plotting helpers (``machine_plot``) and Bokeh toolbar icons
  (``tool_icons``).
* Documentation warnings on redundant symmetric annealing constraints
  (double-null plasma + symmetric circuits + mirrored constraint points).
* **Fix optimiser incumbent ``iteration_num`` indexing** — the displayed
  iteration label now matches the tracking-array index (0 = initial
  equilibrium, :math:`k` = :math:`k`-th simulated-annealing evaluation)
  and is only updated when a new incumbent is accepted.  Previously it
  could be off by one, which showed up in GUI plots and optimisation
  videos.
* Tests for wall I/O, ``Machine.set_wall``, and GEQDSK wall override.

v1.0.1
------
* Fixed NumPy 2.x and Matplotlib 3.8+ compatibility for GUI and optimiser.

v1.0.0
------
* Initial public version.
