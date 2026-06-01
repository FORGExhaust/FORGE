Examples
========

FORGE ships with six progressive example scripts in the ``examples/``
directory. Each script is self-contained and can be run directly:

.. code-block:: bash

   python examples/01_mastu_strike_point.py

The examples use MAST-U geometry and demonstrate increasingly advanced
features.

.. list-table::
   :header-rows: 1
   :widths: 5 40 55

   * - #
     - Script
     - Description
   * - 1
     - ``01_mastu_strike_point.py``
     - Simplest case — optimise to a single strike point.
   * - 2
     - ``02_mastu_strike_surface.py``
     - Replace the strike point with a strike surface.
   * - 3
     - ``03_mastu_strike_surface_buffers.py``
     - Add wall buffers to penalise SOL–wall interaction.
   * - 4
     - ``04_mastu_strike_surface_xpt.py``
     - Enable X-Point Target (XPT) divertor configuration.
   * - 5
     - ``05_mastu_strike_surface_buffers_xpt.py``
     - Combine buffers and X-Point Target — the full feature set.
   * - 6
     - ``06_mastu_strike_surface_tikhonov.py``
     - Demonstrate Tikhonov regularisation constraints.

Each example builds on the previous one. Start with Example 1 to
understand the basic workflow, then work through the rest to learn how
to layer in additional features.
