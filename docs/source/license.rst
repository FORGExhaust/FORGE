License & Acknowledgements
==========================

License
-------

FORGE is released under the
`GNU Lesser General Public License v3.0 (LGPL-3.0-or-later)
<https://www.gnu.org/licenses/lgpl-3.0.html>`_.

A copy of the license is included in the repository as ``LICENSE``.


FreeGS Attribution
------------------

Several modules in FORGE are derived from or inspired by
`FreeGS <https://github.com/freegs-plasma/freegs>`_, an open-source
Grad-Shafranov equilibrium solver also released under LGPL-3.0.

The following modules contain code originally authored by the FreeGS
contributors (primarily Ben Dudson, University of York):

* ``forge.greens`` — the ``Greens`` function is based on the equivalent
  in gradshafranov.py from FreeGS; all other functions are original to FORGE.
* ``forge.magnets`` — Coil, ShapedCoil, Solenoid and Circuit classes are
  based on the FreeGS equivalents.
* ``forge.quadrature`` — quadrature rules based on the FreeGS quadrature
  module, modified for FORGE.
* ``forge.critical`` — critical-point finding routines based on critical.py
  from FreeGS, modified for FORGE.
* ``forge.machine`` — inspired by the Machine class in FreeGS, but
  substantially rewritten for FORGE.

Each of these files retains the original FreeGS copyright notice
alongside the FORGE copyright, as required by the LGPL.


Bundled Data
------------

FORGE ships with MAST-U coil and wall geometry data for use in the
examples and tests. These data are derived from open-source data
published by UKAEA and are licensed separately from the FORGE source
code:

* **License:** `Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)
  <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_
* **Citation:** Harrison, J. (2026). *MAST-U freegs coil configuration*
  (Version 1.0) [Data set]. UKAEA.
  https://doi.org/10.14468/JMPP-4C57

The NonCommercial clause means this data may not be used for commercial
purposes. The data is provided for example and testing purposes only and
is not required for FORGE to function — users may substitute their own
machine data. A full copyright notice is included alongside the data
files at ``src/forge/data/json/MAST-U/MAST-U_UKAEA_open_data_copyright_notice.txt``.

.. note::

   The MAST-U machine configuration represented by the bundled data
   corresponds to the MAST-U device as operated during the period
   2020–2025. Any subsequent modifications to the machine are not
   reflected in these data.


Authors and Contributors
------------------------

* **Chris Marsden** — Project lead and primary developer.
* **Sebastien Shaw** — Summer student who developed the foundational
  prototype during his FOSTER placement (2025).
* **Nathan Welch** — `SCOPE <https://arxiv.org/pdf/2512.16546>`_ project
  lead, whose guidance on simulated annealing underpins the optimisation
  approach.
* **Ben Dudson & FreeGS contributors** — Original authors of the
  modules from which FORGE borrows.


GUI Framework
-------------

The FORGE GUI is built with `Panel <https://panel.holoviz.org/>`_ and
`Bokeh <https://bokeh.org/>`_.

.. image:: https://panel.holoviz.org/_static/logo_horizontal_light_theme.png
   :alt: Panel
   :height: 80px
   :target: https://panel.holoviz.org/

|

.. image:: https://static.bokeh.org/logos/logotype.svg
   :alt: Bokeh
   :height: 80px
   :target: https://bokeh.org/

The documentation and GUI sidebar reproduce a small number of toolbar-tool
icons (|icon-point-draw| |icon-poly-draw| |icon-poly-edit|) originally
shipped with `Bokeh <https://github.com/bokeh/bokeh>`_, which is released
under the `BSD 3-Clause licence
<https://github.com/bokeh/bokeh/blob/main/LICENSE.txt>`_.
These icons are redistributed here under the terms of that licence.

.. |icon-point-draw| image:: _static/icon-point-draw.png
   :height: 14px
.. |icon-poly-draw| image:: _static/icon-poly-draw.png
   :height: 14px
.. |icon-poly-edit| image:: _static/icon-poly-edit.png
   :height: 14px
