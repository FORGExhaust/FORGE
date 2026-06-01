Installation
============

Requirements
------------

* Python >= 3.10.12
* NumPy >= 1.26.3
* Matplotlib >= 3.7.5
* SciPy >= 1.11.4
* Shapely >= 2.0.1
* FreeQDSK >= 0.4.0

Install from source
-------------------

In the near future, FORGE will be available on PyPI for easy installation via pip. In the meantime, you
can install it directly from the source code:

.. code-block:: bash

   git clone https://github.com/FORGExhaust/FORGE.git
   cd forge
   pip install .

For an editable (development) install:

.. code-block:: bash

   pip install -e .

GUI (optional)
--------------

FORGE includes an optional graphical user interface built with
`Panel <https://panel.holoviz.org/>`_ and
`Bokeh <https://bokeh.org/>`_. To install the GUI dependencies:

.. code-block:: bash

   pip install ".[gui]"

Or, for an editable install with GUI support:

.. code-block:: bash

   pip install -e ".[gui]"

Once installed, launch the GUI with:

.. code-block:: bash

   forge-gui

See the :doc:`gui` page for full usage instructions.

Running the tests
-----------------

.. code-block:: bash

   pip install ".[test]"
   pytest

Building the documentation locally
-----------------------------------

.. code-block:: bash

   pip install ".[docs]"
   cd docs
   make html

Then open ``build/html/index.html`` in your browser.
