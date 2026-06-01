FORGE Data Directory
====================

This directory contains reference data files used by FORGE and its example
scripts.  It is organised into subdirectories by file type.


geqdsk/
-------
GEQDSK equilibrium files (G-EQDSK format).

  examples/
    01-mastu.geqdsk  –  MAST-U equilibrium used by example script 01.
    02-mastu.geqdsk  –  MAST-U equilibrium used by example script 02.
    03-mastu.geqdsk  –  MAST-U equilibrium used by example script 03.
    04-mastu.geqdsk  –  MAST-U equilibrium used by example script 04.
    05-mastu.geqdsk  –  MAST-U equilibrium used by example script 05.
    06-mastu.geqdsk  –  MAST-U equilibrium used by example script 06.


json/
-----
JSON data files.

  MAST-U/
    Raw MAST-U reference data extracted from the UKAEA open data set
    (see the copyright notice in that directory).  These files are are
    NOT directly compatible with FORGE's ``read_magnets()`` / ``Machine()`` interface.
    The example scripts use the pre-processed files in ``json/examples/`` instead.

    Note: The MAST-U machine configuration represented by these data
    files corresponds to the MAST-U device as operated during the period
    2020–2025. Any subsequent modifications to the machine are not
    reflected in these data.

    MAST-U_active_coils.json        –  MAST-U PF coil definitions (positions,
                                       sizes, turns, circuits) in the
                                       upstream UKAEA format.
    MAST-U_active_coils_masks.json  –  Current masks for each MAST-U coil.
                                       Produced by FORGE (not part of the
                                       UKAEA data set).
    MAST-U_wall.json                –  MAST-U first-wall polygon (R, Z) in
                                       the upstream UKAEA format.
    MAST-U_UKAEA_open_data_copyright_notice.txt
                                    –  Copyright notice for the MAST-U data
                                       (UKAEA open data licence).

  examples/
    01-mastu.json  –  Coil & circuit definitions for example script 01.
    02-mastu.json  –  Coil & circuit definitions for example script 02.
    03-mastu.json  –  Coil & circuit definitions for example script 03.
    04-mastu.json  –  Coil & circuit definitions for example script 04.
    05-mastu.json  –  Coil & circuit definitions for example script 05.
    06-mastu.json  –  Coil & circuit definitions for example script 06.


images/
-------
Static images used in documentation and the GUI.

  FORGE_logo.svg             –  FORGE project logo.
  FORGE_GUI.png              –  Screenshot of the FORGE GUI.
  mastu_initial_opt.svg      –  Initial vs optimised equilibrium comparison plot.


pickle/
-------
Pre-serialised Python objects (pickle format).

  MAST-U/
    Raw MAST-U reference data in pickle format, corresponding to the
    original UKAEA open data release.  These are NOT directly compatible with
    FORGE's ``Machine()`` interface.  Use the ``json/examples/`` files
    for running FORGE.

    MAST-U_active_coils.pickle  –  MAST-U coil data in the upstream
                                   UKAEA pickle format.
    MAST-U_wall.pickle          –  MAST-U first-wall polygon in the
                                   upstream UKAEA pickle format.
    MAST-U_UKAEA_open_data_copyright_notice.txt
                                –  Copyright notice for the MAST-U data
                                    (UKAEA open data licence).
