"""Contains routines for reading-from and writing-to files.

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

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
from freeqdsk import geqdsk


def read_geqdsk(
    path_to_geqdsk = None
):
    """Reads data from a GEQDSK file.

    Function to read data from a GEQDSK file. Makes use of the FreeQDSK package's GEQDSK reader.
    Some additional fields not directly returned by FreeQDSK are derived, and some fields are
    ignored as they are not relevant to FORGE.

    Parameters
    ----------
    path_to_geqdsk : str
        Path to the GEQDSK file to be read.

    Returns
    -------
    eq_data : dict
        Dictionary of data extracted from the GEQDSK file.
    """
    if not isinstance(path_to_geqdsk, str):
        raise TypeError(
            "Error. path_to_geqdsk was not of the required type - str. Type passed was ",
            str(type(path_to_geqdsk))
        )

    # Load the GEQDSK
    with open(path_to_geqdsk,"r") as f:

        data = geqdsk.read(f)

    # Extract the R, Z grid
    R_min = data["rleft"]
    R_max = R_min + data["rdim"]
    nR = data["nx"]
    R_1D = np.linspace(R_min,R_max,nR,endpoint=True)
    dR = abs(R_1D[1] - R_1D[0])

    Z_mid = data["zmid"]
    nZ = data["ny"]
    Z_dim = data["zdim"]
    Z_min = Z_mid - 0.5 * Z_dim
    Z_max = Z_mid + 0.5 * Z_dim
    Z_1D = np.linspace(Z_min,Z_max,nZ,endpoint=True)
    dZ = abs(Z_1D[1] - Z_1D[0])

    # Generate the 2D R,Z grid
    R_2D, Z_2D = np.meshgrid(R_1D,Z_1D,indexing="ij")

    # Extract the wall
    wall_R = data["rlim"].tolist()
    wall_Z = data["zlim"].tolist()

    # Check if the wall is joined up
    if not (wall_R[-1] == wall_R[0] and wall_Z[-1] == wall_Z[0]):

        wall_R.append(wall_R[0])
        wall_Z.append(wall_Z[0])

    # Get the 1D psin data for the plasma profiles
    psin_data = np.linspace(0.0,1.0,nR,endpoint=True)

    # Get fvac - the toroidal field function in vacuum.
    # This is constant outside the plasma, wherein fvac = R * B_toroidal.
    # The GEQDSK stores the toroidal field at a reference point.
    fvac = data["rcentr"] * data["bcentr"]

    # Store data a dictionary
    eq_data = {
        "R_min": R_min,
        "R_max": R_max,
        "nR": nR,
        "R_1D": R_1D,
        "dR": dR,
        "R_2D": R_2D,

        "Z_min": Z_min,
        "Z_max": Z_max,
        "nZ": nZ,
        "Z_1D": Z_1D,
        "dZ": dZ,
        "Z_2D": Z_2D,

        "wall_R": wall_R,
        "wall_Z": wall_Z,

        "psi_2D": data["psi"],
        "psi_lcfs": data["sibdry"],
        "psi_axis": data["simagx"],

        "psin_data": psin_data,
        "pprime_data": data["pprime"],
        "ffprime_data": data["ffprime"],
        "q_data": data["qpsi"],
        "pressure_data": data["pres"],
        "fpol_data": data["fpol"],

        "plasma_current": data["cpasma"],

        "fvac": fvac
    }

    return eq_data

def read_magnets(
    path_to_magnets = None # Path to magnets file [str]
):
    """Reads a PF coil set from a JSON file.

    Function to read data from a JSON file containing data on the PF coil set.

    Parameters
    ----------
    path_to_magnets : str
        Path to the JSON file to be read.

    Returns
    -------
    magnets_data :dict
        Dictionary of data extracted from the JSON file [dict].
    """

    if not isinstance(path_to_magnets, str):
        raise TypeError(
            "Error. path_to_magnets was not of the required type - str. Type passed was ",
            str(type(path_to_magnets))
        )

    with open(path_to_magnets, 'r') as f:
        magnets_data = json.load(f)

    # In addition to loading the magnets data, we will identify which coils appear to have
    # upper/lower corresponding pairs based off of their names (e.g. P1U and P1L), and produce
    # a list of suggested circuits [("P1U","P1L"),("P2U","P2L")].

    # Extract the coil data
    coils_data = magnets_data["coils"]

    # Extract the circuits data - the user may not have defined any circuits, hence it may not be present.
    try:
        circuits_data = magnets_data["circuits"]
    except KeyError:
        circuits_data = None

    # Create some suggest circuits
    suggested_circuits = {}
    coil_names = list(coils_data.keys())
    indeces_checked = []

    for coil_name in coil_names:

        # Get the circuit name, which is just the coil name with the last character dropped
        circuit_name = coil_name[0:-1]

        circuit_coils_list = []

        if len(circuit_name) > 1:

            # Get names of all of the coils in this circuit
            for i in range(len(coil_names)):

                if i not in indeces_checked:

                    # Potential coil to check
                    coil_name_to_check = coil_names[i]

                    if circuit_name in coil_name_to_check:

                        circuit_coils_list.append(coil_name_to_check)
                        indeces_checked.append(i)

            if len(circuit_coils_list) > 1:

                # There were more than 1 coils containing the circuit name, hence this is a viable circuit.
                suggested_circuits[circuit_name] = circuit_coils_list

    return coils_data, circuits_data, suggested_circuits

def write_magnets(machine, path):
    """Save a Machine's coils and circuits to a JSON file.

    The output format matches the magnets JSON schema used by
    :func:`read_magnets`, so a saved file can be loaded back directly.
    Current values are read from the live coilset, reflecting any
    post-optimisation changes.

    Parameters
    ----------
    machine : forge.machine.Machine
        The machine to serialise.
    path : str or Path
        Output file path.
    """
    save_fancy_json(machine.to_dict(), path)

def write_geqdsk(
        eq,
        path_to_geqdsk,
):
    """Takes a forge.equilibrium Equilibrium object and creates a GEQDSK file.

    Parameters
    ----------
    eq : forge.equilibrium.Equilibrium object
        Equilibrium to be output as a GEQDSK.
    path_to_geqdsk : str
        Output path for the GEQDSK file.

    """

    # Define a reference radius
    rcentr = 1.0

    data = {
        'nx': eq.nR,
        'ny': eq.nZ,
        'rdim': eq.R_max - eq.R_min,
        'zdim': eq.Z_max - eq.Z_min,
        'rcentr': rcentr,
        'rleft': eq.R_min,
        'zmid': 0.5 * (eq.Z_min + eq.Z_max),
        'rmagx': eq.R_mag,
        'zmagx': eq.Z_mag,
        'simagx': eq.psi_axis,
        'sibdry': eq.psi_lcfs,
        'bcentr': eq.fvac / rcentr,
        'cpasma': eq.plasma_current,
        'fpol': eq.fpol_data,
        'pres': eq.pressure_data,
        'ffprime': eq.ffprime_data,
        'pprime': eq.pprime_data,
        'psi': eq.psi_2D,
        'qpsi': eq.q_data,
        'nbdry': len(eq.R_lcfs),
        'nlim': len(eq.wall_R),
        'rbdry': eq.R_lcfs,
        'zbdry': eq.Z_lcfs,
        'rlim': eq.wall_R,
        'zlim': eq.wall_Z,
    }

    time = int(0)

    with open(path_to_geqdsk,"w+") as f:

        geqdsk.write(data,f,"FORGE",0,time)

    f.close()

def fancy_json_string(data: Any, indent: int = 2) -> str:
    """Serialize *data* to a JSON string with compact arrays and pretty dicts.

    Produces JSON where:
      - Dictionary entries split across new lines according to `indent`.
      - Lists/tuples/NumPy arrays kept on a single line (compact).
      - NumPy scalars converted to Python scalars.
      - Dicts that appear inside lists serialized in a compact (minified) form so the parent list
        remains single-line.

    Parameters
    ----------
    data : Any
        The (typically dict) object to serialize.
    indent : int
        Number of spaces used for indenting dictionary entries.

    Returns
    -------
    str
        The formatted JSON string.

    Notes
    -----
    - This function prefers valid JSON. NaN and ±Inf are converted to null.
    - Large lists will be written on a single (long) line by design.
    - Key order follows the insertion order of the input dict (Python 3.7+).

    Examples
    --------
    >>> d = {
    ...     "meta": {"shot": 12345, "machine": "MAST-U"},
    ...     "channels": ["A", "B", "C"],
    ...     "numbers": [1, 2, 3.5],
    ... }
    >>> save_json_compact_arrays(d, "out.json", indent=4)
    # Result (schematic):
    # {
    #     "meta": {
    #         "shot": 12345,
    #         "machine": "MAST-U"
    #     },
    #     "channels": ["A", "B", "C"],
    #     "numbers": [1, 2, 3.5]
    # }
    """

    if not isinstance(indent, int) or indent < 0:
        raise ValueError("indent must be a non-negative integer")

    def _to_python_basic(obj: Any) -> Any:
        """Convert NumPy types to native Python types; leave others unchanged."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            # Keep valid JSON: map NaN/Inf to null
            if val != val or val in (float("inf"), float("-inf")):
                return None
            return val
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if obj is np.nan or obj is np.inf or obj is -np.inf:
            return None
        return obj

    def _encode_atomic(x: Any) -> str:
        """Encode non-container JSON atomics (str, int, float, bool, null) with no extra spaces.

        Throws TypeError if a container is passed.
        """
        x = _to_python_basic(x)
        if isinstance(x, (dict, list, tuple)):
            raise TypeError("encode_atomic called with container type")
        # ensure_ascii=False keeps unicode; separators minify; allow_nan=False enforces valid JSON
        return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def _encode(obj: Any, level: int = 0, compact: bool = False) -> str:
        """Recursive encoder.

        - Dicts:
            * pretty with indentation/newlines unless compact=True
        - Lists/Tuples:
            * always single-line
            * elements encoded compactly; dict elements inside lists are minified
        - Atomics:
            * minified via _encode_atomic
        """
        obj = _to_python_basic(obj)
        sp = " " * indent

        # Dictionaries
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            if compact:
                # Minified dict (no newlines) — used when dict appears inside a list
                parts = []
                for k, v in obj.items():
                    key = json.dumps(str(k), ensure_ascii=False, separators=(",", ":"))
                    # Inside a compact dict, encode children compactly too
                    val = _encode(v, level=0, compact=True)
                    parts.append(f"{key}:{val}")
                return "{" + ",".join(parts) + "}"
            else:
                # Pretty dict with newlines/indentation
                lines = []
                for k, v in obj.items():
                    key = json.dumps(str(k), ensure_ascii=False, separators=(",", ":"))
                    # If the child is a list/tuple, force compact=True to keep it single-line.
                    v_basic = _to_python_basic(v)
                    if isinstance(v_basic, (list, tuple)):
                        val_str = _encode(v, level + 1, compact=True)
                    else:
                        val_str = _encode(v, level + 1, compact=False)
                    lines.append(f"{sp * (level + 1)}{key}: {val_str}")
                return "{\n" + ",\n".join(lines) + "\n" + sp * level + "}"

        # Lists/Tuples — always one line
        if isinstance(obj, (list, tuple)):
            elems = []
            for el in obj:
                el_basic = _to_python_basic(el)
                if isinstance(el_basic, dict):
                    elems.append(_encode(el, level=level, compact=True))  # minified dict in list
                elif isinstance(el_basic, (list, tuple)):
                    elems.append(_encode(el, level=level, compact=True))  # nested list stays one line
                else:
                    elems.append(_encode_atomic(el))
            return "[" + ", ".join(elems) + "]"

        # Atomics
        return _encode_atomic(obj)

    text = _encode(data, level=0, compact=False)

    return text


def save_fancy_json(data: Any, out_path: Union[str, Path], indent: int = 2) -> None:
    """Write *data* to a JSON file using :func:`fancy_json_string` formatting.

    Parameters
    ----------
    data : Any
        The object to serialize (typically a dict).
    out_path : str or pathlib.Path
        Destination file path.
    indent : int
        Number of spaces for indenting dictionary entries.
    """
    text = fancy_json_string(data, indent=indent)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


# -- Attributes that cannot be pickled (prepared geometries, threads, callbacks).
_UNPICKLABLE_ATTRS = ("_prepared_wall", "_prepared_buffers", "_stop_event", "on_iteration")


def save_optimiser(optimiser, path):
    """Save an Optimiser object to a pickle file.

    A shallow copy of the object is made and any unpicklable attributes
    (Shapely prepared geometries, threading events, callbacks) are stripped
    from the copy before serialisation.  The original object is never
    modified.

    Parameters
    ----------
    optimiser : forge.optimise.Optimiser
        The optimiser instance to save.
    path : str or pathlib.Path
        Destination file path (typically ending in ``.pkl``).

    See Also
    --------
    load_optimiser : Load an Optimiser object from a pickle file.
    """

    path = Path(path)

    # Work on a shallow copy so the caller's object is untouched.
    opt_copy = copy.copy(optimiser)
    for attr in _UNPICKLABLE_ATTRS:
        if hasattr(opt_copy, attr):
            delattr(opt_copy, attr)

    with open(path, "wb") as f:
        pickle.dump(opt_copy, f)


def load_optimiser(path):
    """Load an Optimiser object from a pickle file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the pickle file to be read.

    Returns
    -------
    optimiser : forge.optimise.Optimiser
        The deserialised optimiser instance.

    See Also
    --------
    save_optimiser : Save an Optimiser object to a pickle file.
    """

    path = Path(path)

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Strike geometry I/O
# ---------------------------------------------------------------------------

def save_strike_geometry(divertor_data, path, *, buffers=None, xpoint_regions=None):
    """Save strike geometry definitions to a JSON file.

    The file uses a region-first layout::

        {
            "lower_outer": {
                "strike": {"strike_R": [...], "strike_Z": [...]},
                "buffers": [{"R": [...], "Z": [...], "distance": d}, ...],
                "xpoint_regions": {"R": [...], "Z": [...]},
                "connection_length_multiplication_factor_zero": 2.0,
                "weight_connection_length": null,
                ...
            },
            ...
        }

    Parameters
    ----------
    divertor_data : dict
        Divertor data dict as returned by
        :meth:`~forge.gui.geometry_tab.GeometryTab.build_divertor_data` or
        constructed manually in a script.  Only the user-facing keys are
        written; internal keys added by the Optimiser are stripped.
    path : str or pathlib.Path
        Destination file path (typically ending in ``.json``).
    buffers : dict or None
        Optional buffer definitions keyed by region name.
    xpoint_regions : dict or None
        Optional X-point region polygon definitions keyed by region name.
    """
    path = Path(path)

    _STRIKE_KEYS = {"strike_R", "strike_Z"}
    _SETTING_KEYS = {
        "connection_length_multiplication_factor_zero",
        "weight_connection_length",
        "weight_strike_point_distance",
        "weight_xpoint_region",
    }

    # Collect every region name mentioned in any of the inputs.
    all_regions = set(divertor_data)
    if buffers is not None:
        all_regions |= set(buffers)
    if xpoint_regions is not None:
        all_regions |= set(xpoint_regions)

    payload: dict[str, Any] = {}
    for region in sorted(all_regions):
        entry: dict[str, Any] = {}

        # Strike sub-dict
        dd = divertor_data.get(region, {})
        strike = {k: _to_serialisable(dd[k]) for k in _STRIKE_KEYS if k in dd}
        if strike:
            entry["strike"] = strike

        # Per-region settings (flat keys)
        for k in _SETTING_KEYS:
            if k in dd:
                entry[k] = _to_serialisable(dd[k])

        # Buffers
        if buffers is not None and region in buffers:
            entry["buffers"] = [
                {bk: _to_serialisable(bv) for bk, bv in buf.items()}
                for buf in buffers[region]
            ]

        # X-point regions
        if xpoint_regions is not None and region in xpoint_regions:
            entry["xpoint_regions"] = {
                pk: _to_serialisable(pv)
                for pk, pv in xpoint_regions[region].items()
            }

        payload[region] = entry

    save_fancy_json(payload, path)


def load_strike_geometry(path):
    """Load strike geometry definitions from a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file previously written by :func:`save_strike_geometry`.

    Returns
    -------
    divertor_data : dict
        Divertor data dict suitable for passing to
        :class:`~forge.optimise.Optimiser`.
    buffers : dict or None
        Buffer definitions, or *None* if the file contains none.
    xpoint_regions : dict or None
        X-point region polygon definitions, or *None* if the file contains
        none.
    """
    path = Path(path)

    with open(path, "r") as f:
        payload = json.load(f)

    _SETTING_KEYS = {
        "connection_length_multiplication_factor_zero",
        "weight_connection_length",
        "weight_strike_point_distance",
        "weight_xpoint_region",
    }

    divertor_data = {}
    buffers = {}
    xpoint_regions = {}

    for region, entry in payload.items():
        # Strike geometry
        strike = entry.get("strike", {})
        if "strike_R" in strike and "strike_Z" in strike:
            dd_entry = {"strike_R": strike["strike_R"], "strike_Z": strike["strike_Z"]}
            for k in _SETTING_KEYS:
                if k in entry:
                    dd_entry[k] = entry[k]
            divertor_data[region] = dd_entry

        # Buffers
        if "buffers" in entry:
            buffers[region] = entry["buffers"]

        # X-point regions
        if "xpoint_regions" in entry:
            xpoint_regions[region] = entry["xpoint_regions"]

    return (
        divertor_data if divertor_data else None,
        buffers if buffers else None,
        xpoint_regions if xpoint_regions else None,
    )


def _to_serialisable(obj):
    """Convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Full optimisation configuration I/O
# ---------------------------------------------------------------------------

# Keys from divertor_data that belong in the "strike" sub-dict.
_STRIKE_KEYS = {"strike_R", "strike_Z"}

# Per-region setting keys stored alongside "strike" / "buffers" / "xpoint_regions".
_REGION_SETTING_KEYS = {
    "connection_length_multiplication_factor_zero",
    "weight_connection_length",
    "weight_strike_point_distance",
    "weight_xpoint_region",
}

# Scalar optimiser parameters that map directly to Optimiser.__init__ kwargs.
_OPTIMISER_SCALAR_KEYS = [
    "max_evals",
    "current_step_size_factor",
    "estimate_initial_currents",
    "initial_temperature",
    "min_temperature",
    "threshold_acceptance_rate_decay",
    "initial_threshold_acceptance_rate",
    "n_window",
    "cost_termination_fraction",
    "field_line_trace_step_size",
    "field_line_trace_max_steps",
    "field_line_trace_psi_tollerance",
    "buffer_intersection_penalty_factor",
    "initial_total_connection_length_cost",
    "initial_total_strike_point_distance_cost",
    "initial_coil_currents_cost",
    "initial_xpoint_regions_cost",
    "use_xpoint_regions",
    "use_buffers",
    "max_magnetic_disconnection_factor",
    "initial_alpha",
    "alpha_update_factor",
    "max_cooling_factor",
    "min_cooling_factor",
    "detailed_logging",
]


def save_optimisation_config(
    path,
    *,
    divertor_data=None,
    buffers=None,
    xpoint_regions=None,
    constraints=None,
    **optimiser_kwargs,
):
    """Save a complete optimisation configuration to a JSON file.

    The file captures every setting needed to reproduce an optimisation run
    (except the equilibrium and machine data, which must be loaded
    separately from their own files).

    The JSON layout is::

        {
            "geometry": {
                "lower_outer": {
                    "strike": {"strike_R": ..., "strike_Z": ...},
                    "buffers": [...],
                    "xpoint_regions": {"R": [...], "Z": [...]},
                    <per-region settings>
                },
                ...
            },
            "constraints": { ... },
            "optimiser": {
                "max_evals": 3000,
                "initial_temperature": 10.0,
                ...
            }
        }

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file path (typically ending in ``.json``).
    divertor_data : dict or None
        Divertor data dict (strike geometry + per-region settings).
    buffers : dict or None
        Buffer definitions keyed by region name.
    xpoint_regions : dict or None
        X-point region polygons keyed by region name.
    constraints : dict or None
        Constraints dict with ``"annealing"`` and ``"tikhonov"`` sections.
    **optimiser_kwargs
        Scalar optimiser parameters (temperatures, step sizes, cost weights,
        feature toggles, etc.).  Only recognised keys from
        ``Optimiser.__init__`` are written.
    """
    path = Path(path)

    # --- geometry section (region-first layout) ---
    geometry = {}
    all_regions: set[str] = set()
    if divertor_data is not None:
        all_regions |= set(divertor_data)
    if buffers is not None:
        all_regions |= set(buffers)
    if xpoint_regions is not None:
        all_regions |= set(xpoint_regions)

    for region in sorted(all_regions):
        entry: dict[str, Any] = {}
        dd = (divertor_data or {}).get(region, {})

        strike = {k: _to_serialisable(dd[k]) for k in _STRIKE_KEYS if k in dd}
        if strike:
            entry["strike"] = strike
        for k in _REGION_SETTING_KEYS:
            if k in dd:
                entry[k] = _to_serialisable(dd[k])
        if buffers is not None and region in buffers:
            entry["buffers"] = [
                {bk: _to_serialisable(bv) for bk, bv in buf.items()}
                for buf in buffers[region]
            ]
        if xpoint_regions is not None and region in xpoint_regions:
            entry["xpoint_regions"] = {
                pk: _to_serialisable(pv)
                for pk, pv in xpoint_regions[region].items()
            }
        geometry[region] = entry

    # --- optimiser scalars ---
    opt_section = {}
    for key in _OPTIMISER_SCALAR_KEYS:
        if key in optimiser_kwargs:
            opt_section[key] = _to_serialisable(optimiser_kwargs[key])

    # --- assemble ---
    payload: dict[str, Any] = {}
    if geometry:
        payload["geometry"] = geometry
    if constraints is not None:
        payload["constraints"] = constraints
    if opt_section:
        payload["optimiser"] = opt_section

    save_fancy_json(payload, path)


def load_optimisation_config(path):
    """Load a complete optimisation configuration from a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file previously written by
        :func:`save_optimisation_config`.

    Returns
    -------
    config : dict
        Dictionary with the following keys (any may be *None* if absent
        from the file):

        - ``"divertor_data"`` — dict suitable for ``Optimiser(divertor_data=...)``.
        - ``"buffers"`` — dict suitable for ``Optimiser(buffers=...)``.
        - ``"xpoint_regions"`` — dict suitable for ``Optimiser(xpoint_regions=...)``.
        - ``"constraints"`` — dict suitable for ``Optimiser(constraints=...)``.
        - ``"optimiser"`` — dict of scalar kwargs to unpack into ``Optimiser(**cfg["optimiser"])``.
    """
    path = Path(path)

    with open(path, "r") as f:
        payload = json.load(f)

    # --- geometry ---
    geometry = payload.get("geometry", {})
    divertor_data = {}
    buffers_out = {}
    xpt_out = {}

    for region, entry in geometry.items():
        strike = entry.get("strike", {})
        if "strike_R" in strike and "strike_Z" in strike:
            dd_entry = {"strike_R": strike["strike_R"], "strike_Z": strike["strike_Z"]}
            for k in _REGION_SETTING_KEYS:
                if k in entry:
                    dd_entry[k] = entry[k]
            divertor_data[region] = dd_entry

        if "buffers" in entry:
            buffers_out[region] = entry["buffers"]

        if "xpoint_regions" in entry:
            xpt_out[region] = entry["xpoint_regions"]

    return {
        "divertor_data": divertor_data or None,
        "buffers": buffers_out or None,
        "xpoint_regions": xpt_out or None,
        "constraints": payload.get("constraints"),
        "optimiser": payload.get("optimiser", {}),
    }
