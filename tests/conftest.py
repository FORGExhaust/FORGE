"""Shared fixtures for the FORGE test suite.

Fixtures defined here are automatically available to every test module
under the ``tests/`` directory without needing an explicit import.
"""

import json
import os

import numpy as np
import pytest

from forge._paths import data_dir_path

# ---------------------------------------------------------------------------
# Paths to bundled data files
# ---------------------------------------------------------------------------

DATA_DIR = data_dir_path()
GEQDSK_DIR = os.path.join(DATA_DIR, "geqdsk", "examples")
JSON_DIR = os.path.join(DATA_DIR, "json", "examples")

# Pick a representative GEQDSK / JSON pair that ships with the package
SAMPLE_GEQDSK = os.path.join(GEQDSK_DIR, "01-mastu.geqdsk")
SAMPLE_JSON = os.path.join(JSON_DIR, "01-mastu.json")


@pytest.fixture
def sample_geqdsk_path():
    """Path to a bundled GEQDSK file for integration tests."""
    if not os.path.isfile(SAMPLE_GEQDSK):
        pytest.skip(f"Sample GEQDSK not found: {SAMPLE_GEQDSK}")
    return SAMPLE_GEQDSK


@pytest.fixture
def sample_json_path():
    """Path to a bundled magnet-definition JSON file for integration tests."""
    if not os.path.isfile(SAMPLE_JSON):
        pytest.skip(f"Sample JSON not found: {SAMPLE_JSON}")
    return SAMPLE_JSON


@pytest.fixture
def sample_eq_data(sample_geqdsk_path):
    """Loaded equilibrium data dict from ``read_geqdsk``."""
    from forge.io import read_geqdsk

    return read_geqdsk(sample_geqdsk_path)


@pytest.fixture
def sample_magnets_data(sample_json_path):
    """Loaded magnets data from ``read_magnets``."""
    from forge.io import read_magnets

    return read_magnets(sample_json_path)


@pytest.fixture
def simple_coil():
    """A single point-source ``Coil`` at a typical tokamak PF-coil location."""
    from forge.magnets import Coil

    return Coil(R=1.0, Z=1.5, name="TestCoil", current=1000.0, turns=10)


@pytest.fixture
def simple_shaped_coil():
    """A ``ShapedCoil`` with a small rectangular cross-section."""
    from forge.magnets import ShapedCoil

    shape = [
        (0.95, 1.45),
        (1.05, 1.45),
        (1.05, 1.55),
        (0.95, 1.55),
    ]
    return ShapedCoil(shape=shape, name="TestShapedCoil", current=1000.0, turns=10)


@pytest.fixture
def simple_solenoid():
    """A ``Solenoid`` spanning a short vertical extent."""
    from forge.magnets import Solenoid

    return Solenoid(R=0.2, Z_min=-1.0, Z_max=1.0, name="TestSolenoid", current=500.0, turns=100, npoints=11)


@pytest.fixture
def simple_filament_coil():
    """A ``FilamentPointCoil`` with a handful of filaments."""
    from forge.magnets import FilamentPointCoil

    R_fils = [0.95, 0.95, 1.05, 1.05]
    Z_fils = [1.45, 1.55, 1.45, 1.55]
    return FilamentPointCoil(
        name="TestFilamentCoil",
        current=1000.0,
        turns=10,
        R_filaments=R_fils,
        Z_filaments=Z_fils,
        dR=0.1,
        dZ=0.1,
    )


@pytest.fixture
def simple_circuit(simple_coil):
    """A ``Circuit`` containing two symmetric coils."""
    from forge.magnets import Coil, Circuit

    coil_upper = Coil(R=1.0, Z=1.5, name="PF1U", current=2000.0, turns=10)
    coil_lower = Coil(R=1.0, Z=-1.5, name="PF1L", current=-2000.0, turns=10)
    return Circuit(magnets=[coil_upper, coil_lower], multipliers=[1.0, -1.0], name="PF1")


@pytest.fixture
def evaluation_point():
    """A (R, Z) point away from all coil locations, safe for field evaluation."""
    return (0.7, 0.0)
