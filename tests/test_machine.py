"""Tests for ``forge.machine`` — the Machine class."""

import numpy as np
import pytest

from forge.machine import Machine


@pytest.fixture
def machine(sample_eq_data, sample_magnets_data):
    """A ``Machine`` built from the bundled sample files."""
    coils_data, circuits_data, _ = sample_magnets_data
    return Machine(
        magnets_data=coils_data,
        wall_R=sample_eq_data["wall_R"],
        wall_Z=sample_eq_data["wall_Z"],
        circuits=circuits_data,
    )


# A closed unit square, deliberately offset from any real machine wall
SQUARE_R = [1.0, 3.0, 3.0, 1.0, 1.0]
SQUARE_Z = [-2.0, -2.0, 2.0, 2.0, -2.0]


@pytest.mark.integration
class TestSetWall:
    """Tests for ``Machine.set_wall``."""

    def test_constructor_populates_derived_state(self, machine, sample_eq_data):
        assert machine.wall_R_min == np.amin(sample_eq_data["wall_R"])
        assert machine.wall_R_max == np.amax(sample_eq_data["wall_R"])
        assert machine.wall_Z_min == np.amin(sample_eq_data["wall_Z"])
        assert machine.wall_Z_max == np.amax(sample_eq_data["wall_Z"])
        assert len(machine.wall.coords) == len(sample_eq_data["wall_R"])

    def test_coordinates_replaced(self, machine):
        machine.set_wall(SQUARE_R, SQUARE_Z)
        assert machine.wall_R == SQUARE_R
        assert machine.wall_Z == SQUARE_Z

    def test_bounds_recomputed(self, machine):
        machine.set_wall(SQUARE_R, SQUARE_Z)
        assert machine.wall_R_min == 1.0
        assert machine.wall_R_max == 3.0
        assert machine.wall_Z_min == -2.0
        assert machine.wall_Z_max == 2.0

    def test_shapely_wall_recomputed(self, machine):
        machine.set_wall(SQUARE_R, SQUARE_Z)
        assert len(machine.wall.coords) == len(SQUARE_R)
        np.testing.assert_allclose(machine.wall.bounds, (1.0, -2.0, 3.0, 2.0))

    def test_no_stale_state_after_shrinking_wall(self, machine):
        """A smaller wall must not leave the old, larger bounds behind."""
        original_max = machine.wall_R_max
        machine.set_wall([0.5, 0.6, 0.6, 0.5], [0.0, 0.0, 0.1, 0.1])
        assert machine.wall_R_max == 0.6 < original_max

    def test_mismatched_lengths_raise(self, machine):
        with pytest.raises(ValueError, match="same length"):
            machine.set_wall([1.0, 2.0, 3.0], [0.0, 1.0])

    def test_none_raises(self, machine):
        with pytest.raises(ValueError, match="wall_R and wall_Z"):
            machine.set_wall(None, None)

    def test_coilset_untouched(self, machine):
        names = list(machine.coilset)
        machine.set_wall(SQUARE_R, SQUARE_Z)
        assert list(machine.coilset) == names
