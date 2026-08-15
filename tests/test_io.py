"""Tests for ``forge.io`` — file I/O routines."""

import json
import os

import numpy as np
import pytest

from forge.io import (
    geqdsk_dict, read_geqdsk, read_magnets, read_wall, save_fancy_json,
    wall_from_dict, write_geqdsk, write_magnets, write_wall,
)


# ---------------------------------------------------------------------------
# read_geqdsk
# ---------------------------------------------------------------------------


class TestReadGeqdsk:
    """Tests for ``read_geqdsk``."""

    def test_invalid_type_raises_TypeError(self):
        with pytest.raises(TypeError):
            read_geqdsk(123)

    def test_none_raises_TypeError(self):
        with pytest.raises(TypeError):
            read_geqdsk(None)

    @pytest.mark.integration
    def test_loads_expected_keys(self, sample_geqdsk_path):
        eq_data = read_geqdsk(sample_geqdsk_path)
        required_keys = [
            "R_min", "R_max", "nR", "R_1D", "dR", "R_2D",
            "Z_min", "Z_max", "nZ", "Z_1D", "dZ", "Z_2D",
            "wall_R", "wall_Z",
            "psi_2D", "psi_lcfs", "psi_axis",
            "psin_data", "pprime_data", "ffprime_data", "q_data",
            "pressure_data", "fpol_data",
            "plasma_current", "fvac",
        ]
        for key in required_keys:
            assert key in eq_data, f"Missing key: {key}"

    @pytest.mark.integration
    def test_grid_dimensions_consistent(self, sample_eq_data):
        eq = sample_eq_data
        assert eq["R_1D"].shape[0] == eq["nR"]
        assert eq["Z_1D"].shape[0] == eq["nZ"]
        assert eq["R_2D"].shape == (eq["nR"], eq["nZ"])
        assert eq["Z_2D"].shape == (eq["nR"], eq["nZ"])
        assert eq["psi_2D"].shape[0] > 0

    @pytest.mark.integration
    def test_wall_is_closed(self, sample_eq_data):
        eq = sample_eq_data
        assert eq["wall_R"][0] == eq["wall_R"][-1]
        assert eq["wall_Z"][0] == eq["wall_Z"][-1]

    @pytest.mark.integration
    def test_R_min_less_than_R_max(self, sample_eq_data):
        assert sample_eq_data["R_min"] < sample_eq_data["R_max"]

    @pytest.mark.integration
    def test_psin_ranges_from_zero_to_one(self, sample_eq_data):
        np.testing.assert_allclose(sample_eq_data["psin_data"][0], 0.0)
        np.testing.assert_allclose(sample_eq_data["psin_data"][-1], 1.0)


# ---------------------------------------------------------------------------
# read_magnets
# ---------------------------------------------------------------------------


class TestReadMagnets:
    """Tests for ``read_magnets``."""

    def test_invalid_type_raises_TypeError(self):
        with pytest.raises(TypeError):
            read_magnets(42)

    @pytest.mark.integration
    def test_returns_three_items(self, sample_json_path):
        result = read_magnets(sample_json_path)
        assert len(result) == 3  # coils_data, circuits_data, suggested_circuits

    @pytest.mark.integration
    def test_coils_data_is_dict(self, sample_json_path):
        coils_data, _, _ = read_magnets(sample_json_path)
        assert isinstance(coils_data, dict)
        assert len(coils_data) > 0

    @pytest.mark.integration
    def test_suggested_circuits_is_dict(self, sample_json_path):
        _, _, suggested = read_magnets(sample_json_path)
        assert isinstance(suggested, dict)


# ---------------------------------------------------------------------------
# save_fancy_json
# ---------------------------------------------------------------------------


class TestSaveFancyJson:
    """Tests for the custom JSON serialiser."""

    def test_simple_dict_roundtrip(self, tmp_path):
        data = {"a": 1, "b": [2, 3, 4], "c": "hello"}
        out = tmp_path / "test.json"
        save_fancy_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_numpy_arrays_serialized(self, tmp_path):
        data = {"arr": np.array([1.0, 2.0, 3.0])}
        out = tmp_path / "test.json"
        save_fancy_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["arr"] == [1.0, 2.0, 3.0]

    def test_numpy_scalars_serialized(self, tmp_path):
        data = {"val": np.float64(3.14), "idx": np.int64(7)}
        out = tmp_path / "test.json"
        save_fancy_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["val"] == pytest.approx(3.14)
        assert loaded["idx"] == 7

    def test_nan_inside_array_raises(self, tmp_path):
        """NaN values inside arrays raise ValueError because allow_nan=False.

        This documents current behaviour — ``save_fancy_json`` prefers
        strict JSON compliance and will reject bare NaN floats that
        survive the ``_to_python_basic`` conversion.
        """
        data = {"vals": np.array([1.0, np.nan, 3.0])}
        out = tmp_path / "test.json"
        with pytest.raises(ValueError):
            save_fancy_json(data, out)

    def test_nested_dict(self, tmp_path):
        data = {"outer": {"inner": [1, 2]}}
        out = tmp_path / "test.json"
        save_fancy_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded == data


# ---------------------------------------------------------------------------
# write_magnets / Machine.to_dict round-trip
# ---------------------------------------------------------------------------


class TestWriteMagnets:
    """Tests for ``write_magnets`` and ``Machine.to_dict``."""

    @pytest.mark.integration
    def test_round_trip_preserves_coil_names(self, sample_eq_data, sample_magnets_data, tmp_path):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        machine = Machine(
            magnets_data=coils_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
            circuits=circuits_data,
        )
        out = tmp_path / "machine.json"
        write_magnets(machine, out)
        loaded = json.loads(out.read_text())
        assert set(loaded["coils"].keys()) == set(coils_data.keys())

    @pytest.mark.integration
    def test_round_trip_preserves_circuits(self, sample_eq_data, sample_magnets_data, tmp_path):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        machine = Machine(
            magnets_data=coils_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
            circuits=circuits_data,
        )
        out = tmp_path / "machine.json"
        write_magnets(machine, out)
        loaded = json.loads(out.read_text())
        if circuits_data is not None:
            assert "circuits" in loaded
            assert set(loaded["circuits"].keys()) == set(circuits_data.keys())

    @pytest.mark.integration
    def test_round_trip_coil_types_preserved(self, sample_eq_data, sample_magnets_data, tmp_path):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        machine = Machine(
            magnets_data=coils_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
            circuits=circuits_data,
        )
        out = tmp_path / "machine.json"
        write_magnets(machine, out)
        loaded = json.loads(out.read_text())
        for name, coil in loaded["coils"].items():
            assert coil["type"] == coils_data[name]["type"]

    @pytest.mark.integration
    def test_round_trip_currents_match(self, sample_eq_data, sample_magnets_data, tmp_path):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        machine = Machine(
            magnets_data=coils_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
            circuits=circuits_data,
        )
        out = tmp_path / "machine.json"
        write_magnets(machine, out)
        loaded = json.loads(out.read_text())
        for name, coil in loaded["coils"].items():
            np.testing.assert_allclose(
                coil["current"], coils_data[name]["current"], rtol=1e-10
            )


# ---------------------------------------------------------------------------
# read_wall / write_wall
# ---------------------------------------------------------------------------


# A closed unit square, and the same square without its closing point
CLOSED_SQUARE_R = [1.0, 2.0, 2.0, 1.0, 1.0]
CLOSED_SQUARE_Z = [0.0, 0.0, 1.0, 1.0, 0.0]
OPEN_SQUARE_R = CLOSED_SQUARE_R[:-1]
OPEN_SQUARE_Z = CLOSED_SQUARE_Z[:-1]


class TestReadWall:
    """Tests for ``read_wall`` and its shared validation in ``wall_from_dict``."""

    def test_reads_coordinates(self, tmp_path):
        out = tmp_path / "wall.json"
        out.write_text(json.dumps({"R": CLOSED_SQUARE_R, "Z": CLOSED_SQUARE_Z}))
        wall_R, wall_Z = read_wall(out)
        assert wall_R == CLOSED_SQUARE_R
        assert wall_Z == CLOSED_SQUARE_Z

    def test_open_wall_is_closed_on_read(self, tmp_path):
        out = tmp_path / "wall.json"
        out.write_text(json.dumps({"R": OPEN_SQUARE_R, "Z": OPEN_SQUARE_Z}))
        wall_R, wall_Z = read_wall(out)
        assert len(wall_R) == len(OPEN_SQUARE_R) + 1
        assert wall_R[0] == wall_R[-1]
        assert wall_Z[0] == wall_Z[-1]

    def test_already_closed_wall_not_closed_again(self, tmp_path):
        out = tmp_path / "wall.json"
        out.write_text(json.dumps({"R": CLOSED_SQUARE_R, "Z": CLOSED_SQUARE_Z}))
        wall_R, _ = read_wall(out)
        assert len(wall_R) == len(CLOSED_SQUARE_R)

    def test_integer_coordinates_become_floats(self, tmp_path):
        out = tmp_path / "wall.json"
        out.write_text(json.dumps({"R": [1, 2, 2], "Z": [0, 0, 1]}))
        wall_R, wall_Z = read_wall(out)
        assert all(isinstance(v, float) for v in wall_R)
        assert all(isinstance(v, float) for v in wall_Z)

    def test_bundled_wall_file_loads(self):
        from forge._paths import data_dir_path

        path = os.path.join(data_dir_path(), "json", "MAST-U", "MAST-U_wall.json")
        if not os.path.isfile(path):
            pytest.skip(f"Bundled wall file not found: {path}")
        wall_R, wall_Z = read_wall(path)
        assert len(wall_R) == len(wall_Z) >= 3
        assert wall_R[0] == wall_R[-1]

    def test_non_object_raises(self):
        with pytest.raises(ValueError, match="object"):
            wall_from_dict([1, 2, 3])

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="'R' and 'Z'"):
            wall_from_dict({"R": [1.0, 2.0, 3.0]})

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            wall_from_dict({"R": [1.0, 2.0, 3.0], "Z": [0.0, 1.0]})

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            wall_from_dict({"R": [1.0, 2.0], "Z": [0.0, 1.0]})

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="lists of numbers"):
            wall_from_dict({"R": [1.0, "x", 3.0], "Z": [0.0, 1.0, 2.0]})


class TestWriteWall:
    """Tests for ``write_wall``."""

    def test_round_trip(self, tmp_path):
        out = tmp_path / "wall.json"
        write_wall(CLOSED_SQUARE_R, CLOSED_SQUARE_Z, out)
        wall_R, wall_Z = read_wall(out)
        np.testing.assert_allclose(wall_R, CLOSED_SQUARE_R)
        np.testing.assert_allclose(wall_Z, CLOSED_SQUARE_Z)

    def test_schema_matches_read_wall(self, tmp_path):
        out = tmp_path / "wall.json"
        write_wall(CLOSED_SQUARE_R, CLOSED_SQUARE_Z, out)
        loaded = json.loads(out.read_text())
        assert sorted(loaded) == ["R", "Z"]

    def test_accepts_numpy_arrays(self, tmp_path):
        out = tmp_path / "wall.json"
        write_wall(np.array(CLOSED_SQUARE_R), np.array(CLOSED_SQUARE_Z), out)
        wall_R, _ = read_wall(out)
        np.testing.assert_allclose(wall_R, CLOSED_SQUARE_R)

    def test_mismatched_lengths_raise(self, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            write_wall([1.0, 2.0, 3.0], [0.0, 1.0], tmp_path / "wall.json")

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "wall.json"
        write_wall(CLOSED_SQUARE_R, CLOSED_SQUARE_Z, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# geqdsk_dict / write_geqdsk wall override
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_equilibrium(sample_eq_data, sample_magnets_data):
    """An ``Equilibrium`` built from the bundled sample files."""
    from forge.equilibrium import Equilibrium
    from forge.machine import Machine

    coils_data, circuits_data, _ = sample_magnets_data
    machine = Machine(
        magnets_data=coils_data,
        wall_R=sample_eq_data["wall_R"],
        wall_Z=sample_eq_data["wall_Z"],
        circuits=circuits_data,
    )
    return Equilibrium(
        eq_data=sample_eq_data, tokamak=machine, calculate_flux_from_coils=True
    )


@pytest.mark.integration
class TestGeqdskWallOverride:
    """Tests for writing a GEQDSK with a substituted wall."""

    def test_default_uses_equilibrium_wall(self, sample_equilibrium):
        data = geqdsk_dict(sample_equilibrium)
        assert data["rlim"] == sample_equilibrium.wall_R
        assert data["nlim"] == len(sample_equilibrium.wall_R)

    def test_override_replaces_wall(self, sample_equilibrium):
        data = geqdsk_dict(
            sample_equilibrium, wall_R=CLOSED_SQUARE_R, wall_Z=CLOSED_SQUARE_Z
        )
        assert data["rlim"] == CLOSED_SQUARE_R
        assert data["zlim"] == CLOSED_SQUARE_Z
        assert data["nlim"] == len(CLOSED_SQUARE_R)

    def test_override_does_not_mutate_equilibrium(self, sample_equilibrium):
        before = list(sample_equilibrium.wall_R)
        geqdsk_dict(sample_equilibrium, wall_R=CLOSED_SQUARE_R, wall_Z=CLOSED_SQUARE_Z)
        assert sample_equilibrium.wall_R == before

    def test_partial_override_raises(self, sample_equilibrium):
        with pytest.raises(ValueError, match="together"):
            geqdsk_dict(sample_equilibrium, wall_R=CLOSED_SQUARE_R)

    def test_mismatched_override_lengths_raise(self, sample_equilibrium):
        with pytest.raises(ValueError, match="same length"):
            geqdsk_dict(sample_equilibrium, wall_R=[1.0, 2.0, 3.0], wall_Z=[0.0, 1.0])

    def test_written_geqdsk_carries_override(self, sample_equilibrium, tmp_path):
        out = tmp_path / "override.geqdsk"
        write_geqdsk(
            sample_equilibrium, str(out),
            wall_R=CLOSED_SQUARE_R, wall_Z=CLOSED_SQUARE_Z,
        )
        reloaded = read_geqdsk(str(out))
        np.testing.assert_allclose(reloaded["wall_R"], CLOSED_SQUARE_R)
        np.testing.assert_allclose(reloaded["wall_Z"], CLOSED_SQUARE_Z)

    def test_written_geqdsk_defaults_to_equilibrium_wall(self, sample_equilibrium, tmp_path):
        out = tmp_path / "default.geqdsk"
        write_geqdsk(sample_equilibrium, str(out))
        reloaded = read_geqdsk(str(out))
        np.testing.assert_allclose(
            reloaded["wall_R"], sample_equilibrium.wall_R, atol=1e-6
        )


# ---------------------------------------------------------------------------
# save_fancy_json (continued)
# ---------------------------------------------------------------------------


class TestSaveFancyJsonExtra:
    """Additional tests for save_fancy_json that were displaced by insertion."""

    def test_nested_dict(self, tmp_path):
        data = {"outer": {"inner": [1, 2]}}
        out = tmp_path / "test.json"
        save_fancy_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["outer"]["inner"] == [1, 2]

    def test_invalid_indent_raises(self):
        with pytest.raises(ValueError, match="indent"):
            save_fancy_json({}, "dummy.json", indent=-1)

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "dir" / "file.json"
        save_fancy_json({"x": 1}, out)
        assert out.exists()
