"""Tests for ``forge.io`` — file I/O routines."""

import json

import numpy as np
import pytest

from forge.io import read_geqdsk, read_magnets, save_fancy_json, write_magnets


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
