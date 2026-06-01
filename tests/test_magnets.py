"""Tests for ``forge.magnets`` — Coil, ShapedCoil, Solenoid, FilamentPointCoil, Circuit."""

import numpy as np
import pytest

from forge.magnets import Circuit, Coil, FilamentPointCoil


# ---------------------------------------------------------------------------
# Coil
# ---------------------------------------------------------------------------


class TestCoil:
    """Tests for the point-source ``Coil``."""

    def test_construction(self, simple_coil):
        assert simple_coil.R == 1.0
        assert simple_coil.Z == 1.5
        assert simple_coil.name == "TestCoil"
        assert simple_coil.current == 1000.0
        assert simple_coil.turns == 10

    def test_control_psi_is_finite(self, simple_coil, evaluation_point):
        R, Z = evaluation_point
        psi = simple_coil.control_psi(R, Z)
        assert np.isfinite(psi)

    def test_psi_equals_control_psi_times_current(self, simple_coil, evaluation_point):
        R, Z = evaluation_point
        np.testing.assert_allclose(
            simple_coil.psiRZ(R, Z),
            simple_coil.control_psi(R, Z) * simple_coil.current,
        )

    def test_Br_equals_control_Br_times_current(self, simple_coil, evaluation_point):
        R, Z = evaluation_point
        np.testing.assert_allclose(
            simple_coil.BrRZ(R, Z),
            simple_coil.control_Br(R, Z) * simple_coil.current,
        )

    def test_Bz_equals_control_Bz_times_current(self, simple_coil, evaluation_point):
        R, Z = evaluation_point
        np.testing.assert_allclose(
            simple_coil.BzRZ(R, Z),
            simple_coil.control_Bz(R, Z) * simple_coil.current,
        )

    @pytest.mark.parametrize("deriv", ["dBr_dR", "dBr_dZ", "dBz_dR", "dBz_dZ"])
    def test_dBp_returns_finite(self, simple_coil, evaluation_point, deriv):
        R, Z = evaluation_point
        val = simple_coil.control_dBp(R, Z, deriv=deriv)
        assert np.isfinite(val)

    def test_dBp_invalid_deriv_raises(self, simple_coil, evaluation_point):
        R, Z = evaluation_point
        with pytest.raises(ValueError, match="Unknown derivative"):
            simple_coil.control_dBp(R, Z, deriv="invalid")

    def test_jacobian_shape(self, simple_coil, evaluation_point):
        R, Z = evaluation_point
        J = simple_coil.control_Bp_jacobian(R, Z)
        assert J.shape == (2, 2)
        assert np.all(np.isfinite(J))

    def test_zero_current_gives_zero_flux(self, evaluation_point):
        coil = Coil(R=1.0, Z=1.0, current=0.0, turns=10)
        R, Z = evaluation_point
        assert coil.psiRZ(R, Z) == 0.0

    def test_set_fill_colour(self, simple_coil):
        simple_coil.set_fill_colour("red")
        assert simple_coil.fill_colour == "red"

    def test_set_edge_colour(self, simple_coil):
        simple_coil.set_edge_colour("blue")
        assert simple_coil.edge_colour == "blue"

    def test_linearity_in_current(self, evaluation_point):
        """Doubling the current should double psi."""
        R, Z = evaluation_point
        c1 = Coil(R=1.0, Z=1.0, current=1000.0, turns=1)
        c2 = Coil(R=1.0, Z=1.0, current=2000.0, turns=1)
        np.testing.assert_allclose(c2.psiRZ(R, Z), 2.0 * c1.psiRZ(R, Z), rtol=1e-12)


# ---------------------------------------------------------------------------
# ShapedCoil
# ---------------------------------------------------------------------------


class TestShapedCoil:
    """Tests for the ``ShapedCoil`` with a polygonal cross-section."""

    def test_construction(self, simple_shaped_coil):
        assert simple_shaped_coil.name == "TestShapedCoil"
        assert simple_shaped_coil.area > 0

    def test_centroid_is_at_centre(self, simple_shaped_coil):
        np.testing.assert_allclose(simple_shaped_coil.R, 1.0, atol=1e-10)
        np.testing.assert_allclose(simple_shaped_coil.Z, 1.5, atol=1e-10)

    def test_control_psi_finite(self, simple_shaped_coil, evaluation_point):
        R, Z = evaluation_point
        assert np.isfinite(simple_shaped_coil.control_psi(R, Z))

    def test_psi_equals_control_times_current(self, simple_shaped_coil, evaluation_point):
        R, Z = evaluation_point
        np.testing.assert_allclose(
            simple_shaped_coil.psiRZ(R, Z),
            simple_shaped_coil.control_psi(R, Z) * simple_shaped_coil.current,
        )

    def test_shaped_coil_close_to_point_coil(self, simple_shaped_coil, evaluation_point):
        """For a tiny cross-section far from the evaluation point, shaped and point coils should agree closely."""
        R, Z = evaluation_point
        point_coil = Coil(
            R=simple_shaped_coil.R,
            Z=simple_shaped_coil.Z,
            current=simple_shaped_coil.current,
            turns=simple_shaped_coil.turns,
        )
        # Allow up to ~5% difference because the shaped coil distributes current over a small area
        np.testing.assert_allclose(
            simple_shaped_coil.psiRZ(R, Z),
            point_coil.psiRZ(R, Z),
            rtol=0.05,
        )

    @pytest.mark.parametrize("deriv", ["dBr_dR", "dBr_dZ", "dBz_dR", "dBz_dZ"])
    def test_shaped_dBp_returns_finite(self, simple_shaped_coil, evaluation_point, deriv):
        R, Z = evaluation_point
        val = simple_shaped_coil.control_dBp(R, Z, deriv=deriv)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Solenoid
# ---------------------------------------------------------------------------


class TestSolenoid:
    """Tests for the ``Solenoid`` class."""

    def test_construction(self, simple_solenoid):
        assert simple_solenoid.R == 0.2
        assert simple_solenoid.Z_min == -1.0
        assert simple_solenoid.Z_max == 1.0
        assert len(simple_solenoid.Z_points) == 11

    def test_weights_sum_to_one(self, simple_solenoid):
        total = simple_solenoid.weight * simple_solenoid.npoints
        np.testing.assert_allclose(total, 1.0, atol=1e-14)

    def test_control_psi_finite(self, simple_solenoid, evaluation_point):
        R, Z = evaluation_point
        assert np.isfinite(simple_solenoid.control_psi(R, Z))

    def test_psi_equals_control_times_current(self, simple_solenoid, evaluation_point):
        R, Z = evaluation_point
        np.testing.assert_allclose(
            simple_solenoid.psiRZ(R, Z),
            simple_solenoid.control_psi(R, Z) * simple_solenoid.current,
        )

    def test_solenoid_Bz_on_axis_midplane(self, simple_solenoid):
        """On the magnetic axis near the solenoid centre, Bz should dominate."""
        Br = simple_solenoid.control_Br(simple_solenoid.R + 0.01, 0.0)
        Bz = simple_solenoid.control_Bz(simple_solenoid.R + 0.01, 0.0)
        assert abs(Bz) > abs(Br)


# ---------------------------------------------------------------------------
# FilamentPointCoil
# ---------------------------------------------------------------------------


class TestFilamentPointCoil:
    """Tests for ``FilamentPointCoil``."""

    def test_construction(self, simple_filament_coil):
        assert simple_filament_coil.N_filaments == 4
        assert simple_filament_coil.dR == 0.1
        assert simple_filament_coil.dZ == 0.1

    def test_weights_sum_to_one(self, simple_filament_coil):
        total = simple_filament_coil.weight * simple_filament_coil.N_filaments
        np.testing.assert_allclose(total, 1.0, atol=1e-14)

    def test_control_psi_finite(self, simple_filament_coil, evaluation_point):
        R, Z = evaluation_point
        assert np.isfinite(simple_filament_coil.control_psi(R, Z))

    def test_psi_linear_in_current(self, evaluation_point):
        R, Z = evaluation_point
        R_fils = [1.0, 1.0]
        Z_fils = [1.0, 1.1]
        c1 = FilamentPointCoil(current=1000.0, turns=1, R_filaments=R_fils, Z_filaments=Z_fils)
        c2 = FilamentPointCoil(current=3000.0, turns=1, R_filaments=R_fils, Z_filaments=Z_fils)
        np.testing.assert_allclose(c2.psiRZ(R, Z), 3.0 * c1.psiRZ(R, Z), rtol=1e-12)


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------


class TestCircuit:
    """Tests for the ``Circuit`` class."""

    def test_construction(self, simple_circuit):
        assert simple_circuit.name == "PF1"
        assert len(simple_circuit.coilset) == 2

    def test_circuit_current_estimation(self):
        """When no explicit circuit_current, the Circuit infers it from coil currents and multipliers."""
        c_up = Coil(R=1.0, Z=1.0, name="U", current=5000.0, turns=1)
        c_lo = Coil(R=1.0, Z=-1.0, name="L", current=-5000.0, turns=1)
        circ = Circuit(magnets=[c_up, c_lo], multipliers=[1.0, -1.0], name="X")
        np.testing.assert_allclose(circ.current, 5000.0, rtol=1e-12)

    def test_control_psi_is_sum_of_coils(self, simple_circuit, evaluation_point):
        """Circuit's control_psi = sum(coil.control_psi * multiplier)."""
        R, Z = evaluation_point
        expected = 0.0
        for coil_dict in simple_circuit.coilset.values():
            coil = coil_dict["magnet"]
            mult = coil_dict["current_multiplier"]
            expected += coil.control_psi(R, Z) * mult
        actual = simple_circuit.control_psi(R, Z)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_psi_equals_control_times_current(self, simple_circuit, evaluation_point):
        R, Z = evaluation_point
        np.testing.assert_allclose(
            simple_circuit.psiRZ(R, Z),
            simple_circuit.control_psi(R, Z) * simple_circuit.current,
        )

    def test_antisymmetric_circuit_has_zero_Bz_on_midplane(self):
        """Two coils at ±Z with multipliers [1, -1]: Bz cancels on Z=0."""
        c_up = Coil(R=2.0, Z=1.0, name="U", current=1000.0, turns=1)
        c_lo = Coil(R=2.0, Z=-1.0, name="L", current=-1000.0, turns=1)
        circ = Circuit(magnets=[c_up, c_lo], multipliers=[1.0, -1.0])
        # At midplane (Z=0) the Bz contributions should cancel by symmetry
        # Note: this is control_Bz (unit current), not BzRZ (actual current)
        Bz = circ.control_Bz(0.7, 0.0)
        np.testing.assert_allclose(Bz, 0.0, atol=1e-14)

    def test_set_fill_colour_propagates(self, simple_circuit):
        simple_circuit.set_fill_colour("green")
        for coil_dict in simple_circuit.coilset.values():
            assert coil_dict["magnet"].fill_colour == "green"

    def test_set_edge_colour_propagates(self, simple_circuit):
        simple_circuit.set_edge_colour("purple")
        for coil_dict in simple_circuit.coilset.values():
            assert coil_dict["magnet"].edge_colour == "purple"

    @pytest.mark.parametrize("deriv", ["dBr_dR", "dBr_dZ", "dBz_dR", "dBz_dZ"])
    def test_circuit_dBp_finite(self, simple_circuit, evaluation_point, deriv):
        R, Z = evaluation_point
        val = simple_circuit.control_dBp(R, Z, deriv=deriv)
        assert np.isfinite(val)

    def test_circuit_jacobian_shape(self, simple_circuit, evaluation_point):
        R, Z = evaluation_point
        J = simple_circuit.control_Bp_jacobian(R, Z)
        assert J.shape == (2, 2)


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for ``to_dict`` serialisation on magnet classes."""

    def test_coil_to_dict(self, simple_coil):
        d = simple_coil.to_dict()
        assert d["type"] == "point"
        assert d["R"] == simple_coil.R
        assert d["Z"] == simple_coil.Z
        assert d["current"] == simple_coil.current
        assert d["turns"] == simple_coil.turns

    def test_shaped_coil_to_dict(self, simple_shaped_coil):
        d = simple_shaped_coil.to_dict()
        assert d["type"] == "shaped"
        assert len(d["R"]) == len(simple_shaped_coil.shape)
        assert len(d["Z"]) == len(simple_shaped_coil.shape)
        assert d["current"] == simple_shaped_coil.current
        assert d["turns"] == simple_shaped_coil.turns

    def test_solenoid_to_dict(self, simple_solenoid):
        d = simple_solenoid.to_dict()
        assert d["type"] == "solenoid"
        assert d["R"] == simple_solenoid.R
        assert d["Z_min"] == simple_solenoid.Z_min
        assert d["Z_max"] == simple_solenoid.Z_max
        assert d["current"] == simple_solenoid.current
        assert d["turns"] == simple_solenoid.turns

    def test_filament_coil_to_dict(self, simple_filament_coil):
        d = simple_filament_coil.to_dict()
        assert d["type"] == "filament"
        assert d["R"] == [float(r) for r in simple_filament_coil.R_filaments]
        assert d["Z"] == [float(z) for z in simple_filament_coil.Z_filaments]
        assert d["current"] == simple_filament_coil.current
        assert d["dR"] == simple_filament_coil.dR
        assert d["dZ"] == simple_filament_coil.dZ

    def test_coil_to_dict_values_are_json_serialisable(self, simple_coil):
        import json
        d = simple_coil.to_dict()
        json.dumps(d)  # should not raise

    def test_filament_to_dict_values_are_json_serialisable(self, simple_filament_coil):
        import json
        d = simple_filament_coil.to_dict()
        json.dumps(d)  # should not raise
