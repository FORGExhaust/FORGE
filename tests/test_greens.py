"""Tests for ``forge.greens`` — Green's function calculations.

These tests validate the mathematical properties of the Green's functions
used to compute poloidal magnetic flux and field components from current
filaments. The functions are the physics backbone of every coil class.
"""

import numpy as np
import pytest

from forge import greens
from forge.greens import MU0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# A coil at a typical PF-coil location
RC, ZC = 1.0, 1.5

# Field-evaluation points (must not coincide with the coil)
R_EVAL = np.array([0.5, 0.7, 1.5, 2.0])
Z_EVAL = np.array([0.0, -0.5, 0.5, 1.0])

# Single-point evaluation
R1, Z1 = 0.7, 0.0


class TestGreensBasic:
    """Smoke tests that every Green's function returns finite values."""

    @pytest.mark.parametrize("func", [
        greens.Greens,
        greens.Greens_dpsi_dR,
        greens.Greens_dpsi_dZ,
        greens.Greens_d2psi_dR2,
        greens.Greens_d2psi_dZ2,
        greens.Greens_d2psi_dR_dZ,
        greens.Greens_Br,
        greens.Greens_Bz,
        greens.Greens_dBr_dR,
        greens.Greens_dBr_dZ,
        greens.Greens_dBz_dR,
        greens.Greens_dBz_dZ,
    ])
    def test_returns_finite_scalar(self, func):
        """Each function should produce a finite number for a well-separated point."""
        result = func(RC, ZC, R1, Z1)
        assert np.isfinite(result), f"{func.__name__} returned non-finite: {result}"

    @pytest.mark.parametrize("func", [
        greens.Greens,
        greens.Greens_Br,
        greens.Greens_Bz,
    ])
    def test_returns_array_for_array_input(self, func):
        """Functions should broadcast over NumPy arrays."""
        result = func(RC, ZC, R_EVAL, Z_EVAL)
        assert isinstance(result, np.ndarray)
        assert result.shape == R_EVAL.shape
        assert np.all(np.isfinite(result))


class TestGreensSymmetry:
    """Physical symmetry properties of the Green's functions."""

    def test_psi_symmetric_in_Z(self):
        """Flux from a coil at (Rc, 0) is symmetric about Z = 0."""
        psi_pos = greens.Greens(RC, 0.0, R1, 0.5)
        psi_neg = greens.Greens(RC, 0.0, R1, -0.5)
        np.testing.assert_allclose(psi_pos, psi_neg, rtol=1e-12)

    def test_dpsi_dZ_antisymmetric(self):
        """dpsi/dZ from a coil at (Rc, 0) is antisymmetric about Z = 0."""
        dp_pos = greens.Greens_dpsi_dZ(RC, 0.0, R1, 0.5)
        dp_neg = greens.Greens_dpsi_dZ(RC, 0.0, R1, -0.5)
        np.testing.assert_allclose(dp_pos, -dp_neg, rtol=1e-12)

    def test_dpsi_dR_symmetric_in_Z(self):
        """dpsi/dR from a coil at (Rc, 0) is symmetric about Z = 0."""
        dp_pos = greens.Greens_dpsi_dR(RC, 0.0, R1, 0.5)
        dp_neg = greens.Greens_dpsi_dR(RC, 0.0, R1, -0.5)
        np.testing.assert_allclose(dp_pos, dp_neg, rtol=1e-12)

    def test_Br_antisymmetric_in_Z(self):
        """Br from a symmetric coil at Z=0 must flip sign when Z flips."""
        Br_pos = greens.Greens_Br(RC, 0.0, R1, 0.5)
        Br_neg = greens.Greens_Br(RC, 0.0, R1, -0.5)
        np.testing.assert_allclose(Br_pos, -Br_neg, rtol=1e-12)

    def test_Bz_symmetric_in_Z(self):
        """Bz from a symmetric coil at Z=0 has same sign at ±Z."""
        Bz_pos = greens.Greens_Bz(RC, 0.0, R1, 0.5)
        Bz_neg = greens.Greens_Bz(RC, 0.0, R1, -0.5)
        np.testing.assert_allclose(Bz_pos, Bz_neg, rtol=1e-12)


class TestGreensFieldRelations:
    """Relations between flux derivatives and field components (Br, Bz)."""

    def test_Br_from_dpsi_dZ(self):
        """Br = -(1 / 2piR) * dpsi/dZ."""
        dpsi_dZ = greens.Greens_dpsi_dZ(RC, ZC, R1, Z1)
        Br_expected = -(1.0 / (2.0 * np.pi * R1)) * dpsi_dZ
        Br_actual = greens.Greens_Br(RC, ZC, R1, Z1)
        np.testing.assert_allclose(Br_actual, Br_expected, rtol=1e-12)

    def test_Bz_from_dpsi_dR(self):
        """Bz = (1 / 2piR) * dpsi/dR."""
        dpsi_dR = greens.Greens_dpsi_dR(RC, ZC, R1, Z1)
        Bz_expected = (1.0 / (2.0 * np.pi * R1)) * dpsi_dR
        Bz_actual = greens.Greens_Bz(RC, ZC, R1, Z1)
        np.testing.assert_allclose(Bz_actual, Bz_expected, rtol=1e-12)


class TestGreensNumericalDerivatives:
    """Check analytic derivatives against finite-difference approximations."""

    DELTA = 1e-6

    def test_dpsi_dR_via_finite_difference(self):
        """dpsi/dR ≈ [psi(R+h) - psi(R-h)] / 2h."""
        h = self.DELTA
        fd = (greens.Greens(RC, ZC, R1 + h, Z1) - greens.Greens(RC, ZC, R1 - h, Z1)) / (2 * h)
        analytic = greens.Greens_dpsi_dR(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-4)

    def test_dpsi_dZ_via_finite_difference(self):
        """dpsi/dZ ≈ [psi(Z+h) - psi(Z-h)] / 2h."""
        h = self.DELTA
        fd = (greens.Greens(RC, ZC, R1, Z1 + h) - greens.Greens(RC, ZC, R1, Z1 - h)) / (2 * h)
        analytic = greens.Greens_dpsi_dZ(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-4)

    def test_d2psi_dR2_via_finite_difference(self):
        """d²psi/dR² ≈ [psi(R+h) - 2*psi(R) + psi(R-h)] / h²."""
        h = self.DELTA
        fd = (
            greens.Greens(RC, ZC, R1 + h, Z1)
            - 2.0 * greens.Greens(RC, ZC, R1, Z1)
            + greens.Greens(RC, ZC, R1 - h, Z1)
        ) / (h**2)
        analytic = greens.Greens_d2psi_dR2(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3)

    def test_d2psi_dZ2_via_finite_difference(self):
        """d²psi/dZ² ≈ [psi(Z+h) - 2*psi(Z) + psi(Z-h)] / h²."""
        h = self.DELTA
        fd = (
            greens.Greens(RC, ZC, R1, Z1 + h)
            - 2.0 * greens.Greens(RC, ZC, R1, Z1)
            + greens.Greens(RC, ZC, R1, Z1 - h)
        ) / (h**2)
        analytic = greens.Greens_d2psi_dZ2(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-2)

    def test_d2psi_dRdZ_via_finite_difference(self):
        """d²psi/dRdZ ≈ [psi(R+h,Z+h) - psi(R+h,Z-h) - psi(R-h,Z+h) + psi(R-h,Z-h)] / 4h²."""
        h = self.DELTA
        fd = (
            greens.Greens(RC, ZC, R1 + h, Z1 + h)
            - greens.Greens(RC, ZC, R1 + h, Z1 - h)
            - greens.Greens(RC, ZC, R1 - h, Z1 + h)
            + greens.Greens(RC, ZC, R1 - h, Z1 - h)
        ) / (4 * h**2)
        analytic = greens.Greens_d2psi_dR_dZ(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3)

    def test_dBr_dR_via_finite_difference(self):
        h = self.DELTA
        fd = (greens.Greens_Br(RC, ZC, R1 + h, Z1) - greens.Greens_Br(RC, ZC, R1 - h, Z1)) / (2 * h)
        analytic = greens.Greens_dBr_dR(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3)

    def test_dBr_dZ_via_finite_difference(self):
        h = self.DELTA
        fd = (greens.Greens_Br(RC, ZC, R1, Z1 + h) - greens.Greens_Br(RC, ZC, R1, Z1 - h)) / (2 * h)
        analytic = greens.Greens_dBr_dZ(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3)

    def test_dBz_dR_via_finite_difference(self):
        h = self.DELTA
        fd = (greens.Greens_Bz(RC, ZC, R1 + h, Z1) - greens.Greens_Bz(RC, ZC, R1 - h, Z1)) / (2 * h)
        analytic = greens.Greens_dBz_dR(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3)

    def test_dBz_dZ_via_finite_difference(self):
        h = self.DELTA
        fd = (greens.Greens_Bz(RC, ZC, R1, Z1 + h) - greens.Greens_Bz(RC, ZC, R1, Z1 - h)) / (2 * h)
        analytic = greens.Greens_dBz_dZ(RC, ZC, R1, Z1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3)


class TestGreensSingularPoint:
    """Behaviour when source and evaluation points coincide."""

    def test_psi_at_source_returns_finite_near_zero(self):
        """Greens() returns a finite, near-zero value when R == Rc, Z == Zc.

        Due to k^2 clipping the result is not exactly zero but should be
        finite and very small.
        """
        result = greens.Greens(RC, ZC, RC, ZC)
        assert np.isfinite(result)
        assert abs(result) < 1e-3


class TestGreensScaling:
    """Verify that MU0 constant and units are sensible."""

    def test_mu0_value(self):
        expected = 4.0 * np.pi * 1e-7
        np.testing.assert_allclose(MU0, expected, rtol=1e-14)

    def test_psi_positive_for_positive_source(self):
        """Flux from a positive current loop should be positive at a nearby point."""
        psi = greens.Greens(RC, ZC, R1, Z1)
        assert psi > 0
