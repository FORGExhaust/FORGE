"""Tests for ``forge.critical`` — O-point and X-point finding."""

import numpy as np
import pytest

from forge.critical import find_critical


class TestFindCritical:
    """Test the critical-point finder with synthetic flux maps."""

    @staticmethod
    def _make_tokamak_like_grid():
        """Create a flux map that mimics a simple tokamak equilibrium.

        psi = (R - R0)^2 + Z^2 / kappa^2 + an X-point-inducing perturbation.
        This produces at least one O-point near (R0, 0) and one or more
        X-points.
        """
        R0 = 1.0
        kappa = 1.5
        R_1d = np.linspace(0.2, 1.8, 129)
        Z_1d = np.linspace(-1.5, 1.5, 129)
        R, Z = np.meshgrid(R_1d, Z_1d, indexing="ij")
        # base elliptic flux surfaces + dipole-like perturbation
        psi = (R - R0) ** 2 + (Z / kappa) ** 2 - 0.8 * Z ** 2 * (R - R0)
        return R, Z, psi, R0

    def test_finds_opoint(self):
        R, Z, psi, R0 = self._make_tokamak_like_grid()
        opoints, xpoints = find_critical(R, Z, psi, discard_xpoints=False)
        assert len(opoints) >= 1
        Ro, Zo, _ = opoints[0]
        # Primary O-point should be near (R0, 0)
        assert abs(Ro - R0) < 0.15
        assert abs(Zo) < 0.15

    def test_returns_tuples_with_three_elements(self):
        R, Z, psi, _ = self._make_tokamak_like_grid()
        opoints, xpoints = find_critical(R, Z, psi, discard_xpoints=False)
        for pt in opoints + xpoints:
            assert len(pt) == 3  # (R, Z, psi_value)

    def test_xpoint_detection(self):
        """The perturbed map should contain at least one X-point."""
        R, Z, psi, _ = self._make_tokamak_like_grid()
        opoints, xpoints = find_critical(R, Z, psi, discard_xpoints=False)
        assert len(xpoints) >= 1
