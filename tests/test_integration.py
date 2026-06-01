"""Integration tests for ``forge.machine`` and ``forge.equilibrium``.

These tests load real data files shipped with the package and exercise
the higher-level Machine and Equilibrium classes end-to-end.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMachineFromData:
    """Build a Machine from a real JSON file and validate it."""

    def test_machine_creation(self, sample_magnets_data, sample_eq_data):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        mach = Machine(
            magnets_data=coils_data,
            circuits=circuits_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
        )
        assert mach is not None

    def test_machine_has_coils(self, sample_magnets_data, sample_eq_data):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        mach = Machine(
            magnets_data=coils_data,
            circuits=circuits_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
        )
        names = mach.get_coil_names()
        assert len(names) > 0

    def test_machine_psi_finite(self, sample_magnets_data, sample_eq_data):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        mach = Machine(
            magnets_data=coils_data,
            circuits=circuits_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
        )
        psi = mach.psiRZ(0.5, 0.0)
        assert np.isfinite(psi)

    def test_machine_Br_Bz_finite(self, sample_magnets_data, sample_eq_data):
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        mach = Machine(
            magnets_data=coils_data,
            circuits=circuits_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
        )
        assert np.isfinite(mach.BrRZ(0.5, 0.0))
        assert np.isfinite(mach.BzRZ(0.5, 0.0))


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEquilibriumFromData:
    """Build an Equilibrium from real GEQDSK + Machine data."""

    @pytest.fixture
    def eq_and_machine(self, sample_eq_data, sample_magnets_data):
        from forge.equilibrium import Equilibrium
        from forge.machine import Machine

        coils_data, circuits_data, _ = sample_magnets_data
        mach = Machine(
            magnets_data=coils_data,
            circuits=circuits_data,
            wall_R=sample_eq_data["wall_R"],
            wall_Z=sample_eq_data["wall_Z"],
        )
        eq = Equilibrium(eq_data=sample_eq_data, tokamak=mach)
        return eq, mach

    def test_equilibrium_creation(self, eq_and_machine):
        eq, _ = eq_and_machine
        assert eq is not None

    def test_psi_on_grid_is_finite(self, eq_and_machine):
        eq, _ = eq_and_machine
        # Evaluate psi at a point well inside the grid
        R_mid = 0.5 * (eq.R_min + eq.R_max)
        Z_mid = 0.5 * (eq.Z_min + eq.Z_max)
        psi = eq.psiRZ(R_mid, Z_mid)
        assert np.isfinite(psi)

    def test_magnetic_field_components_finite(self, eq_and_machine):
        eq, _ = eq_and_machine
        R_mid = 0.5 * (eq.R_min + eq.R_max)
        Z_mid = 0.5 * (eq.Z_min + eq.Z_max)
        assert np.isfinite(eq.BrRZ(R_mid, Z_mid))
        assert np.isfinite(eq.BzRZ(R_mid, Z_mid))
        assert np.isfinite(eq.BpolRZ(R_mid, Z_mid))

    def test_Bpol_positive(self, eq_and_machine):
        """Poloidal field magnitude should always be non-negative."""
        eq, _ = eq_and_machine
        R_mid = 0.5 * (eq.R_min + eq.R_max)
        Z_mid = 0.5 * (eq.Z_min + eq.Z_max)
        assert eq.BpolRZ(R_mid, Z_mid) >= 0.0
