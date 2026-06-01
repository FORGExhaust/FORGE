"""Plotting utilities.

This module provides standalone, data-driven plotting functions that can be
used by both the original matplotlib-based workflow and the Panel/Bokeh GUI.
Functions accept plain data (numpy arrays, dicts) and some accept optional
matplotlib axes for embedding into existing figures.
"""

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Equilibrium-level plots
# ---------------------------------------------------------------------------

def plot_profiles(psin_data, pprime_data, ffprime_data, pprime_func, ffprime_func, MU0=4.0 * np.pi * 1e-7):
    """Plot pprime and ffprime profiles.

    Parameters
    ----------
    psin_data : array-like
        Normalised poloidal flux values.
    pprime_data : array-like
        pprime profile data.
    ffprime_data : array-like
        ffprime profile data.
    pprime_func : callable
        Interpolator for pprime.
    ffprime_func : callable
        Interpolator for ffprime.
    MU0 : float
        Vacuum permeability.
    """
    ffprime_check = ffprime_func(psin_data)
    pprime_check = pprime_func(psin_data)

    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel(r"$\rm\Psi_{N}$")
    ax[0].set_ylabel(r"FF'($\rm\Psi_{N})/\mu_{0}$")
    ax[0].scatter(psin_data, ffprime_data / MU0, marker='x', color='r', label='Data')
    ax[0].plot(psin_data, ffprime_check / MU0, 'b', label='Interpolator')
    ax[0].legend()

    ax[1].set_xlabel(r"$\rm\Psi_{N}$")
    ax[1].set_ylabel(r"p'($\rm\Psi_{N})$")
    ax[1].scatter(psin_data, pprime_data, marker='x', color='r', label='Data')
    ax[1].plot(psin_data, pprime_check, 'b', label='Interpolator')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_equilibrium(R_2D, Z_2D, psi_2D, psi_lcfs, wall_R, wall_Z, axis=None):
    """Plot the equilibrium contours.

    Parameters
    ----------
    R_2D, Z_2D : 2D arrays
        Meshgrid arrays.
    psi_2D : 2D array
        Poloidal magnetic flux.
    psi_lcfs : float
        Flux value at the last closed flux surface.
    wall_R, wall_Z : array-like
        Wall coordinates.
    axis : matplotlib Axes or None
        Axes to plot onto.  Created if *None*.

    Returns
    -------
    axis : matplotlib Axes
    """
    if axis is None:
        fig, axis = plt.subplots()
        axis.set_xlabel('R (m)')
        axis.set_ylabel('Z (m)')
        axis.set_aspect('equal')

    axis.contour(R_2D, Z_2D, psi_2D, levels=100)
    axis.contour(R_2D, Z_2D, psi_2D, levels=[psi_lcfs], colors='r')
    axis.plot(wall_R, wall_Z, color='k')

    return axis


def plot_fluxes(R_2D, Z_2D, psi_2D, psi_plas_2D, psi_mach_2D, psi_lcfs, wall_R, wall_Z, R_lcfs, Z_lcfs):
    """Plot the total, plasma and machine flux surfaces.

    Parameters
    ----------
    R_2D, Z_2D : 2D arrays
        Meshgrid arrays.
    psi_2D : 2D array
        Total poloidal magnetic flux.
    psi_plas_2D : 2D array
        Plasma contribution to flux.
    psi_mach_2D : 2D array
        Machine contribution to flux.
    psi_lcfs : float
        Flux at the LCFS.
    wall_R, wall_Z : array-like
        Wall coordinates.
    R_lcfs, Z_lcfs : array-like
        LCFS coordinates.
    """
    fig, ax = plt.subplots(1, 3)

    ax[0].set_aspect('equal')
    ax[0].set_xlabel('R (m)')
    ax[0].set_ylabel('Z (m)')
    ax[0].contour(R_2D, Z_2D, psi_2D, levels=100)
    ax[0].contour(R_2D, Z_2D, psi_2D, levels=[psi_lcfs], colors='r')
    ax[0].plot(wall_R, wall_Z, color='k')
    ax[0].scatter(R_lcfs, Z_lcfs, marker='x', color='b', zorder=2)
    ax[0].title.set_text(r"$\rm\psi$")

    ax[1].set_aspect('equal')
    ax[1].set_xlabel('R (m)')
    ax[1].set_ylabel('Z (m)')
    ax[1].contour(R_2D, Z_2D, psi_plas_2D, levels=100)
    ax[1].plot(wall_R, wall_Z, color='k')
    ax[1].title.set_text(r"$\rm\psi_{plasma}$")

    ax[2].set_aspect('equal')
    ax[2].set_xlabel('R (m)')
    ax[2].set_ylabel('Z (m)')
    ax[2].contour(R_2D, Z_2D, psi_mach_2D, levels=100)
    ax[2].plot(wall_R, wall_Z, color='k')
    ax[2].title.set_text(r"$\rm\psi_{machine}$")

    plt.show()


def plot_fields(R_2D, Z_2D, wall_R, wall_Z, R_lcfs, Z_lcfs,
                total_Br, plasma_Br, machine_Br,
                total_Bz, plasma_Bz, machine_Bz):
    """Plot the total, plasma and machine Br and Bz field components.

    Parameters
    ----------
    R_2D, Z_2D : 2D arrays
        Meshgrid arrays.
    wall_R, wall_Z : array-like
        Wall coordinates.
    R_lcfs, Z_lcfs : array-like
        LCFS coordinates.
    total_Br, plasma_Br, machine_Br : 2D arrays
        Radial field components.
    total_Bz, plasma_Bz, machine_Bz : 2D arrays
        Vertical field components.
    """
    R_min = R_2D[0, 0]
    R_max = R_2D[-1, 0]
    Z_min = Z_2D[0, 0]
    Z_max = Z_2D[0, -1]

    # Br
    fig, ax = plt.subplots(1, 3)
    for a, data, title in zip(ax, [total_Br, plasma_Br, machine_Br],
                              [r"$\rm B_{r}$", r"$\rm B_{r,plasma}$", r"$\rm B_{r,machine}$"]):
        a.set_aspect('equal')
        a.set_xlabel('R (m)')
        a.set_ylabel('Z (m)')
        a.contour(R_2D, Z_2D, data, levels=100)
        a.imshow(data.T, extent=(R_min, R_max, Z_min, Z_max), origin='lower')
        a.plot(wall_R, wall_Z, color='k')
        a.scatter(R_lcfs, Z_lcfs, marker='x', color='b', zorder=2)
        a.title.set_text(title)
    plt.show()

    # Bz
    fig, ax = plt.subplots(1, 3)
    for a, data, title in zip(ax, [total_Bz, plasma_Bz, machine_Bz],
                              [r"$\rm B_{z}$", r"$\rm B_{z,plasma}$", r"$\rm B_{z,machine}$"]):
        a.set_aspect('equal')
        a.set_xlabel('R (m)')
        a.set_ylabel('Z (m)')
        a.contour(R_2D, Z_2D, data, levels=100)
        a.imshow(data.T, extent=(R_min, R_max, Z_min, Z_max), origin='lower')
        a.plot(wall_R, wall_Z, color='k')
        a.scatter(R_lcfs, Z_lcfs, marker='x', color='b', zorder=2)
        a.title.set_text(title)
    plt.show()


# ---------------------------------------------------------------------------
# Optimiser-level plots
# ---------------------------------------------------------------------------

def plot_optimiser_state(
    R_2D, Z_2D, flux_map, psi_lcfs,
    wall_R, wall_Z,
    constraint_points_R, constraint_points_Z,
    divertor_regions=None, divertor_data=None, state_data=None,
    plot_field_lines=False, ax=None, show=False,
):
    """Plot a single equilibrium state snapshot.

    Parameters
    ----------
    R_2D, Z_2D : 2D arrays
        Meshgrid arrays.
    flux_map : 2D array
        The poloidal magnetic flux to contour.
    psi_lcfs : float
        LCFS flux level.
    wall_R, wall_Z : array-like
        Wall boundary.
    constraint_points_R, constraint_points_Z : array-like
        Constraint point locations.
    divertor_regions : list of str, optional
        Names of divertor regions.
    divertor_data : dict, optional
        Per-region data including field lines.
    state_data : dict, optional
        Current state data containing ``"divertors"``.
    plot_field_lines : bool
        Whether to overlay traced field lines.
    ax : matplotlib Axes or None
        If *None*, a new figure is created.
    show : bool
        If True, call ``plt.show()``; otherwise return *ax*.

    Returns
    -------
    ax : matplotlib Axes (only if *show* is False)
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')

    ax.contour(R_2D, Z_2D, flux_map, levels=100, alpha=0.4, colors='k')
    ax.contour(R_2D, Z_2D, flux_map, levels=[psi_lcfs], colors='k')
    ax.plot(wall_R, wall_Z, color='k')
    ax.scatter(constraint_points_R, constraint_points_Z, color='k', marker='s', zorder=2)

    if plot_field_lines and state_data is not None and divertor_regions is not None:
        for region in divertor_regions:
            fl_R = state_data["divertors"][region]["field_line_R"]
            fl_Z = state_data["divertors"][region]["field_line_Z"]
            ax.plot(fl_R, fl_Z)
            ax.scatter(fl_R, fl_Z, marker='x')

    if show:
        plt.show()
    else:
        return ax


def plot_currents_comparison_bar_chart(x_labels, initial_data, optimised_data, scale_kA=True):
    """Bar chart comparing initial and optimised coil currents.

    Parameters
    ----------
    x_labels : list
        Coil names.
    initial_data : list
        Initial per-turn coil currents (A).
    optimised_data : list
        Optimised per-turn coil currents (A).
    scale_kA : bool
        Display in kA if True, MA otherwise.
    """
    factor = 1.0e-03 if scale_kA else 1.0e-06
    initial_abs = [abs(v) * factor for v in initial_data]
    optimised_abs = [abs(v) * factor for v in optimised_data]

    initial_ss = sum(v ** 2 for v in initial_abs)
    optimised_ss = sum(v ** 2 for v in optimised_abs)
    ratio = initial_ss / optimised_ss if optimised_ss != 0 else float('inf')
    if ratio >= 1.0:
        ratio_label = f'Improvement factor: {np.round(ratio, 2)}'
    else:
        ratio_label = f'Worsening factor: {np.round(1.0 / ratio, 2)}'

    x = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, initial_abs, width, label='Initial', color='orange', edgecolor='black')
    ax.bar(x + width / 2, optimised_abs, width, label='Optimised', color='blue', edgecolor='black')
    ax.set_ylabel('|I| (kA)' if scale_kA else '|I| (MA)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_title(
        f'Sum of Squares - Initial: {initial_ss:.2f}, Optimised: {optimised_ss:.2f}'
        f' - {ratio_label}'
    )
    plt.tight_layout()


def plot_optimisation_summary(opt, show=True):
    """Produce a multi-panel summary of a completed optimisation.

    Generates four matplotlib figures:

    1. **Equilibrium comparison** — initial and optimised equilibria side by
       side with wall, separatrix, contours, coils, and traced field lines.
    2. **Cost history** — total cost and per-component breakdown vs iteration,
       plus connection length, temperature, and acceptance rate.
    3. **Coil current comparison** — grouped bar chart of initial vs optimised
       coil currents (kA).
    4. **Flux decomposition** — total, plasma, and machine flux for the
       optimised equilibrium.

    Parameters
    ----------
    opt : forge.optimise.Optimiser
        A completed Optimiser instance (after ``opt.optimise()``).
    show : bool
        If *True* (default) call ``plt.show()`` after creating all figures.
        Set to *False* to allow further customisation before displaying.
    """
    eq = opt.eq
    tokamak_init = opt.tokamak_initial
    wall_R = np.append(tokamak_init.wall_R, tokamak_init.wall_R[0])
    wall_Z = np.append(tokamak_init.wall_Z, tokamak_init.wall_Z[0])
    R_2D, Z_2D = eq.R_2D, eq.Z_2D
    psi_lcfs = eq.psi_lcfs
    incumbent = opt.incumbent_data

    # ------------------------------------------------------------------
    # 1. Equilibrium comparison (initial vs optimised)
    # ------------------------------------------------------------------
    fig_eq, (ax_init, ax_opt) = plt.subplots(1, 2, figsize=(12, 8))

    for ax, title, psi_2D in [
        (ax_init, "Initial Equilibrium", eq.psi_2D),
        (ax_opt, "Optimised Equilibrium", incumbent["psi_2D"]),
    ]:
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title)
        ax.fill(wall_R, wall_Z, color="white", zorder=0)
        ax.contour(R_2D, Z_2D, psi_2D, levels=40, alpha=0.35, colors="gray")
        ax.contour(R_2D, Z_2D, psi_2D, levels=[psi_lcfs], colors="red", linewidths=2)
        ax.plot(wall_R, wall_Z, color="black", linewidth=1.5)

    # Field lines on the optimised panel (from the incumbent, not the last eval)
    for region in opt.divertor_regions:
        colour = opt.divertor_data[region].get("colour", "gray")
        inc_div = incumbent.get("divertors", {}).get(region, {})
        fl_R = inc_div.get("field_line_R")
        fl_Z = inc_div.get("field_line_Z")
        if fl_R is not None and fl_Z is not None:
            ax_opt.plot(fl_R, fl_Z, color=colour, linewidth=1.5, linestyle="--")

    # Coils on both panels
    tokamak_init.plot(ax=ax_init, show=False)
    if hasattr(opt, "optimised_tokamak"):
        opt.optimised_tokamak.plot(ax=ax_opt, show=False)

    # Connection length annotation on optimised panel
    ann_parts = []
    for region in opt.divertor_regions:
        short = opt.divertor_data[region].get("short_label", region)
        cl = incumbent["divertors"][region]["connection_length"]
        ann_parts.append(f"$L_{{//,\\mathrm{{{short}}}}}$ = {cl:.2f} m")
    ax_opt.set_title("Optimised Equilibrium\n" + ",  ".join(ann_parts))

    fig_eq.tight_layout()

    # ------------------------------------------------------------------
    # 2. Cost / tracking history
    # ------------------------------------------------------------------
    n = len(opt.tracking_cost)
    iters = list(range(n))
    inc_iter = incumbent.get("iteration_num")

    fig_hist, axes_hist = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Total + component costs
    ax_c = axes_hist[0]
    ax_c.plot(iters, opt.tracking_cost, color="black", label="Total")
    ax_c.plot(iters, opt.tracking_cost_strike_point_distance, color="red", label="Strike")
    ax_c.plot(iters, opt.tracking_cost_connection_length, color="green", label="Conn. length")
    ax_c.plot(iters, opt.tracking_cost_coil_currents, color="orange", label="Coil I²")
    ax_c.plot(iters, opt.tracking_cost_xpoint_regions, color="blue", label="XPT")
    if inc_iter is not None:
        ax_c.axvline(inc_iter, color="black", linestyle="--", alpha=0.5)
    ax_c.set_ylabel("Cost")
    ax_c.legend(fontsize=8, ncol=5)
    ax_c.set_title("Optimisation History")

    # Connection length per region
    ax_cl = axes_hist[1]
    for region in opt.divertor_regions:
        colour = opt.divertor_data[region].get("colour", "gray")
        ax_cl.plot(iters, opt.tracking_connection_length[region], color=colour, label=region)
    if inc_iter is not None:
        ax_cl.axvline(inc_iter, color="black", linestyle="--", alpha=0.5)
    ax_cl.set_ylabel("$L_{//}$ (m)")
    ax_cl.legend(fontsize=8)

    # Temperature
    ax_t = axes_hist[2]
    ax_t.plot(iters, opt.tracking_temperature, color="blue")
    if inc_iter is not None:
        ax_t.axvline(inc_iter, color="black", linestyle="--", alpha=0.5)
    ax_t.set_ylabel("Temperature")

    # Acceptance rate
    ax_a = axes_hist[3]
    ax_a.scatter(iters, opt.tracking_acceptance_rate, color="blue", s=0.5)
    if hasattr(opt, "threshold_acceptance_rate"):
        ax_a.axhline(opt.threshold_acceptance_rate, color="red", linestyle="--", alpha=0.7)
    if inc_iter is not None:
        ax_a.axvline(inc_iter, color="black", linestyle="--", alpha=0.5)
    ax_a.set_ylabel("Acceptance rate")
    ax_a.set_xlabel("Iteration")

    fig_hist.tight_layout()

    # ------------------------------------------------------------------
    # 3. Coil current comparison bar chart
    # ------------------------------------------------------------------
    if hasattr(opt, "optimised_tokamak"):
        coil_names = list(opt.tokamak_opt.coilset.keys())
        initial_currents = opt.tokamak_initial.get_currents()
        optimised_currents = opt.optimised_tokamak.get_currents()
        plot_currents_comparison_bar_chart(
            coil_names, initial_currents, optimised_currents,
        )

    # ------------------------------------------------------------------
    # 4. Flux decomposition of the optimised equilibrium
    # ------------------------------------------------------------------
    if hasattr(opt, "optimised_eq"):
        oeq = opt.optimised_eq
        plot_fluxes(
            oeq.R_2D, oeq.Z_2D,
            oeq.psi_2D, oeq.psi_plas_2D, oeq.psi_mach_2D,
            oeq.psi_lcfs,
            tokamak_init.wall_R, tokamak_init.wall_Z,
            oeq.R_lcfs, oeq.Z_lcfs,
        )

    if show:
        plt.show()
    plt.show()
