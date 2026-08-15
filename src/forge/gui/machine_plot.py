"""Shared Bokeh layers for drawing a machine and its equilibrium.

Provides the static background common to the editor tabs: vessel interior,
flux contours, separatrix, wall outline, coils and coil masks.  The wall
layers are optional so that a tab which edits the wall can own that renderer
itself and drive it from its own data source.
"""

import numpy as np
from bokeh.models import ColumnDataSource

from forge.gui.contours import contour_xy_lists
from forge.utils import orthogonalised_convex_hull_from_rects

COIL_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


class MachineLayers:
    """Bokeh renderers and data sources for a machine/equilibrium background.

    Renderers are added to *fig* in the same order used by the Setup and
    Geometry tabs: vessel interior, flux contours, separatrix, wall outline,
    then coils.

    Parameters
    ----------
    fig : bokeh.plotting.figure
        Figure to add the renderers to.
    include_wall : bool
        If True, the vessel interior and wall outline are created and updated
        by :meth:`update`.  Set to False when the caller draws an editable
        wall from its own source.

    Attributes
    ----------
    wall_fill_source, wall_source : bokeh.models.ColumnDataSource or None
        Wall interior and outline sources, or None when *include_wall* is
        False.
    contour_source, lcfs_source : bokeh.models.ColumnDataSource
        Flux contour and separatrix line sources.
    mask_data_cache : dict
        Most recently computed coil mask polygons, so a tab can toggle mask
        visibility without recomputing the hulls.
    """

    def __init__(self, fig, include_wall=True):
        self.fig = fig
        self.include_wall = include_wall

        self.wall_fill_source = None
        self.wall_source = None
        if include_wall:
            self.wall_fill_source = ColumnDataSource(data=dict(R=[], Z=[]))
            self.wall_source = ColumnDataSource(data=dict(R=[], Z=[]))
            fig.patch("R", "Z", source=self.wall_fill_source,
                      fill_color="white", line_color=None)

        self.contour_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.lcfs_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.coil_source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self.fil_cross_source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self.fil_rect_source = ColumnDataSource(
            data=dict(xs=[], ys=[], fill=[], edge=[], name=[])
        )
        self.mask_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.mask_data_cache = {"xs": [], "ys": []}

        fig.multi_line("xs", "ys", source=self.contour_source,
                       line_alpha=0.35, line_color="gray")
        fig.multi_line("xs", "ys", source=self.lcfs_source,
                       line_color="red", line_width=2)

        if include_wall:
            fig.line("R", "Z", source=self.wall_source,
                     line_color="black", line_width=2)

        fig.scatter("x", "y", source=self.coil_source, color="color", size=10,
                    marker="circle")
        fig.scatter("x", "y", source=self.fil_cross_source, color="color", size=8,
                    marker="x")
        fig.patches("xs", "ys", source=self.fil_rect_source, fill_color="fill",
                    line_color="edge", line_width=0.3)
        fig.patches("xs", "ys", source=self.mask_source, fill_color="orange",
                    fill_alpha=1.0, line_color="black", line_width=1.0)

        # Extents of the coils, cached so view bounds can be recomputed
        # without rebuilding the coil geometry.
        self._coil_R = []
        self._coil_Z = []

    def update(self, eq, tokamak, show_masks=True):
        """Populate every layer from an equilibrium and machine.

        Parameters
        ----------
        eq : forge.equilibrium.Equilibrium
            Equilibrium supplying the flux map and separatrix.
        tokamak : forge.machine.Machine
            Machine supplying the wall and coilset.
        show_masks : bool
            Whether coil mask polygons are drawn.  The polygons are cached in
            :attr:`mask_data_cache` regardless.
        """
        if self.include_wall:
            self.set_wall(tokamak.wall_R, tokamak.wall_Z)

        self._update_contours(eq)
        self._update_coils(tokamak)
        self._update_masks(tokamak, show_masks=show_masks)

    def set_wall(self, wall_R, wall_Z):
        """Update the wall interior and outline sources."""
        if not self.include_wall:
            raise RuntimeError("These layers were created with include_wall=False.")
        data = dict(R=[float(v) for v in wall_R], Z=[float(v) for v in wall_Z])
        self.wall_fill_source.data = dict(data)
        self.wall_source.data = dict(data)

    def set_masks_visible(self, visible):
        """Show or hide the cached coil mask polygons."""
        if visible and self.mask_data_cache["xs"]:
            self.mask_source.data = dict(self.mask_data_cache)
        else:
            self.mask_source.data = dict(xs=[], ys=[])

    def _update_contours(self, eq):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(eq.R_2D, eq.Z_2D, eq.psi_2D, levels=60)
        xs, ys = contour_xy_lists(cs)
        cs_sep = ax_tmp.contour(eq.R_2D, eq.Z_2D, eq.psi_2D, levels=[eq.psi_lcfs])
        sep_xs, sep_ys = contour_xy_lists(cs_sep)
        plt.close(fig_tmp)

        self.contour_source.data = dict(xs=xs, ys=ys)
        self.lcfs_source.data = dict(xs=sep_xs, ys=sep_ys)

    def _update_coils(self, tokamak):
        from forge.magnets import Circuit

        pt_x, pt_y, pt_c, pt_n = [], [], [], []
        cr_x, cr_y, cr_c, cr_n = [], [], [], []
        rc_xs, rc_ys, rc_f, rc_e, rc_n = [], [], [], [], []

        def _add_coil(coil, colour, label):
            if hasattr(coil, "R_filaments"):
                fill_col = getattr(coil, "fill_colour", "orange")
                edge_col = getattr(coil, "edge_colour", "grey")
                if coil.dR is None and coil.dZ is None:
                    for Rf, Zf in zip(coil.R_filaments, coil.Z_filaments):
                        cr_x.append(float(Rf)); cr_y.append(float(Zf))
                        cr_c.append(edge_col); cr_n.append(label)
                else:
                    for Rf, Zf in zip(coil.R_filaments, coil.Z_filaments):
                        r1, r2 = Rf - 0.5 * coil.dR, Rf + 0.5 * coil.dR
                        z1, z2 = Zf - 0.5 * coil.dZ, Zf + 0.5 * coil.dZ
                        rc_xs.append([r1, r2, r2, r1, r1])
                        rc_ys.append([z1, z1, z2, z2, z1])
                        rc_f.append(fill_col); rc_e.append(edge_col); rc_n.append(label)
            elif hasattr(coil, "Z_min") and hasattr(coil, "Z_max"):
                pt_x.extend([coil.R, coil.R]); pt_y.extend([coil.Z_min, coil.Z_max])
                pt_c.extend([colour, colour]); pt_n.extend([label, label])
            else:
                pt_x.append(coil.R); pt_y.append(coil.Z)
                pt_c.append(colour); pt_n.append(label)

        for i, (name, entry) in enumerate(tokamak.coilset.items()):
            colour = COIL_COLOURS[i % len(COIL_COLOURS)]
            if isinstance(entry, Circuit):
                for coil_dict in entry.coilset.values():
                    _add_coil(coil_dict["magnet"], colour, name)
            else:
                _add_coil(entry, colour, name)

        self.coil_source.data = dict(x=pt_x, y=pt_y, color=pt_c, name=pt_n)
        self.fil_cross_source.data = dict(x=cr_x, y=cr_y, color=cr_c, name=cr_n)
        self.fil_rect_source.data = dict(
            xs=rc_xs, ys=rc_ys, fill=rc_f, edge=rc_e, name=rc_n
        )

        coil_R = list(pt_x) + list(cr_x)
        coil_Z = list(pt_y) + list(cr_y)
        for rxs in rc_xs:
            coil_R.extend(rxs)
        for rys in rc_ys:
            coil_Z.extend(rys)
        self._coil_R = coil_R
        self._coil_Z = coil_Z

    def _update_masks(self, tokamak, show_masks=True):
        from forge.magnets import Circuit

        mask_xs, mask_ys = [], []
        for _name, entry in tokamak.coilset.items():
            if isinstance(entry, Circuit):
                coils = [cd["magnet"] for cd in entry.coilset.values()]
            else:
                coils = [entry]
            for coil in coils:
                if not hasattr(coil, "R_filaments"):
                    continue
                if coil.dR is None or coil.dZ is None:
                    continue
                xc = [float(r) for r in coil.R_filaments]
                yc = [float(z) for z in coil.Z_filaments]
                dx = [float(coil.dR)] * len(xc)
                dy = [float(coil.dZ)] * len(xc)
                hx, hy = orthogonalised_convex_hull_from_rects(xc, yc, dx, dy, closed=True)
                if hx:
                    mask_xs.append(hx)
                    mask_ys.append(hy)

        self.mask_data_cache = {"xs": mask_xs, "ys": mask_ys}
        self.set_masks_visible(show_masks)

    def view_bounds(self, wall_R, wall_Z, margin=0.5):
        """Return ``(r_min, r_max, z_min, z_max)`` covering wall and coils."""
        all_R = [float(v) for v in wall_R] + self._coil_R
        all_Z = [float(v) for v in wall_Z] + self._coil_Z
        if not all_R or not all_Z:
            return None
        return (
            float(np.min(all_R)) - margin,
            float(np.max(all_R)) + margin,
            float(np.min(all_Z)) - margin,
            float(np.max(all_Z)) + margin,
        )


def apply_view_bounds(fig, bounds, fixed_height=900):
    """Set the figure ranges, tick density and aspect-matched width.

    Parameters
    ----------
    fig : bokeh.plotting.figure
        Figure to resize.
    bounds : tuple
        ``(r_min, r_max, z_min, z_max)`` as returned by
        :meth:`MachineLayers.view_bounds`.
    fixed_height : int
        Height in pixels; the width is derived from the data aspect ratio so
        that the plot is not distorted.
    """
    if bounds is None:
        return

    r_min, r_max, z_min, z_max = bounds

    fig.x_range.start = fig.x_range.reset_start = r_min
    fig.x_range.end = fig.x_range.reset_end = r_max
    fig.y_range.start = fig.y_range.reset_start = z_min
    fig.y_range.end = fig.y_range.reset_end = z_max

    r_extent = r_max - r_min
    tick_interval = 0.5 if r_extent > 2 else (0.25 if r_extent > 1 else 0.1)
    fig.xaxis.ticker.desired_num_ticks = max(3, int(r_extent / tick_interval) + 1)
    fig.yaxis.ticker.desired_num_ticks = max(3, int((z_max - z_min) / tick_interval) + 1)

    data_width = r_max - r_min
    data_height = z_max - z_min
    if data_height > 0:
        fig.width = max(200, int(fixed_height * data_width / data_height))
    fig.height = fixed_height
