"""Setup tab — load GEQDSK & magnets, build Machine & Equilibrium, configure constraints."""

import logging

import numpy as np
import panel as pn
from bokeh.layouts import row as bk_row
from bokeh.models import ColumnDataSource, Range1d, LinearColorMapper, HoverTool
from bokeh.plotting import figure as bk_figure

from forge.equilibrium import Equilibrium
from forge.io import read_geqdsk, read_magnets
from forge.machine import Machine
from forge.utils import orthogonalised_convex_hull_from_rects

logger = logging.getLogger(__name__)

# Colours for up to 12 coils/circuits — recycled if more are present
_COIL_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


class SetupTab:
    """Panel component for the machine / equilibrium setup phase."""

    def __init__(self, shared_state, geometry_tab=None):
        self.state = shared_state
        self._geometry_tab = geometry_tab

        # --- Widgets ---
        self.geqdsk_input = pn.widgets.FileInput(accept=".geqdsk,.GEQDSK,.eqdsk", name="GEQDSK file")
        self.magnets_input = pn.widgets.FileInput(accept=".json", name="Magnets JSON file")
        self.geqdsk_label = pn.pane.Markdown("**GEQDSK equilibrium file** — select a `.geqdsk` / `.eqdsk` file")
        self.magnets_label = pn.pane.Markdown("**Magnets JSON file** — select the machine magnets `.json` file")
        self.load_btn = pn.widgets.Button(name="Load & Build", button_type="primary")
        self.status = pn.pane.Alert("Upload a GEQDSK and magnets JSON to begin.", alert_type="info")

        # Coils table (populated after loading)
        self.coils_table = pn.pane.HTML("<em>No machine loaded yet.</em>")

        # Equilibrium info panel (right of plot)
        self.eq_info_pane = pn.pane.HTML("<em>No equilibrium loaded yet.</em>", width=320)

        # 1D profile figures (pprime, ffprime/mu0, pressure, fpol)
        self._profile_sources = {}
        self._profile_figs = {}
        profile_height = 200
        profile_width = 300
        for name, ylabel, xlabel in [("pprime", "p' (kPa/Wb)", "Normalised flux"),
                                      ("ffprime", "FF'/mu0 (kA/m)", "Normalised flux"),
                                      ("pressure", "p (kPa)", "Normalised flux"),
                                      ("fpol", "F (T m)", "Normalised flux")]:
            src = ColumnDataSource(data=dict(x=[], y=[]))
            self._profile_sources[name] = src
            fig = bk_figure(
                title=ylabel, x_axis_label=xlabel, y_axis_label=ylabel,
                height=profile_height, width=profile_width,
                tools="pan,wheel_zoom,reset",
            )
            fig.title.text_font_style = "normal"
            fig.title.text_font_size = "9pt"
            fig.xaxis.axis_label_text_font_style = "normal"
            fig.yaxis.axis_label_text_font_style = "normal"
            fig.xaxis.axis_label_text_font_size = "8pt"
            fig.yaxis.axis_label_text_font_size = "8pt"
            fig.line("x", "y", source=src, line_width=1.5, color="navy")
            self._profile_figs[name] = fig

        # Flux decomposition figures (total, plasma, coil) with companion colorbar figures
        self._decomp_figures = {}
        self._decomp_sources = {}
        self._decomp_cbar_figs = {}
        self._decomp_cbar_srcs = {}
        for label in ("Total \u03c8", "Plasma \u03c8", "Coil \u03c8"):
            f = bk_figure(
                title=label,
                x_axis_label="R (m)",
                y_axis_label="Z (m)",
                match_aspect=True,
                width=420,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset",
                background_fill_color="#d9d9d9",
            )
            f.xaxis.axis_label_text_font_style = "normal"
            f.yaxis.axis_label_text_font_style = "normal"
            # Wall fill + outline, contour multi_line, and separatrix overlay
            wf_src = ColumnDataSource(data=dict(R=[], Z=[]))
            wo_src = ColumnDataSource(data=dict(R=[], Z=[]))
            contour_src = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
            sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
            f.patch("R", "Z", source=wf_src, fill_color="white", line_color=None)
            f.multi_line("xs", "ys", source=contour_src, line_color="color", line_alpha=0.6, line_width=0.8)
            f.multi_line("xs", "ys", source=sep_src, line_color="red", line_width=2)
            f.line("R", "Z", source=wo_src, line_color="black", line_width=1.5)
            # Coil glyphs (same three types as the main eq plot)
            d_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
            d_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
            d_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
            d_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))
            f.scatter("x", "y", source=d_pt_src, color="color", size=7,
                      marker="circle")
            f.scatter("x", "y", source=d_cr_src, color="color", size=6,
                      marker="x")
            f.patches("xs", "ys", source=d_rc_src, fill_color="fill",
                      line_color="edge", line_width=0.3)
            f.patches("xs", "ys", source=d_mask_src, fill_color="orange",
                      line_color="black", line_width=1.0, fill_alpha=1.0)

            # Transparent image for hover readout of ψ(R,Z)
            hover_img_src = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
            hover_renderer = f.image(
                "image", source=hover_img_src, x="x", y="y", dw="dw", dh="dh",
                color_mapper=LinearColorMapper(palette=["#00000000", "#00000000"], low=0, high=1),
            )
            hover_tool = HoverTool(
                renderers=[hover_renderer],
                tooltips=[
                    ("R", "$x{0.4f} m"),
                    ("Z", "$y{0.4f} m"),
                    ("ψ", "@image{0.6f} Wb"),
                ],
            )
            f.add_tools(hover_tool)

            # Companion colorbar figure — y-axis label auto-centres vertically
            cf = bk_figure(
                width=90,
                height=400,
                y_range=Range1d(0, 1),
                x_range=Range1d(0, 1),
                y_axis_location="right",
                y_axis_label="\u03c8 (Wb)",
                toolbar_location=None,
                min_border_left=2,
                min_border_right=5,
                outline_line_color=None,
            )
            cf.xaxis.visible = False
            cf.yaxis.axis_label_text_font_style = "normal"
            cf.background_fill_color = None
            cf.xgrid.visible = False
            cf.ygrid.visible = False
            cbar_src = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
            cf.image("image", source=cbar_src, x="x", y="y", dw="dw", dh="dh",
                     color_mapper=LinearColorMapper(palette="Viridis256", low=0, high=1))

            self._decomp_figures[label] = f
            self._decomp_sources[label] = {
                "wall_fill": wf_src, "wall_outline": wo_src,
                "contour": contour_src, "separatrix": sep_src,
                "coil_pt": d_pt_src, "coil_cross": d_cr_src,
                "coil_rect": d_rc_src, "mask": d_mask_src,
                "hover_img": hover_img_src,
            }
            self._decomp_cbar_figs[label] = cf
            self._decomp_cbar_srcs[label] = cbar_src

        # Equilibrium plot placeholder
        self.eq_figure = bk_figure(
            title="Equilibrium",
            x_axis_label="R (m)",
            y_axis_label="Z (m)",
            x_range=Range1d(0, 1),
            y_range=Range1d(0, 1),
            match_aspect=True,
            width=560,
            height=900,
            tools="pan,wheel_zoom,box_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        # Wall interior fill (white polygon)
        self._eq_wall_fill_source = ColumnDataSource(data=dict(R=[], Z=[]))
        self._eq_wall_source = ColumnDataSource(data=dict(R=[], Z=[]))
        self._eq_contour_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._eq_lcfs_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        # Point / shaped coils — circles
        self._eq_coil_source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        # FilamentPointCoil without dR/dZ — small crosses
        self._eq_fil_cross_source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        # FilamentPointCoil with dR/dZ — filled rectangles (patches)
        self._eq_fil_rect_source = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        # Coil masks — orthogonalised convex hulls around grouped filament rectangles
        self._eq_mask_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        # Constraint overlay data sources
        self._eq_flux_constraint_source = ColumnDataSource(data=dict(x=[], y=[]))
        self._eq_xpt_constraint_source = ColumnDataSource(data=dict(x=[], y=[]))

        # Render order: wall fill (white) first, then contours, then separatrix, then wall outline, then coils
        self.eq_figure.patch("R", "Z", source=self._eq_wall_fill_source,
                             fill_color="white", line_color=None)
        self.eq_figure.multi_line("xs", "ys", source=self._eq_contour_source, line_alpha=0.35, line_color="gray")
        self.eq_figure.multi_line("xs", "ys", source=self._eq_lcfs_source, line_color="red", line_width=2)
        self.eq_figure.line("R", "Z", source=self._eq_wall_source, line_color="black", line_width=2)
        self.eq_figure.scatter("x", "y", source=self._eq_coil_source, color="color", size=10,
                               marker="circle")
        self.eq_figure.scatter("x", "y", source=self._eq_fil_cross_source, color="color", size=8,
                               marker="x")
        self.eq_figure.patches("xs", "ys", source=self._eq_fil_rect_source, fill_color="fill",
                               line_color="edge", line_width=0.3)
        self.eq_figure.patches("xs", "ys", source=self._eq_mask_source, fill_color="orange",
                               line_color="black", line_width=1.0, fill_alpha=1.0)

        # Transparent image for hover readout of ψ(R,Z)
        self._eq_hover_img_source = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        eq_hover_renderer = self.eq_figure.image(
            "image", source=self._eq_hover_img_source, x="x", y="y", dw="dw", dh="dh",
            color_mapper=LinearColorMapper(palette=["#00000000", "#00000000"], low=0, high=1),
        )
        eq_hover_tool = HoverTool(
            renderers=[eq_hover_renderer],
            tooltips=[
                ("R", "$x{0.4f} m"),
                ("Z", "$y{0.4f} m"),
                ("ψ", "@image{0.6f} Wb"),
            ],
        )
        self.eq_figure.add_tools(eq_hover_tool)

        # Constraint overlays — ψ constraints: blue squares with black edges
        self.eq_figure.scatter("x", "y", source=self._eq_flux_constraint_source,
                               marker="square", size=8,
                               fill_color="#1f77b4", line_color="black", line_width=1.0,
                               legend_label="ψ constraint")
        # Constraint overlays — X-point constraints: red triangles with black edges
        self.eq_figure.scatter("x", "y", source=self._eq_xpt_constraint_source,
                               marker="triangle", size=10,
                               fill_color="red", line_color="black", line_width=1.0,
                               legend_label="X-point constraint")

        # Non-italic axis labels
        self.eq_figure.xaxis.axis_label_text_font_style = "normal"
        self.eq_figure.yaxis.axis_label_text_font_style = "normal"

        # --- Coil mask toggle (hidden until filament coils with dR/dZ are loaded) ---
        self.show_masks = pn.widgets.Checkbox(name="Show coil masks", value=True)
        self.show_masks.param.watch(self._on_mask_toggle, "value")
        self._mask_data_cache = {}   # label -> {"xs": [...], "ys": [...]}
        self._eq_mask_data_cache = {"xs": [], "ys": []}  # for the eq figure
        self._has_maskable_coils = False

        # --- Null-space info (N coils, N constraints, null-space size) ---
        self._null_space_info = pn.pane.HTML("<em>Load files to see constraint summary.</em>")

        # --- Annealing constraints configurator ---
        self.constrain_omp = pn.widgets.Checkbox(name="Constrain OMP", value=True)
        self.constrain_imp = pn.widgets.Checkbox(name="Constrain IMP", value=True)
        self.constrain_upper_point = pn.widgets.Checkbox(name="Constrain upper point", value=False)
        self.constrain_lower_point = pn.widgets.Checkbox(name="Constrain lower point", value=False)
        self.constrain_ur = pn.widgets.Checkbox(name="Constrain upper-right quadrant", value=True)
        self.n_ur = pn.widgets.IntSlider(name="N upper-right", start=1, end=5, value=1)
        self.constrain_ul = pn.widgets.Checkbox(name="Constrain upper-left quadrant", value=True)
        self.n_ul = pn.widgets.IntSlider(name="N upper-left", start=1, end=5, value=1)
        self.constrain_ll = pn.widgets.Checkbox(name="Constrain lower-left quadrant", value=True)
        self.n_ll = pn.widgets.IntSlider(name="N lower-left", start=1, end=5, value=1)
        self.constrain_lr = pn.widgets.Checkbox(name="Constrain lower-right quadrant", value=True)
        self.n_lr = pn.widgets.IntSlider(name="N lower-right", start=1, end=5, value=1)
        self.xpoint_constraint = pn.widgets.Select(
            name="X-point constraint",
            options=["primary", "lower", "upper"],
            value="primary",
        )

        # Wiring
        self.load_btn.on_click(self._on_load)
        # Re-plot constraints when any constraint widget changes
        _constraint_widgets = [
            self.constrain_omp, self.constrain_imp,
            self.constrain_upper_point, self.constrain_lower_point,
            self.constrain_ur, self.n_ur,
            self.constrain_ul, self.n_ul,
            self.constrain_ll, self.n_ll,
            self.constrain_lr, self.n_lr,
            self.xpoint_constraint,
        ]
        for w in _constraint_widgets:
            w.param.watch(self._on_constraint_change, "value")

    # ------------------------------------------------------------------
    def _on_load(self, event):
        """Handle the Load & Build button click."""
        if self.geqdsk_input.value is None or self.magnets_input.value is None:
            self.status.object = "Please upload both a GEQDSK and a magnets JSON file."
            self.status.alert_type = "warning"
            return

        try:
            self.status.object = "Loading files…"
            self.status.alert_type = "info"

            # Write uploaded bytes to temp files (freeqdsk needs a file path)
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".geqdsk", delete=False) as f:
                f.write(self.geqdsk_input.value)
                geqdsk_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                f.write(self.magnets_input.value)
                magnets_path = f.name

            eq_data = read_geqdsk(geqdsk_path)
            coils_data, circuits_data, suggested_circuits = read_magnets(magnets_path)

            os.unlink(geqdsk_path)
            os.unlink(magnets_path)

            tokamak = Machine(
                magnets_data=coils_data,
                wall_R=eq_data["wall_R"],
                wall_Z=eq_data["wall_Z"],
                circuits=circuits_data,
            )

            eq = Equilibrium(eq_data=eq_data, tokamak=tokamak, calculate_flux_from_coils=True)

            # Store in shared state for other tabs
            self.state["eq_data"] = eq_data
            self.state["coils_data"] = coils_data
            self.state["circuits_data"] = circuits_data
            self.state["tokamak"] = tokamak
            self.state["eq"] = eq

            # Batch all Bokeh model updates into a single websocket message.
            # Without this, each ColumnDataSource assignment triggers its own
            # patch over the wire, and rapid-fire patches can arrive after the
            # document has started processing earlier ones, producing harmless
            # but noisy "Dropping a patch" warnings.
            doc = pn.state.curdoc
            if doc is not None:
                doc.hold("combine")

            try:
                self._update_eq_plot(eq, tokamak)

                # Only expose lower/upper X-point options for DND equilibria
                if eq.DND:
                    self.xpoint_constraint.options = ["primary", "lower", "upper"]
                else:
                    self.xpoint_constraint.options = ["primary"]
                    self.xpoint_constraint.value = "primary"

                self._update_constraint_overlay(eq)
                self._update_coils_table(tokamak)
                self._update_eq_info(eq, tokamak)
                self._update_profile_plots(eq)
                self._update_decomp_plots(eq, tokamak)
                self._update_null_space_info(tokamak)
                # Auto-refresh geometry tab (inside the hold block so its
                # model updates are also batched).
                if self._geometry_tab is not None:
                    self._geometry_tab._on_refresh(None)
            finally:
                if doc is not None:
                    doc.unhold()

            self.status.object = "Loaded successfully."
            self.status.alert_type = "success"

        except Exception as exc:
            logger.exception("Failed to load files")
            self.status.object = f"Error: {exc}"
            self.status.alert_type = "danger"

    # ------------------------------------------------------------------
    @staticmethod
    def _collect_coil_data(tokamak):
        """Collect coil glyph data and mask polygons from a Machine.

        Returns
        -------
        coil_data : dict
            Keys: pt_x/y/c/n (circles), cr_x/y/c/n (crosses),
            rc_xs/ys/f/e/n (rectangles).
        mask_xs, mask_ys : lists
            Orthogonalised convex‑hull outlines for each group of
            FilamentPointCoils that have dR/dZ set.
        has_maskable : bool
            True if at least one maskable group was found.
        """
        from forge.magnets import Circuit

        pt_x, pt_y, pt_c, pt_n = [], [], [], []
        cr_x, cr_y, cr_c, cr_n = [], [], [], []
        rc_xs, rc_ys, rc_f, rc_e, rc_n = [], [], [], [], []
        mask_xs, mask_ys = [], []

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
                        rc_f.append(fill_col); rc_e.append(edge_col)
                        rc_n.append(label)
            elif hasattr(coil, "Z_min") and hasattr(coil, "Z_max"):
                pt_x.extend([coil.R, coil.R])
                pt_y.extend([coil.Z_min, coil.Z_max])
                pt_c.extend([colour, colour])
                pt_n.extend([label, label])
            else:
                pt_x.append(coil.R); pt_y.append(coil.Z)
                pt_c.append(colour); pt_n.append(label)

        def _collect_masks(coil):
            """If coil is a FilamentPointCoil with dR/dZ, compute hull mask."""
            if not hasattr(coil, "R_filaments"):
                return
            if coil.dR is None or coil.dZ is None:
                return
            xc = [float(r) for r in coil.R_filaments]
            yc = [float(z) for z in coil.Z_filaments]
            dx = [float(coil.dR)] * len(xc)
            dy = [float(coil.dZ)] * len(xc)
            hx, hy = orthogonalised_convex_hull_from_rects(xc, yc, dx, dy, closed=True)
            if hx:
                mask_xs.append(hx)
                mask_ys.append(hy)

        for i, (name, entry) in enumerate(tokamak.coilset.items()):
            colour = _COIL_COLOURS[i % len(_COIL_COLOURS)]
            if isinstance(entry, Circuit):
                for coil_dict in entry.coilset.values():
                    _add_coil(coil_dict["magnet"], colour, name)
                    _collect_masks(coil_dict["magnet"])
            else:
                _add_coil(entry, colour, name)
                _collect_masks(entry)

        coil_data = dict(
            pt_x=pt_x, pt_y=pt_y, pt_c=pt_c, pt_n=pt_n,
            cr_x=cr_x, cr_y=cr_y, cr_c=cr_c, cr_n=cr_n,
            rc_xs=rc_xs, rc_ys=rc_ys, rc_f=rc_f, rc_e=rc_e, rc_n=rc_n,
        )
        has_maskable = len(mask_xs) > 0
        return coil_data, mask_xs, mask_ys, has_maskable

    # ------------------------------------------------------------------
    def _on_mask_toggle(self, event):
        """Show / hide coil mask overlays on all plots."""
        show = event.new
        # Decomp figures
        for label in self._decomp_sources:
            src = self._decomp_sources[label]["mask"]
            if show:
                cached = self._mask_data_cache.get(label, {"xs": [], "ys": []})
                src.data = dict(xs=cached["xs"], ys=cached["ys"])
            else:
                src.data = dict(xs=[], ys=[])
        # Eq figure
        if show:
            self._eq_mask_source.data = dict(
                xs=self._eq_mask_data_cache["xs"],
                ys=self._eq_mask_data_cache["ys"],
            )
        else:
            self._eq_mask_source.data = dict(xs=[], ys=[])

    # ------------------------------------------------------------------
    def _update_eq_plot(self, eq, tokamak):
        """Refresh the Bokeh equilibrium figure after loading."""
        # Wall outline and white interior fill
        self._eq_wall_fill_source.data = dict(R=list(tokamak.wall_R), Z=list(tokamak.wall_Z))
        self._eq_wall_source.data = dict(R=list(tokamak.wall_R), Z=list(tokamak.wall_Z))

        # Extract contour lines via matplotlib (Agg backend, no display)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(eq.R_2D, eq.Z_2D, eq.psi_2D, levels=60)
        xs_all, ys_all = [], []
        for collection in cs.collections:
            for path in collection.get_paths():
                verts = path.vertices
                xs_all.append(verts[:, 0].tolist())
                ys_all.append(verts[:, 1].tolist())

        # Separatrix contour at psi_lcfs
        cs_sep = ax_tmp.contour(eq.R_2D, eq.Z_2D, eq.psi_2D, levels=[eq.psi_lcfs])
        sep_xs, sep_ys = [], []
        for collection in cs_sep.collections:
            for path in collection.get_paths():
                verts = path.vertices
                sep_xs.append(verts[:, 0].tolist())
                sep_ys.append(verts[:, 1].tolist())
        plt.close(fig_tmp)

        self._eq_contour_source.data = dict(xs=xs_all, ys=ys_all)
        self._eq_lcfs_source.data = dict(xs=sep_xs, ys=sep_ys)

        # Transparent hover image for ψ readout
        hover_img = eq.psi_2D.T  # Bokeh image expects (nY, nX)
        self._eq_hover_img_source.data = dict(
            image=[hover_img],
            x=[float(eq.R_min)],
            y=[float(eq.Z_min)],
            dw=[float(eq.R_max - eq.R_min)],
            dh=[float(eq.Z_max - eq.Z_min)],
        )

        # Coil positions and mask polygons (shared helper)
        cd, mask_xs, mask_ys, has_maskable = self._collect_coil_data(tokamak)
        self._eq_coil_source.data = dict(x=cd["pt_x"], y=cd["pt_y"], color=cd["pt_c"], name=cd["pt_n"])
        self._eq_fil_cross_source.data = dict(x=cd["cr_x"], y=cd["cr_y"], color=cd["cr_c"], name=cd["cr_n"])
        self._eq_fil_rect_source.data = dict(xs=cd["rc_xs"], ys=cd["rc_ys"], fill=cd["rc_f"], edge=cd["rc_e"], name=cd["rc_n"])

        # Cache and apply mask data
        self._eq_mask_data_cache = {"xs": mask_xs, "ys": mask_ys}
        self._has_maskable_coils = has_maskable
        if self.show_masks.value and has_maskable:
            self._eq_mask_source.data = dict(xs=mask_xs, ys=mask_ys)
        else:
            self._eq_mask_source.data = dict(xs=[], ys=[])

        # --- Size the plot to encompass wall + all coils, preserving aspect ratio ---
        all_R = list(tokamak.wall_R) + cd["pt_x"] + cd["cr_x"]
        all_Z = list(tokamak.wall_Z) + cd["pt_y"] + cd["cr_y"]
        for rxs in cd["rc_xs"]:
            all_R.extend(rxs)
        for rys in cd["rc_ys"]:
            all_Z.extend(rys)

        margin_r = 0.25
        margin_z = 0.1
        r_min = max(0.0, float(np.min(all_R)) - margin_r)
        r_max = float(np.max(all_R)) + margin_r
        z_min = float(np.min(all_Z)) - margin_z
        z_max = float(np.max(all_Z)) + margin_z

        self.eq_figure.x_range.start = self.eq_figure.x_range.reset_start = r_min
        self.eq_figure.x_range.end = self.eq_figure.x_range.reset_end = r_max
        self.eq_figure.y_range.start = self.eq_figure.y_range.reset_start = z_min
        self.eq_figure.y_range.end = self.eq_figure.y_range.reset_end = z_max

        r_extent = r_max - r_min
        if r_extent > 2:
            tick_interval = 0.5
        elif r_extent > 1:
            tick_interval = 0.25
        else:
            tick_interval = 0.1
        n_ticks_r = max(3, int(r_extent / tick_interval) + 1)
        n_ticks_z = max(3, int((z_max - z_min) / tick_interval) + 1)
        self.eq_figure.xaxis.ticker.desired_num_ticks = n_ticks_r
        self.eq_figure.yaxis.ticker.desired_num_ticks = n_ticks_z

        data_width = r_max - r_min
        data_height = z_max - z_min
        fixed_height = 900
        computed_width = max(200, int(fixed_height * data_width / data_height))
        self.eq_figure.width = computed_width
        self.eq_figure.height = fixed_height

    # ------------------------------------------------------------------
    def _on_constraint_change(self, event):
        """Called when any constraint widget value changes — refresh the overlay."""
        eq = self.state.get("eq")
        if eq is not None:
            self._update_constraint_overlay(eq)
            tokamak = self.state.get("tokamak")
            if tokamak is not None:
                self._update_null_space_info(tokamak)

    # ------------------------------------------------------------------
    def _update_constraint_overlay(self, eq):
        """Compute annealing constraint positions and update the plot overlay."""
        flux_R, flux_Z = [], []
        xpt_R, xpt_Z = [], []

        # --- X-point constraint (always present) ---
        # "lower" / "upper" only meaningful for DND; fall back to primary otherwise.
        xpt_choice = self.xpoint_constraint.value
        if xpt_choice == "lower" and getattr(eq, "DND", False):
            R_xpt, Z_xpt = eq.R_xpt_lower, eq.Z_xpt_lower
        elif xpt_choice == "upper" and getattr(eq, "DND", False):
            R_xpt, Z_xpt = eq.R_xpt_upper, eq.Z_xpt_upper
        else:
            R_xpt, Z_xpt = eq.R_xpt_lcfs, eq.Z_xpt_lcfs
        xpt_R.append(float(R_xpt))
        xpt_Z.append(float(Z_xpt))

        # --- Quadrant flux constraints ---
        if self.constrain_ur.value:
            N = self.n_ur.value
            nd = np.linspace(0.0, eq.separatrix_dist_norm_vertical_upper_lcfs, N + 2)[1:-1]
            for r, z in eq.separatrix_interpolator(nd):
                flux_R.append(float(r)); flux_Z.append(float(z))

        if self.constrain_ul.value:
            N = self.n_ul.value
            nd = np.linspace(eq.separatrix_dist_norm_vertical_upper_lcfs,
                             eq.separatrix_dist_norm_imp_lcfs, N + 2)[1:-1]
            for r, z in eq.separatrix_interpolator(nd):
                flux_R.append(float(r)); flux_Z.append(float(z))

        if self.constrain_ll.value:
            N = self.n_ll.value
            nd = np.linspace(eq.separatrix_dist_norm_imp_lcfs,
                             eq.separatrix_dist_norm_vertical_lower_lcfs, N + 2)[1:-1]
            for r, z in eq.separatrix_interpolator(nd):
                flux_R.append(float(r)); flux_Z.append(float(z))

        if self.constrain_lr.value:
            N = self.n_lr.value
            nd = np.linspace(eq.separatrix_dist_norm_vertical_lower_lcfs, 1, N + 2)[1:-1]
            for r, z in eq.separatrix_interpolator(nd):
                flux_R.append(float(r)); flux_Z.append(float(z))

        # --- OMP / IMP ---
        if self.constrain_omp.value:
            flux_R.append(float(eq.R_OMP)); flux_Z.append(float(eq.Z_OMP))
        if self.constrain_imp.value:
            flux_R.append(float(eq.R_IMP)); flux_Z.append(float(eq.Z_IMP))

        # --- Upper / lower point ---
        if self.constrain_upper_point.value:
            flux_R.append(float(eq.R_vertical_upper)); flux_Z.append(float(eq.Z_vertical_upper))
        if self.constrain_lower_point.value:
            flux_R.append(float(eq.R_vertical_lower)); flux_Z.append(float(eq.Z_vertical_lower))

        self._eq_flux_constraint_source.data = dict(x=flux_R, y=flux_Z)
        self._eq_xpt_constraint_source.data = dict(x=xpt_R, y=xpt_Z)

    # ------------------------------------------------------------------
    def _update_eq_info(self, eq, tokamak):
        """Build an HTML summary of all key equilibrium data."""
        def _fmt(val, unit="", decimals=4):
            if isinstance(val, (float, np.floating)):
                return f"{val:.{decimals}f} {unit}".strip()
            return f"{val} {unit}".strip()

        rows = ""
        def _row(label, value):
            nonlocal rows
            rows += (
                f'<tr><td style="padding:2px 8px 2px 0;white-space:nowrap;">'
                f'{label}</td><td style="padding:2px 0;">{value}</td></tr>'
            )

        # Grid
        _row("Grid (nR \u00d7 nZ)", f"{eq.nR} \u00d7 {eq.nZ}")
        _row("R range", f"{eq.R_min:.4f} \u2013 {eq.R_max:.4f} m")
        _row("Z range", f"{eq.Z_min:.4f} \u2013 {eq.Z_max:.4f} m")
        _row("dR", _fmt(eq.dR, "m"))
        _row("dZ", _fmt(eq.dZ, "m"))

        # Magnetic configuration
        _row("Config", eq.mag_con)
        _row("DND", str(eq.DND))
        _row("N active X-points", str(eq.N_active_xpoints))

        # Magnetic axis
        _row("R<sub>mag</sub>", _fmt(eq.R_mag, "m"))
        _row("Z<sub>mag</sub>", _fmt(eq.Z_mag, "m"))
        _row("\u03c8<sub>axis</sub>", _fmt(eq.psi_axis, "Wb"))

        # Primary X-point
        _row("R<sub>xpt</sub> (primary)", _fmt(eq.R_xpt_lcfs, "m"))
        _row("Z<sub>xpt</sub> (primary)", _fmt(eq.Z_xpt_lcfs, "m"))
        _row("\u03c8<sub>lcfs</sub>", _fmt(eq.psi_lcfs, "Wb"))

        # Lower / upper X-points with primary/secondary labels and ψ values
        lower_label = "primary" if eq.lower_xpoint_primary else "secondary"
        upper_label = "primary" if eq.upper_xpoint_primary else "secondary"
        _row(f"R<sub>xpt</sub> (lower, {lower_label})", _fmt(eq.R_xpt_lower, "m"))
        _row(f"Z<sub>xpt</sub> (lower, {lower_label})", _fmt(eq.Z_xpt_lower, "m"))
        _row(f"\u03c8<sub>xpt</sub> (lower)", _fmt(eq.psi_xpt_lower, "Wb"))
        _row(f"R<sub>xpt</sub> (upper, {upper_label})", _fmt(eq.R_xpt_upper, "m"))
        _row(f"Z<sub>xpt</sub> (upper, {upper_label})", _fmt(eq.Z_xpt_upper, "m"))
        _row(f"\u03c8<sub>xpt</sub> (upper)", _fmt(eq.psi_xpt_upper, "Wb"))

        # DND ψ balance — relative difference
        if eq.DND:
            psi_diff = abs(eq.psi_xpt_upper - eq.psi_xpt_lower)
            psi_ref = abs(eq.psi_xpt_lower) if abs(eq.psi_xpt_lower) > 0 else 1.0
            rel_diff_pct = 100.0 * psi_diff / psi_ref
            _row("\u03c8 balance", f"{rel_diff_pct:.4f}%")

        # Key LCFS points
        _row("R<sub>OMP</sub>", _fmt(eq.R_OMP, "m"))
        _row("Z<sub>OMP</sub>", _fmt(eq.Z_OMP, "m"))
        _row("R<sub>IMP</sub>", _fmt(eq.R_IMP, "m"))
        _row("Z<sub>IMP</sub>", _fmt(eq.Z_IMP, "m"))
        _row("R<sub>upper</sub>", _fmt(eq.R_vertical_upper, "m"))
        _row("Z<sub>upper</sub>", _fmt(eq.Z_vertical_upper, "m"))
        _row("R<sub>lower</sub>", _fmt(eq.R_vertical_lower, "m"))
        _row("Z<sub>lower</sub>", _fmt(eq.Z_vertical_lower, "m"))

        # Shape parameters
        _row("Minor radius", _fmt(eq.minor_radius, "m"))

        # Plasma current
        _row("I<sub>p</sub>", f"{eq.plasma_current:.1f} A ({eq.plasma_current/1e3:.2f} kA)")
        _row("I<sub>p</sub> (check)", f"{eq.plasma_current_check:.1f} A")

        # COCOS
        _row("COCOS (input)", eq.cocos_input)

        # Vacuum toroidal field
        _row("f<sub>vac</sub>", _fmt(eq.fvac, "T·m"))

        # Coil count
        _row("N coils", str(tokamak.N_coils))

        html = (
            '<div style="font-size:0.85em;">'
            '<h4 style="margin:0 0 6px 0;">Equilibrium Data</h4>'
            '<table style="border-collapse:collapse;">'
            + rows
            + '</table></div>'
        )
        self.eq_info_pane.object = html

    # ------------------------------------------------------------------
    def _update_profile_plots(self, eq):
        """Populate the 1D profile figures with equilibrium data."""
        import numpy as np
        MU0 = 4.0 * np.pi * 1e-7
        psin = list(eq.psin_data)
        self._profile_sources["pprime"].data = dict(x=psin, y=list(eq.pprime_data / 1e3))
        self._profile_sources["ffprime"].data = dict(x=psin, y=list(eq.ffprime_data / (MU0 * 1e3)))
        self._profile_sources["pressure"].data = dict(x=psin, y=list(eq.pressure_data / 1e3))
        self._profile_sources["fpol"].data = dict(x=psin, y=list(eq.fpol_data))

    # ------------------------------------------------------------------
    def _update_decomp_plots(self, eq, tokamak):
        """Build the three flux-decomposition contour plots with colorbars."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import cm

        datasets = {
            "Total \u03c8": eq.psi_2D,
            "Plasma \u03c8": eq.psi_plas_2D,
            "Coil \u03c8": eq.psi_mach_2D,
        }

        wall_R = list(tokamak.wall_R)
        wall_Z = list(tokamak.wall_Z)

        # Coil data (shared helper) — same data for every decomp plot
        cd, mask_xs, mask_ys, has_maskable = self._collect_coil_data(tokamak)

        # Compute extents including wall + coils
        all_R = wall_R + cd["pt_x"] + cd["cr_x"]
        all_Z = list(tokamak.wall_Z) + cd["pt_y"] + cd["cr_y"]
        for rxs in cd["rc_xs"]:
            all_R.extend(rxs)
        for rys in cd["rc_ys"]:
            all_Z.extend(rys)

        margin_r = 0.05
        margin_z = 0.05
        r_min = max(0.0, float(np.min(all_R)) - margin_r)
        r_max = float(np.max(all_R)) + margin_r
        z_min = float(np.min(all_Z)) - margin_z
        z_max = float(np.max(all_Z)) + margin_z
        data_width = r_max - r_min
        data_height = z_max - z_min

        # Pixel dimensions — account for chrome (axes, title) so that
        # match_aspect fills the canvas correctly without squishing.
        chrome_w = 80
        chrome_h = 75
        plot_area_h = 500
        plot_area_w = max(100, int(plot_area_h * data_width / data_height))
        total_width = plot_area_w + chrome_w
        total_height = plot_area_h + chrome_h

        n_levels = 80

        for label, data_2D in datasets.items():
            fig = self._decomp_figures[label]
            srcs = self._decomp_sources[label]
            cbar_fig = self._decomp_cbar_figs[label]
            cbar_src = self._decomp_cbar_srcs[label]

            # Wall
            srcs["wall_fill"].data = dict(R=wall_R, Z=wall_Z)
            srcs["wall_outline"].data = dict(R=wall_R, Z=wall_Z)

            # Coil glyphs
            srcs["coil_pt"].data = dict(x=cd["pt_x"], y=cd["pt_y"], color=cd["pt_c"], name=cd["pt_n"])
            srcs["coil_cross"].data = dict(x=cd["cr_x"], y=cd["cr_y"], color=cd["cr_c"], name=cd["cr_n"])
            srcs["coil_rect"].data = dict(xs=cd["rc_xs"], ys=cd["rc_ys"], fill=cd["rc_f"], edge=cd["rc_e"], name=cd["rc_n"])

            # Cache mask data per label and apply
            self._mask_data_cache[label] = {"xs": mask_xs, "ys": mask_ys}
            if self.show_masks.value and has_maskable:
                srcs["mask"].data = dict(xs=mask_xs, ys=mask_ys)
            else:
                srcs["mask"].data = dict(xs=[], ys=[])

            # Extract contour lines via matplotlib Agg backend
            vmin = float(np.nanmin(data_2D))
            vmax = float(np.nanmax(data_2D))
            levels = np.linspace(vmin, vmax, n_levels)
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("viridis")

            fig_tmp, ax_tmp = plt.subplots()
            cs = ax_tmp.contour(eq.R_2D, eq.Z_2D, data_2D, levels=levels)

            xs_all, ys_all, colors_all = [], [], []
            for level_val, collection in zip(cs.levels, cs.collections):
                hex_col = matplotlib.colors.to_hex(cmap(norm(level_val)))
                for path in collection.get_paths():
                    verts = path.vertices
                    xs_all.append(verts[:, 0].tolist())
                    ys_all.append(verts[:, 1].tolist())
                    colors_all.append(hex_col)

            # Separatrix on the Total \u03c8 plot
            if label == "Total \u03c8":
                cs_sep = ax_tmp.contour(
                    eq.R_2D, eq.Z_2D, eq.psi_2D, levels=[eq.psi_lcfs],
                )
                sep_xs, sep_ys = [], []
                for coll in cs_sep.collections:
                    for path in coll.get_paths():
                        verts = path.vertices
                        sep_xs.append(verts[:, 0].tolist())
                        sep_ys.append(verts[:, 1].tolist())
                srcs["separatrix"].data = dict(xs=sep_xs, ys=sep_ys)

            plt.close(fig_tmp)

            srcs["contour"].data = dict(xs=xs_all, ys=ys_all, color=colors_all)

            # Transparent hover image — Bokeh image expects (nY, nX) row-major
            hover_img = data_2D.T
            srcs["hover_img"].data = dict(
                image=[hover_img],
                x=[float(eq.R_min)],
                y=[float(eq.Z_min)],
                dw=[float(eq.R_max - eq.R_min)],
                dh=[float(eq.Z_max - eq.Z_min)],
            )

            # Update companion colorbar figure
            gradient = np.linspace(vmin, vmax, 256).reshape(256, 1)
            # Update the existing colorbar mapper in-place (avoid replacing
            # the model object, which orphans the old ID and causes
            # "Dropping a patch" warnings).
            cbar_fig.renderers[0].glyph.color_mapper.low = vmin
            cbar_fig.renderers[0].glyph.color_mapper.high = vmax
            cbar_src.data = dict(
                image=[gradient], x=[0], y=[vmin], dw=[1], dh=[vmax - vmin],
            )
            cbar_fig.y_range.start = cbar_fig.y_range.reset_start = vmin
            cbar_fig.y_range.end = cbar_fig.y_range.reset_end = vmax
            cbar_fig.height = total_height

            # Axis ranges and pixel dimensions for main figure
            # Decomposition figures use DataRange1d (no reset_start/reset_end)
            fig.x_range.start = r_min
            fig.x_range.end = r_max
            fig.y_range.start = z_min
            fig.y_range.end = z_max
            fig.width = total_width
            fig.height = total_height

    # ------------------------------------------------------------------
    def _count_annealing_constraints(self):
        """Count the number of annealing constraints from the current widget state.

        The X-point always contributes 3 constraints (Br, Bz, psi at the X-point).
        Each enabled quadrant contributes N flux constraints. OMP, IMP, upper point,
        and lower point each contribute 1 flux constraint when enabled.

        Returns
        -------
        int
            Total number of annealing constraints.
        """
        n = 3  # X-point: Br, Bz, psi
        if self.constrain_ur.value:
            n += self.n_ur.value
        if self.constrain_ul.value:
            n += self.n_ul.value
        if self.constrain_ll.value:
            n += self.n_ll.value
        if self.constrain_lr.value:
            n += self.n_lr.value
        if self.constrain_omp.value:
            n += 1
        if self.constrain_imp.value:
            n += 1
        if self.constrain_upper_point.value:
            n += 1
        if self.constrain_lower_point.value:
            n += 1
        return n

    # ------------------------------------------------------------------
    def _update_null_space_info(self, tokamak):
        """Update the null-space info pane with N coils, N constraints and null-space size."""
        n_coils = tokamak.N_coils
        n_constraints = self._count_annealing_constraints()
        null_space_size = n_coils - n_constraints

        if null_space_size >= 2:
            status_colour = "#2ca02c"  # green
            status_icon = "&#10003;"   # check mark
            status_text = "OK"
        elif null_space_size == 1:
            status_colour = "#ff7f0e"  # amber
            status_icon = "&#9888;"    # warning
            status_text = "Marginal — only one degree of freedom to explore"
        else:
            status_colour = "#d62728"  # red
            status_icon = "&#9888;"    # warning
            status_text = "Too many constraints — null space must be ≥ 1"

        html = (
            '<div style="font-size:0.85em; margin-top:4px;">'
            '<table style="border-collapse:collapse;">'
            f'<tr><td style="padding:2px 8px 2px 0;">N coils / circuits</td>'
            f'<td style="padding:2px 0;"><b>{n_coils}</b></td></tr>'
            f'<tr><td style="padding:2px 8px 2px 0;">N annealing constraints</td>'
            f'<td style="padding:2px 0;"><b>{n_constraints}</b></td></tr>'
            f'<tr><td style="padding:2px 8px 2px 0;">Null-space size</td>'
            f'<td style="padding:2px 0;"><b>{null_space_size}</b></td></tr>'
            '</table>'
            f'<div style="margin-top:4px; padding:4px 8px; '
            f'background:{status_colour}20; border-left:3px solid {status_colour}; '
            f'color:{status_colour}; font-weight:bold;">'
            f'{status_icon} {status_text}'
            '</div>'
            '<div style="margin-top:4px; font-size:0.9em; color:#555;">'
            'The null-space approach requires N<sub>coils</sub> − N<sub>constraints</sub> ≥ 1 '
            'for the null space to have finite size. A size of 2 or more is recommended '
            'so that the optimiser has sufficient degrees of freedom to explore.'
            '</div>'
            '</div>'
        )
        self._null_space_info.object = html

    # ------------------------------------------------------------------
    def _update_coils_table(self, tokamak):
        """Build an HTML table of coil names and currents (kA)."""
        rows = ""
        for name, coil in tokamak.coilset.items():
            current_kA = coil.current / 1.0e3
            rows += f'<tr><td>{name}</td><td style="text-align:right;">{current_kA:.2f}</td></tr>'
        html = (
            '<table style="border-collapse:collapse;">'
            '<tr style="background:#eee;"><th style="text-align:left; padding-right:16px;">Coil / Circuit</th>'
            '<th style="text-align:right;">Current (kA)</th></tr>'
            + rows
            + "</table>"
        )
        self.coils_table.object = html

    # ------------------------------------------------------------------
    def get_constraints_dict(self):
        """Build the constraints dictionary from the current widget values."""
        return {
            "annealing": {
                "constrain_omp": self.constrain_omp.value,
                "constrain_imp": self.constrain_imp.value,
                "constrain_upper_point": self.constrain_upper_point.value,
                "constrain_lower_point": self.constrain_lower_point.value,
                "constrain_upper_right_quadrant": self.constrain_ur.value,
                "N_constraints_upper_right_quadrant": self.n_ur.value,
                "constrain_upper_left_quadrant": self.constrain_ul.value,
                "N_constraints_upper_left_quadrant": self.n_ul.value,
                "constrain_lower_left_quadrant": self.constrain_ll.value,
                "N_constraints_lower_left_quadrant": self.n_ll.value,
                "constrain_lower_right_quadrant": self.constrain_lr.value,
                "N_constraints_lower_right_quadrant": self.n_lr.value,
                "additional_divertor_constraint_points": None,
                "additional_divertor_xpoints": None,
                "xpoint_constraint": self.xpoint_constraint.value,
            },
            "tikhonov": {
                "constrain_omp": self.constrain_omp.value,
                "constrain_imp": self.constrain_imp.value,
                "constrain_upper_point": self.constrain_upper_point.value,
                "constrain_lower_point": self.constrain_lower_point.value,
                "constrain_upper_right_quadrant": self.constrain_ur.value,
                "N_constraints_upper_right_quadrant": self.n_ur.value,
                "constrain_upper_left_quadrant": self.constrain_ul.value,
                "N_constraints_upper_left_quadrant": self.n_ul.value,
                "constrain_lower_left_quadrant": self.constrain_ll.value,
                "N_constraints_lower_left_quadrant": self.n_ll.value,
                "constrain_lower_right_quadrant": self.constrain_lr.value,
                "N_constraints_lower_right_quadrant": self.n_lr.value,
                "additional_divertor_constraint_points": None,
                "additional_divertor_xpoints": None,
                "xpoint_constraint": self.xpoint_constraint.value,
                "exclude_coils": None,
            },
        }

    def _apply_constraints(self, ann):
        """Set constraint widget values from an annealing constraints dict."""
        _MAP = {
            "constrain_omp": self.constrain_omp,
            "constrain_imp": self.constrain_imp,
            "constrain_upper_point": self.constrain_upper_point,
            "constrain_lower_point": self.constrain_lower_point,
            "constrain_upper_right_quadrant": self.constrain_ur,
            "N_constraints_upper_right_quadrant": self.n_ur,
            "constrain_upper_left_quadrant": self.constrain_ul,
            "N_constraints_upper_left_quadrant": self.n_ul,
            "constrain_lower_left_quadrant": self.constrain_ll,
            "N_constraints_lower_left_quadrant": self.n_ll,
            "constrain_lower_right_quadrant": self.constrain_lr,
            "N_constraints_lower_right_quadrant": self.n_lr,
            "xpoint_constraint": self.xpoint_constraint,
        }
        for key, widget in _MAP.items():
            if key in ann:
                widget.value = ann[key]

    # ------------------------------------------------------------------
    @property
    def panel(self):
        """Return the Panel layout for this tab."""
        if hasattr(self, "_panel"):
            return self._panel
        file_col = pn.Column(
            "### 1. Load Files",
            self.geqdsk_label,
            self.geqdsk_input,
            self.magnets_label,
            self.magnets_input,
            self.load_btn,
            self.status,
            pn.layout.Divider(),
            "### Coils / Circuits",
            self.coils_table,
            self.show_masks,
            self._null_space_info,
            pn.layout.Divider(),
            "### Annealing Constraints",
            self.xpoint_constraint,
            self.constrain_omp,
            self.constrain_imp,
            self.constrain_upper_point,
            self.constrain_lower_point,
            pn.Row(self.constrain_ur, self.n_ur),
            pn.Row(self.constrain_ul, self.n_ul),
            pn.Row(self.constrain_ll, self.n_ll),
            pn.Row(self.constrain_lr, self.n_lr),
            width=420,
            height=900,
            scroll=True,
        )
        eq_pane = pn.pane.Bokeh(self.eq_figure, sizing_mode="stretch_height")
        info_col = pn.Column(self.eq_info_pane, width=320, styles={"overflow-y": "auto", "overflow-x": "hidden"}, sizing_mode="stretch_height")
        profiles_col = pn.Column(
            pn.pane.Bokeh(self._profile_figs["pprime"]),
            pn.pane.Bokeh(self._profile_figs["ffprime"]),
            pn.pane.Bokeh(self._profile_figs["pressure"]),
            pn.pane.Bokeh(self._profile_figs["fpol"]),
            sizing_mode="stretch_height",
        )
        top_row = pn.Row(file_col, eq_pane, info_col, profiles_col, sizing_mode="stretch_width")
        decomp_row = pn.Row(
            *(pn.pane.Bokeh(bk_row(self._decomp_figures[k], self._decomp_cbar_figs[k]))
              for k in self._decomp_figures),
            sizing_mode="stretch_width",
        )
        self._panel = pn.Column(top_row, pn.layout.Divider(), "### Flux Decomposition", decomp_row, sizing_mode="stretch_both")
        return self._panel
