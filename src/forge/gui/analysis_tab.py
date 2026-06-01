"""Analysis tab — post-optimisation plots, current comparison, save/load.

Provides cost breakdown charts, connection length histories, a coil current
comparison bar chart, and buttons to save/load the full optimiser state.
"""

import copy
import io
import logging
import os
import pickle
import time

import numpy as np
import panel as pn
from bokeh.models import BasicTickFormatter, ColumnDataSource, HoverTool, Range1d, ColorBar, LinearColorMapper
from bokeh.plotting import figure as bk_figure
from bokeh.transform import linear_cmap

from forge.gui.setup_tab import SetupTab
from forge.io import (
    _UNPICKLABLE_ATTRS,
    fancy_json_string,
    load_optimiser,
    save_optimiser,
    write_geqdsk,
    write_magnets,
)

logger = logging.getLogger(__name__)


class _ProgressBytesIO(io.BytesIO):
    """BytesIO wrapper that logs progress during large writes."""

    def __init__(self, *args, logger=None, log_interval_mb=50, **kwargs):
        super().__init__(*args, **kwargs)
        self._log = logger
        self._interval = log_interval_mb * 1024 * 1024
        self._next_threshold = self._interval

    def write(self, b):
        n = super().write(b)
        if self._log is not None and self.tell() >= self._next_threshold:
            mb = self.tell() / (1024 * 1024)
            msg = f"  ... {mb:.0f} MB written"
            self._log.info(msg)
            print(msg, flush=True)
            self._next_threshold = self.tell() + self._interval
        return n


class AnalysisTab:
    """Panel component for post-optimisation analysis."""

    def __init__(self, shared_state):
        self.state = shared_state

        # --- Buttons ---
        self.refresh_btn = pn.widgets.Button(name="Refresh from Optimisation", button_type="primary")
        self.save_pickle_btn = pn.widgets.Button(
            name="Save Optimiser (.pkl)",
            button_type="success",
        )
        # Hidden trigger: when its value changes, JS opens the download URL.
        self._download_trigger = pn.widgets.TextInput(value="", visible=False)
        self._download_trigger.jscallback(
            value="if (cb_obj.value) { window.open(cb_obj.value, '_blank'); }"
        )
        self.load_pickle_path_input = pn.widgets.TextInput(
            name="Server file path", placeholder="/path/to/optimiser.pkl"
        )
        self.load_pickle_btn = pn.widgets.Button(name="Load (server path)", button_type="primary")
        # Browser upload: uses fetch() POST to /upload to stream the file
        # without loading it entirely into JS memory via base64.
        self.upload_pickle_btn = pn.widgets.Button(name="Upload from browser", button_type="success")
        self._upload_id_trigger = pn.widgets.TextInput(value="", visible=False)
        self.upload_pickle_btn.jscallback(
            clicks="""
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pkl,.pickle';
            input.onchange = async () => {
                const file = input.files[0];
                if (!file) return;
                trigger.value = 'uploading';
                const form = new FormData();
                form.append('file', file);
                try {
                    const resp = await fetch('/upload', {method: 'POST', body: form});
                    const data = await resp.json();
                    trigger.value = data.upload_id;
                } catch(e) {
                    trigger.value = 'ERROR:' + e.message;
                }
            };
            input.click();
            """,
            args={"trigger": self._upload_id_trigger},
        )
        self.save_geqdsk_btn = pn.widgets.FileDownload(
            callback=self._save_geqdsk_callback,
            filename="forge_optimised.geqdsk",
            label="Save Equilibrium (.geqdsk)",
            button_type="warning",
        )
        self.save_machine_btn = pn.widgets.FileDownload(
            callback=self._save_machine_callback,
            filename="forge_machine.json",
            label="Save Machine (.json)",
            button_type="warning",
        )

        self.status = pn.pane.Alert("Run an optimisation first, then click Refresh.", alert_type="info")

        # --- Bokeh figures ---
        # Cost breakdown
        self._cost_source = ColumnDataSource(data=dict(x=[], total=[], strike=[], cl=[], coil=[], xpt=[]))
        self.cost_fig = self._make_fig("Cost Breakdown", "Iteration", "Cost")
        self.cost_fig.line("x", "total", source=self._cost_source, color="black", legend_label="Total")
        self.cost_fig.line("x", "strike", source=self._cost_source, color="red", legend_label="Strike")
        self.cost_fig.line("x", "cl", source=self._cost_source, color="green", legend_label="Conn. length")
        self.cost_fig.line("x", "coil", source=self._cost_source, color="orange", legend_label="Coil I²")
        self.cost_fig.line("x", "xpt", source=self._cost_source, color="blue", legend_label="XPT")
        self.cost_fig.legend.click_policy = "hide"

        # Individual cost term figures
        self.strike_cost_fig = self._make_fig("Strike Point Distance Cost", "Iteration", "Cost")
        self.strike_cost_fig.line("x", "strike", source=self._cost_source, color="red")

        self.cl_cost_fig = self._make_fig("Connection Length Cost", "Iteration", "Cost")
        self.cl_cost_fig.line("x", "cl", source=self._cost_source, color="green")

        self.coil_cost_fig = self._make_fig("Coil Currents Cost", "Iteration", "Cost")
        self.coil_cost_fig.line("x", "coil", source=self._cost_source, color="orange")

        self.xpt_cost_fig = self._make_fig("X-Point Regions Cost", "Iteration", "Cost")
        self.xpt_cost_fig.line("x", "xpt", source=self._cost_source, color="blue")

        # Incumbent iteration vertical line (shared source for all time-series)
        self._vline_src = ColumnDataSource(data=dict(x=[]))

        # Temperature
        self._temp_source = ColumnDataSource(data=dict(x=[], temperature=[]))
        self.temp_fig = self._make_fig("Temperature Schedule", "Iteration", "T")
        self.temp_fig.line("x", "temperature", source=self._temp_source, color="blue")

        # Acceptance rate
        self._accept_source = ColumnDataSource(data=dict(x=[], rate=[]))
        self.accept_fig = self._make_fig("Acceptance Rate", "Iteration", "Rate")
        self.accept_fig.scatter("x", "rate", source=self._accept_source, color="blue", size=2)

        # Connection length per region
        self._cl_source = ColumnDataSource(data=dict(x=[]))
        self.cl_fig = self._make_fig("Connection Length per Region", "Iteration", "L (m)")
        # Invisible placeholder so Bokeh doesn't warn about an empty plot.
        self.cl_fig.line([], [], visible=False)

        # Add incumbent vertical line to all time-series figures
        for fig in (
            self.cost_fig, self.strike_cost_fig, self.cl_cost_fig,
            self.coil_cost_fig, self.xpt_cost_fig, self.temp_fig,
            self.accept_fig, self.cl_fig,
        ):
            fig.ray(x="x", y=0, length=0, angle=1.5708, source=self._vline_src,
                    color="black", line_dash="dashed", line_width=1)

        # --- Initial equilibrium figure ---
        self._init_wall_fill_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._init_wall_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._init_contour_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._init_sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._init_coil_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._init_coil_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._init_coil_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        self._init_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._init_fl_sources = {}

        self.init_eq_fig = bk_figure(
            title="Initial Equilibrium",
            x_axis_label="R (m)", y_axis_label="Z (m)",
            height=600, width=300,
            x_range=Range1d(0, 1), y_range=Range1d(0, 1),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        self.init_eq_fig.xaxis.axis_label_text_font_style = "normal"
        self.init_eq_fig.yaxis.axis_label_text_font_style = "normal"
        self.init_eq_fig.title.text_font_style = "normal"

        self.init_eq_fig.patch("R", "Z", source=self._init_wall_fill_src,
                               fill_color="white", line_color=None)
        self.init_eq_fig.multi_line("xs", "ys", source=self._init_contour_src,
                                    line_alpha=0.35, line_color="gray")
        self.init_eq_fig.multi_line("xs", "ys", source=self._init_sep_src,
                                    line_color="red", line_width=2)
        self.init_eq_fig.line("R", "Z", source=self._init_wall_src,
                              line_color="black", line_width=2)
        self.init_eq_fig.scatter("x", "y", source=self._init_coil_pt_src, color="color",
                                 size=10, marker="circle")
        self.init_eq_fig.scatter("x", "y", source=self._init_coil_cr_src, color="color",
                                 size=8, marker="x")
        self.init_eq_fig.patches("xs", "ys", source=self._init_coil_rc_src,
                                  fill_color="fill", line_color="edge", line_width=0.3)
        self.init_eq_fig.patches("xs", "ys", source=self._init_mask_src,
                                  fill_color="orange", line_color="black",
                                  line_width=1.0, fill_alpha=1.0)

        # --- Incumbent equilibrium figure ---
        self._inc_wall_fill_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._inc_wall_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._inc_contour_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._inc_sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._inc_coil_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._inc_coil_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._inc_coil_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        self._inc_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._inc_fl_sources = {}

        self.inc_eq_fig = bk_figure(
            title="Incumbent Equilibrium",
            x_axis_label="R (m)", y_axis_label="Z (m)",
            height=600, width=300,
            x_range=Range1d(0, 1), y_range=Range1d(0, 1),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        self.inc_eq_fig.xaxis.axis_label_text_font_style = "normal"
        self.inc_eq_fig.yaxis.axis_label_text_font_style = "normal"
        self.inc_eq_fig.title.text_font_style = "normal"

        self.inc_eq_fig.patch("R", "Z", source=self._inc_wall_fill_src,
                              fill_color="white", line_color=None)
        self.inc_eq_fig.multi_line("xs", "ys", source=self._inc_contour_src,
                                   line_alpha=0.35, line_color="gray")
        self.inc_eq_fig.multi_line("xs", "ys", source=self._inc_sep_src,
                                   line_color="red", line_width=2)
        self.inc_eq_fig.line("R", "Z", source=self._inc_wall_src,
                             line_color="black", line_width=2)
        self.inc_eq_fig.scatter("x", "y", source=self._inc_coil_pt_src, color="color",
                                size=10, marker="circle")
        self.inc_eq_fig.scatter("x", "y", source=self._inc_coil_cr_src, color="color",
                                size=8, marker="x")
        self.inc_eq_fig.patches("xs", "ys", source=self._inc_coil_rc_src,
                                 fill_color="fill", line_color="edge", line_width=0.3)
        self.inc_eq_fig.patches("xs", "ys", source=self._inc_mask_src,
                                 fill_color="orange", line_color="black",
                                 line_width=1.0, fill_alpha=1.0)
        # Target strike geometry on the incumbent plot
        self._inc_strike_src = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
        self.inc_eq_fig.multi_line("xs", "ys", source=self._inc_strike_src,
                                   line_color="color", line_width=2.5,
                                   line_dash="dashed")

        # --- Separatrix comparison figure ---
        self._cmp_wall_fill_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._cmp_wall_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._cmp_init_sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._cmp_inc_sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._cmp_coil_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._cmp_coil_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._cmp_coil_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        self._cmp_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))

        self.cmp_eq_fig = bk_figure(
            title="Separatrix Comparison",
            x_axis_label="R (m)", y_axis_label="Z (m)",
            height=600, width=300,
            x_range=Range1d(0, 1), y_range=Range1d(0, 1),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        self.cmp_eq_fig.xaxis.axis_label_text_font_style = "normal"
        self.cmp_eq_fig.yaxis.axis_label_text_font_style = "normal"
        self.cmp_eq_fig.title.text_font_style = "normal"

        self.cmp_eq_fig.patch("R", "Z", source=self._cmp_wall_fill_src,
                              fill_color="white", line_color=None)
        self.cmp_eq_fig.line("R", "Z", source=self._cmp_wall_src,
                             line_color="black", line_width=2)
        self.cmp_eq_fig.multi_line("xs", "ys", source=self._cmp_init_sep_src,
                                   line_color="#E75480", line_width=2,
                                   legend_label="Initial")
        self.cmp_eq_fig.multi_line("xs", "ys", source=self._cmp_inc_sep_src,
                                   line_color="#89CFF0", line_width=2,
                                   legend_label="Incumbent")
        self.cmp_eq_fig.scatter("x", "y", source=self._cmp_coil_pt_src, color="color",
                                size=10, marker="circle")
        self.cmp_eq_fig.scatter("x", "y", source=self._cmp_coil_cr_src, color="color",
                                size=8, marker="x")
        self.cmp_eq_fig.patches("xs", "ys", source=self._cmp_coil_rc_src,
                                 fill_color="fill", line_color="edge", line_width=0.3)
        self.cmp_eq_fig.patches("xs", "ys", source=self._cmp_mask_src,
                                 fill_color="orange", line_color="black",
                                 line_width=1.0, fill_alpha=1.0)
        self.cmp_eq_fig.legend.click_policy = "hide"

        # --- Coil flux change figure (boundary coloured by relative change) ---
        self._flux_wall_fill_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._flux_wall_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._flux_coil_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._flux_coil_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._flux_coil_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        self._flux_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._flux_sep_src = ColumnDataSource(data=dict(R=[], Z=[], change_pct=[]))

        self._flux_cmap = LinearColorMapper(palette="Viridis256", low=0, high=1.0)

        self.flux_eq_fig = bk_figure(
            title="Relative Separatrix Coil Flux Change (%)",
            x_axis_label="R (m)", y_axis_label="Z (m)",
            height=600, width=300,
            x_range=Range1d(0, 1), y_range=Range1d(0, 1),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        self.flux_eq_fig.xaxis.axis_label_text_font_style = "normal"
        self.flux_eq_fig.yaxis.axis_label_text_font_style = "normal"
        self.flux_eq_fig.title.text_font_style = "normal"

        self.flux_eq_fig.patch("R", "Z", source=self._flux_wall_fill_src,
                               fill_color="white", line_color=None)
        self.flux_eq_fig.line("R", "Z", source=self._flux_wall_src,
                              line_color="black", line_width=2)
        self.flux_eq_fig.scatter("x", "y", source=self._flux_coil_pt_src, color="color",
                                 size=10, marker="circle")
        self.flux_eq_fig.scatter("x", "y", source=self._flux_coil_cr_src, color="color",
                                 size=8, marker="x")
        self.flux_eq_fig.patches("xs", "ys", source=self._flux_coil_rc_src,
                                  fill_color="fill", line_color="edge", line_width=0.3)
        self.flux_eq_fig.patches("xs", "ys", source=self._flux_mask_src,
                                  fill_color="orange", line_color="black",
                                  line_width=1.0, fill_alpha=1.0)
        flux_scatter = self.flux_eq_fig.scatter(
            "R", "Z", source=self._flux_sep_src,
            color={"field": "change_pct", "transform": self._flux_cmap},
            size=6, marker="circle",
        )
        flux_hover = HoverTool(
            renderers=[flux_scatter],
            tooltips=[
                ("R", "@R{0.4f} m"),
                ("Z", "@Z{0.4f} m"),
                ("|Δψ_mach/ψ_mach|", "@change_pct{0.4f} %"),
            ],
        )
        self.flux_eq_fig.add_tools(flux_hover)
        color_bar = ColorBar(color_mapper=self._flux_cmap,
                             label_standoff=8, width=12, location=(0, 0))
        self.flux_eq_fig.add_layout(color_bar, "right")

        # --- Coil mask toggle ---
        self.show_masks = pn.widgets.Checkbox(name="Show coil masks", value=True)
        self.show_masks.param.watch(self._on_mask_toggle, "value")
        self._cached_mask_xs = []
        self._cached_mask_ys = []

        # Current comparison bar chart
        self._bar_source = ColumnDataSource(data=dict(
            names=[], x_initial=[], x_optimised=[], initial=[], optimised=[],
        ))
        self.bar_fig = bk_figure(
            title="Coil Current Comparison",
            x_axis_label="I (kA)",
            height=350,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,reset",
        )
        self.bar_fig.title.text_font_style = "normal"
        self.bar_fig.xaxis.axis_label_text_font_style = "normal"
        self.bar_fig.yaxis.axis_label_text_font_style = "normal"
        self.bar_fig.hbar(
            y="x_initial", right="initial", source=self._bar_source,
            height=0.35, color="orange", legend_label="Initial",
        )
        self.bar_fig.hbar(
            y="x_optimised", right="optimised", source=self._bar_source,
            height=0.35, color="steelblue", legend_label="Optimised",
        )
        self.bar_fig.add_tools(HoverTool(
            tooltips=[
                ("Coil", "@names"),
                ("Initial (kA)", "@initial{0.000}"),
                ("Optimised (kA)", "@optimised{0.000}"),
            ],
        ))

        # Incumbent info
        self.incumbent_pane = pn.pane.HTML("<em>No data yet.</em>")

        # Wire
        self.refresh_btn.on_click(self._on_refresh)
        self.save_pickle_btn.on_click(self._on_save_pickle)

        self.load_pickle_btn.on_click(self._on_load_pickle)
        self.upload_pickle_btn.on_click(self._on_upload_pickle_click)
        self._upload_id_trigger.param.watch(self._on_upload_complete, "value")

    # ------------------------------------------------------------------
    @staticmethod
    def _make_fig(title, xlabel, ylabel):
        f = bk_figure(
            title=title, x_axis_label=xlabel, y_axis_label=ylabel,
            height=220, sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset",
        )
        f.title.text_font_style = "normal"
        f.xaxis.axis_label_text_font_style = "normal"
        f.yaxis.axis_label_text_font_style = "normal"
        f.xaxis.formatter = BasicTickFormatter(use_scientific=False)
        return f

    # ------------------------------------------------------------------
    def _on_mask_toggle(self, event):
        """Show / hide coil mask overlays on the analysis equilibrium plots."""
        if event.new:
            data = dict(xs=self._cached_mask_xs, ys=self._cached_mask_ys)
        else:
            data = dict(xs=[], ys=[])
        self._init_mask_src.data = data
        self._inc_mask_src.data = data
        self._cmp_mask_src.data = data
        self._flux_mask_src.data = data

    # ------------------------------------------------------------------
    def _compute_contours(self, R_2D, Z_2D, psi_2D, psi_lcfs=None):
        """Compute contour and separatrix line data from a 2-D flux array."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(R_2D, Z_2D, psi_2D, levels=40)
        xs_all, ys_all = [], []
        for collection in cs.collections:
            for path in collection.get_paths():
                verts = path.vertices
                xs_all.append(verts[:, 0].tolist())
                ys_all.append(verts[:, 1].tolist())

        sep_xs, sep_ys = [], []
        if psi_lcfs is not None:
            cs_sep = ax_tmp.contour(R_2D, Z_2D, psi_2D, levels=[psi_lcfs])
            for collection in cs_sep.collections:
                for path in collection.get_paths():
                    verts = path.vertices
                    sep_xs.append(verts[:, 0].tolist())
                    sep_ys.append(verts[:, 1].tolist())

        plt.close(fig_tmp)
        return xs_all, ys_all, sep_xs, sep_ys

    # ------------------------------------------------------------------
    def _populate_eq_figures(self, opt):
        """Populate the initial and incumbent equilibrium figures."""
        eq = opt.eq
        tokamak_initial = opt.tokamak_initial

        wall_r = list(tokamak_initial.wall_R) + [tokamak_initial.wall_R[0]]
        wall_z = list(tokamak_initial.wall_Z) + [tokamak_initial.wall_Z[0]]

        # Coil data from the initial tokamak
        cd_init, _, _, _ = SetupTab._collect_coil_data(tokamak_initial)

        # --- Initial equilibrium ---
        self._init_wall_fill_src.data = dict(R=wall_r, Z=wall_z)
        self._init_wall_src.data = dict(R=wall_r, Z=wall_z)
        self._init_coil_pt_src.data = dict(x=cd_init["pt_x"], y=cd_init["pt_y"],
                                            color=cd_init["pt_c"], name=cd_init["pt_n"])
        self._init_coil_cr_src.data = dict(x=cd_init["cr_x"], y=cd_init["cr_y"],
                                            color=cd_init["cr_c"], name=cd_init["cr_n"])
        self._init_coil_rc_src.data = dict(xs=cd_init["rc_xs"], ys=cd_init["rc_ys"],
                                            fill=cd_init["rc_f"], edge=cd_init["rc_e"],
                                            name=cd_init["rc_n"])

        # Initial contours
        c_xs, c_ys, s_xs, s_ys = self._compute_contours(
            eq.R_2D, eq.Z_2D, eq.psi_2D, psi_lcfs=eq.psi_lcfs,
        )
        self._init_contour_src.data = dict(xs=c_xs, ys=c_ys)
        self._init_sep_src.data = dict(xs=s_xs, ys=s_ys)

        # Initial field lines (from tracking data)
        for src in self._init_fl_sources.values():
            src.data = dict(r=[], z=[])
        self._init_fl_sources.clear()
        for region in opt.divertor_regions:
            init_data = opt.divertor_data[region]
            fl_R = init_data.get("initial_field_line_R")
            fl_Z = init_data.get("initial_field_line_Z")
            if fl_R is not None and fl_Z is not None:
                src = ColumnDataSource(data=dict(r=list(fl_R), z=list(fl_Z)))
            else:
                src = ColumnDataSource(data=dict(r=[], z=[]))
            self._init_fl_sources[region] = src
            colour = init_data.get("colour", "gray")
            self.init_eq_fig.line("r", "z", source=src, color=colour, line_width=1.5)

        # --- Incumbent equilibrium ---
        if hasattr(opt, "optimised_tokamak"):
            tokamak_opt = opt.optimised_tokamak
            cd_opt, _, _, _ = SetupTab._collect_coil_data(tokamak_opt)
        else:
            tokamak_opt = tokamak_initial
            cd_opt = cd_init

        self._inc_wall_fill_src.data = dict(R=wall_r, Z=wall_z)
        self._inc_wall_src.data = dict(R=wall_r, Z=wall_z)
        self._inc_coil_pt_src.data = dict(x=cd_opt["pt_x"], y=cd_opt["pt_y"],
                                           color=cd_opt["pt_c"], name=cd_opt["pt_n"])
        self._inc_coil_cr_src.data = dict(x=cd_opt["cr_x"], y=cd_opt["cr_y"],
                                           color=cd_opt["cr_c"], name=cd_opt["cr_n"])
        self._inc_coil_rc_src.data = dict(xs=cd_opt["rc_xs"], ys=cd_opt["rc_ys"],
                                           fill=cd_opt["rc_f"], edge=cd_opt["rc_e"],
                                           name=cd_opt["rc_n"])

        # Incumbent contours
        inc = opt.incumbent_data
        if inc and "psi_2D" in inc:
            psi_lcfs = eq.psi_lcfs
            c_xs, c_ys, s_xs, s_ys = self._compute_contours(
                eq.R_2D, eq.Z_2D, inc["psi_2D"], psi_lcfs=psi_lcfs,
            )
            self._inc_contour_src.data = dict(xs=c_xs, ys=c_ys)
            self._inc_sep_src.data = dict(xs=s_xs, ys=s_ys)

        # Incumbent field lines (from the incumbent state, not the last iteration)
        for src in self._inc_fl_sources.values():
            src.data = dict(r=[], z=[])
        self._inc_fl_sources.clear()
        inc = opt.incumbent_data
        for region in opt.divertor_regions:
            fl_R = inc["divertors"][region].get("field_line_R") if inc else None
            fl_Z = inc["divertors"][region].get("field_line_Z") if inc else None
            if fl_R is not None and fl_Z is not None:
                src = ColumnDataSource(data=dict(r=list(fl_R), z=list(fl_Z)))
            else:
                src = ColumnDataSource(data=dict(r=[], z=[]))
            self._inc_fl_sources[region] = src
            colour = opt.divertor_data[region].get("colour", "gray")
            self.inc_eq_fig.line("r", "z", source=src, color=colour, line_width=1.5)

        # Target strike geometries on the incumbent figure
        strike_xs, strike_ys, strike_colors = [], [], []
        for region in opt.divertor_regions:
            sr = opt.divertor_data[region].get("strike_R")
            sz = opt.divertor_data[region].get("strike_Z")
            if sr is not None and sz is not None:
                sr = list(sr) if hasattr(sr, '__iter__') else [sr]
                sz = list(sz) if hasattr(sz, '__iter__') else [sz]
                colour = opt.divertor_data[region].get("colour", "green")
                strike_xs.append(sr)
                strike_ys.append(sz)
                strike_colors.append(colour)
        self._inc_strike_src.data = dict(xs=strike_xs, ys=strike_ys, color=strike_colors)

        # Set plot bounds from wall geometry
        r_min = float(np.min(tokamak_initial.wall_R))
        r_max = float(np.max(tokamak_initial.wall_R))
        z_min = float(np.min(tokamak_initial.wall_Z))
        z_max = float(np.max(tokamak_initial.wall_Z))
        dr = r_max - r_min
        dz = z_max - z_min
        pad = 0.05
        for fig in (self.init_eq_fig, self.inc_eq_fig, self.cmp_eq_fig, self.flux_eq_fig):
            fig.x_range.start = fig.x_range.reset_start = r_min - pad * dr
            fig.x_range.end = fig.x_range.reset_end = r_max + pad * dr
            fig.y_range.start = fig.y_range.reset_start = z_min - pad * dz
            fig.y_range.end = fig.y_range.reset_end = z_max + pad * dz

        # Compute pixel dimensions from wall aspect ratio
        aspect = dr / dz if dz > 0 else 1.0
        eq_height = 600
        eq_width = max(300, int(eq_height * aspect))
        for fig in (self.init_eq_fig, self.inc_eq_fig, self.cmp_eq_fig):
            fig.height = eq_height
            fig.width = eq_width
        # Extra width for the flux figure to accommodate the colour bar
        self.flux_eq_fig.height = eq_height
        self.flux_eq_fig.width = eq_width + 80

        # --- Separatrix comparison figure ---
        self._cmp_wall_fill_src.data = dict(R=wall_r, Z=wall_z)
        self._cmp_wall_src.data = dict(R=wall_r, Z=wall_z)
        self._cmp_coil_pt_src.data = dict(x=cd_init["pt_x"], y=cd_init["pt_y"],
                                           color=cd_init["pt_c"], name=cd_init["pt_n"])
        self._cmp_coil_cr_src.data = dict(x=cd_init["cr_x"], y=cd_init["cr_y"],
                                           color=cd_init["cr_c"], name=cd_init["cr_n"])
        self._cmp_coil_rc_src.data = dict(xs=cd_init["rc_xs"], ys=cd_init["rc_ys"],
                                           fill=cd_init["rc_f"], edge=cd_init["rc_e"],
                                           name=cd_init["rc_n"])
        # Reuse the separatrix data already computed for the individual figures
        self._cmp_init_sep_src.data = dict(
            xs=list(self._init_sep_src.data["xs"]),
            ys=list(self._init_sep_src.data["ys"]),
        )
        if inc and "psi_2D" in inc:
            self._cmp_inc_sep_src.data = dict(
                xs=list(self._inc_sep_src.data["xs"]),
                ys=list(self._inc_sep_src.data["ys"]),
            )

        # --- Coil flux change figure ---
        self._flux_wall_fill_src.data = dict(R=wall_r, Z=wall_z)
        self._flux_wall_src.data = dict(R=wall_r, Z=wall_z)
        self._flux_coil_pt_src.data = dict(x=cd_init["pt_x"], y=cd_init["pt_y"],
                                            color=cd_init["pt_c"], name=cd_init["pt_n"])
        self._flux_coil_cr_src.data = dict(x=cd_init["cr_x"], y=cd_init["cr_y"],
                                            color=cd_init["cr_c"], name=cd_init["cr_n"])
        self._flux_coil_rc_src.data = dict(xs=cd_init["rc_xs"], ys=cd_init["rc_ys"],
                                            fill=cd_init["rc_f"], edge=cd_init["rc_e"],
                                            name=cd_init["rc_n"])
        if hasattr(opt, "optimised_eq"):
            flux_diag = opt.separatrix_coil_flux_change()
            change_pct = list(np.abs(flux_diag["delta_psi_mach_rel"]) * 100.0)
            self._flux_sep_src.data = dict(
                R=list(flux_diag["R_lcfs"]),
                Z=list(flux_diag["Z_lcfs"]),
                change_pct=change_pct,
            )
            # Set colour map range from 0 to max magnitude
            vmax = max(change_pct) if max(change_pct) > 0 else 1.0
            self._flux_cmap.low = 0.0
            self._flux_cmap.high = vmax
        else:
            self._flux_sep_src.data = dict(R=[], Z=[], change_pct=[])

        # --- Coil masks ---
        cd_init_masks = SetupTab._collect_coil_data(tokamak_initial)
        self._cached_mask_xs = cd_init_masks[1]
        self._cached_mask_ys = cd_init_masks[2]
        if self.show_masks.value:
            mask_data = dict(xs=self._cached_mask_xs, ys=self._cached_mask_ys)
        else:
            mask_data = dict(xs=[], ys=[])
        self._init_mask_src.data = mask_data
        self._inc_mask_src.data = mask_data
        self._cmp_mask_src.data = mask_data
        self._flux_mask_src.data = mask_data

    # ------------------------------------------------------------------
    def _on_refresh(self, event):
        opt = self.state.get("optimiser")
        if opt is None:
            self.status.object = "No optimiser available. Run one first."
            self.status.alert_type = "warning"
            return
        self._populate(opt)

    # ------------------------------------------------------------------
    def _populate(self, opt):
        """Fill all charts from an Optimiser instance."""
        n = len(opt.tracking_cost)
        xs = list(range(n))

        self._cost_source.data = dict(
            x=xs,
            total=list(opt.tracking_cost),
            strike=list(opt.tracking_cost_strike_point_distance),
            cl=list(opt.tracking_cost_connection_length),
            coil=list(opt.tracking_cost_coil_currents),
            xpt=list(opt.tracking_cost_xpoint_regions),
        )
        self._temp_source.data = dict(x=xs, temperature=list(opt.tracking_temperature))
        self._accept_source.data = dict(x=xs, rate=list(opt.tracking_acceptance_rate))

        # Incumbent vertical line
        inc = opt.incumbent_data
        inc_iter = inc.get("iteration_num") if inc else None
        if inc_iter is not None:
            self._vline_src.data = dict(x=[inc_iter])
        else:
            self._vline_src.data = dict(x=[])

        cl_data = dict(x=xs)
        for region in opt.divertor_regions:
            cl_data[region] = list(opt.tracking_connection_length[region])
            colour = opt.divertor_data[region].get("colour", "gray")
            self.cl_fig.line("x", region, source=self._cl_source, color=colour, legend_label=region)
        self._cl_source.data = cl_data

        # Bar chart — current comparison
        if hasattr(opt, "optimised_tokamak"):
            names = list(opt.tokamak_opt.coilset.keys())
            initial = [c * 1e-3 for c in opt.tokamak_initial.get_currents()]
            optimised = [c * 1e-3 for c in opt.optimised_tokamak.get_currents()]
            y_pos = list(range(len(names)))
            self._bar_source.data = dict(
                names=names,
                x_initial=[y - 0.18 for y in y_pos],
                x_optimised=[y + 0.18 for y in y_pos],
                initial=initial,
                optimised=optimised,
            )
            self.bar_fig.yaxis.ticker = y_pos
            self.bar_fig.yaxis.major_label_overrides = {i: n for i, n in enumerate(names)}

        # Incumbent summary
        inc = opt.incumbent_data
        if inc:
            html = "<h4>Incumbent State</h4><ul>"
            html += f"<li><b>Iteration:</b> {inc.get('iteration_num', '?')}</li>"
            html += f"<li><b>Cost:</b> {inc.get('cost', '?'):.6f}</li>"
            for region in opt.divertor_regions:
                cl = inc["divertors"][region]["connection_length"]
                html += f"<li><b>L<sub>//,{region}</sub>:</b> {cl:.3f} m</li>"
            html += "</ul>"

            # Separatrix coil flux change diagnostic
            if hasattr(opt, "optimised_eq"):
                flux_diag = opt.separatrix_coil_flux_change()
                html += "<h4>Separatrix Coil Flux Change</h4><ul>"
                html += f"<li><b>Max |Δψ<sub>mach</sub>|:</b> {flux_diag['max_abs_change']:.4e} Wb ({flux_diag['max_rel_change']*100:.4f}%)</li>"
                html += f"<li><b>Mean |Δψ<sub>mach</sub>|:</b> {flux_diag['mean_abs_change']:.4e} Wb ({flux_diag['mean_rel_change']*100:.4f}%)</li>"
                html += "</ul>"

            self.incumbent_pane.object = html
        # Equilibrium figures
        self._populate_eq_figures(opt)
        self.status.object = f"Analysis loaded — {n - 1} iterations."
        self.status.alert_type = "success"

    # ------------------------------------------------------------------
    def _on_save_pickle(self, event):
        """Serialise the optimiser to memory and trigger an HTTP download."""
        from forge.gui.app import _DOWNLOAD_STORE

        opt = self.state.get("optimiser")
        if opt is None:
            self.status.object = "Nothing to save."
            self.status.alert_type = "warning"
            return
        self.status.object = "Pickling Optimiser..."
        self.status.alert_type = "warning"
        print("Pickling Optimiser...", flush=True)
        t0 = time.monotonic()
        opt_copy = copy.copy(opt)
        for attr in _UNPICKLABLE_ATTRS:
            if hasattr(opt_copy, attr):
                delattr(opt_copy, attr)
        filename = "forge_optimiser.pkl"
        buf = _ProgressBytesIO(logger=logger, log_interval_mb=50)
        pickle.dump(opt_copy, buf)
        data = buf.getvalue()
        size_mb = len(data) / (1024 * 1024)
        _DOWNLOAD_STORE[filename] = data
        del buf  # free the duplicate copy
        elapsed = time.monotonic() - t0
        msg = f"Optimiser pickling complete ({size_mb:.1f} MB, {elapsed:.1f} s). Starting download..."
        print(msg, flush=True)
        self.status.object = msg
        self.status.alert_type = "success"
        # Trigger the browser download by setting the hidden TextInput value,
        # which fires a jscallback that opens the download URL.
        self._download_trigger.value = ""
        self._download_trigger.value = f"/download/{filename}"

    # ------------------------------------------------------------------
    def _save_geqdsk_callback(self):
        """Generate a GEQDSK BytesIO for the equilibrium FileDownload."""
        from freeqdsk import geqdsk

        opt = self.state.get("optimiser")
        if opt is None or not hasattr(opt, "optimised_eq"):
            self.status.object = "No optimised equilibrium available."
            self.status.alert_type = "warning"
            return io.BytesIO(b"")
        logger.info("Preparing GEQDSK...")
        print("Preparing GEQDSK...", flush=True)
        t0 = time.monotonic()
        eq = opt.optimised_eq
        rcentr = 1.0
        data = {
            'nx': eq.nR,
            'ny': eq.nZ,
            'rdim': eq.R_max - eq.R_min,
            'zdim': eq.Z_max - eq.Z_min,
            'rcentr': rcentr,
            'rleft': eq.R_min,
            'zmid': 0.5 * (eq.Z_min + eq.Z_max),
            'rmagx': eq.R_mag,
            'zmagx': eq.Z_mag,
            'simagx': eq.psi_axis,
            'sibdry': eq.psi_lcfs,
            'bcentr': eq.fvac / rcentr,
            'cpasma': eq.plasma_current,
            'fpol': eq.fpol_data,
            'pres': eq.pressure_data,
            'ffprime': eq.ffprime_data,
            'pprime': eq.pprime_data,
            'psi': eq.psi_2D,
            'qpsi': eq.q_data,
            'nbdry': len(eq.R_lcfs),
            'nlim': len(eq.wall_R),
            'rbdry': eq.R_lcfs,
            'zbdry': eq.Z_lcfs,
            'rlim': eq.wall_R,
            'zlim': eq.wall_Z,
        }
        sbuf = io.StringIO()
        geqdsk.write(data, sbuf, "FORGE", 0, 0)
        buf = io.BytesIO(sbuf.getvalue().encode("utf-8"))
        elapsed = time.monotonic() - t0
        msg = f"GEQDSK ready ({elapsed:.1f} s)."
        logger.info(msg)
        print(msg, flush=True)
        self.status.object = "GEQDSK ready for download."
        self.status.alert_type = "success"
        return buf

    # ------------------------------------------------------------------
    def _save_machine_callback(self):
        """Generate a JSON BytesIO for the machine FileDownload."""
        opt = self.state.get("optimiser")
        if opt is None or not hasattr(opt, "optimised_tokamak"):
            self.status.object = "No optimised machine available."
            self.status.alert_type = "warning"
            return io.BytesIO(b"")
        logger.info("Preparing Machine JSON...")
        print("Preparing Machine JSON...", flush=True)
        t0 = time.monotonic()
        content = fancy_json_string(opt.optimised_tokamak.to_dict())
        buf = io.BytesIO(content.encode("utf-8"))
        elapsed = time.monotonic() - t0
        msg = f"Machine JSON ready ({elapsed:.1f} s)."
        logger.info(msg)
        print(msg, flush=True)
        self.status.object = "Machine JSON ready for download."
        self.status.alert_type = "success"
        return buf

    # ------------------------------------------------------------------
    def _on_load_pickle(self, event):
        path = (self.load_pickle_path_input.value or "").strip()
        if not path:
            self.status.object = "Please enter a pickle file path."
            self.status.alert_type = "warning"
            return
        import os
        if not os.path.isfile(path):
            self.status.object = f"File not found: {path}"
            self.status.alert_type = "danger"
            return
        try:
            opt = load_optimiser(path)
            self.state["optimiser"] = opt
            self._populate(opt)
            self.status.object = "Loaded optimiser from pickle."
            self.status.alert_type = "success"
        except Exception as exc:
            self.status.object = f"Failed to load: {exc}"
            self.status.alert_type = "danger"

    # ------------------------------------------------------------------
    def _on_upload_pickle_click(self, event):
        """Open a native file picker and POST the file via fetch() to /upload.

        This avoids the Panel FileInput base64 path which crashes the browser
        on large files.  The fetch FormData API streams without encoding.
        """
        # The actual work is done by the jscallback attached in __init__.
        # The Python-side click handler just resets the trigger.
        self._upload_id_trigger.value = ""

    def _on_upload_complete(self, event):
        """Called when the browser upload finishes and sets the upload_id trigger."""
        upload_id = (event.new or "").strip()
        if not upload_id or upload_id == "uploading":
            if upload_id == "uploading":
                self.status.object = "Uploading pickle to server..."
                self.status.alert_type = "info"
            return
        if upload_id.startswith("ERROR:"):
            self.status.object = f"Upload failed: {upload_id[6:]}"
            self.status.alert_type = "danger"
            return
        from forge.gui.app import _UPLOAD_STORE
        data = _UPLOAD_STORE.pop(upload_id, None)
        if data is None:
            self.status.object = "Upload completed but data not found on server."
            self.status.alert_type = "danger"
            return
        try:
            opt = pickle.loads(data)
            del data  # free the raw bytes immediately
            self.state["optimiser"] = opt
            self._populate(opt)
            self.status.object = "Loaded optimiser from uploaded pickle."
            self.status.alert_type = "success"
        except Exception as exc:
            self.status.object = f"Failed to load uploaded pickle: {exc}"
            self.status.alert_type = "danger"

    # ------------------------------------------------------------------
    @property
    def panel(self):
        if hasattr(self, "_panel"):
            return self._panel
        self._sidebar_col = pn.Column(
            "### Actions",
            self.refresh_btn,
            pn.layout.Divider(),
            self.save_pickle_btn,
            self.save_geqdsk_btn,
            self.save_machine_btn,
            self._download_trigger,
            pn.layout.Divider(),
            "**Load Optimiser from pickle (.pkl)**",
            self.load_pickle_path_input,
            self.load_pickle_btn,
            self.upload_pickle_btn,
            self._upload_id_trigger,
            pn.layout.Divider(),
            self.show_masks,
            self.incumbent_pane,
            self.status,
            width=340,
            scroll=True,
        )
        sidebar = self._sidebar_col
        charts = pn.Column(
            self.cost_fig,
            self.strike_cost_fig,
            self.cl_cost_fig,
            self.coil_cost_fig,
            self.xpt_cost_fig,
            self.cl_fig,
            self.temp_fig,
            self.accept_fig,
            self.bar_fig,
            pn.Row(
                pn.pane.Bokeh(self.init_eq_fig),
                pn.pane.Bokeh(self.inc_eq_fig),
                pn.pane.Bokeh(self.cmp_eq_fig),
                pn.pane.Bokeh(self.flux_eq_fig),
            ),
            sizing_mode="stretch_both",
            scroll=True,
        )
        self._panel = pn.Row(sidebar, charts, sizing_mode="stretch_both")
        return self._panel
