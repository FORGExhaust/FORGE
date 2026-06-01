"""Optimisation tab — configure, run, and live-monitor a FORGE optimisation.

The optimisation runs in a background thread.  A periodic Panel callback
polls the Optimiser's tracking data and pushes updates to Bokeh
ColumnDataSources, giving smooth real-time chart updates over WebSocket.
"""

import io as _io
import json
import logging
import threading
import time

import numpy as np
import panel as pn
from bokeh.models import BasicTickFormatter, ColumnDataSource, Range1d
from bokeh.plotting import figure as bk_figure

from forge.io import fancy_json_string
from forge.optimise import Optimiser

logger = logging.getLogger(__name__)


class OptimisationTab:
    """Panel component for running and monitoring an optimisation."""

    def __init__(self, shared_state, setup_tab, geometry_tab):
        self.state = shared_state
        self.setup_tab = setup_tab
        self.geometry_tab = geometry_tab

        self._optimiser = None
        self._thread = None
        self._periodic_cb = None

        # --- Parameter widgets ---
        # Annealing schedule
        self.max_evals = pn.widgets.IntInput(name="Max evaluations", value=3000, start=100, step=100)
        self.step_size_factor = pn.widgets.FloatInput(name="Current step size factor", value=0.05, step=0.01, start=0.001)
        self.initial_temp = pn.widgets.FloatInput(name="Initial temperature", value=10.0, step=1.0, start=0.1)
        self.min_temp = pn.widgets.FloatInput(name="Min temperature", value=1.0, step=0.1, start=0.01)
        self.n_window = pn.widgets.IntInput(name="Window size", value=50, start=10, step=10)
        self.threshold_decay = pn.widgets.FloatInput(name="Threshold acceptance decay, \u03bb", value=2.0, step=0.1)
        self.init_threshold = pn.widgets.FloatInput(name="Initial threshold acceptance rate", value=0.75, step=0.05)
        self.cost_frac = pn.widgets.FloatInput(name="Cost termination fraction", value=0.01, step=0.005)
        self.max_cooling = pn.widgets.FloatInput(name="Max cooling factor, \u03b2_max", value=0.99, step=0.01, start=0.5, end=1.0)
        self.min_cooling = pn.widgets.FloatInput(name="Min cooling factor, \u03b2_min", value=0.9, step=0.01, start=0.5, end=1.0)

        # Cost weights
        self.w_strike = pn.widgets.FloatInput(name="Initial strike distance cost", value=1.0, step=0.1, start=0.0)
        self.w_conn = pn.widgets.FloatInput(name="Initial connection length cost", value=1.0, step=0.1, start=0.0)
        self.w_coil = pn.widgets.FloatInput(name="Initial coil currents cost", value=1.0, step=0.1, start=0.0)
        self.w_xpt = pn.widgets.FloatInput(name="Initial XPT regions cost", value=1.0, step=0.1, start=0.0)

        # Alpha / regularisation
        self.initial_alpha = pn.widgets.FloatInput(name="Initial \u03b1", value=50.0, step=5.0)
        self.alpha_update = pn.widgets.FloatInput(name="\u03b1 update factor", value=1.05, step=0.01, start=1.0)

        # Field-line tracing
        self.trace_step_size = pn.widgets.FloatInput(name="Trace step size (m)", value=0.15, step=0.01, start=0.01)
        self.trace_max_steps = pn.widgets.IntInput(name="Trace max steps", value=1000, start=100, step=100)
        self.psi_tol = pn.widgets.FloatInput(name="\u03C8 trace tolerance", value=0.01, step=0.005, start=0.001)

        # X-point regions
        self.max_disconnection = pn.widgets.FloatInput(name="Max magnetic disconnection factor", value=0.5, step=0.05, start=0.0, end=1.0)

        # Buffer penalty
        self.buffer_penalty = pn.widgets.FloatInput(name="Buffer penalty factor", value=1.05, step=0.01, start=1.0)

        # Feature toggles
        self.estimate_currents = pn.widgets.Checkbox(name="Estimate initial currents", value=False)
        self.use_buffers = pn.widgets.Checkbox(name="Use buffers", value=False)
        self.use_xpt = pn.widgets.Checkbox(name="Use X-point regions", value=False)
        self.detailed_logging = pn.widgets.Checkbox(name="Detailed logging", value=False)

        # --- Control buttons ---
        self.run_btn = pn.widgets.Button(name="▶ Run", button_type="success")
        self.stop_btn = pn.widgets.Button(name="■ Stop", button_type="danger", disabled=True)

        # --- Config save / load ---
        self.load_config_file = pn.widgets.FileInput(
            accept=".json", multiple=False, width=220,
        )
        self.load_config_file.param.watch(self._on_config_file_load, "value")
        self.save_config_btn = pn.widgets.FileDownload(
            callback=self._save_config_callback,
            filename="forge_config.json",
            label="Save config",
            button_type="primary",
            width=120,
        )

        self.status = pn.pane.Alert("Configure parameters and press Run.", alert_type="info")
        self.progress = pn.indicators.Progress(name="Progress", value=0, max=100, sizing_mode="stretch_width")

        # --- Live Bokeh plots ---
        self._cost_source = ColumnDataSource(data=dict(x=[], total=[], strike=[], cl=[], coil=[], xpt=[]))
        self._temp_source = ColumnDataSource(data=dict(x=[], temperature=[]))
        self._acceptance_source = ColumnDataSource(data=dict(x=[], rate=[]))
        self._cl_source = ColumnDataSource(data=dict(x=[]))  # per-region columns added dynamically
        self._alpha_source = ColumnDataSource(data=dict(x=[], alpha=[]))

        self.cost_fig = self._make_fig("Cost", "Iteration", "Cost (-)")
        self.cost_fig.line("x", "total", source=self._cost_source, color="black", legend_label="Total")
        self.cost_fig.line("x", "strike", source=self._cost_source, color="red", legend_label="Strike")
        self.cost_fig.line("x", "cl", source=self._cost_source, color="green", legend_label="L")
        self.cost_fig.line("x", "coil", source=self._cost_source, color="orange", legend_label="Coil I²")
        self.cost_fig.line("x", "xpt", source=self._cost_source, color="blue", legend_label="XPT")
        self.cost_fig.legend.click_policy = "hide"
        self.cost_fig.legend.orientation = "horizontal"
        self.cost_fig.legend.label_text_font_size = "8pt"
        self.cost_fig.legend.spacing = 8
        self.cost_fig.legend.padding = 2
        self.cost_fig.legend.margin = 2
        self.cost_fig.legend.background_fill_alpha = 0.7

        self.temp_fig = self._make_fig("Temperature", "Iteration", "T (-)")
        self.temp_fig.line("x", "temperature", source=self._temp_source, color="blue")

        self.accept_fig = self._make_fig("Acceptance Rate", "Iteration", "R (-)")
        self.accept_fig.scatter("x", "rate", source=self._acceptance_source, color="blue", size=2)
        # Dashed horizontal line for the current threshold acceptance rate
        self._threshold_rate_src = ColumnDataSource(data=dict(y=[]))
        self.accept_fig.ray(x=0, y="y", length=0, angle=0, source=self._threshold_rate_src,
                            color="red", line_dash="dashed", line_width=1)

        self.cl_fig = self._make_fig("Parallel Connection Length", "Iteration", "L (m)")
        # Invisible placeholder so Bokeh doesn't warn about an empty plot
        # at startup. Real renderers are added when the user clicks Run.
        self.cl_fig.line([], [], visible=False)

        self.alpha_fig = self._make_fig("\u03b1", "Iteration", "\u03b1 (-)")
        self.alpha_fig.line("x", "alpha", source=self._alpha_source, color="blue")

        # --- Live equilibrium figure (matches Setup-tab style) ---
        self._eq_wall_fill_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._eq_wall_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._eq_contour_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._eq_sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._eq_coil_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._eq_coil_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._eq_coil_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        self._eq_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._fl_sources = {}  # per-region field-line ColumnDataSources

        # Incumbent vertical line source for time-series plots
        self._incumbent_vline_src = ColumnDataSource(data=dict(x=[], y0=[], y1=[]))

        self.eq_fig = bk_figure(
            title="Equilibrium (live)",
            x_axis_label="R (m)", y_axis_label="Z (m)",
            height=600, width=500,
            x_range=Range1d(0, 1),
            y_range=Range1d(0, 1),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        self.eq_fig.xaxis.axis_label_text_font_style = "normal"
        self.eq_fig.yaxis.axis_label_text_font_style = "normal"
        self.eq_fig.title.text_font_style = "normal"

        # Render order: white wall fill → contours → separatrix → wall outline → coils → masks → field lines
        self.eq_fig.patch("R", "Z", source=self._eq_wall_fill_src,
                          fill_color="white", line_color=None)
        self.eq_fig.multi_line("xs", "ys", source=self._eq_contour_src,
                               line_alpha=0.35, line_color="gray")
        self.eq_fig.multi_line("xs", "ys", source=self._eq_sep_src,
                               line_color="red", line_width=2)
        self.eq_fig.line("R", "Z", source=self._eq_wall_src,
                         line_color="black", line_width=2)
        self.eq_fig.scatter("x", "y", source=self._eq_coil_pt_src, color="color",
                            size=10, marker="circle")
        self.eq_fig.scatter("x", "y", source=self._eq_coil_cr_src, color="color",
                            size=8, marker="x")
        self.eq_fig.patches("xs", "ys", source=self._eq_coil_rc_src,
                            fill_color="fill", line_color="edge", line_width=0.3)
        self.eq_fig.patches("xs", "ys", source=self._eq_mask_src,
                            fill_color="orange", line_color="black",
                            line_width=1.0, fill_alpha=1.0)

        # Cache the R_2D / Z_2D / psi_lcfs for contour recomputation
        self._eq_R2D = None
        self._eq_Z2D = None
        self._eq_psi_lcfs = None

        # --- Incumbent equilibrium figure ---
        self._inc_wall_fill_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._inc_wall_src = ColumnDataSource(data=dict(R=[], Z=[]))
        self._inc_contour_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._inc_sep_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._inc_coil_pt_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._inc_coil_cr_src = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._inc_coil_rc_src = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))
        self._inc_mask_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._inc_fl_sources = {}  # per-region field-line ColumnDataSources

        self.inc_fig = bk_figure(
            title="Incumbent",
            x_axis_label="R (m)", y_axis_label="Z (m)",
            height=600, width=500,
            x_range=Range1d(0, 1),
            y_range=Range1d(0, 1),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
            background_fill_color="#d9d9d9",
        )
        self.inc_fig.xaxis.axis_label_text_font_style = "normal"
        self.inc_fig.yaxis.axis_label_text_font_style = "normal"
        self.inc_fig.title.text_font_style = "normal"

        self.inc_fig.patch("R", "Z", source=self._inc_wall_fill_src,
                           fill_color="white", line_color=None)
        self.inc_fig.multi_line("xs", "ys", source=self._inc_contour_src,
                                line_alpha=0.35, line_color="gray")
        self.inc_fig.multi_line("xs", "ys", source=self._inc_sep_src,
                                line_color="red", line_width=2)
        self.inc_fig.line("R", "Z", source=self._inc_wall_src,
                          line_color="black", line_width=2)
        self.inc_fig.scatter("x", "y", source=self._inc_coil_pt_src, color="color",
                             size=10, marker="circle")
        self.inc_fig.scatter("x", "y", source=self._inc_coil_cr_src, color="color",
                             size=8, marker="x")
        self.inc_fig.patches("xs", "ys", source=self._inc_coil_rc_src,
                             fill_color="fill", line_color="edge", line_width=0.3)
        self.inc_fig.patches("xs", "ys", source=self._inc_mask_src,
                             fill_color="orange", line_color="black",
                             line_width=1.0, fill_alpha=1.0)

        # --- Initial LCFS overlay (cyan) on both eq plots ---
        self._init_lcfs_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.eq_fig.multi_line("xs", "ys", source=self._init_lcfs_src,
                               line_color="cyan", line_width=2, line_dash="dashed")
        self.inc_fig.multi_line("xs", "ys", source=self._init_lcfs_src,
                                line_color="cyan", line_width=2, line_dash="dashed")
        self._cached_init_lcfs_xs = []
        self._cached_init_lcfs_ys = []

        # --- Coil mask toggle for Optimise tab eq plots ---
        self.show_masks = pn.widgets.Checkbox(name="Show coil masks", value=True)
        self.show_masks.param.watch(self._on_mask_toggle, "value")
        self._cached_mask_xs = []
        self._cached_mask_ys = []

        # --- Initial LCFS overlay toggle ---
        self.show_initial_lcfs = pn.widgets.Checkbox(name="Show initial LCFS", value=True)
        self.show_initial_lcfs.param.watch(self._on_lcfs_toggle, "value")

        # Cache the last known incumbent iteration to avoid redundant refreshes
        self._last_incumbent_iter = -1
        self._last_live_contour_iter = -1
        self._last_streamed_idx = 0
        self._last_contour_time = 0.0
        self._contour_scheduled = False
        self._pending_live_psi = None
        self._pending_live_n = 0
        self._pending_live_fl = {}
        self._pending_inc = None

        # --- Log pane ---
        self.log_pane = pn.pane.HTML(
            "<pre style='max-height:200px;overflow-y:auto;font-size:11px;'></pre>",
            sizing_mode="stretch_width",
        )
        self._log_lines = []
        self._log_handler = _PanelLogHandler(self._log_lines)
        logging.getLogger("forge").addHandler(self._log_handler)

        # Wire buttons
        self.run_btn.on_click(self._on_run)
        self.stop_btn.on_click(self._on_stop)

        # Record the number of static renderers on each figure so that
        # dynamic glyphs added during a run can be stripped before the
        # next run (prevents renderer / legend-item accumulation).
        self._eq_fig_static_n = len(self.eq_fig.renderers)
        self._inc_fig_static_n = len(self.inc_fig.renderers)
        self._cl_fig_static_n = len(self.cl_fig.renderers)
        self._cost_fig_static_n = len(self.cost_fig.renderers)
        self._temp_fig_static_n = len(self.temp_fig.renderers)
        self._accept_fig_static_n = len(self.accept_fig.renderers)
        self._alpha_fig_static_n = len(self.alpha_fig.renderers)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_fig(title, xlabel, ylabel):
        f = bk_figure(
            title=title, x_axis_label=xlabel, y_axis_label=ylabel,
            height=200, sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset",
        )
        f.title.text_font_style = "normal"
        f.xaxis.axis_label_text_font_style = "normal"
        f.yaxis.axis_label_text_font_style = "normal"
        f.x_range.range_padding = 0.0
        f.x_range.start = 0
        f.xaxis.formatter = BasicTickFormatter(use_scientific=False)
        return f

    # ------------------------------------------------------------------
    def _refresh_contours(self, psi_2D, target="live"):
        """Recompute contour lines and separatrix from a 2-D flux array.

        Uses matplotlib's Agg backend (no display) with 40 levels for speed.

        Parameters
        ----------
        psi_2D : 2-D numpy array
            The flux map to contour.
        target : str
            ``"live"`` updates the live equilibrium plot sources;
            ``"incumbent"`` updates the incumbent plot sources.
        """
        if self._eq_R2D is None:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_tmp, ax_tmp = plt.subplots()

        # Contour lines (40 levels — fast yet detailed enough for live feedback)
        cs = ax_tmp.contour(self._eq_R2D, self._eq_Z2D, psi_2D, levels=40)
        xs_all, ys_all = [], []
        for collection in cs.collections:
            for path in collection.get_paths():
                verts = path.vertices
                xs_all.append(verts[:, 0].tolist())
                ys_all.append(verts[:, 1].tolist())

        # Separatrix at psi_lcfs
        sep_xs, sep_ys = [], []
        if self._eq_psi_lcfs is not None:
            cs_sep = ax_tmp.contour(
                self._eq_R2D, self._eq_Z2D, psi_2D,
                levels=[self._eq_psi_lcfs],
            )
            for collection in cs_sep.collections:
                for path in collection.get_paths():
                    verts = path.vertices
                    sep_xs.append(verts[:, 0].tolist())
                    sep_ys.append(verts[:, 1].tolist())

        plt.close(fig_tmp)

        if target == "incumbent":
            self._inc_contour_src.data = dict(xs=xs_all, ys=ys_all)
            self._inc_sep_src.data = dict(xs=sep_xs, ys=sep_ys)
        else:
            self._eq_contour_src.data = dict(xs=xs_all, ys=ys_all)
            self._eq_sep_src.data = dict(xs=sep_xs, ys=sep_ys)

    # ------------------------------------------------------------------
    def _on_run(self, event):
        """Construct the Optimiser and launch the background thread."""
        self._log_lines.clear()
        logger.info("Launching optimisation...")
        eq = self.state.get("eq")
        tokamak = self.state.get("tokamak")
        if eq is None or tokamak is None:
            self.status.object = "Load a machine in the Setup tab first."
            self.status.alert_type = "danger"
            return

        logger.info("Collecting geometry data...")
        divertor_data = self.geometry_tab.build_divertor_data()
        if divertor_data is None:
            self.status.object = "Define at least one divertor region with a strike geometry in the Geometry tab."
            self.status.alert_type = "danger"
            return

        constraints = self.setup_tab.get_constraints_dict()
        buffers = self.geometry_tab.get_buffer_defs() if self.use_buffers.value else None
        xpt_regions = self.geometry_tab.get_xpoint_region_defs() if self.use_xpt.value else None

        stop_event = threading.Event()

        logger.info("Building Optimiser (this may take a few seconds)...")
        try:
            opt = Optimiser(
                eq=eq,
                tokamak_initial=tokamak,
                divertor_data=divertor_data,
                max_evals=self.max_evals.value,
                current_step_size_factor=self.step_size_factor.value,
                estimate_initial_currents=False,
                initial_temperature=self.initial_temp.value,
                min_temperature=self.min_temp.value,
                n_window=self.n_window.value,
                threshold_acceptance_rate_decay=self.threshold_decay.value,
                initial_threshold_acceptance_rate=self.init_threshold.value,
                cost_termination_fraction=self.cost_frac.value,
                constraints=constraints,
                field_line_trace_step_size=self.trace_step_size.value,
                field_line_trace_max_steps=self.trace_max_steps.value,
                field_line_trace_psi_tollerance=self.psi_tol.value,
                buffer_intersection_penalty_factor=self.buffer_penalty.value,
                initial_total_connection_length_cost=self.w_conn.value,
                initial_total_strike_point_distance_cost=self.w_strike.value,
                initial_coil_currents_cost=self.w_coil.value,
                initial_xpoint_regions_cost=self.w_xpt.value,
                use_buffers=self.use_buffers.value,
                buffers=buffers,
                use_xpoint_regions=self.use_xpt.value,
                xpoint_regions=xpt_regions,
                max_magnetic_disconnection_factor=self.max_disconnection.value,
                initial_alpha=self.initial_alpha.value,
                alpha_update_factor=self.alpha_update.value,
                max_cooling_factor=self.max_cooling.value,
                min_cooling_factor=self.min_cooling.value,
                detailed_logging=self.detailed_logging.value,
                stop_event=stop_event,
            )
        except Exception as exc:
            logger.exception("Failed to construct Optimiser")
            self.status.object = f"Error building optimiser: {exc}"
            self.status.alert_type = "danger"
            return

        logger.info("Optimiser built successfully.")
        self._optimiser = opt
        self.state["optimiser"] = opt

        # Batch all Bokeh model updates into a single combined websocket
        # message.  Without this, each ColumnDataSource / renderer / range
        # assignment triggers its own patch, and rapid-fire patches can
        # produce "Dropping a patch" warnings.
        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")

        logger.info("Setting up plots...")
        try:
            self._setup_run_plots(opt, eq, tokamak)
        finally:
            if doc is not None:
                doc.unhold()

        # UI state
        self.run_btn.disabled = True
        self.stop_btn.disabled = False
        self.progress.value = 0
        self._last_streamed_idx = 0
        self._last_incumbent_iter = -1
        self._last_live_contour_iter = -1
        self._last_contour_time = 0.0
        self._contour_scheduled = False
        self.status.object = "Optimisation running\u2026"
        self.status.alert_type = "info"

        # Launch background thread
        logger.info("Starting optimisation thread...")
        self._thread = threading.Thread(target=self._run_optimisation, daemon=True)
        self._thread.start()

        # Flush log pane so launch messages appear immediately
        if self._log_lines:
            html = "<pre style='max-height:200px;overflow-y:auto;font-size:11px;'>"
            html += "\n".join(self._log_lines[-50:])
            html += "</pre>"
            self.log_pane.object = html

        # Start periodic UI update (every 1 s — fast enough for visual
        # feedback, slow enough to let the browser process user input).
        self._periodic_cb = pn.state.add_periodic_callback(self._poll_updates, period=1000)

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------
    def _gather_optimiser_kwargs(self):
        """Collect all scalar widget values as an Optimiser-kwarg dict."""
        return {
            "max_evals": self.max_evals.value,
            "current_step_size_factor": self.step_size_factor.value,
            "estimate_initial_currents": self.estimate_currents.value,
            "initial_temperature": self.initial_temp.value,
            "min_temperature": self.min_temp.value,
            "threshold_acceptance_rate_decay": self.threshold_decay.value,
            "initial_threshold_acceptance_rate": self.init_threshold.value,
            "n_window": self.n_window.value,
            "cost_termination_fraction": self.cost_frac.value,
            "max_cooling_factor": self.max_cooling.value,
            "min_cooling_factor": self.min_cooling.value,
            "initial_total_connection_length_cost": self.w_conn.value,
            "initial_total_strike_point_distance_cost": self.w_strike.value,
            "initial_coil_currents_cost": self.w_coil.value,
            "initial_xpoint_regions_cost": self.w_xpt.value,
            "initial_alpha": self.initial_alpha.value,
            "alpha_update_factor": self.alpha_update.value,
            "field_line_trace_step_size": self.trace_step_size.value,
            "field_line_trace_max_steps": self.trace_max_steps.value,
            "field_line_trace_psi_tollerance": self.psi_tol.value,
            "buffer_intersection_penalty_factor": self.buffer_penalty.value,
            "max_magnetic_disconnection_factor": self.max_disconnection.value,
            "use_buffers": self.use_buffers.value,
            "use_xpoint_regions": self.use_xpt.value,
            "detailed_logging": self.detailed_logging.value,
        }

    def _apply_optimiser_kwargs(self, kwargs):
        """Set widget values from a dict of Optimiser kwargs."""
        _MAP = {
            "max_evals": self.max_evals,
            "current_step_size_factor": self.step_size_factor,
            "estimate_initial_currents": self.estimate_currents,
            "initial_temperature": self.initial_temp,
            "min_temperature": self.min_temp,
            "threshold_acceptance_rate_decay": self.threshold_decay,
            "initial_threshold_acceptance_rate": self.init_threshold,
            "n_window": self.n_window,
            "cost_termination_fraction": self.cost_frac,
            "max_cooling_factor": self.max_cooling,
            "min_cooling_factor": self.min_cooling,
            "initial_total_connection_length_cost": self.w_conn,
            "initial_total_strike_point_distance_cost": self.w_strike,
            "initial_coil_currents_cost": self.w_coil,
            "initial_xpoint_regions_cost": self.w_xpt,
            "initial_alpha": self.initial_alpha,
            "alpha_update_factor": self.alpha_update,
            "field_line_trace_step_size": self.trace_step_size,
            "field_line_trace_max_steps": self.trace_max_steps,
            "field_line_trace_psi_tollerance": self.psi_tol,
            "buffer_intersection_penalty_factor": self.buffer_penalty,
            "max_magnetic_disconnection_factor": self.max_disconnection,
            "use_buffers": self.use_buffers,
            "use_xpoint_regions": self.use_xpt,
            "detailed_logging": self.detailed_logging,
        }
        for key, widget in _MAP.items():
            if key in kwargs:
                widget.value = kwargs[key]

    def _save_config_callback(self):
        """Generate a JSON BytesIO for the config FileDownload."""
        from forge.io import _STRIKE_KEYS, _REGION_SETTING_KEYS, _to_serialisable

        divertor_data = self.geometry_tab.build_divertor_data() or {}
        buffers = self.geometry_tab.get_buffer_defs()
        xpoint_regions = self.geometry_tab.get_xpoint_region_defs()
        constraints = self.setup_tab.get_constraints_dict()
        opt_kwargs = self._gather_optimiser_kwargs()

        # --- geometry section ---
        all_regions: set[str] = set(divertor_data)
        if buffers is not None:
            all_regions |= set(buffers)
        if xpoint_regions is not None:
            all_regions |= set(xpoint_regions)

        geometry = {}
        for region in sorted(all_regions):
            entry = {}
            dd = divertor_data.get(region, {})
            strike = {k: _to_serialisable(dd[k]) for k in _STRIKE_KEYS if k in dd}
            if strike:
                entry["strike"] = strike
            for k in _REGION_SETTING_KEYS:
                if k in dd:
                    entry[k] = _to_serialisable(dd[k])
            if buffers is not None and region in buffers:
                entry["buffers"] = [
                    {bk: _to_serialisable(bv) for bk, bv in buf.items()}
                    for buf in buffers[region]
                ]
            if xpoint_regions is not None and region in xpoint_regions:
                entry["xpoint_regions"] = {
                    pk: _to_serialisable(pv)
                    for pk, pv in xpoint_regions[region].items()
                }
            geometry[region] = entry

        payload = {}
        if geometry:
            payload["geometry"] = geometry
        if constraints is not None:
            payload["constraints"] = constraints
        if opt_kwargs:
            payload["optimiser"] = opt_kwargs

        content = fancy_json_string(payload)
        return _io.BytesIO(content.encode("utf-8"))

    def _on_config_file_load(self, event):
        """Load a full optimisation config from a JSON file."""
        raw = event.new
        if raw is None:
            return

        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            data = json.loads(text)
        except Exception as exc:
            self.status.object = f"Failed to parse config JSON: {exc}"
            self.status.alert_type = "danger"
            return

        if not isinstance(data, dict):
            self.status.object = "Config JSON must be a dict."
            self.status.alert_type = "danger"
            return

        parts = []

        # --- geometry → geometry tab ---
        geometry = data.get("geometry", {})
        if geometry:
            # Build a synthetic event-like load via the geometry tab's internal API
            loaded_regions = 0
            loaded_buffers = 0
            loaded_xpt = 0
            for region, entry in geometry.items():
                if region not in self.geometry_tab._region_data:
                    continue
                d = self.geometry_tab._region_data[region]

                strike = entry.get("strike", {})
                sr = strike.get("strike_R")
                sz = strike.get("strike_Z")
                if sr is not None and sz is not None:
                    d["strike"]["x"] = sr if isinstance(sr, list) else [sr]
                    d["strike"]["y"] = sz if isinstance(sz, list) else [sz]
                else:
                    d["strike"]["x"] = []
                    d["strike"]["y"] = []

                if "connection_length_multiplication_factor_zero" in entry:
                    d["cl_mult_factor_zero"] = entry["connection_length_multiplication_factor_zero"]
                if "weight_connection_length" in entry:
                    d["weight_connection_length"] = entry["weight_connection_length"]
                if "weight_strike_point_distance" in entry:
                    d["weight_strike_point_distance"] = entry["weight_strike_point_distance"]
                if "weight_xpoint_region" in entry:
                    d["weight_xpoint_region"] = entry["weight_xpoint_region"]

                bufs = entry.get("buffers")
                if bufs is not None and isinstance(bufs, list):
                    d["buffers"] = list(bufs)
                    loaded_buffers += len(bufs)

                xpt = entry.get("xpoint_regions")
                if xpt is not None:
                    R = xpt.get("R", [])
                    Z = xpt.get("Z", [])
                    if R and Z:
                        d["xpt"]["xs"] = [list(R)]
                        d["xpt"]["ys"] = [list(Z)]
                        loaded_xpt += 1

                if region in self.geometry_tab.region_enabled:
                    self.geometry_tab.region_enabled[region].value = True
                loaded_regions += 1

            # Refresh geometry canvas
            doc = pn.state.curdoc
            if doc is not None:
                doc.hold("combine")
            try:
                self.geometry_tab._load_region(self.geometry_tab._active_region)
                self.geometry_tab._update_ghost_layers()
            finally:
                if doc is not None:
                    doc.unhold()

            geo_parts = [f"{loaded_regions} region(s)"]
            if loaded_buffers:
                geo_parts.append(f"{loaded_buffers} buffer(s)")
            if loaded_xpt:
                geo_parts.append(f"{loaded_xpt} XPT region(s)")
            parts.append("Geometry: " + ", ".join(geo_parts))

        # --- constraints → setup tab ---
        constraints = data.get("constraints", {})
        if constraints:
            ann = constraints.get("annealing", {})
            if ann:
                self.setup_tab._apply_constraints(ann)
            parts.append("Constraints")

        # --- optimiser scalars → this tab's widgets ---
        opt_kwargs = data.get("optimiser", {})
        if opt_kwargs:
            self._apply_optimiser_kwargs(opt_kwargs)
            parts.append(f"{len(opt_kwargs)} optimiser setting(s)")

        self.load_config_file.value = None

        if parts:
            self.status.object = "Loaded config: " + "; ".join(parts) + "."
            self.status.alert_type = "success"
        else:
            self.status.object = "Config file was empty or had no recognised sections."
            self.status.alert_type = "warning"

    # ------------------------------------------------------------------
    def _setup_run_plots(self, opt, eq, tokamak):
        """Initialise / reset all Bokeh plots for a new optimisation run.

        Extracted so the caller can wrap it in ``doc.hold("combine")``.
        """
        # Strip dynamic renderers added by a previous run so they don't
        # accumulate (this also eliminates the E-1006 legend-source mismatch
        # error caused by stale cl_fig legend items).
        self.eq_fig.renderers = self.eq_fig.renderers[:self._eq_fig_static_n]
        self.inc_fig.renderers = self.inc_fig.renderers[:self._inc_fig_static_n]
        self.cl_fig.renderers = self.cl_fig.renderers[:self._cl_fig_static_n]
        self.cost_fig.renderers = self.cost_fig.renderers[:self._cost_fig_static_n]
        self.temp_fig.renderers = self.temp_fig.renderers[:self._temp_fig_static_n]
        self.accept_fig.renderers = self.accept_fig.renderers[:self._accept_fig_static_n]
        self.alpha_fig.renderers = self.alpha_fig.renderers[:self._alpha_fig_static_n]
        if self.cl_fig.legend:
            self.cl_fig.legend[0].items = []

        # Reset plot sources
        self._cost_source.data = dict(x=[], total=[], strike=[], cl=[], coil=[], xpt=[])
        self._temp_source.data = dict(x=[], temperature=[])
        self._acceptance_source.data = dict(x=[], rate=[])
        self._alpha_source.data = dict(x=[], alpha=[])

        # Prepare per-region connection length lines
        cl_data = dict(x=[])
        for region in opt.divertor_regions:
            cl_data[region] = []
            colour = opt.divertor_data[region]["colour"]
            short_label = opt.divertor_data[region]["short_label"]
            self.cl_fig.line("x", region, source=self._cl_source, color=colour, legend_label=short_label)
        self._cl_source.data = cl_data

        # Initialise the live equilibrium plot (contour style matching Setup tab)
        wall_r = list(tokamak.wall_R)
        wall_z = list(tokamak.wall_Z)
        self._eq_wall_fill_src.data = dict(R=wall_r, Z=wall_z)
        self._eq_wall_src.data = dict(R=wall_r, Z=wall_z)

        # Coil glyphs + masks (reuse the Setup tab helper)
        cd, mask_xs, mask_ys, _ = self.setup_tab._collect_coil_data(tokamak)
        self._eq_coil_pt_src.data = dict(x=cd["pt_x"], y=cd["pt_y"],
                                          color=cd["pt_c"], name=cd["pt_n"])
        self._eq_coil_cr_src.data = dict(x=cd["cr_x"], y=cd["cr_y"],
                                          color=cd["cr_c"], name=cd["cr_n"])
        self._eq_coil_rc_src.data = dict(xs=cd["rc_xs"], ys=cd["rc_ys"],
                                          fill=cd["rc_f"], edge=cd["rc_e"],
                                          name=cd["rc_n"])
        self._eq_mask_src.data = dict(xs=mask_xs, ys=mask_ys)

        # Cache mask data for the toggle and apply to incumbent plot too
        self._cached_mask_xs = mask_xs
        self._cached_mask_ys = mask_ys
        self._inc_mask_src.data = dict(xs=mask_xs, ys=mask_ys)

        # Cache grid arrays for fast contour recomputation
        self._eq_R2D = eq.R_2D
        self._eq_Z2D = eq.Z_2D
        self._eq_psi_lcfs = eq.psi_lcfs

        # Compute initial contours
        self._refresh_contours(eq.psi_2D)

        # Capture the initial LCFS boundary directly from the equilibrium
        init_R = list(eq.R_lcfs)
        init_Z = list(eq.Z_lcfs)
        self._cached_init_lcfs_xs = [init_R]
        self._cached_init_lcfs_ys = [init_Z]
        self._init_lcfs_src.data = dict(xs=[init_R], ys=[init_Z])

        # Set equilibrium plot bounds to match Setup tab
        r_min = float(np.min(tokamak.wall_R))
        r_max = float(np.max(tokamak.wall_R))
        z_min = float(np.min(tokamak.wall_Z))
        z_max = float(np.max(tokamak.wall_Z))
        dr = r_max - r_min
        dz = z_max - z_min
        pad = 0.05
        for fig in (self.eq_fig, self.inc_fig):
            fig.x_range.start = fig.x_range.reset_start = r_min - pad * dr
            fig.x_range.end = fig.x_range.reset_end = r_max + pad * dr
            fig.y_range.start = fig.y_range.reset_start = z_min - pad * dz
            fig.y_range.end = fig.y_range.reset_end = z_max + pad * dz

        # Compute correct pixel dimensions from wall aspect ratio
        aspect = dr / dz if dz > 0 else 1.0
        eq_height = 600
        eq_width = max(300, int(eq_height * aspect))
        self.eq_fig.height = eq_height
        self.eq_fig.width = eq_width
        self.inc_fig.height = eq_height
        self.inc_fig.width = eq_width

        # Initialise the incumbent equilibrium plot
        self._inc_wall_fill_src.data = dict(R=wall_r, Z=wall_z)
        self._inc_wall_src.data = dict(R=wall_r, Z=wall_z)
        self._inc_coil_pt_src.data = dict(x=cd["pt_x"], y=cd["pt_y"],
                                           color=cd["pt_c"], name=cd["pt_n"])
        self._inc_coil_cr_src.data = dict(x=cd["cr_x"], y=cd["cr_y"],
                                           color=cd["cr_c"], name=cd["cr_n"])
        self._inc_coil_rc_src.data = dict(xs=cd["rc_xs"], ys=cd["rc_ys"],
                                           fill=cd["rc_f"], edge=cd["rc_e"],
                                           name=cd["rc_n"])
        self._refresh_contours(eq.psi_2D, target="incumbent")
        self._last_incumbent_iter = -1

        # Plot strike geometries, buffers, and XPT regions on both eq figures
        for region in opt.divertor_regions:
            colour = opt.divertor_data[region]["colour"]
            strike_R = opt.divertor_data[region].get("strike_R")
            strike_Z = opt.divertor_data[region].get("strike_Z")
            if strike_R is not None and strike_Z is not None:
                sr = list(strike_R) if isinstance(strike_R, (list, np.ndarray)) else [strike_R]
                sz = list(strike_Z) if isinstance(strike_Z, (list, np.ndarray)) else [strike_Z]
                if len(sr) > 1:
                    self.eq_fig.line(sr, sz, line_color=colour, line_width=2)
                    self.inc_fig.line(sr, sz, line_color=colour, line_width=2)
                self.eq_fig.scatter(sr, sz, marker="square", size=8,
                                    fill_color=colour, line_color="black")
                self.inc_fig.scatter(sr, sz, marker="square", size=8,
                                     fill_color=colour, line_color="black")

        # Buffer outlines (only if buffers are enabled)
        if self.use_buffers.value and opt.buffers is not None:
            for region_geoms in opt.buffers.values():
                for buffer_geom in region_geoms:
                    coords = list(buffer_geom.coords)
                    br = [c[0] for c in coords]
                    bz = [c[1] for c in coords]
                    self.eq_fig.line(br, bz, line_color="black", line_dash="dashed",
                                     line_alpha=0.5, line_width=1)
                    self.inc_fig.line(br, bz, line_color="black", line_dash="dashed",
                                      line_alpha=0.5, line_width=1)

        # XPT region outlines (only if xpt regions are enabled)
        if self.use_xpt.value:
            for region in opt.divertor_regions:
                xpt_data = opt.divertor_data[region].get("xpoint_region")
                if xpt_data and opt.divertor_data[region].get("xpoint_region_present", False):
                    xr = list(xpt_data["R_region"])
                    xz = list(xpt_data["Z_region"])
                    self.eq_fig.patch(xr, xz, fill_color="gray", fill_alpha=0.3,
                                       line_color="black", line_alpha=0.3)
                    self.inc_fig.patch(xr, xz, fill_color="gray", fill_alpha=0.3,
                                        line_color="black", line_alpha=0.3)

        for region in opt.divertor_regions:
            colour = opt.divertor_data[region]["colour"]
            src = ColumnDataSource(data=dict(r=[], z=[]))
            self._fl_sources[region] = src
            self.eq_fig.line("r", "z", source=src, color=colour, line_width=1.5)

            # Incumbent field-line glyphs
            inc_src = ColumnDataSource(data=dict(r=[], z=[]))
            self._inc_fl_sources[region] = inc_src
            self.inc_fig.line("r", "z", source=inc_src, color=colour, line_width=1.5)

        # Add incumbent vertical line to each time-series chart
        for fig in (self.cost_fig, self.cl_fig, self.temp_fig, self.accept_fig, self.alpha_fig):
            fig.ray(x="x", y=0, length=0, angle=1.5708, source=self._incumbent_vline_src,
                    color="black", line_dash="dashed", line_width=1)


    # ------------------------------------------------------------------
    def _on_stop(self, event):
        if self._optimiser is not None:
            self._optimiser.request_stop()
            self.status.object = "Stop requested — waiting for current iteration to finish…"
            self.status.alert_type = "warning"

    # ------------------------------------------------------------------
    def _on_mask_toggle(self, event):
        """Show / hide coil mask overlays on the Optimise tab eq plots."""
        if event.new:
            data = dict(xs=self._cached_mask_xs, ys=self._cached_mask_ys)
        else:
            data = dict(xs=[], ys=[])
        self._eq_mask_src.data = data
        self._inc_mask_src.data = data

    # ------------------------------------------------------------------
    def _on_lcfs_toggle(self, event):
        """Show / hide the initial LCFS overlay on both eq plots."""
        if event.new:
            data = dict(xs=self._cached_init_lcfs_xs, ys=self._cached_init_lcfs_ys)
        else:
            data = dict(xs=[], ys=[])
        self._init_lcfs_src.data = data

    # ------------------------------------------------------------------
    def _run_optimisation(self):
        """Target function for the background thread."""
        try:
            self._optimiser.optimise()
        except Exception:
            logger.exception("Optimisation failed")

    # ------------------------------------------------------------------
    # Minimum interval (seconds) between expensive contour refreshes.
    _CONTOUR_INTERVAL = 2.0

    def _poll_updates(self):
        """Periodic callback that streams new data to the Bokeh sources.

        Lightweight work (streaming, log, progress bar) runs immediately
        inside a ``doc.hold`` batch.  Expensive contour refreshes are
        deferred to a separate event-loop tick via
        ``add_next_tick_callback`` so that pending user-interaction
        callbacks (stop button, toggle switches) can fire in between.
        """
        opt = self._optimiser
        if opt is None:
            return

        doc = pn.state.curdoc

        # ---- lightweight batch (streaming, log, vline) ----
        if doc is not None:
            doc.hold("combine")
        try:
            self._poll_stream_updates(opt, doc)
        finally:
            if doc is not None:
                doc.unhold()

        # Check if thread finished
        if self._thread is not None and not self._thread.is_alive():
            self._finish()

    # ------------------------------------------------------------------
    def _poll_stream_updates(self, opt, doc):
        """Fast, non-blocking updates that run every poll tick (200 ms)."""
        n = getattr(opt, "num_evals", 0)
        max_n = opt.max_evals

        # Update progress bar
        pct = int(100 * n / max_n) if max_n else 0
        self.progress.value = min(pct, 100)

        # How many points we have already streamed.
        existing = self._last_streamed_idx
        new_n = len(opt.tracking_cost)

        if new_n > existing:
            sl = slice(existing, new_n)
            xs = list(range(existing, new_n))

            self._cost_source.stream(dict(
                x=xs,
                total=opt.tracking_cost[sl],
                strike=opt.tracking_cost_strike_point_distance[sl],
                cl=opt.tracking_cost_connection_length[sl],
                coil=opt.tracking_cost_coil_currents[sl],
                xpt=opt.tracking_cost_xpoint_regions[sl],
            ))
            self._temp_source.stream(dict(x=xs, temperature=opt.tracking_temperature[sl]))
            self._acceptance_source.stream(dict(x=xs, rate=opt.tracking_acceptance_rate[sl]))
            self._alpha_source.stream(dict(x=xs, alpha=opt.tracking_alpha[sl]))

            # Update the threshold acceptance rate horizontal line
            if hasattr(opt, "threshold_acceptance_rate"):
                self._threshold_rate_src.data = dict(y=[opt.threshold_acceptance_rate])

            cl_stream = dict(x=xs)
            for region in opt.divertor_regions:
                cl_stream[region] = opt.tracking_connection_length[region][sl]
            self._cl_source.stream(cl_stream)

            self._last_streamed_idx = new_n

        # Update log pane
        if self._log_lines:
            html = "<pre style='max-height:200px;overflow-y:auto;font-size:11px;'>"
            html += "\n".join(self._log_lines[-50:])
            html += "</pre>"
            self.log_pane.object = html

        # Move the incumbent vertical line on the cost plot (lightweight).
        # This is on the *cost* plot, not the eq plot, so update immediately.
        incumbent_data = getattr(opt, "incumbent_data", None)
        if incumbent_data is not None:
            inc_iter = incumbent_data.get("iteration_num", -1)
            if inc_iter != self._last_incumbent_iter:
                self._incumbent_vline_src.data = dict(x=[inc_iter], y0=[0], y1=[0])

        # ---- schedule contour refresh on the *next* event-loop tick ----
        now = time.monotonic()
        contour_due = (now - self._last_contour_time) >= self._CONTOUR_INTERVAL

        if contour_due and not self._contour_scheduled:
            need_live = (
                hasattr(opt, "psi_2D")
                and opt.psi_2D is not None
                and n > self._last_live_contour_iter
            )
            need_inc = (
                incumbent_data is not None
                and incumbent_data.get("iteration_num", -1) != self._last_incumbent_iter
            )

            if need_live or need_inc:
                # Snapshot data now — the optimiser thread may mutate it.
                if need_live:
                    self._pending_live_psi = opt.psi_2D.copy()
                    self._pending_live_n = n
                    self._pending_live_fl.clear()
                    for region in opt.divertor_regions:
                        fl_r = getattr(opt, "tracking_field_lines_R", {}).get(region)
                        fl_z = getattr(opt, "tracking_field_lines_Z", {}).get(region)
                        if fl_r is not None and fl_z is not None and region in self._fl_sources:
                            self._pending_live_fl[region] = (list(fl_r), list(fl_z))

                if need_inc:
                    inc_divs = incumbent_data.get("divertors", {})
                    self._pending_inc = {
                        "iteration_num": incumbent_data["iteration_num"],
                        "cost": incumbent_data.get("cost", 0),
                        "psi_2D": (
                            incumbent_data["psi_2D"].copy()
                            if incumbent_data.get("psi_2D") is not None
                            else None
                        ),
                        "divertors": {
                            region: {
                                "field_line_R": list(ddata.get("field_line_R", [])),
                                "field_line_Z": list(ddata.get("field_line_Z", [])),
                            }
                            for region, ddata in inc_divs.items()
                        },
                    }

                self._contour_scheduled = True
                if doc is not None:
                    doc.add_next_tick_callback(self._apply_contour_update)

    # ------------------------------------------------------------------
    def _apply_contour_update(self):
        """Render pending contour updates.

        Runs on a *separate* event-loop tick from the streaming update,
        which means user-interaction callbacks (stop button, toggles) can
        fire in between.
        """
        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")
        try:
            opt = self._optimiser

            # --- Live equilibrium contour + field lines ---
            if self._pending_live_psi is not None:
                self._last_live_contour_iter = self._pending_live_n
                self.eq_fig.title.text = (
                    f"Equilibrium (live) — iteration {self._pending_live_n}"
                )
                self._refresh_contours(self._pending_live_psi)
                for region, (fl_r, fl_z) in self._pending_live_fl.items():
                    self._fl_sources[region].data = dict(r=fl_r, z=fl_z)
                self._pending_live_psi = None

            # --- Incumbent contour + field lines + title ---
            if self._pending_inc is not None:
                inc = self._pending_inc
                inc_iter = inc["iteration_num"]
                self._last_incumbent_iter = inc_iter
                self.inc_fig.title.text = (
                    f"Incumbent — iteration {inc_iter}, cost {inc['cost']:.4f}"
                )
                if inc["psi_2D"] is not None:
                    self._refresh_contours(inc["psi_2D"], target="incumbent")
                if opt is not None:
                    for region in opt.divertor_regions:
                        div_data = inc["divertors"].get(region, {})
                        fl_r = div_data.get("field_line_R")
                        fl_z = div_data.get("field_line_Z")
                        if fl_r and fl_z and region in self._inc_fl_sources:
                            self._inc_fl_sources[region].data = dict(
                                r=list(fl_r), z=list(fl_z)
                            )
                self._pending_inc = None
        finally:
            if doc is not None:
                doc.unhold()
            self._last_contour_time = time.monotonic()
            self._contour_scheduled = False

    # ------------------------------------------------------------------
    def _finish(self):
        """Clean up after optimisation completes."""
        if self._periodic_cb is not None:
            self._periodic_cb.stop()
            self._periodic_cb = None

        # Final flush: stream any remaining data points and render the
        # latest incumbent contour so nothing is missed.
        opt = self._optimiser
        if opt is not None:
            doc = pn.state.curdoc
            if doc is not None:
                doc.hold("combine")
            try:
                self._poll_stream_updates(opt, doc)
            finally:
                if doc is not None:
                    doc.unhold()
            # Force a final contour update if there's a pending incumbent
            if self._pending_inc is not None or self._pending_live_psi is not None:
                self._apply_contour_update()

        self.run_btn.disabled = False
        self.stop_btn.disabled = True
        self.progress.value = 100

        opt = self._optimiser
        if opt is not None and hasattr(opt, "incumbent_data"):
            cost = opt.incumbent_data.get("cost", "?")
            it = opt.incumbent_data.get("iteration_num", "?")

            # Determine the stop reason
            reason = self._determine_stop_reason(opt)
            logger.info("Optimisation stopped. Reason: %s. Incumbent cost: %.4f at iteration %s.",
                        reason, cost, it)

            self.status.object = f"Optimisation complete ({reason}). Incumbent cost: {cost:.4f} at iteration {it}."
            self.status.alert_type = "success"
            self.state["optimiser"] = opt
        else:
            self.status.object = "Optimisation finished (no incumbent data found)."
            self.status.alert_type = "warning"

        # Flush the log pane so the stop-reason message is visible
        if self._log_lines:
            html = "<pre style='max-height:200px;overflow-y:auto;font-size:11px;'>"
            html += "\n".join(self._log_lines[-50:])
            html += "</pre>"
            self.log_pane.object = html

    # ------------------------------------------------------------------
    @staticmethod
    def _determine_stop_reason(opt):
        """Determine why the optimisation stopped.

        Parameters
        ----------
        opt : Optimiser
            The optimiser instance after the run has finished.

        Returns
        -------
        str
            A human-readable description of the stop reason.
        """
        if opt._stop_event.is_set():
            return "stopped by user"
        if opt.num_evals >= opt.max_evals:
            return "max iterations reached"
        if opt.temperature <= opt.min_temperature:
            return "min temperature reached"
        if opt.incumbent_data["cost"] <= opt.cost_termination_fraction * opt.initial_cost:
            return "cost target reached"
        return "unknown"

    # ------------------------------------------------------------------
    @property
    def panel(self):
        if hasattr(self, "_panel"):
            return self._panel
        params_col = pn.Column(
            "### Annealing Schedule",
            self.max_evals,
            self.step_size_factor,
            self.initial_temp,
            self.min_temp,
            self.n_window,
            self.threshold_decay,
            self.init_threshold,
            self.cost_frac,
            self.max_cooling,
            self.min_cooling,
            pn.layout.Divider(),
            "### Initial Cost Values",
            self.w_strike,
            self.w_conn,
            self.w_coil,
            self.w_xpt,
            pn.layout.Divider(),
            "### Regularisation (\u03b1)",
            self.initial_alpha,
            self.alpha_update,
            pn.layout.Divider(),
            "### Field-line Tracing",
            self.trace_step_size,
            self.trace_max_steps,
            self.psi_tol,
            self.buffer_penalty,
            pn.layout.Divider(),
            "### X-point Regions",
            self.max_disconnection,
            pn.layout.Divider(),
            "### Feature Toggles",
            self.use_buffers,
            self.use_xpt,
            self.detailed_logging,
            pn.layout.Divider(),
            pn.Row(self.run_btn, self.stop_btn),
            self.progress,
            self.status,
            pn.layout.Divider(),
            "### Load / Save Config",
            pn.pane.HTML(
                "<small>Save or load the complete optimisation configuration "
                "(geometry, constraints, and all settings) as a JSON file. "
                "Equilibrium and magnet data must still be loaded separately "
                "on the Setup tab.</small>"
            ),
            pn.Row(self.load_config_file, self.save_config_btn),
            width=400,
            height=1200,
            scroll=True,
        )
        charts_col = pn.Column(
            self.cost_fig,
            self.cl_fig,
            self.temp_fig,
            self.accept_fig,
            self.alpha_fig,
            self.log_pane,
            sizing_mode="stretch_both",
            scroll=True,
        )
        eq_col = pn.Column(
            pn.Row(self.show_masks, self.show_initial_lcfs, margin=(0, 0)),
            self.eq_fig,
            self.inc_fig,
            sizing_mode="stretch_height",
            scroll=True,
        )
        self._panel = pn.Row(params_col, charts_col, eq_col, sizing_mode="stretch_both")
        return self._panel


# ======================================================================
# Tiny logging handler that captures messages for the GUI log pane
# ======================================================================

class _PanelLogHandler(logging.Handler):
    """Append formatted log records to a shared list (thread-safe via GIL)."""

    def __init__(self, line_list):
        super().__init__(level=logging.INFO)
        self._lines = line_list
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            self._lines.append(self.format(record))
        except Exception:
            pass
