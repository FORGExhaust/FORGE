"""Geometry Editor tab — interactive wall buffers, strike surfaces, and X-point regions.

Uses Bokeh's built-in drawing tools (PolyDrawTool, PolyEditTool, PointDrawTool)
so that all editing happens in the browser canvas — no blocking matplotlib windows.

Geometry is defined **per divertor region**.  The user selects a region from the
dropdown, draws/edits strike geometry and XPT polygons for that region, then
switches to the next one.  Each region's data is stored independently and
assembled into the ``divertor_data`` dict consumed by the optimiser.
"""

import io
import json
import logging
import math

import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.models import (
    BoxZoomTool, ColumnDataSource, CustomJS, HoverTool, MultiLine,
    PolyDrawTool, PolyEditTool, PointDrawTool, Range1d,
)
from bokeh.plotting import figure as bk_figure
from shapely.geometry import LineString

from forge.io import fancy_json_string
from forge.utils import orthogonalised_convex_hull_from_rects

# Inline base64 data URIs for Bokeh toolbar tool icons (from Bokeh 3.7.2).
# Bokeh is released under the BSD 3-Clause licence; these icons are
# redistributed here under the terms of that licence.
# Used in sidebar help text so the user can identify the correct button.
_ICON_POINT_DRAW = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0"
    "AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH"
    "4gEMEiERGWPELgAAA4RJREFUWMO1lr1uG1cQhb9ztdRSP7AF1QxgwKlcuZSqRC"
    "9gWUUUINWqTh5AnaFOnVPEteQmRuhCURqWsSqqc9IolREXdEvQBElxtdw7KURS"
    "FEVKu4w8wAKLxdw9Z+bMnRmZGXfZ29//II8th4WwGVNyIoQLYB5vxA9Caq04"
    "iUd9A+7ZlsNC2I7TdSd2hZXMJKlnTqp9jtl/GBaqoyQ0noFKpUIzBicYYc+D"
    "EFpxkglc4oVJa5gvDn8v1xV2irG3FM4NSVwjUKlUaMcpJhCGmSEJQ6QGD8M5"
    "WnHCd8+f3QCXpPLx8WNwv0j6Bm9FMK7FJ3WBE+R/2t7c/GBmFvSBrzRTCsy"
    "TDjXrxUgEMtpxynJYmJoBJ4VAybwVARgvL7Oik0okCodnKpVKX7P0leiVMb0V"
    "vbJT+upznK4vh0GIeQwwQStJkHQD3MwsCALTJRG7Qrdrj5m/djgYaIa0hlkR"
    "dJk26XEgC9txurccBtVW3IudBImmZuACUP+ZlIDBt9FKcubYNTcAH/X0RYM1E"
    "7utJPlqe+uZzPxUcEkiSS4sTT95n15Mud0xWC0o2PAWOCdK3KYZlFxfM+tHO"
    "cnMzNr1es18ug+cgsVjP4yBU/Ppfrter1m/+l0+zYygML1xRVHU7TSb1cSzBz"
    "oBzszsH+AMdJJ49jrNZjWKou6wBnwOzcyndBpNbuueURR1Dw8Pq35p9cc5p/D"
    "y9Dypt7jXrtdGwQECS9NPhr6Gq6txUzNigE6zydLK6lTw12/KT4FGFEUfJX2Y"
    "JNONq5tVs4ODA7sD/DnwJ/BoADZuE3tHFs12dna6d4C/BI6AlbyzI8ii2TTw"
    "12/KK33gb2cdXsNZoAntbZC2SeO4c9592k/5eNQbiwvFd1kJuFGwLJr1wSPg/S"
    "wpvyFBHufOeXcFeAlE97U/uCxOY+P3b+Bn4B3Q+L8EdJfD4a+/AbC4UBzPxi"
    "Pg3wlHZquB28Cn2IuR9x3gr3uV4DbwfvSDOvi4uFA8BDZmIRHkjHpS9Ht9iR"
    "qd8+5G3g05mAGcQbsdiX5QJ428G7Kygo8XYdb1/K4NWVmjzkNge2sz84bs+E"
    "LmpDDLtqWsNZBXgvmw8CTtpWVMT7x5YWBjLARnwZfKQNYN2U2LPvrh+5nBt"
    "7c2M2/It9bArCTKR8eZN+SJ13AScPnoODeRdqNenH+wul5w2gUr2WUjMFAt8b"
    "Z/0axX/wNnv4H8vTFb1QAAAABJRU5ErkJggg=="
)
_ICON_POLY_DRAW = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0"
    "AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH"
    "4gEMEjglo9eZgwAAAc5JREFUWMPt1zFrU1EUB/DfS4OmVTGDIChCP4BgnQXRxV"
    "HqIJUupp9AB8VBQcRBQUXIB9DWQoMRiXZzcnQSA34A7aAuHSJKkgo2LvfBrU3a"
    "JnlYkBy4vHcP557zP/9z3r33JdXa647N0kHSZd5Nn0rSxc8G3cXp85sMcnZZ8v"
    "ge3osZ+l3vB8CWFA0iL14t79h210swAjACMAIwAjACkB90D/8/GchI9ve4nPwTB"
    "h5E9ws7OepzGWb9EddSn51Op9ZstadSg4VK1UKlKkmSDSMLALewiuNh/hVJq71W"
    "xttmqz0dG88vPc+MgWP4grvYG3SLOBrZFFFrttqPe4HIDxh4GSei+98iSlusuY"
    "opXEAjBtEPA3tQwUpwluAbDm4TPJUz+BTW9l2Ce6G7L0X/Bw8D3T/7SKKIDzHg"
    "7QCcxjvcQAEtXAnrrg/RP0/DKPbqgcN4iVOR7gcO4dcQgRuoh7HSqwlP4n20m6"
    "3jJu5n8MkWMYfP3UowhzdR8FU8w9iQwevBdyq3/27CMRzAE5yLuvsRLg+ZcR1n"
    "J8YL81HWJUzGAPaFZwe/Q5MdyYDyNHgjzO90YyGHtVDncuiJchaHw8R4oREFV5"
    "qdiVmYLM3OgD9k5209/atmIAAAAABJRU5ErkJggg=="
)
_ICON_POLY_EDIT = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0"
    "AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH"
    "4gELFi46qJmxxAAABV9JREFUWMOdl19vFFUYxn9n9u9sCyylUIzWUoMQBAWCMd"
    "EEIt6xIRQSLIEKtvHe6AcA4yeQb7CAUNJy0daLeomJN8SEULAC2kBBapBKoLvb"
    "mdl/c14vdmY7u91tF95kknPOnHmf95znPc97Ro2OTeBbdjFDT3c32ZxVHUOE9k"
    "SMB0/m6ExuoJn1H+ur6Y+OTfD50SMN5168OgrAlyf7CfuD+z7+iDs3p8hkLUQ0"
    "iFQ/yFl5Nm/qonfHVva+s32Zw9GxCYILsZ08tpNfBhbs+1YN4OH9+7huGdECSB"
    "VfqUosbsllfmauBqiR+cCNwOr7AEo8pPHJnymXykhg5fUWjoQpl0vVvhZhbSzG"
    "oUOHqgBlt6B6uruj2Zy1E9jo0fhfeyL2x4Mnc8VErK0KUEOB64JSyptfG4RSyt"
    "sJjUJVxw2lsFy3urL9nx1Qd25ObctkrVMi+jQivd7U2ZyV/3Hzpq7h3h1b/7p"
    "9Y0o8v8rwAbTWrGpSocN/FGDlbAI0Rl23PCBan0Ok158H9Ipwzi25A/Mzc9Gl/B"
    "Yx/E4kYqC1NKRARNAaDCNUM27Z+Zr+ouXs0q4+LSLBHPYCFkTkC6uU39kwCdsS"
    "7WRKmaYUiAhdnZ3MPX2K4+QjQI+C94A93rMzm8ltMwyDeDzWjMZeEb2pYQDdW3"
    "vITU2jtUZ5QThOPgm8C7wP7J15OPsBsB3oWpGnVWisCeDS1VHj4vBI92+/3tgB"
    "7Ab2AruAXiDBK5oIOkhtkEYRNRuJhObrd8Dl9ewf4D5wG7hVLpen29vb5wzD+B"
    "rkbBMaL3d1dk5nsrnlFDTTFWAWmAZueWD3gCemGde2k2fw1Al1YXhEvjozoO49"
    "eczdqekrWmsc2zlrmvEKOGoW1GUjFLqSk2KpJrCLwyMCPAP+BO54QL8DM6YZX/C"
    "lsP9YnwKkXnIBP4jdIpJRpdJTCYdMwwi98KU0Hjc/dDILNyUcwTCWdOSMJ0TR"
    "mBktGRhLugu0xyLk7CIqVNm+0bGJptl1YXikD0grpY4Rjc4a8Fbgdab/6OGbAJ"
    "eCUuyJnnHmZH9pbSyGuBXV8NUwlUpR1EWyixmSyTWEwqGlJ2Swbo2JXbAAfgDG"
    "gGQA9I1A9t1tlq0AxrXxn0ilUpw4fhQqYkH/sT41OTnJJwf2s6FjI5mshdYa7b"
    "qVR2uezr9MJmJt14FvGrh/O9D+e6UkM/xyCuCqEKCYnJyUTKFQrZDHjxzGshwW"
    "LQcRsOz8Hi85P23id0ug/XilAMLBmm4tPGdoaKjSH5+oAGrhwvBI9SjZTn4QSK"
    "9yenoD7dlrExPoJlXW8G8ytpNHxRKk02lGxsdRKFwXLNvx5yY94HQLGhGk4LFC"
    "YQSqaE0AwWM1eOoEbR0dKBSW7bC4mKuffxs4D/wCLKwQQPAUzIkslfp6cVomRO"
    "WSolh0GjldAM4nzDi2k9/i5UAzC9aKfwNJ3zgJg9YEvN6+C7SHgKm69+sD7RfN"
    "nKTTaZRPQfAut4oFV//IS7gkcB34VlVo8kGzphlfB+DU+TfNGBpZtRastvrvAR"
    "JmfMF28ge9sc2B9/PNnCilMIDwK6y8/ow/Ai4kvILTljAXvDvEvrqKSUs60Kol"
    "zPjBxspavQD2tKqCAGF/Ba+xE/Wbilu54wZV8NEKF5fXzQHl/bh4hUsE0WAXSl"
    "DMYcQSrQXgCmsTseXHsJkNnjqBFGwKJaHsKlxtUHYVhbLCzr1kaOA4bcn1y1Sw"
    "mb+iLpJKpVrfgdpfsiVVCYcgluwgnU7jEgJ4s5UkLFtWYyHyEg0/N1q1tmQH+Y"
    "XnAMFr97Nmv3p+0QsHQRsF8qpBOE5+rb9Nkaj50tVQKjqh4OU3GNL/1/So3vuU"
    "gbAAAAAASUVORK5CYII="
)

logger = logging.getLogger(__name__)


def _is_finite(v):
    """Return True if *v* is a finite number (not NaN/Inf)."""
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# Region display colours (muted, for ghost renderers)
_REGION_COLOURS = {
    "lower_outer": "#1f77b4",
    "lower_inner": "#ff7f0e",
    "upper_outer": "#2ca02c",
    "upper_inner": "#d62728",
}


class GeometryTab:
    """Panel component for geometric editing (buffers, strike surfaces, XPT regions).

    Geometry is scoped to the **active divertor region**.  Switching regions
    saves the current canvas state and loads the target region's data.
    """

    # Template for empty per-region data
    @staticmethod
    def _empty_region():
        return {
            "strike": {"x": [], "y": []},
            "xpt": {"xs": [], "ys": []},
            "buffers": [],  # list of {"R": [...], "Z": [...], "distance": d}
            "cl_mult_factor_zero": 2.0,
            "weight_connection_length": None,  # None → auto (1/N)
            "weight_strike_point_distance": None,
            "weight_xpoint_region": None,
        }

    def __init__(self, shared_state):
        self.state = shared_state

        # --- Per-region data store ---
        self._region_data = {
            "lower_outer": self._empty_region(),
            "lower_inner": self._empty_region(),
            "upper_outer": self._empty_region(),
            "upper_inner": self._empty_region(),
        }
        self._active_region = "lower_outer"
        self._switching_region = False  # guard against recursive data updates

        # --- Main Bokeh figure ---
        self.fig = bk_figure(
            title="Geometry Editor",
            x_axis_label="R (m)",
            y_axis_label="Z (m)",
            x_range=Range1d(0, 1),
            y_range=Range1d(0, 1),
            match_aspect=True,
            width=700,
            height=900,
            tools="pan,wheel_zoom,reset,save",
            background_fill_color="#d9d9d9",
        )
        self.fig.xaxis.axis_label_text_font_style = "normal"
        self.fig.yaxis.axis_label_text_font_style = "normal"
        self.fig.add_tools(BoxZoomTool(match_aspect=True))

        # Static layers (updated when machine is loaded) — same as setup tab
        self._wall_fill_source = ColumnDataSource(data=dict(R=[], Z=[]))
        self._wall_source = ColumnDataSource(data=dict(R=[], Z=[]))
        self._contour_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._lcfs_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._coil_source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._fil_cross_source = ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self._fil_rect_source = ColumnDataSource(data=dict(xs=[], ys=[], fill=[], edge=[], name=[]))

        # Render order: wall fill, contours, separatrix, wall outline, coils
        self.fig.patch("R", "Z", source=self._wall_fill_source,
                       fill_color="white", line_color=None)
        self.fig.multi_line("xs", "ys", source=self._contour_source,
                            line_alpha=0.35, line_color="gray")
        self.fig.multi_line("xs", "ys", source=self._lcfs_source,
                            line_color="red", line_width=2)
        self.fig.line("R", "Z", source=self._wall_source,
                      line_color="black", line_width=2)
        self.fig.scatter("x", "y", source=self._coil_source, color="color", size=10,
                         marker="circle")
        self.fig.scatter("x", "y", source=self._fil_cross_source, color="color", size=8,
                         marker="x")
        self.fig.patches("xs", "ys", source=self._fil_rect_source, fill_color="fill",
                         line_color="edge", line_width=0.3)

        # Coil mask overlays
        self._mask_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.fig.patches("xs", "ys", source=self._mask_source, fill_color="orange",
                         fill_alpha=1.0, line_color="black", line_width=1.0)
        self._mask_data_cache = {"xs": [], "ys": []}

        # --- Ghost layers for inactive regions ---
        self._ghost_strike_source = ColumnDataSource(data=dict(x=[], y=[], color=[]))
        self.fig.scatter(
            "x", "y", source=self._ghost_strike_source,
            color="color", size=7, marker="square", alpha=0.35,
        )
        self._ghost_strike_line_source = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
        self.fig.multi_line(
            "xs", "ys", source=self._ghost_strike_line_source,
            line_color="color", line_width=1, line_alpha=0.35,
        )
        self._ghost_xpt_source = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
        self.fig.patches(
            "xs", "ys", source=self._ghost_xpt_source,
            fill_alpha=0.1, fill_color="color", line_color="color",
            line_alpha=0.3, line_width=1,
        )
        self._ghost_buffer_source = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
        self.fig.multi_line(
            "xs", "ys", source=self._ghost_buffer_source,
            line_color="color", line_width=1.5, line_alpha=0.55, line_dash="dashed",
        )

        # --- Editable layers (active region only) ---

        # Strike surface points
        self._strike_source = ColumnDataSource(data=dict(x=[], y=[]))
        strike_renderer = self.fig.scatter(
            "x", "y", source=self._strike_source,
            color="red", size=10, marker="square",
        )
        strike_draw_tool = PointDrawTool(renderers=[strike_renderer], description="Draw strike geometry")
        self.fig.add_tools(strike_draw_tool)

        # Line connecting strike geometry — shares the same source so it
        # updates live during drag (Bokeh mutates data in-place).
        self.fig.line(
            "x", "y", source=self._strike_source,
            line_color="red", line_width=1.5, line_dash="solid",
        )

        # --- Snap-to-wall machinery ---
        # A single-element data source acts as a boolean flag readable by JS.
        self._snap_flag_source = ColumnDataSource(data=dict(active=[False]))

        # Client-side callback: when snap is active, project every strike
        # point onto the nearest segment of the wall polyline.
        self._snap_guard_source = ColumnDataSource(data=dict(snapping=[False]))
        self._strike_source.js_on_change("data", CustomJS(
            args=dict(strike=self._strike_source,
                      wall=self._wall_source,
                      snap_flag=self._snap_flag_source,
                      guard=self._snap_guard_source),
            code="""
                if (!snap_flag.data['active'][0]) return;
                if (guard.data['snapping'][0]) return;

                const wx = wall.data['R'];
                const wy = wall.data['Z'];
                if (wx.length < 2) return;

                const sx = strike.data['x'].slice();
                const sy = strike.data['y'].slice();
                let changed = false;

                for (let i = 0; i < sx.length; i++) {
                    const px = sx[i];
                    const py = sy[i];
                    if (!isFinite(px) || !isFinite(py)) continue;

                    let best_d2 = Infinity;
                    let best_x = px;
                    let best_y = py;

                    for (let j = 0; j < wx.length - 1; j++) {
                        const ax = wx[j],   ay = wy[j];
                        const bx = wx[j+1], by = wy[j+1];
                        const dx = bx - ax, dy = by - ay;
                        const len2 = dx*dx + dy*dy;
                        let t = 0;
                        if (len2 > 0) {
                            t = ((px - ax)*dx + (py - ay)*dy) / len2;
                            t = Math.max(0, Math.min(1, t));
                        }
                        const cx = ax + t*dx;
                        const cy = ay + t*dy;
                        const d2 = (px - cx)*(px - cx) + (py - cy)*(py - cy);
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_x = cx;
                            best_y = cy;
                        }
                    }

                    if (best_x !== px || best_y !== py) {
                        sx[i] = best_x;
                        sy[i] = best_y;
                        changed = true;
                    }
                }

                if (changed) {
                    guard.data['snapping'] = [true];
                    strike.data = {x: sx, y: sy};
                    guard.data['snapping'] = [false];
                }
            """,
        ))

        # Buffer outlines — represented as multi-line (Shapely polygon rings)
        self._buffer_line_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.fig.multi_line(
            "xs", "ys", source=self._buffer_line_source,
            line_color="blue", line_dash="dashed", line_width=2,
        )

        # --- Wall segment selection for buffer workflow ---
        # Each wall segment is a separate entry in this multi_line source so
        # the HoverTool can identify individual segments via hover_glyph.
        self._wall_seg_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._wall_seg_renderer = self.fig.multi_line(
            "xs", "ys", source=self._wall_seg_source,
            line_color=None, line_alpha=0, line_width=10,  # invisible hit area
        )
        self._wall_seg_renderer.hover_glyph = MultiLine(
            line_color="yellow", line_alpha=0.8, line_width=5,
        )
        self._wall_seg_renderer.visible = False  # hidden until buffer mode
        seg_hover = HoverTool(
            renderers=[self._wall_seg_renderer], tooltips=None,
        )
        self.fig.add_tools(seg_hover)

        # Highlight for the currently selected (clicked) segment
        self._selected_seg_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.fig.multi_line(
            "xs", "ys", source=self._selected_seg_source,
            line_color="cyan", line_alpha=1.0, line_width=4,
        )
        # Index of selected segment (-1 = none).  Updated by JS, read by Python.
        self._selected_seg_idx_source = ColumnDataSource(data=dict(idx=[-1]))

        # Committed (added) buffer segments — permanent highlight
        self._committed_seg_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.fig.multi_line(
            "xs", "ys", source=self._committed_seg_source,
            line_color="lime", line_alpha=0.9, line_width=3,
        )

        # Buffer-mode flag readable by JS
        self._buffer_mode_source = ColumnDataSource(data=dict(active=[False]))

        # Client-side tap: find nearest wall segment and mark it selected.
        self.fig.js_on_event(Tap, CustomJS(
            args=dict(
                wall=self._wall_seg_source,
                sel=self._selected_seg_source,
                sel_idx=self._selected_seg_idx_source,
                mode=self._buffer_mode_source,
            ),
            code="""
                if (!mode.data['active'][0]) return;
                const xs = wall.data['xs'];
                const ys = wall.data['ys'];
                if (xs.length === 0) return;

                const mx = cb_obj.x, my = cb_obj.y;
                let best_d2 = Infinity, best_i = -1;

                for (let i = 0; i < xs.length; i++) {
                    const ax = xs[i][0], ay = ys[i][0];
                    const bx = xs[i][1], by = ys[i][1];
                    const dx = bx - ax, dy = by - ay;
                    const len2 = dx*dx + dy*dy;
                    let t = 0;
                    if (len2 > 0) {
                        t = ((mx - ax)*dx + (my - ay)*dy) / len2;
                        t = Math.max(0, Math.min(1, t));
                    }
                    const cx = ax + t*dx, cy = ay + t*dy;
                    const d2 = (mx-cx)*(mx-cx) + (my-cy)*(my-cy);
                    if (d2 < best_d2) { best_d2 = d2; best_i = i; }
                }

                if (best_i >= 0) {
                    sel.data = {xs: [xs[best_i]], ys: [ys[best_i]]};
                    sel_idx.data = {idx: [best_i]};
                    sel.change.emit();
                    sel_idx.change.emit();
                }
            """,
        ))

        # X-point region polygons
        self._xpt_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        xpt_renderer = self.fig.patches(
            "xs", "ys", source=self._xpt_source,
            fill_alpha=0.3, fill_color="gray", line_color="black",
        )
        # PolyDrawTool — no vertex_renderer so no circles clutter the canvas
        self._xpt_draw_tool = PolyDrawTool(
            renderers=[xpt_renderer],
            description="Draw X-point region",
        )
        self.fig.add_tools(self._xpt_draw_tool)

        # PolyEditTool — has its own vertex_renderer; red circles appear
        # only when a polygon is selected for editing
        self._xpt_edit_vertex_source = ColumnDataSource(data=dict(x=[], y=[]))
        xpt_edit_vertex_renderer = self.fig.scatter(
            "x", "y", source=self._xpt_edit_vertex_source,
            color="red", size=10, marker="circle",
        )
        self._xpt_edit_tool = PolyEditTool(
            renderers=[xpt_renderer],
            vertex_renderer=xpt_edit_vertex_renderer,
            description="Edit X-point region vertices",
        )
        self.fig.add_tools(self._xpt_edit_tool)

        # Flag source: when a delete handler sets flag=[1], the JS callback
        # deactivates the PolyEditTool, clears vertex dots, clears source
        # selection, and forces a renderer repaint after all synchronous
        # signals have completed.
        self._xpt_delete_flag = ColumnDataSource(data=dict(flag=[0]))
        # Attach an invisible scatter to the figure so this source is part of
        # the Bokeh document model tree and gets serialized to the client.
        self.fig.scatter(
            "flag", "flag", source=self._xpt_delete_flag,
            size=0, alpha=0, visible=False,
        )
        self._xpt_delete_flag.js_on_change(
            "data",
            CustomJS(
                args=dict(
                    edit_tool=self._xpt_edit_tool,
                    vertex_source=self._xpt_edit_vertex_source,
                    main_source=self._xpt_source,
                ),
                code="""
                    if (cb_obj.data['flag'][0] > 0) {
                        edit_tool.active = false;
                        vertex_source.data = {x: [], y: []};
                        vertex_source.change.emit();
                        main_source.selected.indices = [];
                        // Force renderer repaint after all sync handlers finish
                        setTimeout(function() {
                            vertex_source.data = {x: [], y: []};
                            vertex_source.change.emit();
                            main_source.change.emit();
                        }, 0);
                    }
                """,
            ),
        )

        # Clear stale vertex dots when main source data changes while the
        # edit tool is inactive.  The PolyEditTool leaves a signal connection
        # that repopulates vertex dots even after deactivation.  setTimeout(0)
        # fires after the stale handler but before the next canvas paint.
        self._xpt_source.js_on_change(
            "data",
            CustomJS(
                args=dict(
                    edit_tool=self._xpt_edit_tool,
                    vertex_source=self._xpt_edit_vertex_source,
                ),
                code="""
                    if (!edit_tool.active) {
                        setTimeout(function() {
                            if (!edit_tool.active) {
                                vertex_source.data = {x: [], y: []};
                                vertex_source.change.emit();
                            }
                        }, 0);
                    }
                """,
            ),
        )

        # --- Strike geometry widgets ---
        self.snap_to_wall = pn.widgets.Checkbox(name="Snap to wall", value=False)
        self.snap_to_wall.param.watch(self._on_snap_toggle, "value")
        self.clear_strike_btn = pn.widgets.Button(name="Clear All", button_type="warning")
        self._strike_point_column = pn.Column(sizing_mode="stretch_width")
        self._strike_editors = []  # list of per-point editor dicts

        self._strike_source.on_change("data", self._on_strike_source_change)
        self._suppressing_strike_rebuild = False
        self.clear_strike_btn.on_click(self._on_clear_strike_points)

        # --- X-point region widgets ---
        self.clear_xpt_btn = pn.widgets.Button(name="Clear All", button_type="warning")
        self._xpt_region_column = pn.Column(sizing_mode="stretch_width")
        self._xpt_editors = []  # list of per-region editor widgets

        # Listen for changes to the xpt data source so we can rebuild the list
        self._xpt_source.on_change("data", self._on_xpt_source_change)
        self._suppressing_xpt_rebuild = False

        # --- Buffer creation widgets ---
        self.buffer_select_mode = pn.widgets.Checkbox(
            name="Select segment", value=False,
        )
        self.buffer_select_mode.param.watch(self._on_buffer_mode_toggle, "value")
        self.buffer_distance = pn.widgets.FloatInput(
            name="Buffer distance (m)", value=0.05, step=0.01, start=0.001,
        )
        self.add_buffer_btn = pn.widgets.Button(
            name="Add Buffer", button_type="primary",
        )
        self.clear_buffers_btn = pn.widgets.Button(
            name="Clear All", button_type="warning",
        )
        self._buffer_list_column = pn.Column(sizing_mode="stretch_width")

        # --- Strike geometry load / save (all regions) ---
        self.load_geometry_file = pn.widgets.FileInput(
            accept=".json", multiple=False, width=220,
        )
        self.load_geometry_file.param.watch(self._on_geometry_file_load, "value")
        self.save_geometry_btn = pn.widgets.FileDownload(
            callback=self._save_geometry_callback,
            filename="geometry.json",
            label="Save",
            button_type="success",
            width=80,
        )

        # --- Region enable/disable toggles ---
        self._region_names = ["lower_outer", "lower_inner", "upper_outer", "upper_inner"]
        self._region_display = {"lower_outer": "Lower Outer", "lower_inner": "Lower Inner",
                                "upper_outer": "Upper Outer", "upper_inner": "Upper Inner"}
        self.region_enabled = {}
        for name in self._region_names:
            cb = pn.widgets.Checkbox(name=self._region_display[name], value=False)
            self.region_enabled[name] = cb

        # --- Per-region cost settings (shown for active region) ---
        self.cl_mult_factor_zero = pn.widgets.FloatInput(
            name="Connection length multiplication factor zero", value=2.0, step=0.1, start=1.01,
        )
        self.weight_cl = pn.widgets.FloatInput(
            name="Weight: connection length", value=0.0, step=0.05, start=0.0, end=1.0,
        )
        self.weight_strike = pn.widgets.FloatInput(
            name="Weight: strike distance", value=0.0, step=0.05, start=0.0, end=1.0,
        )
        self.weight_xpt = pn.widgets.FloatInput(
            name="Weight: X-point region", value=0.0, step=0.05, start=0.0, end=1.0,
        )

        # Divertor region selector — single select, drives per-region editing
        self.divertor_selector = pn.widgets.Select(
            name="Active divertor region",
            options=self._region_names,
            value="lower_outer",
        )
        self.divertor_selector.param.watch(self._on_region_switch, "value")

        # Mask toggle
        self.show_masks = pn.widgets.Checkbox(name="Show coil masks", value=True)
        self.show_masks.param.watch(self._on_mask_toggle, "value")

        # Status
        self.status = pn.pane.Alert("Load a machine in the Setup tab first.", alert_type="info")

        # Wire events
        self.add_buffer_btn.on_click(self._on_add_buffer)
        self.clear_buffers_btn.on_click(self._on_clear_buffers)
        self.clear_xpt_btn.on_click(self._on_clear_xpt_regions)

    # ------------------------------------------------------------------
    def _on_snap_toggle(self, event):
        """Sync the snap-to-wall checkbox to the JS-readable flag source."""
        self._snap_flag_source.data = dict(active=[bool(event.new)])

    # ------------------------------------------------------------------
    def _on_buffer_mode_toggle(self, event):
        """Toggle wall-segment selection mode for buffers."""
        active = bool(event.new)
        self._wall_seg_renderer.visible = active
        self._buffer_mode_source.data = dict(active=[active])
        if not active:
            # Clear the selection highlight when leaving buffer mode
            self._selected_seg_source.data = dict(xs=[], ys=[])
            self._selected_seg_idx_source.data = dict(idx=[-1])

    def _populate_wall_segments(self):
        """Split the wall polyline into individual segments for buffer selection."""
        wr = list(self._wall_source.data.get("R", []))
        wz = list(self._wall_source.data.get("Z", []))
        seg_xs, seg_ys = [], []
        for i in range(len(wr) - 1):
            seg_xs.append([float(wr[i]), float(wr[i + 1])])
            seg_ys.append([float(wz[i]), float(wz[i + 1])])
        self._wall_seg_source.data = dict(xs=seg_xs, ys=seg_ys)
        # Clear any stale selection
        self._selected_seg_source.data = dict(xs=[], ys=[])
        self._selected_seg_idx_source.data = dict(idx=[-1])

    # ------------------------------------------------------------------
    # Region switching
    # ------------------------------------------------------------------
    def _save_active_region(self):
        """Persist the current canvas data into _region_data.

        All coordinate values are converted to plain Python ``float`` and any
        NaN/Inf sentinels that Bokeh draw tools may append are stripped so that
        the saved data round-trips cleanly through Bokeh's serialisation.
        """
        d = self._region_data[self._active_region]
        d["strike"] = {
            "x": [float(v) for v in self._strike_source.data["x"] if _is_finite(v)],
            "y": [float(v) for v in self._strike_source.data["y"] if _is_finite(v)],
        }
        saved_xs, saved_ys = [], []
        for vx, vy in zip(self._xpt_source.data["xs"],
                          self._xpt_source.data["ys"]):
            cx = [float(c) for c in vx if _is_finite(c)]
            cy = [float(c) for c in vy if _is_finite(c)]
            n = min(len(cx), len(cy))  # keep coordinate pairs aligned
            saved_xs.append(cx[:n])
            saved_ys.append(cy[:n])
        d["xpt"] = {"xs": saved_xs, "ys": saved_ys}

        # Persist per-region settings from widgets
        d["cl_mult_factor_zero"] = self.cl_mult_factor_zero.value
        d["weight_connection_length"] = self.weight_cl.value if self.weight_cl.value > 0 else None
        d["weight_strike_point_distance"] = self.weight_strike.value if self.weight_strike.value > 0 else None
        d["weight_xpoint_region"] = self.weight_xpt.value if self.weight_xpt.value > 0 else None

    def _load_region(self, region_name):
        """Load a region's data onto the canvas."""
        d = self._region_data[region_name]

        self._switching_region = True
        self._suppressing_strike_rebuild = True
        self._suppressing_xpt_rebuild = True

        # Deactivate the edit tool before swapping data — prevents the tool's
        # internal state from corrupting the new region's polygon display.
        self._xpt_delete_flag.data = dict(flag=[1])

        self._strike_source.data = dict(
            x=[float(v) for v in d["strike"]["x"]],
            y=[float(v) for v in d["strike"]["y"]],
        )
        self._xpt_source.data = dict(
            xs=[[float(c) for c in v] for v in d["xpt"]["xs"]],
            ys=[[float(c) for c in v] for v in d["xpt"]["ys"]],
        )
        # Clear edit-tool vertices
        self._xpt_edit_vertex_source.data = dict(x=[], y=[])

        self._suppressing_strike_rebuild = False
        self._suppressing_xpt_rebuild = False
        self._switching_region = False

        # Rebuild sidebar lists
        self._rebuild_strike_point_list()
        self._rebuild_xpt_region_list()

        # Restore buffer visuals for this region
        self._restore_buffer_visuals(d.get("buffers", []))

        # Restore per-region settings to widgets
        self.cl_mult_factor_zero.value = d.get("cl_mult_factor_zero", 2.0)
        w_cl = d.get("weight_connection_length")
        self.weight_cl.value = w_cl if w_cl is not None else 0.0
        w_strike = d.get("weight_strike_point_distance")
        self.weight_strike.value = w_strike if w_strike is not None else 0.0
        w_xpt = d.get("weight_xpoint_region")
        self.weight_xpt.value = w_xpt if w_xpt is not None else 0.0

    def _update_ghost_layers(self):
        """Render inactive regions as faded ghosts on the canvas."""
        gx, gy, gc = [], [], []
        gline_xs, gline_ys, gline_c = [], [], []
        gxpt_xs, gxpt_ys, gxpt_c = [], [], []
        gbuf_xs, gbuf_ys, gbuf_c = [], [], []
        for name, d in self._region_data.items():
            if name == self._active_region:
                continue
            colour = _REGION_COLOURS.get(name, "gray")
            # Ghost strike geometry
            sx = list(d["strike"]["x"])
            sy = list(d["strike"]["y"])
            gx.extend(sx)
            gy.extend(sy)
            gc.extend([colour] * len(sx))
            # Ghost strike connecting line (one line per region)
            if len(sx) >= 2:
                gline_xs.append(sx)
                gline_ys.append(sy)
                gline_c.append(colour)
            # Ghost XPT polygons
            for xs, ys in zip(d["xpt"]["xs"], d["xpt"]["ys"]):
                gxpt_xs.append(list(xs))
                gxpt_ys.append(list(ys))
                gxpt_c.append(colour)
            # Ghost buffer outlines
            for bdef in d.get("buffers", []):
                line_geom = LineString(list(zip(bdef["R"], bdef["Z"])))
                buf_geom = line_geom.buffer(bdef["distance"])
                bx, by = buf_geom.exterior.xy
                gbuf_xs.append(list(bx))
                gbuf_ys.append(list(by))
                gbuf_c.append(colour)

        self._ghost_strike_source.data = dict(x=gx, y=gy, color=gc)
        self._ghost_strike_line_source.data = dict(xs=gline_xs, ys=gline_ys, color=gline_c)
        self._ghost_xpt_source.data = dict(xs=gxpt_xs, ys=gxpt_ys, color=gxpt_c)
        self._ghost_buffer_source.data = dict(xs=gbuf_xs, ys=gbuf_ys, color=gbuf_c)

    def _on_region_switch(self, event):
        """Handle switching the active divertor region."""
        old_region = self._active_region
        new_region = event.new
        if new_region == old_region:
            return

        # Save current canvas -> store
        self._save_active_region()
        self._active_region = new_region

        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")
        try:
            # Load new region -> canvas
            self._load_region(new_region)

            # Update ghosts
            self._update_ghost_layers()
        finally:
            if doc is not None:
                doc.unhold()

        self.status.object = f"Editing region: <b>{new_region}</b>"
        self.status.alert_type = "info"

    # ------------------------------------------------------------------
    # Strike geometry list management
    # ------------------------------------------------------------------
    def _on_strike_source_change(self, attr, old, new):
        """Rebuild the strike geometry list whenever data changes."""
        if self._suppressing_strike_rebuild or self._switching_region:
            return
        self._rebuild_strike_point_list()

    def _rebuild_strike_point_list(self):
        """Rebuild the sidebar list of strike geometry from the data source."""
        data = self._strike_source.data
        xs = list(data.get("x", []))
        ys = list(data.get("y", []))
        n = len(xs)

        self._strike_editors.clear()
        items = []
        orig_xs = [float(v) for v in xs]
        orig_ys = [float(v) for v in ys]

        for j in range(n):
            r_inp = pn.widgets.TextInput(
                name="", value=f"{float(xs[j]):.6f}",
                width=95, margin=(0, 1),
            )
            z_inp = pn.widgets.TextInput(
                name="", value=f"{float(ys[j]):.6f}",
                width=95, margin=(0, 1),
            )
            restore_btn = pn.widgets.Button(
                name="\u21ba", button_type="light", width=26,
                margin=(0, 0),
            )
            del_pt_btn = pn.widgets.Button(
                name="\u2715", button_type="danger", width=26,
                margin=(0, 0),
            )

            # Auto-apply when user edits
            r_inp.param.watch(
                lambda event: self._auto_apply_strike(), "value",
            )
            z_inp.param.watch(
                lambda event: self._auto_apply_strike(), "value",
            )

            # Per-point restore
            _or, _oz = orig_xs[j], orig_ys[j]
            restore_btn.on_click(
                lambda event, _ri=r_inp, _zi=z_inp, _r=_or, _z=_oz:
                    self._restore_strike_point(_ri, _zi, _r, _z),
            )

            # Per-point delete
            del_pt_btn.on_click(
                lambda event, _j=j: self._delete_strike_point(_j),
            )

            self._strike_editors.append({"r_input": r_inp, "z_input": z_inp})
            items.append(
                pn.Row(
                    pn.pane.HTML(
                        f"<b style='font-size:11px'>{j + 1}</b>",
                        width=14, margin=(0, 1, 0, 0),
                    ),
                    pn.pane.HTML("<small>R</small>", width=9, margin=(0, 0)),
                    r_inp,
                    pn.pane.HTML("<small>Z</small>", width=9, margin=(0, 0)),
                    z_inp,
                    restore_btn,
                    del_pt_btn,
                    sizing_mode="stretch_width",
                    margin=(2, 0),
                )
            )

        if n > 0:
            card = pn.Card(
                *items,
                title=f"Strike Geometry ({n})",
                collapsed=True,
                sizing_mode="stretch_width",
            )
            self._strike_point_column.objects = [card]
        else:
            self._strike_point_column.objects = []

    def _auto_apply_strike(self):
        """Push current widget values back into the strike data source."""
        try:
            new_x = [float(ed["r_input"].value) for ed in self._strike_editors]
            new_y = [float(ed["z_input"].value) for ed in self._strike_editors]
        except (ValueError, TypeError):
            return
        self._suppressing_strike_rebuild = True
        self._strike_source.data = dict(x=new_x, y=new_y)
        self._suppressing_strike_rebuild = False

    def _restore_strike_point(self, r_input, z_input, orig_r, orig_z):
        """Restore a single strike geometry entry to its original values."""
        r_input.value = f"{orig_r:.6f}"
        z_input.value = f"{orig_z:.6f}"

    def _delete_strike_point(self, index):
        """Delete a single strike geometry entry by index."""
        xs = list(self._strike_source.data["x"])
        ys = list(self._strike_source.data["y"])
        if 0 <= index < len(xs):
            xs.pop(index)
            ys.pop(index)
        self._strike_source.data = dict(x=xs, y=ys)

    def _on_clear_strike_points(self, event):
        """Remove all strike geometry for the active region."""
        self._strike_source.data = dict(x=[], y=[])
        self.status.object = f"Strike geometry cleared for {self._active_region}."
        self.status.alert_type = "info"

    # ------------------------------------------------------------------
    # X-point region list management
    # ------------------------------------------------------------------
    def _on_xpt_source_change(self, attr, old, new):
        """Rebuild the region list whenever the Bokeh data source changes."""
        if self._suppressing_xpt_rebuild or self._switching_region:
            return
        self._rebuild_xpt_region_list()

    def _rebuild_xpt_region_list(self):
        """Rebuild the sidebar list of X-point regions from the data source."""
        self._suppressing_xpt_rebuild = True
        data = self._xpt_source.data
        xs_all = data.get("xs", [])
        ys_all = data.get("ys", [])
        n = len(xs_all)

        self._xpt_editors.clear()
        items = []
        for i in range(n):
            rx = [float(v) for v in xs_all[i]]
            rz = [float(v) for v in ys_all[i]]
            nv = len(rx)
            label = f"Region {i + 1}  ({nv} vertices)"

            # Store original values for per-point restore
            orig_rx = list(rx)
            orig_rz = list(rz)

            r_inputs = []
            z_inputs = []
            rows = []
            for j in range(nv):
                r_inp = pn.widgets.TextInput(
                    name="", value=f"{rx[j]:.6f}",
                    width=95, margin=(0, 1),
                )
                z_inp = pn.widgets.TextInput(
                    name="", value=f"{rz[j]:.6f}",
                    width=95, margin=(0, 1),
                )
                restore_btn = pn.widgets.Button(
                    name="\u21ba", button_type="light", width=26,
                    margin=(0, 0),
                )
                del_pt_btn = pn.widgets.Button(
                    name="\u2715", button_type="danger", width=26,
                    margin=(0, 0),
                )

                # Auto-apply when user edits a coordinate
                idx = i
                r_inp.param.watch(
                    lambda event, _idx=idx: self._auto_apply_xpt(_idx), "value",
                )
                z_inp.param.watch(
                    lambda event, _idx=idx: self._auto_apply_xpt(_idx), "value",
                )

                # Per-point restore to original values
                _or, _oz = orig_rx[j], orig_rz[j]
                restore_btn.on_click(
                    lambda event, _ri=r_inp, _zi=z_inp, _r=_or, _z=_oz:
                        self._restore_point(_ri, _zi, _r, _z),
                )

                # Per-point delete
                del_pt_btn.on_click(
                    lambda event, _i=i, _j=j: self._delete_xpt_point(_i, _j),
                )

                r_inputs.append(r_inp)
                z_inputs.append(z_inp)
                rows.append(
                    pn.Row(
                        pn.pane.HTML(
                            f"<b style='font-size:11px'>{j + 1}</b>",
                            width=14, margin=(0, 1, 0, 0),
                        ),
                        pn.pane.HTML("<small>R</small>", width=9, margin=(0, 0)),
                        r_inp,
                        pn.pane.HTML("<small>Z</small>", width=9, margin=(0, 0)),
                        z_inp,
                        restore_btn,
                        del_pt_btn,
                        sizing_mode="stretch_width",
                        margin=(2, 0),
                    )
                )

            # Delete button on the card header (next to the title)
            delete_btn = pn.widgets.Button(
                name="\u2715", button_type="danger", width=32, height=28,
                margin=(0, 0, 0, 5),
            )
            delete_btn.on_click(lambda event, _i=i: self._delete_xpt_region(_i))

            card = pn.Card(
                *rows,
                title=label,
                collapsed=True,
                sizing_mode="stretch_width",
            )
            # Delete button is placed OUTSIDE the Card so clicking it does not
            # trigger the Card's internal collapse-toggle (which would mutate
            # the Card model and conflict with removal during rebuild).
            item = pn.Row(
                card, delete_btn,
                sizing_mode="stretch_width",
            )
            items.append(item)
            self._xpt_editors.append({"r_inputs": r_inputs, "z_inputs": z_inputs})

        self._xpt_region_column.objects = items
        self._suppressing_xpt_rebuild = False

    def _auto_apply_xpt(self, region_index):
        """Push current widget values for a region back into the data source."""
        if self._suppressing_xpt_rebuild:
            return
        if region_index >= len(self._xpt_editors):
            return
        editors = self._xpt_editors[region_index]
        try:
            new_R = [float(ri.value) for ri in editors["r_inputs"]]
            new_Z = [float(zi.value) for zi in editors["z_inputs"]]
        except (ValueError, TypeError):
            return  # ignore invalid text while user is still typing

        xs = [list(v) for v in self._xpt_source.data["xs"]]
        ys = [list(v) for v in self._xpt_source.data["ys"]]
        if region_index >= len(xs):
            return
        xs[region_index] = new_R
        ys[region_index] = new_Z

        self._suppressing_xpt_rebuild = True
        self._xpt_source.data = dict(xs=xs, ys=ys)
        self._suppressing_xpt_rebuild = False

    def _restore_point(self, r_input, z_input, orig_r, orig_z):
        """Restore a single point's R and Z to their original values."""
        r_input.value = f"{orig_r:.6f}"
        z_input.value = f"{orig_z:.6f}"

    def _deactivate_and_update_xpt(self, new_data):
        """Set delete flag then update XPT source data.

        The flag tells the client-side JS callback (attached to
        _xpt_delete_flag) to deactivate the PolyEditTool and clear vertex
        dots.  The main source data is updated separately so the patches
        renderer processes it without interference.
        """
        self._xpt_delete_flag.data = dict(flag=[1])
        self._xpt_source.data = new_data

    def _delete_xpt_point(self, region_index, point_index):
        """Delete a single vertex from an X-point region."""
        xs = [list(v) for v in self._xpt_source.data["xs"]]
        ys = [list(v) for v in self._xpt_source.data["ys"]]
        if region_index >= len(xs):
            return
        if point_index >= len(xs[region_index]):
            return
        xs[region_index].pop(point_index)
        ys[region_index].pop(point_index)
        if len(xs[region_index]) == 0:
            xs.pop(region_index)
            ys.pop(region_index)
        self._deactivate_and_update_xpt(dict(xs=xs, ys=ys))
        self._xpt_edit_vertex_source.data = dict(x=[], y=[])
        # Persist to _region_data
        d = self._region_data[self._active_region]
        d["xpt"] = {"xs": list(xs), "ys": list(ys)}
        # Force sidebar rebuild
        self._rebuild_xpt_region_list()

    def _delete_xpt_region(self, index):
        """Delete a single X-point region by index."""
        xs = [list(v) for v in self._xpt_source.data["xs"]]
        ys = [list(v) for v in self._xpt_source.data["ys"]]
        if 0 <= index < len(xs):
            xs.pop(index)
            ys.pop(index)
        self._deactivate_and_update_xpt(dict(xs=xs, ys=ys))
        self._xpt_edit_vertex_source.data = dict(x=[], y=[])
        # Persist deletion to _region_data so it survives region switches
        d = self._region_data[self._active_region]
        d["xpt"] = {"xs": list(xs), "ys": list(ys)}
        # Force sidebar rebuild (on_change may not fire if document not live)
        self._rebuild_xpt_region_list()
        self.status.object = f"XPT polygon {index + 1} deleted from {self._active_region}."
        self.status.alert_type = "info"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Colour palette for coils (same as setup tab)
    _COIL_COLOURS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78",
    ]

    def _on_refresh(self, event):
        """Pull the latest machine/equilibrium from shared state."""
        eq = self.state.get("eq")
        tokamak = self.state.get("tokamak")
        if eq is None or tokamak is None:
            self.status.object = "No machine loaded yet. Go to the Setup tab first."
            self.status.alert_type = "warning"
            return

        # Wall
        self._wall_fill_source.data = dict(R=list(tokamak.wall_R), Z=list(tokamak.wall_Z))
        self._wall_source.data = dict(R=list(tokamak.wall_R), Z=list(tokamak.wall_Z))

        # Contours + separatrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(eq.R_2D, eq.Z_2D, eq.psi_2D, levels=60)
        xs, ys = [], []
        for coll in cs.collections:
            for path in coll.get_paths():
                v = path.vertices
                xs.append(v[:, 0].tolist())
                ys.append(v[:, 1].tolist())
        cs_sep = ax_tmp.contour(eq.R_2D, eq.Z_2D, eq.psi_2D, levels=[eq.psi_lcfs])
        sep_xs, sep_ys = [], []
        for coll in cs_sep.collections:
            for path in coll.get_paths():
                v = path.vertices
                sep_xs.append(v[:, 0].tolist())
                sep_ys.append(v[:, 1].tolist())
        plt.close(fig_tmp)
        self._contour_source.data = dict(xs=xs, ys=ys)
        self._lcfs_source.data = dict(xs=sep_xs, ys=sep_ys)

        # Coil positions (same logic as setup tab)
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
            colour = self._COIL_COLOURS[i % len(self._COIL_COLOURS)]
            if isinstance(entry, Circuit):
                for coil_dict in entry.coilset.values():
                    _add_coil(coil_dict["magnet"], colour, name)
            else:
                _add_coil(entry, colour, name)

        self._coil_source.data = dict(x=pt_x, y=pt_y, color=pt_c, name=pt_n)
        self._fil_cross_source.data = dict(x=cr_x, y=cr_y, color=cr_c, name=cr_n)
        self._fil_rect_source.data = dict(xs=rc_xs, ys=rc_ys, fill=rc_f, edge=rc_e, name=rc_n)

        # Coil mask polygons
        mask_xs, mask_ys = [], []
        for _name, entry in tokamak.coilset.items():
            coils = []
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
        self._mask_data_cache = {"xs": mask_xs, "ys": mask_ys}
        if self.show_masks.value and mask_xs:
            self._mask_source.data = dict(xs=mask_xs, ys=mask_ys)
        else:
            self._mask_source.data = dict(xs=[], ys=[])

        # Axis ranges (same logic as setup tab)
        all_R = list(tokamak.wall_R) + pt_x + cr_x
        all_Z = list(tokamak.wall_Z) + pt_y + cr_y
        for rxs in rc_xs:
            all_R.extend(rxs)
        for rys in rc_ys:
            all_Z.extend(rys)

        margin_r, margin_z = 0.5, 0.5
        r_min = float(np.min(all_R)) - margin_r
        r_max = float(np.max(all_R)) + margin_r
        z_min = float(np.min(all_Z)) - margin_z
        z_max = float(np.max(all_Z)) + margin_z

        self.fig.x_range.start = self.fig.x_range.reset_start = r_min
        self.fig.x_range.end = self.fig.x_range.reset_end = r_max
        self.fig.y_range.start = self.fig.y_range.reset_start = z_min
        self.fig.y_range.end = self.fig.y_range.reset_end = z_max

        r_extent = r_max - r_min
        tick_interval = 0.5 if r_extent > 2 else (0.25 if r_extent > 1 else 0.1)
        self.fig.xaxis.ticker.desired_num_ticks = max(3, int(r_extent / tick_interval) + 1)
        self.fig.yaxis.ticker.desired_num_ticks = max(3, int((z_max - z_min) / tick_interval) + 1)

        data_width = r_max - r_min
        data_height = z_max - z_min
        fixed_height = 900
        self.fig.width = max(200, int(fixed_height * data_width / data_height))
        self.fig.height = fixed_height

        # Build per-segment source for buffer selection
        self._populate_wall_segments()

        self.status.object = f"Geometry editor refreshed. Editing region: <b>{self._active_region}</b>"
        self.status.alert_type = "success"

    # ------------------------------------------------------------------
    def _on_mask_toggle(self, event):
        """Show / hide coil mask overlays."""
        if event.new:
            self._mask_source.data = dict(
                xs=self._mask_data_cache["xs"],
                ys=self._mask_data_cache["ys"],
            )
        else:
            self._mask_source.data = dict(xs=[], ys=[])

    # ------------------------------------------------------------------
    def _on_clear_xpt_regions(self, event):
        """Remove all drawn X-point region polygons for the active region."""
        self._deactivate_and_update_xpt(dict(xs=[], ys=[]))
        self._xpt_edit_vertex_source.data = dict(x=[], y=[])
        # Persist to _region_data
        d = self._region_data[self._active_region]
        d["xpt"] = {"xs": [], "ys": []}
        # Force sidebar rebuild
        self._rebuild_xpt_region_list()
        self.status.object = f"XPT regions cleared for {self._active_region}."
        self.status.alert_type = "info"

    # ------------------------------------------------------------------
    def _on_add_buffer(self, event):
        """Create a buffer from the currently selected wall segment (stored per-region)."""
        idx = int(self._selected_seg_idx_source.data["idx"][0])
        seg_xs = self._wall_seg_source.data["xs"]
        seg_ys = self._wall_seg_source.data["ys"]

        if idx < 0 or idx >= len(seg_xs):
            self.status.object = (
                "No wall segment selected. Enable <b>Select segment</b>, "
                "then click a wall segment before adding a buffer."
            )
            self.status.alert_type = "warning"
            return

        R = [float(seg_xs[idx][0]), float(seg_xs[idx][1])]
        Z = [float(seg_ys[idx][0]), float(seg_ys[idx][1])]
        dist = self.buffer_distance.value

        buf_def = {"R": R, "Z": Z, "distance": dist}
        region_buffers = self._region_data[self._active_region]["buffers"]
        region_buffers.append(buf_def)

        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")
        try:
            # Render the buffer outline (Shapely polygon ring)
            line_geom = LineString(list(zip(R, Z)))
            buf_geom = line_geom.buffer(dist)
            bx, by = buf_geom.exterior.xy
            xs = list(self._buffer_line_source.data["xs"]) + [list(bx)]
            ys = list(self._buffer_line_source.data["ys"]) + [list(by)]
            self._buffer_line_source.data = dict(xs=xs, ys=ys)

            # Add the segment to the committed (permanent) highlight layer
            cxs = list(self._committed_seg_source.data["xs"]) + [R]
            cys = list(self._committed_seg_source.data["ys"]) + [Z]
            self._committed_seg_source.data = dict(xs=cxs, ys=cys)

            # Clear the selection highlight
            self._selected_seg_source.data = dict(xs=[], ys=[])
            self._selected_seg_idx_source.data = dict(idx=[-1])

            self._rebuild_buffer_list()
        finally:
            if doc is not None:
                doc.unhold()
        self.status.object = (
            f"Buffer added to {self._active_region} (distance={dist:.3f} m). "
            f"Total: {len(region_buffers)}"
        )
        self.status.alert_type = "success"

    def _on_clear_buffers(self, event):
        """Remove all buffers for the active region."""
        self._region_data[self._active_region]["buffers"].clear()
        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")
        try:
            self._buffer_line_source.data = dict(xs=[], ys=[])
            self._committed_seg_source.data = dict(xs=[], ys=[])
            self._selected_seg_source.data = dict(xs=[], ys=[])
            self._selected_seg_idx_source.data = dict(idx=[-1])
            self._rebuild_buffer_list()
        finally:
            if doc is not None:
                doc.unhold()
        self.status.object = f"All buffers cleared for {self._active_region}."
        self.status.alert_type = "info"

    # ------------------------------------------------------------------
    # Geometry save / load (all regions)
    # ------------------------------------------------------------------
    def _save_geometry_callback(self):
        """Generate a JSON BytesIO for the geometry FileDownload."""
        self._save_active_region()
        divertor_data = self.build_divertor_data() or {}
        buffers = self.get_buffer_defs()
        xpoint_regions = self.get_xpoint_region_defs()

        # Collect all region names across all data sources.
        all_regions = set(divertor_data)
        if buffers is not None:
            all_regions |= set(buffers)
        if xpoint_regions is not None:
            all_regions |= set(xpoint_regions)

        _STRIKE_KEYS = {"strike_R", "strike_Z"}
        _SETTING_KEYS = {
            "connection_length_multiplication_factor_zero",
            "weight_connection_length",
            "weight_strike_point_distance",
            "weight_xpoint_region",
        }

        payload = {}
        for region in sorted(all_regions):
            entry = {}
            dd = divertor_data.get(region, {})
            strike = {k: dd[k] for k in _STRIKE_KEYS if k in dd}
            if strike:
                entry["strike"] = strike
            for k in _SETTING_KEYS:
                if k in dd:
                    entry[k] = dd[k]
            if buffers is not None and region in buffers:
                entry["buffers"] = buffers[region]
            if xpoint_regions is not None and region in xpoint_regions:
                entry["xpoint_regions"] = xpoint_regions[region]
            payload[region] = entry

        content = fancy_json_string(payload)
        return io.BytesIO(content.encode("utf-8"))

    def _on_geometry_file_load(self, event):
        """Load geometry from a JSON file."""
        raw = event.new
        if raw is None:
            return

        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            data = json.loads(text)
        except Exception as exc:
            self.status.object = f"Failed to parse geometry JSON file: {exc}"
            self.status.alert_type = "danger"
            return

        if not isinstance(data, dict) or not data:
            self.status.object = "Geometry JSON must be a dict keyed by region name."
            self.status.alert_type = "danger"
            return

        loaded_regions = 0
        loaded_buffers = 0
        loaded_xpt = 0

        for region, entry in data.items():
            if region not in self._region_data:
                logger.warning("Geometry file: skipping unknown region '%s'", region)
                continue

            d = self._region_data[region]

            # Strike geometry
            strike = entry.get("strike", {})
            sr = strike.get("strike_R")
            sz = strike.get("strike_Z")
            if sr is not None and sz is not None:
                d["strike"]["x"] = sr if isinstance(sr, list) else [sr]
                d["strike"]["y"] = sz if isinstance(sz, list) else [sz]
            else:
                d["strike"]["x"] = []
                d["strike"]["y"] = []

            # Per-region settings
            if "connection_length_multiplication_factor_zero" in entry:
                d["cl_mult_factor_zero"] = entry["connection_length_multiplication_factor_zero"]
            if "weight_connection_length" in entry:
                d["weight_connection_length"] = entry["weight_connection_length"]
            if "weight_strike_point_distance" in entry:
                d["weight_strike_point_distance"] = entry["weight_strike_point_distance"]
            if "weight_xpoint_region" in entry:
                d["weight_xpoint_region"] = entry["weight_xpoint_region"]

            # Buffers
            bufs = entry.get("buffers")
            if bufs is not None and isinstance(bufs, list):
                d["buffers"] = list(bufs)
                loaded_buffers += len(bufs)

            # XPT regions
            xpt = entry.get("xpoint_regions")
            if xpt is not None:
                R = xpt.get("R", [])
                Z = xpt.get("Z", [])
                if R and Z:
                    d["xpt"]["xs"] = [list(R)]
                    d["xpt"]["ys"] = [list(Z)]
                    loaded_xpt += 1

            # Enable the region
            if region in self.region_enabled:
                self.region_enabled[region].value = True

            loaded_regions += 1

        # Refresh canvas for active region
        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")
        try:
            self._load_region(self._active_region)
            self._update_ghost_layers()
        finally:
            if doc is not None:
                doc.unhold()

        # Clear file input so the same file can be re-loaded
        self.load_geometry_file.value = None

        parts = [f"{loaded_regions} region(s)"]
        if loaded_buffers:
            parts.append(f"{loaded_buffers} buffer(s)")
        if loaded_xpt:
            parts.append(f"{loaded_xpt} XPT region(s)")
        self.status.object = f"Loaded {', '.join(parts)} from geometry file."
        self.status.alert_type = "success"

    def _on_delete_btn_click(self, event):
        """Shared click handler for all buffer delete buttons."""
        self._delete_buffer(event.obj._buf_idx)

    def _delete_buffer(self, index):
        """Delete a single buffer by index from the active region."""
        region_buffers = self._region_data[self._active_region]["buffers"]
        if not (0 <= index < len(region_buffers)):
            return
        region_buffers.pop(index)
        n_remaining = len(region_buffers)

        # Batch all Bokeh model mutations (data-source updates, card.pop,
        # label re-index, title update) into a single combined message.
        # No new models are created here — only existing ones are mutated
        # or removed — so hold("combine") is safe and prevents the
        # individual patches from racing with a dynamic tab switch.
        doc = pn.state.curdoc
        if doc is not None:
            doc.hold("combine")
        try:
            # Update data sources.
            bxs = list(self._buffer_line_source.data["xs"])
            bys = list(self._buffer_line_source.data["ys"])
            if index < len(bxs):
                bxs.pop(index)
                bys.pop(index)
            self._buffer_line_source.data = dict(xs=bxs, ys=bys)

            cxs = list(self._committed_seg_source.data["xs"])
            cys = list(self._committed_seg_source.data["ys"])
            if index < len(cxs):
                cxs.pop(index)
                cys.pop(index)
            self._committed_seg_source.data = dict(xs=cxs, ys=cys)

            # Surgically pop the single deleted Row from the Card.
            if self._buffer_list_column.objects:
                card = self._buffer_list_column.objects[0]
                if index < len(card):
                    card.pop(index)
                if len(card):
                    # Re-index remaining buttons and update labels in-place.
                    for new_i, row in enumerate(card.objects):
                        row.objects[-1]._buf_idx = new_i
                        bdef = region_buffers[new_i]
                        row.objects[0].object = (
                            f"<small><b>{new_i + 1}.</b> "
                            f"({bdef['R'][0]:.3f},\u2009{bdef['Z'][0]:.3f}) \u2192 "
                            f"({bdef['R'][1]:.3f},\u2009{bdef['Z'][1]:.3f})&ensp;"
                            f"d={bdef['distance']:.3f}\u2009m</small>"
                        )
                    card.title = f"Buffers ({len(card)})"
                else:
                    self._buffer_list_column.objects = []

            self.status.object = f"Buffer {index + 1} deleted from {self._active_region}. Remaining: {n_remaining}"
            self.status.alert_type = "info"
        finally:
            if doc is not None:
                doc.unhold()

    def _restore_buffer_visuals(self, buffers):
        """Restore the buffer outline and committed segment renderers from a list of buffer defs."""
        outline_xs, outline_ys = [], []
        seg_xs, seg_ys = [], []
        for bdef in buffers:
            line_geom = LineString(list(zip(bdef["R"], bdef["Z"])))
            buf_geom = line_geom.buffer(bdef["distance"])
            bx, by = buf_geom.exterior.xy
            outline_xs.append(list(bx))
            outline_ys.append(list(by))
            seg_xs.append(list(bdef["R"]))
            seg_ys.append(list(bdef["Z"]))
        self._buffer_line_source.data = dict(xs=outline_xs, ys=outline_ys)
        self._committed_seg_source.data = dict(xs=seg_xs, ys=seg_ys)
        self._selected_seg_source.data = dict(xs=[], ys=[])
        self._selected_seg_idx_source.data = dict(idx=[-1])
        self._rebuild_buffer_list()

    def _rebuild_buffer_list(self):
        """Rebuild the sidebar list of committed buffers for the active region."""
        region_buffers = self._region_data[self._active_region]["buffers"]
        items = []
        for i, bdef in enumerate(region_buffers):
            r1, z1 = bdef["R"][0], bdef["Z"][0]
            r2, z2 = bdef["R"][1], bdef["Z"][1]
            dist = bdef["distance"]
            label_html = (
                f"<small><b>{i + 1}.</b> "
                f"({r1:.3f},\u2009{z1:.3f}) \u2192 "
                f"({r2:.3f},\u2009{z2:.3f})&ensp;"
                f"d={dist:.3f}\u2009m</small>"
            )
            del_btn = pn.widgets.Button(
                name="\u2715", button_type="danger", width=26,
                margin=(0, 0),
            )
            del_btn._buf_idx = i
            del_btn.on_click(self._on_delete_btn_click)
            items.append(
                pn.Row(
                    pn.pane.HTML(label_html, width=370),
                    del_btn,
                    margin=(2, 0),
                )
            )

        if items:
            card = pn.Card(
                *items,
                title=f"Buffers ({len(items)})",
                collapsed=False,
                sizing_mode="stretch_width",
            )
            self._buffer_list_column.objects = [card]
        else:
            self._buffer_list_column.objects = []

    # ------------------------------------------------------------------
    # Public helpers for the optimisation tab to pull geometry definitions
    # ------------------------------------------------------------------

    def _enabled_regions(self):
        """Return the set of region names that are toggled on."""
        return {name for name in self._region_names if self.region_enabled[name].value}

    def get_buffer_defs(self):
        """Return buffer definitions as a dict keyed by region name, or None.

        Each key is a divertor region name and the value is a list of
        ``{"R": [...], "Z": [...], "distance": d}`` dicts.
        Only regions with at least one buffer are included.

        Returns ``None`` if no buffers have been defined in any region.
        """
        self._save_active_region()
        result = {}
        enabled = self._enabled_regions()
        for name, d in self._region_data.items():
            if name not in enabled:
                continue
            bufs = d.get("buffers", [])
            if bufs:
                result[name] = list(bufs)
        return result if result else None

    def get_configured_regions(self):
        """Return the list of enabled region names that have strike geometry defined."""
        # Save current canvas first so data is up to date
        self._save_active_region()
        enabled = self._enabled_regions()
        return [
            name for name, d in self._region_data.items()
            if name in enabled and d["strike"]["x"]  # enabled + at least one strike geometry
        ]

    def get_xpoint_region_defs(self):
        """Return X-point region definitions as a dict keyed by region name (or None).

        Each key is a divertor region name and the value is a ``{R, Z}`` dict
        defining the XPT polygon boundary for that region.  Only the **first**
        XPT polygon drawn in a region is returned (the optimiser supports one
        per divertor region).

        Returns ``None`` if no XPT polygons have been drawn.
        """
        self._save_active_region()
        result = {}
        enabled = self._enabled_regions()
        for name, d in self._region_data.items():
            if name not in enabled:
                continue
            xs_list = d["xpt"]["xs"]
            ys_list = d["xpt"]["ys"]
            if xs_list:
                # Use first polygon only (one XPT region per divertor region)
                result[name] = {"R": list(xs_list[0]), "Z": list(ys_list[0])}
        return result if result else None

    def get_strike_points(self):
        """Return strike geometry coordinates for the active region as (R_list, Z_list)."""
        d = self._strike_source.data
        return list(d["x"]), list(d["y"])

    def get_divertor_regions(self):
        """Return the list of configured divertor region names."""
        return self.get_configured_regions()

    def build_divertor_data(self):
        """Assemble a ``divertor_data`` dict from the current widget state.

        Each configured region gets its own ``strike_R`` / ``strike_Z``.
        Returns *None* if no regions have strike geometry.
        """
        self._save_active_region()

        divertor_data = {}
        enabled = self._enabled_regions()
        for name, d in self._region_data.items():
            if name not in enabled:
                continue
            sx = list(d["strike"]["x"])
            sy = list(d["strike"]["y"])
            if not sx:
                continue  # skip regions without strike geometry
            entry = {
                "strike_R": sx if len(sx) > 1 else sx[0],
                "strike_Z": sy if len(sy) > 1 else sy[0],
                "connection_length_multiplication_factor_zero": d.get("cl_mult_factor_zero", 2.0),
            }
            w = d.get("weight_connection_length")
            if w is not None:
                entry["weight_connection_length"] = w
            w = d.get("weight_strike_point_distance")
            if w is not None:
                entry["weight_strike_point_distance"] = w
            w = d.get("weight_xpoint_region")
            if w is not None:
                entry["weight_xpoint_region"] = w
            divertor_data[name] = entry

        return divertor_data if divertor_data else None

    # ------------------------------------------------------------------
    @property
    def panel(self):
        if hasattr(self, "_panel"):
            return self._panel
        sidebar = pn.Column(
            "### Geometry Tools",
            pn.pane.HTML(
                "<small>Select a divertor region below, then draw strike "
                "geometry and XPT regions for that region. Switch regions "
                "to define geometry for each one independently. Other "
                "regions' geometry is shown faded on the canvas.</small>"
            ),
            "**Divertor Regions**",
            pn.pane.HTML("<small>Toggle regions to include in the optimisation.</small>"),
            pn.Column(*(self.region_enabled[n] for n in self._region_names)),
            "**Active Divertor Region**",
            self.divertor_selector,
            pn.layout.Divider(),
            "**Per-Region Settings**",
            pn.pane.HTML(
                "<small>These are the <b>relative cost weights</b> for each "
                "divertor region, controlling how much each cost term "
                "(connection length, strike distance, X-point region) "
                "contributes from this region versus the others. "
                "For example, if the Lower Outer region has a strike "
                "distance weight of 0.6, then 60% of the total strike "
                "distance cost comes from that region.<br><br>"
                "Leave at <b>0</b> to distribute equally across all "
                "enabled regions (each gets 1/<em>N</em>, where "
                "<em>N</em> is the number of enabled regions).</small>"
            ),
            self.cl_mult_factor_zero,
            self.weight_cl,
            self.weight_strike,
            self.weight_xpt,
            pn.layout.Divider(),
            "**Strike Geometry**",
            pn.pane.HTML(
                "<small>Use the <em>Draw strike geometry</em> tool "
                f"(<img src='{_ICON_POINT_DRAW}' "
                "height='14' style='vertical-align:middle;'/>) "
                "on the canvas to place points. A single point defines a "
                "strike point; two or more points define a strike surface. "
                "A line connects them in order. Drag points to move them; "
                "use the controls below to edit or delete.</small>"
            ),
            self.snap_to_wall,
            self._strike_point_column,
            self.clear_strike_btn,
            pn.layout.Divider(),
            self.show_masks,
            pn.layout.Divider(),
            "**Buffers**",
            pn.pane.HTML(
                "<small>Enable <em>Select segment</em>, hover over the wall "
                "to highlight segments (yellow), then click to select one "
                "(cyan). Set the buffer distance and press <em>Add Buffer</em>. "
                "Use the pan/zoom tools while selecting segments. "
                "Buffers are not selected via the Bokeh toolbar; use the "
                "sidebar controls below.<br><br>"
                "<b>Important:</b> Make sure the <em>Draw strike geometry</em> "
                "tool is <b>not</b> selected in the Bokeh toolbar when "
                "selecting buffer segments, otherwise clicks will also "
                "place strike points.</small>"
            ),
            self.buffer_select_mode,
            self.buffer_distance,
            self.add_buffer_btn,
            self.clear_buffers_btn,
            self._buffer_list_column,
            pn.layout.Divider(),
            "**X-point Regions**",
            pn.pane.HTML(
                "<small><b>Draw X-point region:</b> Select the "
                "<em>Draw X-point region</em> tool "
                f"(<img src='{_ICON_POLY_DRAW}' "
                "height='14' style='vertical-align:middle;'/>) "
                "in the Bokeh toolbar, then <b>click and hold</b> to place the first vertex. "
                "<b>Click</b> to add more. Press <b>Esc</b> or "
                "<b>click and hold</b> to finish.<br><br>"
                "<b>Edit X-point region vertices:</b> Select the "
                "<em>Edit X-point region vertices</em> tool "
                f"(<img src='{_ICON_POLY_EDIT}' "
                "height='14' style='vertical-align:middle;'/>) "
                "in the Bokeh toolbar, then <b>click and hold</b> on the shaded polygon area "
                "(not on a vertex) &mdash; red vertices will appear.<br>"
                "&bull; <b>Drag</b> a vertex to move it (start moving "
                "immediately after pressing down).<br>"
                "&bull; <b>Click and hold on a vertex</b> to add a new "
                "vertex &mdash; then <b>click</b> to place it.<br>"
                "&bull; Press <b>Backspace</b> to delete a selected "
                "vertex.<br>"
                "&bull; Press <b>Esc</b> to cancel / exit insert mode."
                "</small>"
            ),
            self._xpt_region_column,
            self.clear_xpt_btn,
            pn.layout.Divider(),
            "**Load / Save Geometry**",
            pn.pane.HTML(
                "<small>Save or load the complete geometry data "
                "(strike points/surfaces, buffers, XPT regions, and "
                "per-region settings) for all regions as a JSON file.</small>"
            ),
            pn.Row(self.load_geometry_file, self.save_geometry_btn),
            pn.layout.Divider(),
            self.status,
            width=440,
            height=900,
            scroll=True,
        )
        fig_pane = pn.pane.Bokeh(self.fig, sizing_mode="stretch_height")
        self._panel = pn.Row(sidebar, fig_pane, sizing_mode="stretch_both")
        return self._panel
