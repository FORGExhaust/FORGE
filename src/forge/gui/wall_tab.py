"""Wall Editor tab - interactively edit the machine wall outline.

Editing happens on a temporary copy of the wall held by this tab. The machine
and equilibrium are only modified when the user presses *Finish*, so the wall
can be reshaped, loaded from file and exported without disturbing anything
downstream until it is deliberately committed.

The wall polygon is stored here **open** (no repeated closing point). Bokeh
closes patches automatically for display, and the closing point is re-appended
when the wall is committed or written out, matching the convention used by
:func:`forge.io.read_geqdsk`.
"""

import io
import json
import logging
import math
import os

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import (
    BoxZoomTool, ColumnDataSource, CustomJS, PolyEditTool, Range1d,
)
from bokeh.plotting import figure as bk_figure
from shapely.geometry import LinearRing

from forge.gui.machine_plot import MachineLayers, apply_view_bounds
from forge.gui.tool_icons import ICON_POLY_EDIT, tool_icon
from forge.io import fancy_json_string, geqdsk_dict, wall_from_dict, write_geqdsk, write_wall

logger = logging.getLogger(__name__)

_R_COL = "R (m)"
_Z_COL = "Z (m)"


def _is_finite(v):
    """Return True if *v* is a finite number (not NaN/Inf)."""
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _open_wall(wall_R, wall_Z):
    """Strip the repeated closing point, if present."""
    R = [float(v) for v in wall_R]
    Z = [float(v) for v in wall_Z]
    if len(R) > 1 and R[0] == R[-1] and Z[0] == Z[-1]:
        R = R[:-1]
        Z = Z[:-1]
    return R, Z


def _closed_wall(wall_R, wall_Z):
    """Append the first point so the outline is closed."""
    R = [float(v) for v in wall_R]
    Z = [float(v) for v in wall_Z]
    if len(R) > 1 and not (R[0] == R[-1] and Z[0] == Z[-1]):
        R.append(R[0])
        Z.append(Z[0])
    return R, Z


class WallTab:
    """Panel component for editing the machine wall outline.

    All edits are made to a temporary wall owned by this tab. Nothing is
    written back to the machine, equilibrium or GEQDSK data until
    :meth:`_on_finish` runs.
    """

    def __init__(self, shared_state):
        self.state = shared_state

        # Tabs whose plots need refreshing once a new wall is committed.
        # Assigned by forge.gui.app.create_app.
        self._setup_tab = None
        self._geometry_tab = None

        # Guards against table <-> canvas <-> quick-editor feedback loops
        self._suppressing_table_sync = False
        self._suppressing_quick_editor = False

        # --- Main Bokeh figure ---
        self.fig = bk_figure(
            title="Wall Editor",
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

        # Vessel interior, drawn first so everything else sits on top. Driven
        # by the temporary wall so the fill follows edits live.
        self._wall_fill_source = ColumnDataSource(data=dict(R=[], Z=[]))
        self.fig.patch("R", "Z", source=self._wall_fill_source,
                       fill_color="white", line_color=None)

        # Static background (contours, separatrix, coils). The wall is
        # excluded because this tab owns an editable version of it.
        self.layers = MachineLayers(self.fig, include_wall=False)

        # --- Editable wall polygon ---
        self._wall_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        wall_renderer = self.fig.patches(
            "xs", "ys", source=self._wall_source,
            fill_alpha=0.0, line_color="black", line_width=2,
        )

        self._edit_vertex_source = ColumnDataSource(data=dict(x=[], y=[]))
        edit_vertex_renderer = self.fig.scatter(
            "x", "y", source=self._edit_vertex_source,
            color="red", size=9, marker="circle",
        )
        self._edit_tool = PolyEditTool(
            renderers=[wall_renderer],
            vertex_renderer=edit_vertex_renderer,
            description="Edit wall vertices",
        )
        self.fig.add_tools(self._edit_tool)

        # Highlight for the point selected in the table
        self._selected_vertex_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.fig.scatter(
            "x", "y", source=self._selected_vertex_source,
            marker="circle", size=14, fill_alpha=0.0,
            line_color="#00bcd4", line_width=3,
        )

        # When the wall is replaced programmatically while PolyEditTool is
        # active, stale vertex handles remain on the canvas. Setting this flag
        # deactivates the tool and clears them (same approach as the X-point
        # region editor on the Geometry tab).
        self._edit_reset_flag = ColumnDataSource(data=dict(flag=[0]))
        self.fig.scatter(
            "flag", "flag", source=self._edit_reset_flag,
            size=0, alpha=0, visible=False,
        )
        self._edit_reset_flag.js_on_change(
            "data",
            CustomJS(
                args=dict(
                    edit_tool=self._edit_tool,
                    vertex_source=self._edit_vertex_source,
                    main_source=self._wall_source,
                ),
                code="""
                    if (cb_obj.data['flag'][0] > 0) {
                        edit_tool.active = false;
                        vertex_source.data = {x: [], y: []};
                        vertex_source.change.emit();
                        main_source.selected.indices = [];
                        setTimeout(function() {
                            vertex_source.data = {x: [], y: []};
                            vertex_source.change.emit();
                            main_source.change.emit();
                        }, 0);
                    }
                """,
            ),
        )

        self._wall_source.on_change("data", self._on_wall_source_change)

        # --- Sidebar widgets ---
        self.status = pn.pane.Alert(
            "No machine loaded yet. Go to the Setup tab first.",
            alert_type="info",
        )

        # Re-reading the machine both discards edits and re-syncs the
        # background layers, so a separate "undo my edits" button would do
        # exactly the same thing.
        self.reset_btn = pn.widgets.Button(
            name="Reset to machine wall", button_type="default",
        )
        self.reset_btn.on_click(self._on_refresh)

        self.show_masks = pn.widgets.Checkbox(name="Show coil masks", value=False)
        self.show_masks.param.watch(self._on_toggle_masks, "value")

        # Point table
        self._table = pn.widgets.Tabulator(
            self._make_dataframe([], []),
            height=320,
            width=400,
            show_index=True,
            selectable=1,
            layout="fit_columns",
            theme="simple",
        )
        self._table.on_edit(self._on_table_edit)
        self._table.param.watch(self._on_table_selection, "selection")

        # Selected-point quick editor
        self.sel_r_input = pn.widgets.TextInput(name="R (m)", value="", width=120)
        self.sel_z_input = pn.widgets.TextInput(name="Z (m)", value="", width=120)
        self.sel_r_input.param.watch(self._on_quick_edit, "value")
        self.sel_z_input.param.watch(self._on_quick_edit, "value")

        self.insert_btn = pn.widgets.Button(
            name="Insert point after", button_type="primary", width=120,
        )
        self.insert_btn.on_click(self._on_insert_point)
        self.delete_btn = pn.widgets.Button(
            name="Delete point", button_type="danger", width=120,
        )
        self.delete_btn.on_click(self._on_delete_point)

        # --- Load / save widgets ---
        self.load_wall_file = pn.widgets.FileInput(
            accept=".json", multiple=False, width=220,
        )
        self.load_wall_file.param.watch(self._on_wall_file_load, "value")

        self.load_wall_path = pn.widgets.TextInput(
            name="Server file path", placeholder="/path/to/wall.json",
        )
        self.load_wall_path_btn = pn.widgets.Button(
            name="Load (server path)", button_type="primary",
        )
        self.load_wall_path_btn.on_click(self._on_load_wall_path)

        self.save_wall_download = pn.widgets.FileDownload(
            callback=self._save_wall_callback,
            filename="wall.json",
            label="Download wall (.json)",
            button_type="success",
            width=220,
        )
        self.save_wall_path = pn.widgets.TextInput(
            name="Server file path", placeholder="/path/to/wall.json",
        )
        self.save_wall_path_btn = pn.widgets.Button(
            name="Save wall (server path)", button_type="success",
        )
        self.save_wall_path_btn.on_click(self._on_save_wall_path)

        self.save_geqdsk_download = pn.widgets.FileDownload(
            callback=self._save_geqdsk_callback,
            filename="forge_wall_edited.geqdsk",
            label="Download GEQDSK",
            button_type="warning",
            width=220,
        )
        self.save_geqdsk_path = pn.widgets.TextInput(
            name="Server file path", placeholder="/path/to/equilibrium.geqdsk",
        )
        self.save_geqdsk_path_btn = pn.widgets.Button(
            name="Save GEQDSK (server path)", button_type="warning",
        )
        self.save_geqdsk_path_btn.on_click(self._on_save_geqdsk_path)

        self.finish_btn = pn.widgets.Button(
            name="Finish - update machine wall", button_type="danger",
        )
        self.finish_btn.on_click(self._on_finish)

    # ------------------------------------------------------------------
    # Event plumbing
    # ------------------------------------------------------------------
    def _defer_to_document(self, fn):
        """Run *fn* with the Bokeh document lock held.

        Panel dispatches widget events without the document lock, and
        mutating a Bokeh model from such a handler raises inside the server
        session, so the update never reaches the browser. Deferring to a
        next-tick callback runs the work with the lock held. Outside a server
        session (tests, notebooks) there is no lock to acquire and *fn* runs
        immediately.
        """
        doc = pn.state.curdoc
        if doc is not None and doc.session_context is not None:
            doc.add_next_tick_callback(fn)
        else:
            fn()

    def _set_status(self, message, alert_type):
        """Update the status alert, holding the document lock if needed."""
        def _apply():
            self.status.object = message
            self.status.alert_type = alert_type

        self._defer_to_document(_apply)

    # ------------------------------------------------------------------
    # Temporary wall accessors
    # ------------------------------------------------------------------
    def _get_wall(self):
        """Return the temporary wall as open ``(R, Z)`` lists of float."""
        xs = self._wall_source.data.get("xs") or []
        ys = self._wall_source.data.get("ys") or []
        if not xs or not ys:
            return [], []

        # PolyEditTool only edits the existing polygon, but guard anyway so a
        # stray extra ring can never silently become the wall.
        R = [float(v) for v in xs[0] if _is_finite(v)]
        Z = [float(v) for v in ys[0] if _is_finite(v)]
        n = min(len(R), len(Z))  # keep coordinate pairs aligned
        return R[:n], Z[:n]

    def _set_wall(self, wall_R, wall_Z, reset_edit_tool=True):
        """Replace the temporary wall and refresh the canvas and table."""
        R, Z = _open_wall(wall_R, wall_Z)

        if reset_edit_tool:
            # Clear stale PolyEditTool vertex handles before the data changes
            flag = int(self._edit_reset_flag.data["flag"][0])
            self._edit_reset_flag.data = dict(flag=[flag + 1])

        self._suppressing_table_sync = True
        self._wall_source.data = dict(xs=[R], ys=[Z])
        self._suppressing_table_sync = False

        self._sync_wall_fill()
        self._rebuild_table(R, Z)
        self._update_selection_highlight()

    def _sync_wall_fill(self):
        R, Z = self._get_wall()
        self._wall_fill_source.data = dict(R=R, Z=Z)

    # ------------------------------------------------------------------
    # Table synchronisation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_dataframe(wall_R, wall_Z):
        return pd.DataFrame({
            _R_COL: pd.Series(wall_R, dtype="float64"),
            _Z_COL: pd.Series(wall_Z, dtype="float64"),
        })

    def _rebuild_table(self, wall_R, wall_Z):
        """Push the wall into the table, patching in place where possible."""
        df = self._table.value
        if df is not None and len(df) == len(wall_R):
            patches = {}
            r_patch = [
                (i, float(wall_R[i])) for i in range(len(wall_R))
                if float(df[_R_COL].iat[i]) != float(wall_R[i])
            ]
            z_patch = [
                (i, float(wall_Z[i])) for i in range(len(wall_Z))
                if float(df[_Z_COL].iat[i]) != float(wall_Z[i])
            ]
            if r_patch:
                patches[_R_COL] = r_patch
            if z_patch:
                patches[_Z_COL] = z_patch
            if patches:
                self._table.patch(patches)
            return

        selection = list(self._table.selection or [])
        self._table.value = self._make_dataframe(wall_R, wall_Z)
        # Keep the selected row if it still exists
        self._table.selection = [i for i in selection if i < len(wall_R)]

    def _on_wall_source_change(self, attr, old, new):
        """Mirror canvas edits into the fill, table and highlight."""
        if self._suppressing_table_sync:
            return
        R, Z = self._get_wall()
        self._sync_wall_fill()
        self._rebuild_table(R, Z)
        self._update_selection_highlight()

    def _on_table_edit(self, event):
        """Push a table cell edit back into the canvas."""
        row, column, value = event.row, event.column, event.value
        self._defer_to_document(lambda: self._apply_table_edit(row, column, value))

    def _apply_table_edit(self, row, column, value):
        R, Z = self._get_wall()
        if not (0 <= row < len(R)):
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return

        if column == _R_COL:
            R[row] = value
        elif column == _Z_COL:
            Z[row] = value
        else:
            return

        self._suppressing_table_sync = True
        self._wall_source.data = dict(xs=[R], ys=[Z])
        self._suppressing_table_sync = False
        self._sync_wall_fill()
        self._update_selection_highlight()

    # ------------------------------------------------------------------
    # Selected-point editing
    # ------------------------------------------------------------------
    def _selected_index(self):
        selection = self._table.selection or []
        if not selection:
            return None
        idx = int(selection[0])
        n = len(self._get_wall()[0])
        return idx if 0 <= idx < n else None

    def _on_table_selection(self, event):
        self._defer_to_document(self._update_selection_highlight)

    def _update_selection_highlight(self):
        """Highlight the selected point and load it into the quick editor."""
        R, Z = self._get_wall()
        idx = self._selected_index()

        if idx is None:
            self._selected_vertex_source.data = dict(x=[], y=[])
            self._suppressing_quick_editor = True
            self.sel_r_input.value = ""
            self.sel_z_input.value = ""
            self._suppressing_quick_editor = False
            return

        self._selected_vertex_source.data = dict(x=[R[idx]], y=[Z[idx]])
        self._suppressing_quick_editor = True
        self.sel_r_input.value = f"{R[idx]:.6f}"
        self.sel_z_input.value = f"{Z[idx]:.6f}"
        self._suppressing_quick_editor = False

    def _on_quick_edit(self, event):
        """Apply the quick-editor boxes to the selected point."""
        # Checked here rather than in the deferred body so that the flag is
        # still set when the highlight code is the one writing the boxes.
        if self._suppressing_quick_editor:
            return
        self._defer_to_document(self._apply_quick_edit)

    def _apply_quick_edit(self):
        idx = self._selected_index()
        if idx is None:
            return
        try:
            new_r = float(self.sel_r_input.value)
            new_z = float(self.sel_z_input.value)
        except (ValueError, TypeError):
            return  # ignore invalid text while the user is still typing
        if not (math.isfinite(new_r) and math.isfinite(new_z)):
            return

        R, Z = self._get_wall()
        R[idx] = new_r
        Z[idx] = new_z

        self._suppressing_table_sync = True
        self._wall_source.data = dict(xs=[R], ys=[Z])
        self._suppressing_table_sync = False
        self._sync_wall_fill()
        self._rebuild_table(R, Z)
        self._selected_vertex_source.data = dict(x=[new_r], y=[new_z])

    def _on_insert_point(self, event):
        """Insert a point midway between the selected point and the next."""
        self._defer_to_document(self._apply_insert_point)

    def _apply_insert_point(self):
        R, Z = self._get_wall()
        if not R:
            self.status.object = "No wall to edit. Load a machine first."
            self.status.alert_type = "warning"
            return

        # With nothing selected, extend from the end of the wall, i.e. insert
        # on the segment that closes the outline.
        idx = self._selected_index()
        if idx is None:
            idx = len(R) - 1

        nxt = (idx + 1) % len(R)
        mid_r = 0.5 * (R[idx] + R[nxt])
        mid_z = 0.5 * (Z[idx] + Z[nxt])
        R.insert(idx + 1, mid_r)
        Z.insert(idx + 1, mid_z)

        self._set_wall(R, Z)
        self._table.selection = [idx + 1]
        self.status.object = f"Inserted point {idx + 1}. Wall has {len(R)} points."
        self.status.alert_type = "info"

    def _on_delete_point(self, event):
        """Delete the selected point."""
        self._defer_to_document(self._apply_delete_point)

    def _apply_delete_point(self):
        R, Z = self._get_wall()
        idx = self._selected_index()
        if idx is None:
            self.status.object = "Select a point in the table first."
            self.status.alert_type = "warning"
            return
        if len(R) <= 3:
            self.status.object = "A wall needs at least 3 points."
            self.status.alert_type = "danger"
            return

        R.pop(idx)
        Z.pop(idx)
        self._set_wall(R, Z)
        self._table.selection = [min(idx, len(R) - 1)]
        self.status.object = f"Deleted point {idx}. Wall has {len(R)} points."
        self.status.alert_type = "info"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(wall_R, wall_Z):
        """Return an error message describing why a wall is unusable, or None."""
        if len(wall_R) != len(wall_Z):
            return "Wall R and Z have different lengths."

        if any(not _is_finite(v) for v in wall_R) or any(not _is_finite(v) for v in wall_Z):
            return "The wall contains non-finite coordinates."

        unique = {(float(r), float(z)) for r, z in zip(wall_R, wall_Z)}
        if len(unique) < 3:
            return f"A wall needs at least 3 distinct points, but has {len(unique)}."

        bad_R = [r for r in wall_R if float(r) <= 0.0]
        if bad_R:
            return (
                f"All wall R coordinates must be greater than 0 "
                f"({len(bad_R)} point(s) are not)."
            )

        closed_R, closed_Z = _closed_wall(wall_R, wall_Z)
        try:
            ring = LinearRing(list(zip(closed_R, closed_Z)))
        except Exception as exc:
            return f"The wall is not a valid polygon: {exc}"
        if not ring.is_simple:
            return (
                "The wall outline intersects itself. Field-line tracing needs "
                "a simple (non-self-intersecting) wall, so this cannot be "
                "applied to the machine."
            )

        return None

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def _on_refresh(self, event):
        """Pull the latest machine/equilibrium from shared state.

        This doubles as the reset action: re-reading the machine discards any
        edits made to the temporary wall.
        """
        self._defer_to_document(self._apply_refresh)

    def _apply_refresh(self):
        eq = self.state.get("eq")
        tokamak = self.state.get("tokamak")
        if eq is None or tokamak is None:
            self.status.object = "No machine loaded yet. Go to the Setup tab first."
            self.status.alert_type = "warning"
            return

        self.layers.update(eq, tokamak, show_masks=self.show_masks.value)
        self._set_wall(tokamak.wall_R, tokamak.wall_Z)

        bounds = self.layers.view_bounds(tokamak.wall_R, tokamak.wall_Z)
        apply_view_bounds(self.fig, bounds)

        n = len(self._get_wall()[0])
        self.status.object = (
            f"Wall editor refreshed. Editing a copy of the wall ({n} points). "
            "The machine is unchanged until you press Finish."
        )
        self.status.alert_type = "success"

    def _on_toggle_masks(self, event):
        visible = bool(event.new)
        self._defer_to_document(lambda: self.layers.set_masks_visible(visible))

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    def _apply_loaded_wall(self, wall_R, wall_Z, source_desc):
        self._set_wall(wall_R, wall_Z)
        n = len(self._get_wall()[0])
        self.status.object = (
            f"Loaded wall from {source_desc} ({n} points). "
            "Press Finish to apply it to the machine."
        )
        self.status.alert_type = "success"

    def _on_wall_file_load(self, event):
        """Load a wall from an uploaded JSON file."""
        raw = event.new
        if raw is None:
            return
        self._defer_to_document(lambda: self._apply_wall_file_load(raw))

    def _apply_wall_file_load(self, raw):
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            payload = json.loads(text)
            wall_R, wall_Z = wall_from_dict(payload)
        except Exception as exc:
            self.status.object = f"Failed to load wall JSON: {exc}"
            self.status.alert_type = "danger"
            return

        self._apply_loaded_wall(wall_R, wall_Z, "the uploaded file")
        # Allow the same file to be re-selected
        self.load_wall_file.value = None

    def _on_load_wall_path(self, event):
        """Load a wall from a JSON file on the server."""
        self._defer_to_document(self._apply_load_wall_path)

    def _apply_load_wall_path(self):
        from forge.io import read_wall

        path = (self.load_wall_path.value or "").strip()
        if not path:
            self.status.object = "Please enter a wall JSON file path."
            self.status.alert_type = "warning"
            return
        if not os.path.isfile(path):
            self.status.object = f"File not found: {path}"
            self.status.alert_type = "danger"
            return

        try:
            wall_R, wall_Z = read_wall(path)
        except Exception as exc:
            self.status.object = f"Failed to load wall JSON: {exc}"
            self.status.alert_type = "danger"
            return

        self._apply_loaded_wall(wall_R, wall_Z, os.path.abspath(path))

    def _save_wall_callback(self):
        """Generate a wall JSON BytesIO for the FileDownload widget.

        This must return synchronously, so status updates are deferred rather
        than applied inline.
        """
        R, Z = self._get_wall()
        if not R:
            self._set_status("No wall to save.", "warning")
            return io.BytesIO(b"")

        closed_R, closed_Z = _closed_wall(R, Z)
        content = fancy_json_string({"R": closed_R, "Z": closed_Z})
        self._set_status("Wall JSON ready for download.", "success")
        return io.BytesIO(content.encode("utf-8"))

    def _on_save_wall_path(self, event):
        """Write the wall to a JSON file on the server."""
        self._defer_to_document(self._apply_save_wall_path)

    def _apply_save_wall_path(self):
        path = (self.save_wall_path.value or "").strip()
        if not path:
            self.status.object = "Please enter an output file path."
            self.status.alert_type = "warning"
            return

        R, Z = self._get_wall()
        if not R:
            self.status.object = "No wall to save."
            self.status.alert_type = "warning"
            return

        closed_R, closed_Z = _closed_wall(R, Z)
        try:
            write_wall(closed_R, closed_Z, path)
        except Exception as exc:
            self.status.object = f"Failed to save wall JSON: {exc}"
            self.status.alert_type = "danger"
            return

        self.status.object = f"Wall saved to {os.path.abspath(path)}"
        self.status.alert_type = "success"

    def _save_geqdsk_callback(self):
        """Generate a GEQDSK BytesIO carrying the edited wall.

        This must return synchronously, so status updates are deferred rather
        than applied inline.
        """
        from freeqdsk import geqdsk

        eq = self.state.get("eq")
        if eq is None:
            self._set_status(
                "No equilibrium loaded. Go to the Setup tab first.", "warning"
            )
            return io.BytesIO(b"")

        R, Z = self._get_wall()
        if not R:
            self._set_status("No wall to save.", "warning")
            return io.BytesIO(b"")

        closed_R, closed_Z = _closed_wall(R, Z)
        try:
            data = geqdsk_dict(eq, wall_R=closed_R, wall_Z=closed_Z)
            sbuf = io.StringIO()
            geqdsk.write(data, sbuf, "FORGE", 0, 0)
        except Exception as exc:
            logger.exception("Failed to build GEQDSK")
            self._set_status(f"Failed to build GEQDSK: {exc}", "danger")
            return io.BytesIO(b"")

        self._set_status("GEQDSK ready for download.", "success")
        return io.BytesIO(sbuf.getvalue().encode("utf-8"))

    def _on_save_geqdsk_path(self, event):
        """Write a GEQDSK carrying the edited wall to a path on the server."""
        self._defer_to_document(self._apply_save_geqdsk_path)

    def _apply_save_geqdsk_path(self):
        eq = self.state.get("eq")
        if eq is None:
            self.status.object = "No equilibrium loaded. Go to the Setup tab first."
            self.status.alert_type = "warning"
            return

        path = (self.save_geqdsk_path.value or "").strip()
        if not path:
            self.status.object = "Please enter an output file path."
            self.status.alert_type = "warning"
            return

        R, Z = self._get_wall()
        if not R:
            self.status.object = "No wall to save."
            self.status.alert_type = "warning"
            return

        closed_R, closed_Z = _closed_wall(R, Z)
        try:
            write_geqdsk(eq, path, wall_R=closed_R, wall_Z=closed_Z)
        except Exception as exc:
            logger.exception("Failed to write GEQDSK")
            self.status.object = f"Failed to save GEQDSK: {exc}"
            self.status.alert_type = "danger"
            return

        self.status.object = f"GEQDSK saved to {os.path.abspath(path)}"
        self.status.alert_type = "success"

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def _on_finish(self, event):
        """Apply the edited wall to the machine, equilibrium and GEQDSK data."""
        self._defer_to_document(self._apply_finish)

    def _apply_finish(self):
        eq = self.state.get("eq")
        tokamak = self.state.get("tokamak")
        if eq is None or tokamak is None:
            self.status.object = "No machine loaded yet. Go to the Setup tab first."
            self.status.alert_type = "warning"
            return

        R, Z = self._get_wall()
        error = self._validate(R, Z)
        if error is not None:
            self.status.object = f"Cannot apply this wall: {error}"
            self.status.alert_type = "danger"
            return

        closed_R, closed_Z = _closed_wall(R, Z)

        # Machine owns the wall and caches its bounds and Shapely form;
        # Equilibrium holds references to the same lists.
        tokamak.set_wall(closed_R, closed_Z)
        eq.wall_R = tokamak.wall_R
        eq.wall_Z = tokamak.wall_Z

        eq_data = self.state.get("eq_data")
        if eq_data is not None:
            eq_data["wall_R"] = tokamak.wall_R
            eq_data["wall_Z"] = tokamak.wall_Z

        cleared = False
        if self._geometry_tab is not None:
            # Redraw against the new wall first, so that the clearing warning
            # is the message left on the Geometry tab.
            self._geometry_tab._on_refresh(None)
            cleared = self._geometry_tab.clear_wall_dependent_geometry()

        if self._setup_tab is not None:
            try:
                self._setup_tab._update_eq_plot(eq, tokamak)
            except Exception:
                logger.exception("Failed to refresh the Setup tab plot")

        message = (
            f"Machine wall updated ({len(closed_R)} points, including the "
            "repeated closing point)."
        )
        if cleared:
            message += (
                " Strike geometry and buffers on the Geometry tab were cleared "
                "because they were tied to the previous wall."
            )
        self.status.object = message
        self.status.alert_type = "success"

    # ------------------------------------------------------------------
    @property
    def panel(self):
        """Return the Panel layout for this tab."""
        if hasattr(self, "_panel"):
            return self._panel

        sidebar = pn.Column(
            "### Wall Editor",
            pn.pane.HTML(
                "<small>Edit the machine wall here. All changes are made to a "
                "<b>temporary copy</b> &mdash; the machine, equilibrium and "
                "every other tab are untouched until you press "
                "<b>Finish</b> at the bottom.</small>"
            ),
            self.reset_btn,
            self.show_masks,
            pn.layout.Divider(),
            "**Edit on the canvas**",
            pn.pane.HTML(
                "<small>Select the <em>Edit wall vertices</em> tool "
                f"({tool_icon(ICON_POLY_EDIT)}) "
                "in the Bokeh toolbar, then <b>click and hold</b> inside the "
                "wall outline &mdash; red vertices will appear.<br>"
                "&bull; <b>Drag</b> a vertex to move it (start moving "
                "immediately after pressing down).<br>"
                "&bull; <b>Click and hold on a vertex</b> to add a new "
                "vertex &mdash; then <b>click</b> to place it.<br>"
                "&bull; Press <b>Backspace</b> to delete a selected "
                "vertex.<br>"
                "&bull; Press <b>Esc</b> to cancel / exit insert mode."
                "</small>"
            ),
            pn.layout.Divider(),
            "**Wall points**",
            pn.pane.HTML(
                "<small>Edit R/Z directly in the table. Select a row to "
                "highlight that point on the canvas (cyan ring) and edit it "
                "below. The closing point is handled automatically and is not "
                "listed.</small>"
            ),
            self._table,
            "**Selected point**",
            pn.pane.HTML(
                "<small><em>Insert point after</em> adds a point midway "
                "between the selected point and the next one. With no row "
                "selected it adds one after the last point, on the segment "
                "that closes the wall.</small>"
            ),
            pn.Row(self.sel_r_input, self.sel_z_input),
            pn.Row(self.insert_btn, self.delete_btn),
            pn.layout.Divider(),
            "**Load Wall**",
            pn.pane.HTML(
                "<small>A wall JSON file is an object with <code>R</code> and "
                "<code>Z</code> lists of coordinates in metres.</small>"
            ),
            self.load_wall_file,
            self.load_wall_path,
            self.load_wall_path_btn,
            pn.layout.Divider(),
            "**Save Wall**",
            self.save_wall_download,
            self.save_wall_path,
            self.save_wall_path_btn,
            pn.layout.Divider(),
            "**Save GEQDSK with this wall**",
            pn.pane.HTML(
                "<small>Writes the loaded equilibrium out with the wall as "
                "edited here, without needing to press Finish.</small>"
            ),
            self.save_geqdsk_download,
            self.save_geqdsk_path,
            self.save_geqdsk_path_btn,
            pn.layout.Divider(),
            "**Apply to Machine**",
            pn.pane.HTML(
                "<small>Applies the edited wall to the loaded machine and "
                "equilibrium. Strike geometry and buffers on the Geometry tab "
                "are cleared, since they are defined against the wall.</small>"
            ),
            self.finish_btn,
            pn.layout.Divider(),
            self.status,
            width=440,
            height=900,
            scroll=True,
        )
        fig_pane = pn.pane.Bokeh(self.fig, sizing_mode="stretch_height")
        self._panel = pn.Row(sidebar, fig_pane, sizing_mode="stretch_both")
        return self._panel
