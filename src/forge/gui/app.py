"""FORGE GUI — main Panel application entry point.

Launch with one of::

    forge-gui                        # uses the entry point in pyproject.toml
    python -m forge.gui              # via __main__.py
    panel serve src/forge/gui/app.py --show   # raw Panel serve

Within a Jupyter notebook::

    from forge.gui.app import create_app
    create_app().servable()
"""

import io as _io
import os as _os
import logging
import warnings

import panel as pn
import tornado.web
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import MISSING_RENDERERS

# Suppress harmless Bokeh warnings that fire when empty-data figures are
# serialised, or when batched hold/unhold re-registers known models.
silence(MISSING_RENDERERS, True)
warnings.filterwarnings(
    "ignore", message="reference already known", category=UserWarning,
)


class _PatchDropFilter(logging.Filter):
    """Suppress the 'Dropping a patch because it contains a previously known
    reference' log message that Bokeh emits on benign websocket race conditions."""
    def filter(self, record):
        return "previously known reference" not in record.getMessage()


logging.getLogger().addFilter(_PatchDropFilter())

from forge.gui.analysis_tab import AnalysisTab
from forge.gui.geometry_tab import GeometryTab
from forge.gui.optimisation_tab import OptimisationTab
from forge.gui.setup_tab import SetupTab

logger = logging.getLogger(__name__)

# In-memory store for pending downloads.  Keys are filenames, values are
# ``bytes`` objects.  The download handler streams the bytes to the browser
# with a Content-Length header (so the browser shows a progress bar) and
# then removes the entry to free memory.
_DOWNLOAD_STORE: dict[str, bytes] = {}

# Upload store: completed uploads are stored here as raw bytes so the
# analysis tab can unpickle them.  Keys are upload IDs, values are bytes.
_UPLOAD_STORE: dict[str, bytes] = {}

# Panel extensions
pn.extension(sizing_mode="stretch_width")


class _DownloadHandler(tornado.web.RequestHandler):
    """Serve an in-memory blob with Content-Length so the browser shows
    real download progress, then free the memory afterwards."""

    CHUNK_SIZE = 1024 * 1024  # 1 MB

    async def get(self, filename):
        data = _DOWNLOAD_STORE.get(filename)
        if data is None:
            raise tornado.web.HTTPError(404)
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.set_header("Content-Length", str(len(data)))
        reader = _io.BytesIO(data)
        while True:
            chunk = reader.read(self.CHUNK_SIZE)
            if not chunk:
                break
            self.write(chunk)
            await self.flush()
        self.finish()
        # Free the memory
        _DOWNLOAD_STORE.pop(filename, None)


@tornado.web.stream_request_body
class _UploadHandler(tornado.web.RequestHandler):
    """Accept a streamed file upload into an in-memory buffer.

    The browser sends a multipart/form-data POST.  Tornado streams the raw
    body in chunks via ``data_received``.  We accumulate into a BytesIO
    (no disk usage) and store the bytes in ``_UPLOAD_STORE`` when finished.
    """

    def prepare(self):
        # Allow uploads up to 10 GB (with @stream_request_body the body is
        # not buffered, but Tornado still checks Content-Length against this).
        self.request.connection.set_max_body_size(10 * 1024**3)
        self._buffer = _io.BytesIO()
        # Progress tracking
        self._total = int(self.request.headers.get("Content-Length", 0))
        self._received = 0
        # Extract the multipart boundary from the content-type header.
        content_type = self.request.headers.get("Content-Type", "")
        if "boundary=" in content_type:
            self._boundary = ("--" + content_type.split("boundary=")[1].split(";")[0].strip()).encode()
        else:
            self._boundary = None
        self._header_stripped = False
        self._buf = b""

    def data_received(self, chunk):
        self._received += len(chunk)
        if self._total:
            pct = self._received / self._total * 100
            print(f"\rUploading pickle: {self._received / 1e9:.2f} / {self._total / 1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
        # Multipart framing: strip the leading boundary + headers on first chunk,
        # and handle the trailing boundary at the end.
        if not self._header_stripped:
            self._buf += chunk
            # Find end of MIME headers (double CRLF)
            header_end = self._buf.find(b"\r\n\r\n")
            if header_end == -1:
                return  # need more data to find header end
            # Everything after the double-CRLF is file content
            self._buf = self._buf[header_end + 4:]
            self._header_stripped = True
            if self._buf:
                self._buffer.write(self._buf)
            self._buf = b""
        else:
            self._buffer.write(chunk)

    def post(self):
        print(f"\nUpload complete ({self._received / 1e9:.2f} GB). Loading...", flush=True)
        # Trim trailing boundary from the accumulated bytes.
        data = self._buffer.getvalue()
        self._buffer.close()
        if self._boundary and data:
            # The trailing boundary is: \r\n--boundary--\r\n
            trailer = b"\r\n" + self._boundary + b"--"
            idx = data.rfind(trailer)
            if idx != -1:
                data = data[:idx]
            elif data.endswith(self._boundary + b"--\r\n"):
                data = data[:-(len(self._boundary) + 4)]
        import uuid
        upload_id = uuid.uuid4().hex
        _UPLOAD_STORE[upload_id] = data
        self.set_header("Content-Type", "application/json")
        self.write({"upload_id": upload_id})
        self.finish()



def create_app():
    """Build and return the full FORGE GUI application as a Panel ``Tabs`` object.

    Returns
    -------
    pn.Tabs
        The top-level Panel layout ready to be served or embedded in a notebook.
    """
    # Shared mutable state dictionary passed between all tabs
    shared_state = {}

    setup = SetupTab(shared_state)
    geometry = GeometryTab(shared_state)
    setup._geometry_tab = geometry
    optimisation = OptimisationTab(shared_state, setup_tab=setup, geometry_tab=geometry)
    analysis = AnalysisTab(shared_state)

    tabs = pn.Tabs(
        ("Setup", setup.panel),
        ("Geometry", geometry.panel),
        ("Optimise", optimisation.panel),
        ("Analysis", analysis.panel),
        stylesheets=[
            # Stop Bokeh from sizing the tab container to the tallest tab.
            # Each tab panel only takes as much height as its own content.
            ":host .bk-headers + div { height: auto !important; }"
        ],
    )

    # When the user switches to the Geometry tab, rebuild the sidebar widget
    # lists.  Buttons created before the tab is rendered (e.g. during config
    # load) don't have live event wiring; rebuilding with a live document
    # ensures on_click callbacks fire correctly.
    def _on_tab_change(event):
        if event.new == 1:  # Geometry tab
            geometry._rebuild_strike_point_list()
            geometry._rebuild_xpt_region_list()

    tabs.param.watch(_on_tab_change, "active")

    header = pn.pane.HTML(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; width:100%; flex-wrap:wrap; gap:8px;">
            <div>
                <h2 style="margin:0 0 4px;">FORGE</h2>
                <p style="margin:0 0 6px; color:gray; font-size:0.85em;">
                    <b style="color:red;">F</b>ORGE
                    <b style="color:red;">O</b>ptimises
                    <b style="color:red;">R</b>eactor
                    <b style="color:red;">G</b>eometries to improve
                    <b style="color:red;">E</b>xhaust
                </p>
            </div>
            <div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
            <div style="display:flex; align-items:center; gap:16px;">
                <!-- GitHub -->
                <a href="https://github.com/placeholder/forge" target="_blank" title="Source Code"
                   style="color:inherit; text-decoration:none;">
                    <svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15
                        -.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87
                        .51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12
                        0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82
                        2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65
                        3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01
                        0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
                <!-- ReadTheDocs -->
                <a href="https://forge.readthedocs.io" target="_blank" title="Documentation"
                   style="color:inherit; text-decoration:none;">
                    <svg height="24" width="24" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M7 2a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-6-6H7zm0
                        2h6v5h5v11H7V4zm2 7v2h6v-2H9zm0 4v2h6v-2H9z"/>
                    </svg>
                </a>
                <!-- Discord -->
                <a href="https://discord.gg/placeholder" target="_blank" title="Discord"
                   style="color:inherit; text-decoration:none;">
                    <svg height="24" width="24" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M20.317 4.37a19.79 19.79 0 0 0-4.885-1.515.074.074 0 0
                        0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487
                        0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.74 19.74
                        0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099
                        18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078
                        0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0
                        0-.041-.106 13.11 13.11 0 0 1-1.872-.892.077.077 0 0
                        1-.008-.128c.126-.094.252-.192.372-.291a.074.074 0 0
                        1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0
                        1 .078.01c.12.098.246.198.373.292a.077.077 0 0
                        1-.006.127 12.3 12.3 0 0 1-1.873.892.077.077 0 0
                        0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0
                        0 .084.028 19.84 19.84 0 0 0 6.002-3.03.077.077 0 0
                        0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.06.06 0 0
                        0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419
                        0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157
                        2.42 0 1.333-.956 2.418-2.157 2.418zm7.975
                        0c-1.183 0-2.157-1.085-2.157-2.419
                        0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157
                        2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                    </svg>
                </a>
            </div>
            <div style="font-size:0.7em; color:gray; text-align:center;">
                GUI built with
                <a href="https://panel.holoviz.org" target="_blank" style="color:#1f77b4;">Panel</a>
                &amp;
                <a href="https://bokeh.org" target="_blank" style="color:#1f77b4;">Bokeh</a>
            </div>
            </div>
        </div>
        """,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        header,
        tabs,
        sizing_mode="stretch_width",
    )


def main():
    """Entry point called by the ``forge-gui`` console script."""
    # Configure forge logging so INFO messages appear
    logging.getLogger("forge").setLevel(logging.INFO)

    # Default to localhost-only binding.  Override with the environment
    # variable FORGE_GUI_ADDRESS if access from other machines is needed
    # (e.g. FORGE_GUI_ADDRESS=0.0.0.0 forge-gui).
    address = _os.environ.get("FORGE_GUI_ADDRESS", "localhost")

    # Build the websocket-origin list to match the bound address/port.
    port = int(_os.environ.get("FORGE_GUI_PORT", "5006"))
    ws_origin = [f"localhost:{port}", f"127.0.0.1:{port}"]
    if address not in ("localhost", "127.0.0.1", ""):
        ws_origin.append(f"{address}:{port}")

    print(f"FORGE GUI running at: http://{address}:{port}/")

    # Pass the factory function (not a pre-built instance) so that each
    # browser session gets its own fresh set of Panel/Bokeh models.
    pn.serve(
        {"/": create_app},
        port=port,
        address=address,
        show=False,
        verbose=False,
        title="FORGE GUI",
        websocket_origin=ws_origin,
        extra_patterns=[
            (r"/download/(.*)", _DownloadHandler),
            (r"/upload", _UploadHandler),
        ],
    )


# Allow `panel serve app.py`
if __name__.startswith("bokeh") or __name__ == "__main__":
    create_app().servable()
