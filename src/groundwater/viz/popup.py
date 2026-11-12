"""Interactive figures in their own browser window.

A large interactive map or 3D view inside the dashboard page competes with the page for the mouse (the wheel zooms
the figure instead of scrolling the page) and keeps the page busy redrawing, so scrolling past it stalls. The
dashboard shows a still preview instead and opens the interactive figure in a separate popup window, where it has
the whole window and the plain wheel zooms.

The pages are served by a small HTTP server inside the dashboard process (127.0.0.1, a free port), from one
folder in the temp directory: a page opened with window.open() must come from http, not file://.
"""
import functools
import hashlib
import html
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(tempfile.gettempdir()) / "groundwater_views"
PAGE_CONFIG = {"scrollZoom": True, "displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        if self.path.split("?")[0].endswith(".html"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        name = self.path.split("?")[0].lstrip("/")
        # only this machine's own pages, and only the flat .html / .js files of the folder
        if host not in ("127.0.0.1", "localhost", "::1") or "/" in name or "\\" in name or \
                not name.endswith((".html", ".js")):
            return self.send_error(404)
        return super().do_GET()


class ViewServer:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(self.root)))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="groundwater-views").start()

    def publish(self, name: str, page: str) -> str:
        """Write a page as <name>.html and return its URL (with a version, so the window shows the new one)."""
        data = page.encode("utf-8")
        v = hashlib.sha1(data).hexdigest()[:12]
        path = self.root / f"{name}.html"
        if not path.exists() or hashlib.sha1(path.read_bytes()).hexdigest()[:12] != v:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return f"http://127.0.0.1:{self.port}/{path.name}?v={v}"


def plotly_page(fig, title: str, root: Path = ROOT) -> str:
    """Stand-alone page of a 2D Plotly figure filling the window (plotly.js from a file next to it, no internet)."""
    import plotly
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    js = f"plotly-{plotly.__version__}.min.js"
    if not (Path(root) / js).exists():
        (Path(root) / js).write_text(get_plotlyjs(), encoding="utf-8")
    f = go.Figure(fig)
    f.update_layout(height=None, width=None, autosize=True, dragmode="pan")
    if f.layout.title is None or not f.layout.title.text:
        f.update_layout(title=title)
    page = pio.to_html(f, include_plotlyjs=js, full_html=True, config=PAGE_CONFIG,
                       default_width="100%", default_height="100vh")
    head = (f"<title>{html.escape(title)}</title><style>html,body{{margin:0;height:100%;overflow:hidden;"
            "font-family:'Malgun Gothic',sans-serif}</style>")
    return page.replace("<head>", "<head>" + head, 1)


# one click handler for every popup button of the page (Streamlit drops inline onclick attributes): window.open
# runs in the click itself, so no popup blocker stops it
_CLICK_JS = """<script>
if (!window.__gwPopup) {
  window.__gwPopup = true;
  document.addEventListener('click', (e) => {
    const b = e.target.closest && e.target.closest('button.gw-popup');
    if (!b) return;
    e.preventDefault();
    const w = window.open(b.dataset.url, b.dataset.window, `popup,width=${b.dataset.w},height=${b.dataset.h}`);
    if (w) w.focus();
  }, true);
}
</script>"""


def open_button_html(url: str, window: str, label: str, width: int = 1600, height: int = 1000) -> str:
    """A button that opens (or, clicked again, refreshes) the named popup window on `url`."""
    return (f'<button class="gw-popup" type="button" data-url="{html.escape(url)}" data-window="{html.escape(window)}" '
            f'data-w="{width}" data-h="{height}" style="display:inline-flex;align-items:center;gap:6px;'
            'padding:6px 14px;border:1px solid rgba(49,51,63,0.25);border-radius:8px;background:#fff;cursor:pointer;'
            f'font:14px Malgun Gothic,sans-serif;color:#1f2937">&#x2197; {html.escape(label)}</button>' + _CLICK_JS)
