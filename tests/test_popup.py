import base64
import io
import urllib.error
import urllib.request

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from groundwater.viz import popup, static_map


def _jpeg_uri(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_preview_draws_the_map_trace_types():
    img = np.zeros((20, 40, 3), np.uint8)
    fig = go.Figure([
        go.Image(source=_jpeg_uri(img), x0=0, dx=0.5, y0=0, dy=0.5),
        go.Heatmap(z=np.ones((5, 5)), x0=0, dx=1, y0=0, dy=1, colorscale=[[0, "rgba(180,150,110,0.2)"], [1, "rgba(180,150,110,0.2)"]]),
        go.Scattergl(x=np.array([0, 5, np.nan, 6, 9]), y=np.array([0, 5, np.nan, 1, 2]), mode="lines", name="lines",
                     line=dict(color="lime", width=2, dash="dot")),
        go.Scatter(x=[1, 2, 3], y=[1, 2, 1], fill="toself", fillcolor="rgba(0,0,255,0.2)", name="fill"),
        go.Scatter(x=[1, 2], y=[3, 4], mode="markers", marker=dict(size=9, color=["red", "#00f"], line=dict(width=1)),
                   name="pts"),
    ])
    fig.update_yaxes(scaleanchor="x")
    out = static_map.preview_image(fig, width_px=400)
    assert out[:3] == b"\xff\xd8\xff"                      # a photo map comes back as JPEG
    fig2 = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    assert static_map.preview_image(fig2, width_px=300)[:4] == b"\x89PNG"


def test_view_server_serves_its_own_pages_only(tmp_path):
    srv = popup.ViewServer(tmp_path)
    url = srv.publish("page", "<html><body>ok</body></html>")
    with urllib.request.urlopen(url, timeout=5) as r:
        assert r.read() == b"<html><body>ok</body></html>"
    (tmp_path.parent / "secret.html").write_text("no", encoding="utf-8")
    for bad in ("../secret.html", "%2e%2e/secret.html", "page.txt"):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/{bad}", timeout=5)
            raise AssertionError(bad)
        except urllib.error.HTTPError as e:
            assert e.code == 404
    srv.httpd.shutdown()


def test_section_reference_lines_are_labelled_and_drawn_in_the_preview():
    from groundwater.viz import setmap3d

    fig = go.Figure(go.Scatter(x=[0, 10], y=[0, 5], mode="lines", name="joint"))
    setmap3d._ref_line(fig, [0, 10], [2, 2], "A", "표고 2.00 m", where="start", anchor=("left", "bottom"))
    setmap3d._ref_line(fig, [4, 4], [5, 0], "C", "측점 4.00 m", where="start", anchor=("left", "top"))
    assert [a.text for a in fig.layout.annotations] == ["<b>A</b> 표고 2.00 m", "<b>C</b> 측점 4.00 m"]
    assert all(t.showlegend is False for t in fig.data[1:])          # the reference lines stay out of the legend
    fig.update_yaxes(scaleanchor="x")
    assert static_map.preview_image(fig, width_px=400, height_px=300, show_title=False)[:4] == b"\x89PNG"
