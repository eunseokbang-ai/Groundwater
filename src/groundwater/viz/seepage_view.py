"""The seepage-potential map (hydro.seepage_potential) for the dashboard: the grade (상·중), the score or one of
the four indicators over a facade background -- the point-cloud facade image or the wall-parallel orthomosaic
itself (facade.planar.wall_image) -- as a zoomable figure, and the same layers draped on the tiled model.
"""
import base64

import numpy as np

# name -> (label, unit, matplotlib colormap, value shown = f(stored layer))
LAYERS = {
    "grade": ("유출 가능성 등급 (상·중)", "", None, None),
    "score": ("종합 점수", "0~1", "YlOrRd", lambda a: a),
    "p21": ("절리 밀도 P21", "m/m²", "OrRd", lambda a: a),
    "openness": ("열린 정도 (홈 평균 깊이)", "cm", "PuBu", lambda a: a * 100.0),
    "connectivity": ("연결성 (이어진 트레이스망 길이)", "m", "Purples", lambda a: 10 ** a - 1.0),
    "wetness": ("절리 옆 식생 비율", "%", "Greens", lambda a: a * 100.0),
}
GRADE_COLOURS = {2: "#dc2626", 1: "#f59e0b"}


def layer_values(smap, name: str) -> np.ndarray:
    """(rows, cols) values of a layer in display units, NaN off the rock cells."""
    if name == "grade":
        return np.where(smap.classes >= 0, smap.classes, np.nan).astype(float)
    a = LAYERS[name][3](np.asarray(smap.layers[name], float))
    return np.where(smap.rock, a, np.nan)


def _alpha(t):
    """Opacity along the colour scale: the low values nearly see-through, so the photo stays readable and only
    the high values stand out."""
    return np.clip(0.12 + 1.76 * np.asarray(t, float), 0.12, 1.0)


def _colorscale(cmap: str, n: int = 9) -> list:
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (int(255 * c) for c in cm(0.12 + 0.88 * t)[:3])
        out.append([t, f"rgba({r},{g},{b},{float(_alpha(t)):.2f})"])
    return out


def _range(vals: np.ndarray):
    v = vals[np.isfinite(vals)]
    if not len(v):
        return 0.0, 1.0
    lo, hi = np.percentile(v, [2, 98])
    return float(lo), float(hi if hi > lo else lo + 1e-6)


def _jpeg_uri(rgb_top_first: np.ndarray) -> str:
    import cv2

    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb_top_first[::-1]), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def map_figure(smap, layer: str, background, ext, opacity: float = 0.6, zones: int = 10, title: str = ""):
    """Plotly figure: `background` (RGB, top row first) over chainage `ext` = [u0, u1, v0, v1], the layer over it
    (grade: '상' / '중' cells; otherwise the values with a colour bar), the `zones` best '상' zones boxed and
    numbered, and every rock cell's values on hover."""
    import plotly.graph_objects as go

    H, W = background.shape[:2]
    dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
    fig = go.Figure(go.Image(source=_jpeg_uri(background), x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy,
                             hoverinfo="skip"))
    rows, cols = smap.classes.shape
    c = smap.cell_m
    x0, y0 = smap.u_min + c / 2, smap.v_max - rows * c + c / 2
    if layer == "grade":
        for k, name in ((1, "중 (다음 35%)"), (2, "상 (상위 15%)")):
            z = np.where(smap.classes == k, 1.0, np.nan)[::-1]
            fig.add_trace(go.Heatmap(z=z, x0=x0, dx=c, y0=y0, dy=c, colorscale=[[0, GRADE_COLOURS[k]], [1, GRADE_COLOURS[k]]],
                                     showscale=False, opacity=opacity, hoverinfo="skip", name=name))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                     marker=dict(symbol="square", size=13, color=GRADE_COLOURS[k])))
    else:
        label, unit, cmap, _ = LAYERS[layer]
        vals = layer_values(smap, layer)
        lo, hi = _range(vals)
        fig.add_trace(go.Heatmap(z=vals[::-1], x0=x0, dx=c, y0=y0, dy=c, colorscale=_colorscale(cmap), zmin=lo, zmax=hi,
                                 opacity=opacity, hoverinfo="skip", showscale=True, name=label,
                                 colorbar=dict(title=dict(text=f"{label}" + (f" ({unit})" if unit else "")), thickness=14)))
    # the best zones: boxes and numbers
    for z in smap.zones.head(zones).itertuples():
        a, b = (float(v) for v in z.chainage_m.split("–"))
        e, f = (float(v) for v in z.elevation_m.split("–"))
        fig.add_trace(go.Scatter(x=[a, b, b, a, a], y=[e, e, f, f, e], mode="lines", line=dict(color="white", width=2),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=a, y=f, text=f"<b>#{z.zone}</b>", showarrow=False, xanchor="left", yanchor="bottom",
                           bgcolor="#7f1d1d", font=dict(color="white", size=11), borderpad=2)
    # hover: every rock cell's values
    rr, cc = np.nonzero(smap.rock)
    if len(rr):
        txt = []
        vals = {k: layer_values(smap, k) for k in LAYERS if k != "grade"}
        grade = {2: "상", 1: "중", 0: "하"}
        for r, c_ in zip(rr, cc):
            txt.append(f"등급 {grade.get(int(smap.classes[r, c_]), '-')} · 점수 {vals['score'][r, c_]:.2f}<br>"
                       f"절리 밀도 {vals['p21'][r, c_]:.2f} m/m² · 열린 정도 {vals['openness'][r, c_]:.1f} cm<br>"
                       f"연결성 {vals['connectivity'][r, c_]:.0f} m · 절리 옆 식생 {vals['wetness'][r, c_]:.0f}%")
        fig.add_trace(go.Scatter(x=smap.u_min + (cc + 0.5) * c, y=smap.v_max - (rr + 0.5) * c, mode="markers",
                                 marker=dict(size=7, opacity=0), text=txt, hoverinfo="text", showlegend=False))
    fig.update_layout(title=title or f"유출 가능성 — {LAYERS[layer][0]}", legend=dict(orientation="h"))
    fig.update_xaxes(title="측점 (m)", range=[ext[0], ext[1]])
    fig.update_yaxes(title="표고 (m)", range=[ext[2], ext[3]])
    return fig


def _rgba(smap, layer: str, alpha: int = 175) -> np.ndarray:
    """RGBA image of a layer, one pixel per cell (top row = highest elevation), transparent off the rock cells."""
    import matplotlib

    rows, cols = smap.classes.shape
    out = np.zeros((rows, cols, 4), np.uint8)
    if layer == "grade":
        for k, col in GRADE_COLOURS.items():
            m = smap.classes == k
            h = col.lstrip("#")
            out[m, :3] = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
            out[m, 3] = alpha
        return out
    vals = layer_values(smap, layer)
    lo, hi = _range(vals)
    ok = np.isfinite(vals)
    t = np.clip((vals - lo) / (hi - lo), 0, 1)
    rgb = matplotlib.colormaps[LAYERS[layer][2]](0.12 + 0.88 * np.nan_to_num(t))[..., :3]
    out[ok, :3] = (rgb[ok] * 255).astype(np.uint8)
    out[ok, 3] = (alpha * _alpha(t[ok])).astype(np.uint8)
    return out


def tiles_overlays(result, smap, first: str = "grade") -> list:
    """The grade and every indicator draped on the tiled model (tiles3d.scene_json overlays), `first` on top of
    the list (shown when the page opens); each with its colour key (low, middle, high for the indicators)."""
    import matplotlib

    from . import setmap3d

    out = []
    k = max(1, int(round(smap.cell_m / 0.05)))
    for name in [first] + [n for n in LAYERS if n != first]:
        label, unit, cmap, _ = LAYERS[name]
        rgba = np.repeat(np.repeat(_rgba(smap, name), k, axis=0), k, axis=1)
        if name == "grade":
            colors = [GRADE_COLOURS[2], GRADE_COLOURS[1]]
        else:
            cm = matplotlib.colormaps[cmap]
            colors = ["#%02x%02x%02x" % tuple(int(255 * x) for x in cm(t)[:3]) for t in (0.2, 0.55, 0.95)]
        ov = setmap3d._image_overlay(result, rgba, smap.u_min, smap.v_max, smap.cell_m / k, smap.cell_m / k,
                                     f"유출 가능성: {label}" + (f" ({unit}, 낮음→높음)" if unit else ""), "seepage",
                                     0.1, 0.14, colors=colors)
        if ov:
            out.append(ov)
    return out
