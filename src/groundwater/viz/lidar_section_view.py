"""Figures of the LiDAR section tool (lidar_section.py): the plan image to draw the section line on, and the
section -- returns by kind, the rock layer the program keeps, and the analysis points by joint set."""
import numpy as np

from .seepage_view import _jpeg_uri

KIND_STYLE = {0: ("단일 반사", "#9ca3af"), 1: ("첫 반사 (망·잎 등)", "#2563eb"), 2: ("중간 반사", "#93c5fd"),
              3: ("마지막 반사", "#dc2626")}
ROCK, LEDGE, GENTLE, AIR = "#111827", "#7c3aed", "#f59e0b", "#16a34a"
MAX_PLOT = 150_000


def plan_image(index, max_px: int = 1600):
    """(RGB top row first, extent [x0, x1, y0, y1]) of the scan's plan image, cropped to the data, at most
    `max_px` wide; cells without a colour get a grey hillshade of the top height."""
    import cv2

    z, rgb = index["plan_z"], index["plan_rgb"]
    res = float(index["plan_res"])
    x0, ytop = (float(v) for v in index["plan_origin"])
    ok = np.isfinite(z)
    rows, cols = np.flatnonzero(ok.any(axis=1)), np.flatnonzero(ok.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    z, rgb, ok = z[r0:r1, c0:c1], rgb[r0:r1, c0:c1], ok[r0:r1, c0:c1]
    zf = np.where(ok, z, np.nanmedian(z[ok]) if ok.any() else 0).astype(np.float32)
    gy, gx = np.gradient(cv2.GaussianBlur(zf, (0, 0), 1.0), res)
    shade = np.clip(0.55 + 0.45 * (-gx - gy) / np.sqrt(gx ** 2 + gy ** 2 + 1.0), 0, 1)
    img = np.where((rgb.sum(axis=2) > 0)[..., None], rgb, (shade[..., None] * 255).astype(np.uint8))
    img[~ok] = 255
    k = max(1.0, img.shape[1] / max_px)
    if k > 1:
        img = cv2.resize(img, (round(img.shape[1] / k), round(img.shape[0] / k)), interpolation=cv2.INTER_AREA)
    ext = [x0 + c0 * res, x0 + c1 * res, ytop - r1 * res, ytop - r0 * res]
    return img, ext


ORTHO_SKIP = ("dsm", "dtm", "dem", "thermal", "열화상", "temperature", "planar")


def find_orthos(site) -> list:
    """Georeferenced colour GeoTIFFs of the site folder (orthophotos, projected CRS; DSM / DTM / thermal / wall-
    parallel exports skipped): the newest by the date in the name first (YYYYMMDD), then the finest."""
    import re
    from pathlib import Path

    import rasterio

    out = []
    for f in Path(site).glob("*.tif"):
        if any(w in f.name.lower() for w in ORTHO_SKIP):
            continue
        try:
            with rasterio.open(f) as d:
                if d.count < 3 or d.crs is None or not d.crs.is_projected:
                    continue
                res = abs(d.transform.a)
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"(20\d{6})", f.stem)
        out.append((m.group(1) if m else "", -res, f))
    return [f for *_, f in sorted(out, reverse=True)]


def ortho_image(path, ext, max_px: int = 2000):
    """(RGB top row first, extent) of an orthophoto over `ext` [x0, x1, y0, y1], at most `max_px` wide (from its
    overviews); outside the photo white."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    x0, x1, y0, y1 = ext
    w = int(max_px)
    h = max(1, int(round(w * (y1 - y0) / (x1 - x0))))
    px = (x1 - x0) / w
    a = np.full((3, h, w), 255, np.uint8)
    with rasterio.open(path) as d:
        b = d.bounds
        cx0, cx1, cy0, cy1 = max(x0, b.left), min(x1, b.right), max(y0, b.bottom), min(y1, b.top)
        if cx1 > cx0 and cy1 > cy0:            # the part inside the photo, read from its overviews
            c0, c1 = int((cx0 - x0) / px), min(w, int(np.ceil((cx1 - x0) / px)))
            r0, r1 = int((y1 - cy1) / px), min(h, int(np.ceil((y1 - cy0) / px)))
            win = from_bounds(x0 + c0 * px, y1 - r1 * px, x0 + c1 * px, y1 - r0 * px, d.transform)
            part = d.read([1, 2, 3], window=win, out_shape=(3, r1 - r0, c1 - c0), resampling=Resampling.average)
            if d.count >= 4:                   # transparent (no data) -> white
                al = d.read(4, window=win, out_shape=(r1 - r0, c1 - c0), resampling=Resampling.nearest)
                part[:, al == 0] = 255
            a[:, r0:r1, c0:c1] = part
    return np.ascontiguousarray(np.moveaxis(a, 0, -1)), list(ext)


def plan_figure(img, ext, line=None, sections=(), height: int = 640):
    """Plan image with the section polygons and the current line A -> B; lasso-select draws a new line (an
    invisible grid of points lets Plotly report the selection). Opens on the section polygons (+20 m) when there
    are any -- a whole-site scan reaches far beyond the walls -- the rest is a pan away."""
    import plotly.graph_objects as go

    H, W = img.shape[:2]
    dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
    fig = go.Figure(go.Image(source=_jpeg_uri(img), x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy, hoverinfo="skip"))
    gx, gy = np.meshgrid(np.arange(ext[0], ext[1], 5.0), np.arange(ext[2], ext[3], 5.0))
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=2, opacity=0),
                             hoverinfo="skip", showlegend=False, name="grid"))
    for name, poly, colour in sections:
        p = np.vstack([poly, poly[:1]])
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=colour, width=2), name=name,
                                 hoverinfo="skip"))
    if line is not None:
        a, b = line
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines+markers+text", text=["A", "B"],
                                 textposition="top center", textfont=dict(size=16, color="#facc15"),
                                 line=dict(color="#facc15", width=4), marker=dict(size=10, color="#facc15"),
                                 name="단면 선", hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), dragmode="lasso", showlegend=True,
                      legend=dict(orientation="h", y=1.0, x=0, bgcolor="rgba(255,255,255,0.8)"))
    view = list(ext)
    if len(sections):
        allp = np.vstack([np.asarray(poly) for _, poly, _ in sections])
        lo, hi = allp.min(axis=0) - 20, allp.max(axis=0) + 20
        view = [max(ext[0], lo[0]), min(ext[1], hi[0]), max(ext[2], lo[1]), min(ext[3], hi[1])]
        if view[1] <= view[0] or view[3] <= view[2]:
            view = list(ext)
    fig.update_xaxes(range=view[:2], showticklabels=False, constrain="domain")
    fig.update_yaxes(range=view[2:], showticklabels=False, scaleanchor="x", scaleratio=1)
    return fig


def section_figure(slab, kinds, cls, half: float, analysis=None, set_names=None, title: str = "", show_cls: bool = True):
    """The section: returns within `half` of the plane by kind; the rock layer (and the rear layer outside the face
    or above the ground, left out) when `show_cls`; analysis points `analysis` = (s, z, set id) coloured by set."""
    import plotly.graph_objects as go

    from ..report import SET_COLORS

    m = np.abs(slab["d"]) <= half
    s, z = slab["s"], slab["xyz"][:, 2] if len(slab["xyz"]) else np.zeros(0)
    rng = np.random.default_rng(0)
    fig = go.Figure()

    def add(sel, name, colour, size, symbol="circle", outline=None):
        idx = np.flatnonzero(sel)
        n = len(idx)
        if n > MAX_PLOT:
            idx = rng.choice(idx, MAX_PLOT, replace=False)
        fig.add_trace(go.Scattergl(x=s[idx], y=z[idx], mode="markers", name=f"{name} ({n:,})",
                                   marker=dict(size=size, color=colour, symbol=symbol,
                                               line=dict(width=0.6, color=outline) if outline else None)))

    for k, (name, colour) in KIND_STYLE.items():
        add(m & (kinds == k), name, colour, 3)
    if show_cls and cls is not None:
        ledge = cls.get("ledge", np.zeros_like(cls["rock"]))
        add(m & cls["rock"] & ~ledge, "뒤쪽 층 = 암반으로 남김", ROCK, 3.5)
        add(m & ledge, "뒤쪽 층 · 벽면 안 완경사 (소단·사면) = 암반으로 남김", LEDGE, 3.5)
        add(m & cls["outside"], "뒤쪽 층이지만 벽면 범위 밖 (원지반·나무 줄기·바닥) → 제외", GENTLE, 3.5)
        if "air" in cls:
            add(m & cls["air"], "뒤쪽 층이지만 지면 위 (수목) → 제외", AIR, 3.5)
    if analysis is not None and len(analysis[0]):
        a_s, a_z, a_set = analysis
        for sid in np.unique(a_set[a_set >= 0]):                 # the points of a joint set (the rest is not shown)
            sel = a_set == sid
            name = f"분석 점: {(set_names or {}).get(int(sid), f'Set {sid}')}"
            colour = SET_COLORS[int(sid) % len(SET_COLORS)]
            fig.add_trace(go.Scattergl(x=a_s[sel], y=a_z[sel], mode="markers", name=f"{name} ({int(sel.sum()):,})",
                                       marker=dict(size=7, color=colour, symbol="diamond", line=dict(width=0.8, color="#111827"))))
    L = slab["length"]
    if m.any():
        zr = [float(np.percentile(z[m], 0.2)) - 1, float(np.percentile(z[m], 99.8)) + 1]
        for x, t in ((0.0, "A"), (L, "B")):
            fig.add_annotation(x=x, y=zr[1], text=t, showarrow=False, font=dict(size=16, color="#111827"), yanchor="bottom")
    fig.update_layout(title=title, margin=dict(l=60, r=10, t=40, b=50), legend=dict(itemsizing="constant"))
    fig.update_xaxes(title="A → B 거리 (m)")
    fig.update_yaxes(title="표고 (m)", scaleanchor="x", scaleratio=1)
    return fig


def slice_stats(slab, kinds, cls, half: float) -> dict:
    """Counts in the slice: points per return kind, rock layer, rear layer left out; the share of multi-return pulses."""
    m = np.abs(slab["d"]) <= half
    out = {KIND_STYLE[k][0]: int((m & (kinds == k)).sum()) for k in KIND_STYLE}
    out["전체"] = int(m.sum())
    if cls is not None:
        out["암반으로 남김"] = int((m & cls["rock"]).sum())
        out["벽면 범위 밖 제외"] = int((m & cls["outside"]).sum())
        out["벽면 안 완경사 (암반)"] = int((m & cls["ledge"]).sum()) if "ledge" in cls else 0
        out["지면 위 제외"] = int((m & cls["air"]).sum()) if "air" in cls else 0
    return out
