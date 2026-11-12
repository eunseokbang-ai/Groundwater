"""Still image (JPEG for photo maps, otherwise PNG) of a 2D Plotly map figure, drawn with matplotlib, for the dashboard page.

Covers what the dashboard's maps and sections use: Image (data-URI source or z), Heatmap, Scatter / Scattergl
lines (dash, NaN gaps), markers (size, colour, outline), fill="toself", trace opacity, axis titles and ranges,
equal axis scale (scaleanchor), data-anchored annotations (text badges), the title and a legend under the plot.
"""
import base64
import io
import re

import numpy as np

PLOTLY_COLORS = ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880",
                 "#FF97FF", "#FECB52"]
DASH = {"dot": ":", "dash": "--", "dashdot": "-.", "longdash": "--", "longdashdot": "-.", "solid": "-"}
PX = 0.75                       # matplotlib points per CSS pixel


def _colour(c, default):
    if c is None:
        return default
    if isinstance(c, str):
        m = re.match(r"\s*rgba?\(([^)]*)\)", c)
        if m:
            p = [float(x) for x in m.group(1).split(",")]
            return (p[0] / 255, p[1] / 255, p[2] / 255, p[3] if len(p) > 3 else 1.0)
        return c.strip()
    return c


def _typed(a):
    """Plotly's base64 typed-array form {dtype, bdata, shape} (how numpy data is stored) as a numpy array."""
    if isinstance(a, dict) and "bdata" in a:
        arr = np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
        shape = a.get("shape")
        if shape:
            arr = arr.reshape([int(x) for x in str(shape).split(",")] if isinstance(shape, str) else shape)
        return arr
    return a


def _rgba(c, alpha):
    """matplotlib RGBA of a colour with the trace opacity applied on top of the colour's own alpha (passing
    alpha= to matplotlib would replace it)."""
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(c)
    return (r, g, b, a * alpha)


def _num(a):
    a = _typed(a)
    if a is None:
        return np.zeros(0)
    return np.array([np.nan if v is None else v for v in a], dtype=float) if isinstance(a, (list, tuple)) \
        else np.asarray(a, dtype=float)


def _image(t):
    if t.get("source"):
        from PIL import Image

        raw = base64.b64decode(t["source"].split(",", 1)[1])
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
    z = np.asarray(_typed(t.get("z")))
    return z.astype(np.uint8) if z.dtype != np.uint8 else z


def _extent(t, h, w):
    x0, dx = float(t.get("x0", 0) or 0), float(t.get("dx", 1) or 1)
    y0, dy = float(t.get("y0", 0) or 0), float(t.get("dy", 1) or 1)
    return [x0 - dx / 2, x0 + (w - 0.5) * dx, y0 - dy / 2, y0 + (h - 0.5) * dy]


def _cmap(scale):
    from matplotlib.colors import LinearSegmentedColormap

    if not isinstance(scale, (list, tuple)) or not scale:
        return "viridis"
    stops = [(float(p), _colour(c, "#000")) for p, c in scale]
    if stops[0][0] > 0:
        stops.insert(0, (0.0, stops[0][1]))
    if stops[-1][0] < 1:
        stops.append((1.0, stops[-1][1]))
    return LinearSegmentedColormap.from_list("plotly", stops)


FONT = {"title": 11.5, "label": 10, "tick": 9, "legend": 9}   # pt; the same in every preview shown at its own width
INK, SOFT, LINE = "#1f2937", "#475569", "#cbd5e1"


def preview_image(fig, width_px: int = 1200, max_height_px: int = 1100, scale: float = 2.0, height_px: int | None = None,
                  show_title: bool = True, font_scale: float = 1.0, legend: bool = True) -> bytes:
    """JPEG / PNG bytes of the figure, `width_px` CSS pixels wide (rendered at `scale` for sharp text).

    Shown at `width_px` on the page, every preview has the same font size. With `height_px` the image has exactly
    that size (panels side by side line up): an equal-scale plot then fills the box by widening one axis range
    instead of shrinking the box. `font_scale` enlarges all text (slides)."""
    from groundwater.report import _setup_mpl

    import logging

    plt = _setup_mpl()
    # equal scale in a fixed box widens one axis range on purpose; matplotlib logs that it does each time
    logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)
    fs = {k: v * font_scale for k, v in FONT.items()}
    spec = fig.to_plotly_json()
    data, lay = spec.get("data", []), spec.get("layout", {})
    xa, ya = lay.get("xaxis", {}) or {}, lay.get("yaxis", {}) or {}
    equal = bool(ya.get("scaleanchor") or xa.get("scaleanchor"))

    # data extent: sets the figure shape of an equal-scale map
    xs, ys = [], []
    for t in data:
        if t.get("type") in ("image", "heatmap"):
            img = _image(t) if t["type"] == "image" else np.asarray(_typed(t.get("z")), dtype=float)
            e = _extent(t, img.shape[0], img.shape[1])
            xs += e[:2]
            ys += e[2:]
        elif t.get("type") in ("scatter", "scattergl"):
            x, y = _num(t.get("x")), _num(t.get("y"))
            if len(x) and np.isfinite(x).any():
                xs += [np.nanmin(x), np.nanmax(x)]
                ys += [np.nanmin(y), np.nanmax(y)]
    xr = xa.get("range") or ([min(xs), max(xs)] if xs else [0, 1])
    yr = ya.get("range") or ([min(ys), max(ys)] if ys else [0, 1])
    w_in = width_px / 96
    if height_px:
        h_in = height_px / 96
    elif equal and xr[1] != xr[0]:
        h_in = np.clip(w_in * abs(yr[1] - yr[0]) / abs(xr[1] - xr[0]) + 1.2, 3.0, max_height_px / 96)
    else:
        h_in = min(float(lay.get("height") or 620), max_height_px) / 96
    f, ax = plt.subplots(figsize=(w_in, h_in), dpi=96 * scale, layout="constrained")
    if lay.get("plot_bgcolor"):
        ax.set_facecolor(_colour(lay["plot_bgcolor"], "white"))

    handles = []
    for i, t in enumerate(data):
        kind = t.get("type", "scatter")
        alpha = float(t.get("opacity", 1.0) if t.get("opacity") is not None else 1.0)
        name = t.get("name")
        if kind == "image":
            img = _image(t)
            ax.imshow(img, origin="lower", extent=_extent(t, *img.shape[:2]), interpolation="antialiased",
                      alpha=alpha, zorder=0)
            continue
        if kind == "heatmap":
            z = np.asarray(_typed(t.get("z")), dtype=float)
            lo = t.get("zmin", np.nanmin(z) if np.isfinite(z).any() else 0)
            hi = t.get("zmax", np.nanmax(z) if np.isfinite(z).any() else 1)
            im = ax.imshow(np.ma.masked_invalid(z), origin="lower", extent=_extent(t, *z.shape),
                           cmap=_cmap(t.get("colorscale")), vmin=lo, vmax=hi if hi > lo else lo + 1, alpha=alpha,
                           interpolation="nearest", zorder=0)
            if t.get("showscale"):                       # Plotly's colour bar (only when asked for explicitly)
                cb = f.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
                ct = (t.get("colorbar") or {}).get("title") or {}
                cb.set_label(ct.get("text", "") if isinstance(ct, dict) else str(ct), fontsize=fs["tick"], color=INK)
                cb.ax.tick_params(labelsize=fs["tick"], colors=SOFT)
                cb.outline.set_edgecolor(LINE)
            continue
        if kind not in ("scatter", "scattergl"):
            continue
        x, y = _num(t.get("x")), _num(t.get("y"))
        mode = t.get("mode") or ("lines+markers" if len(x) < 20 else "lines")
        line, marker = t.get("line") or {}, t.get("marker") or {}
        default = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
        lc = _colour(line.get("color"), default)
        h = None
        if t.get("fill") == "toself" and len(x):
            ok = np.isfinite(x) & np.isfinite(y)
            h = ax.fill(x[ok], y[ok], color=_rgba(_colour(t.get("fillcolor"), lc), alpha), lw=0, zorder=1)[0]
        if "lines" in mode and len(x):
            lw = float(line.get("width", 2) or 0) * PX
            if lw > 0:
                h = ax.plot(x, y, color=_rgba(lc, alpha), lw=lw, ls=DASH.get(line.get("dash") or "solid", "-"),
                            zorder=2, solid_capstyle="round")[0]
        mop = marker.get("opacity")
        if "markers" in mode and len(x) and (mop is None or not np.isscalar(mop) or float(mop) > 0):
            alpha = alpha * (float(mop) if mop is not None and np.isscalar(mop) else 1.0)
            size = _typed(marker.get("size", 6))
            size = np.asarray(size, float) if isinstance(size, (list, tuple, np.ndarray)) else float(size or 6)
            mc = _typed(marker.get("color"))
            mc = ([_rgba(_colour(c, default), alpha) for c in mc] if isinstance(mc, (list, tuple, np.ndarray))
                  else [_rgba(_colour(mc, default), alpha)])
            ml = marker.get("line") or {}
            sym = {"square": "s", "diamond": "D", "triangle-up": "^", "x": "x", "cross": "P"}.get(
                str(marker.get("symbol", "circle")), "o")
            h = ax.scatter(x, y, s=(np.asarray(size) * PX) ** 2, c=mc, marker=sym, zorder=3,
                           edgecolors=_colour(ml.get("color"), "none") if ml.get("width") else "none",
                           linewidths=float(ml.get("width", 0) or 0) * PX, rasterized=len(x) > 5000)
        if h is not None and name and t.get("showlegend", True) is not False:
            h.set_label(name)
            handles.append(h)

    for a in lay.get("annotations") or []:
        if a.get("xref", "x") != "x" or a.get("yref", "y") != "y" or a.get("x") is None:
            continue
        font = a.get("font") or {}
        ha = {"left": "left", "right": "right"}.get(a.get("xanchor"), "center")
        va = {"top": "top", "bottom": "bottom"}.get(a.get("yanchor"), "center")
        text = re.sub(r"<[^>]+>", "", str(a.get("text", "")))
        box = dict(boxstyle="round,pad=0.3", fc=_colour(a["bgcolor"], "white"), ec="none") if a.get("bgcolor") else None
        ax.annotate(text, (a["x"], a["y"]), xytext=(float(a.get("xshift", 0) or 0) * PX, float(a.get("yshift", 0) or 0) * PX),
                    textcoords="offset points", ha=ha, va=va, fontsize=float(font.get("size", 10)) * PX / 0.85 * font_scale,
                    color=_colour(font.get("color"), INK), fontweight="bold" if "<b>" in str(a.get("text")) else None,
                    bbox=box, zorder=6, annotation_clip=True)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    if equal:
        ax.set_aspect("equal", adjustable="datalim" if height_px else "box")
    if ya.get("autorange") == "reversed":
        ax.invert_yaxis()
    title = (xa.get("title") or {}).get("text") if isinstance(xa.get("title"), dict) else xa.get("title")
    ax.set_xlabel(title or "", fontsize=fs["label"], color=INK)
    title = (ya.get("title") or {}).get("text") if isinstance(ya.get("title"), dict) else ya.get("title")
    ax.set_ylabel(title or "", fontsize=fs["label"], color=INK)
    t = lay.get("title")
    t = t.get("text") if isinstance(t, dict) else t
    if t and show_title:
        ax.set_title(t, fontsize=fs["title"], loc="left", color=INK)
    ax.grid(True, color="#e5e7eb", lw=0.5, zorder=-1)
    ax.tick_params(labelsize=fs["tick"], colors=SOFT, length=3)
    for sp in ax.spines.values():
        sp.set_color(LINE)
    if handles and legend:
        ncol = max(1, min(len(handles), int(width_px // 230)))
        leg = f.legend(handles=handles, loc="outside lower center", ncol=ncol, fontsize=fs["legend"],
                       frameon=False, labelcolor=INK)
        for lh in leg.legend_handles:
            if hasattr(lh, "set_sizes"):
                lh.set_sizes([40])
    buf = io.BytesIO()
    photo = any(t.get("type") == "image" for t in data)     # a photo map: JPEG (a PNG of it is ~5 MB)
    f.savefig(buf, format="jpg" if photo else "png", pad_inches=0.08, bbox_inches=None if height_px else "tight",
              **({"pil_kwargs": {"quality": 88}} if photo else {}))
    plt.close(f)
    return buf.getvalue()
