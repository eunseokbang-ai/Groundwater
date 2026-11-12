"""3D views in the dashboard with the mouse controls of CesiumJS (as in the tiled-model page), drawn with three.js.

Plotly's 3D scene rotates on a left drag and zooms toward the centre; this page works like Cesium instead:
left drag = pan (the grabbed point stays under the cursor), right drag = zoom (down = closer), wheel = zoom
toward the cursor, middle drag or Ctrl + left drag = orbit about the grabbed point, Shift + left drag = look
around. The figures are still built with Plotly (`scene_from_plotly` reads their Scatter3d / Mesh3d traces), so
the same figure can also be written to a stand-alone Plotly HTML.

The data go to the page as base64 float32 / uint8 arrays (a Plotly figure sends them as JSON text, several times
larger), centred on the scene so float32 keeps millimetres. The camera is kept in the browser's local storage per
view, so a rerun of the dashboard does not reset it.
"""
import base64
import json
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).with_name("view3d.html")
_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0), "lightgray": (211, 211, 211), "gray": (128, 128, 128),
          "grey": (128, 128, 128), "red": (255, 0, 0), "orange": (255, 165, 0), "lime": (0, 255, 0),
          "deepskyblue": (0, 191, 255), "yellow": (255, 255, 0), "blue": (0, 0, 255), "green": (0, 128, 0)}
DEFAULT_COLOUR = "#636efa"


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _rgb1(c) -> tuple:
    """One CSS colour ('#rrggbb', '#rgb', 'rgb(r, g, b)', 'rgba(...)' or a few names) -> (r, g, b)."""
    c = str(c).strip().lower()
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if c.startswith("rgb"):
        v = [float(x) for x in c[c.index("(") + 1:c.index(")")].split(",")[:3]]
        return tuple(int(round(x)) for x in v)
    return _NAMED.get(c, (99, 110, 250))


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _colours(c, n):
    """(single '#rrggbb' or None, (n, 3) uint8 or None) from a Plotly colour property."""
    if c is None:
        return DEFAULT_COLOUR, None
    if isinstance(c, str):
        return _hex(_rgb1(c)), None
    seq = list(c)
    if len(seq) != n or not seq:
        return _hex(_rgb1(seq[0])) if seq else DEFAULT_COLOUR, None
    if all(isinstance(s, str) and len(s) == 7 and s[0] == "#" for s in seq[:100]):
        try:                                   # the usual case, '#rrggbb' strings: one hex decode for all
            return None, np.frombuffer(bytes.fromhex("".join(s[1:] for s in seq)), np.uint8).reshape(-1, 3)
        except ValueError:
            pass
    return None, np.array([_rgb1(s) for s in seq], np.uint8)


def _num(a) -> np.ndarray:
    if a is None:
        return np.zeros(0)
    if isinstance(a, np.ndarray) and a.dtype.kind in "fiu":
        return a.astype(float)
    return np.array([np.nan if v is None else v for v in a], float)


def _get(obj, name, default=None):
    v = getattr(obj, name, None) if obj is not None else None
    return default if v is None else v


def scene_from_plotly(fig, key: str = "view") -> dict:
    """The Scatter3d (markers / lines) and Mesh3d traces of a Plotly figure as a scene for view3d.html."""
    raw = []
    for tr in fig.data:
        t = tr.type
        name = tr.name or t
        common = {"name": name, "group": tr.legendgroup or name, "legend": tr.showlegend is not False,
                  "opacity": float(_get(tr, "opacity", 1.0)),
                  # meta="ontop": drawn over everything (markers that a draped map or the wall would hide)
                  "top": tr.meta == "ontop"}
        if t == "scatter3d":
            x, y, z = _num(tr.x), _num(tr.y), _num(tr.z)
            if not len(x):
                continue
            P = np.column_stack([x, y, z])
            mode = tr.mode or "markers"
            if "lines" in mode:
                ok = np.isfinite(P).all(axis=1)
                seg = ok[:-1] & ok[1:]                                   # consecutive points of one polyline
                S = np.stack([P[:-1][seg], P[1:][seg]], axis=1).reshape(-1, 3)
                line = tr.line
                width = float(_get(line, "width", 2))
                dash = _get(line, "dash", None)
                raw.append({**common, "kind": "lines", "P": S, "color": _hex(_rgb1(_get(line, "color", DEFAULT_COLOUR))),
                            "width": max(1.0, width * 0.8), "dash": dash not in (None, "solid")})
            if "markers" in mode:
                ok = np.isfinite(P).all(axis=1)
                m = tr.marker
                single, per = _colours(_get(m, "color", None), len(P))
                size = _get(m, "size", 6)
                size = float(np.median(size)) if not np.isscalar(size) else float(size)
                text = tr.text
                text = [text[i] for i in np.nonzero(ok)[0]] if isinstance(text, (list, tuple)) and len(text) == len(P) else None
                item = {**common, "kind": "points", "P": P[ok], "color": single or _hex(tuple(np.median(per, axis=0).astype(int))),
                        "C": per[ok] if per is not None else None, "size": max(2.0, size * 1.4),
                        "opacity": common["opacity"] * float(_get(m, "opacity", 1.0))}
                if text:
                    item["text"] = text
                raw.append(item)
            if "text" in mode and "markers" not in mode:              # labels drawn as text over the view
                ok = np.isfinite(P).all(axis=1)
                text = list(tr.text) if isinstance(tr.text, (list, tuple)) else [str(tr.text or "")] * len(P)
                font = tr.textfont
                raw.append({**common, "kind": "labels", "P": P[ok], "text": [str(text[i]) for i in np.nonzero(ok)[0]],
                            "color": _hex(_rgb1(_get(font, "color", "#111827")))})
        elif t == "mesh3d":
            P = np.column_stack([_num(tr.x), _num(tr.y), _num(tr.z)])
            if tr.i is None or not len(P):
                continue
            F = np.column_stack([np.asarray(tr.i), np.asarray(tr.j), np.asarray(tr.k)]).astype(np.uint32)
            item = {**common, "kind": "mesh", "P": P, "F": F, "flat": bool(_get(tr, "flatshading", False))}
            if tr.vertexcolor is not None:
                _, item["VC"] = _colours(tr.vertexcolor, len(P))
            elif tr.facecolor is not None:
                _, item["FC"] = _colours(tr.facecolor, len(F))
            single = _hex(_rgb1(tr.color)) if tr.color is not None else None
            if single is None:
                cc = item.get("VC") if item.get("VC") is not None else item.get("FC")
                single = _hex(tuple(np.median(cc, axis=0).astype(int))) if cc is not None else "#9ca3af"
            item["color"] = single
            raw.append(item)
    if raw:
        allp = np.vstack([r["P"] for r in raw if len(r["P"])])
        lo, hi = np.nanmin(allp, axis=0), np.nanmax(allp, axis=0)
    else:
        lo = hi = np.zeros(3)
    centre = 0.5 * (lo + hi)
    radius = float(max(np.linalg.norm(hi - lo) / 2, 1.0))
    items = []
    for r in raw:
        it = {k: v for k, v in r.items() if k not in ("P", "C", "F", "VC", "FC")}
        it["pos"] = _b64((r["P"] - centre).astype(np.float32))
        if r["kind"] == "points":
            it["col"] = _b64(r["C"].astype(np.uint8)) if r.get("C") is not None else None
        elif r["kind"] == "mesh":
            it["idx"] = _b64(r["F"])
            it["vcol"] = _b64(r["VC"]) if r.get("VC") is not None else None
            it["fcol"] = _b64(r["FC"]) if r.get("FC") is not None and r.get("VC") is None else None
        elif r["kind"] == "lines":
            it["dashSize"] = radius * 0.02
        items.append(it)

    sc = fig.layout.scene
    eye = sc.camera.eye if sc.camera is not None and sc.camera.eye is not None else None
    e = np.array([eye.x, eye.y, eye.z], float) if eye is not None and eye.x is not None else np.array([1.25, 1.25, 1.25])
    titles = []
    for ax, d in ((sc.xaxis, "x"), (sc.yaxis, "y"), (sc.zaxis, "z")):
        t = ax.title.text if ax is not None and ax.title is not None and ax.title.text else d
        titles.append(t)
    return {"key": key, "items": items, "center": centre.tolist(), "radius": radius, "eye": (e / np.linalg.norm(e)).tolist(),
            "eyeScale": float(np.clip(np.linalg.norm(e) / 1.9, 0.5, 2.0)), "axes": titles,
            # a different scene extent (another site or chainage window) starts from the figure's own view
            "sig": f"{np.round(lo, 1).tolist()}{np.round(hi, 1).tolist()}"}


def page(scene: dict) -> str:
    data = json.dumps(scene, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__SCENE_JSON__", data)


_CACHE: list = []          # (figure, key, html, digest) of the last few figures: a rerun with the same figure
                           # sends the same page


def page_for(fig, key: str) -> tuple[str, str]:
    """(viewer page, its digest) of a Plotly 3D figure, built once per figure object."""
    import hashlib

    for f, k, html, sig in _CACHE:
        if f is fig and k == key:
            return html, sig
    html = page(scene_from_plotly(fig, key))
    sig = hashlib.sha1(html.encode("utf-8")).hexdigest()
    _CACHE.append((fig, key, html, sig))
    del _CACHE[:-6]
    return html, sig


def show(fig, key: str, height: int = 760):
    """Draw a Plotly 3D figure in Streamlit with the Cesium-style viewer."""
    import streamlit.components.v1 as components

    components.html(page_for(fig, key)[0], height=height)
