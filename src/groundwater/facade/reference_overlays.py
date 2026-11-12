"""Reference images (tracings, photos) placed on the wall-parallel planar orthomosaic / DSM.

A planar folder may hold reference_overlays.json (written by a site script, e.g.
scripts/register_bangudae_overlays.py): per image a mapping from image pixels to plane coordinates (the
planar rasters' own x, y in metres) -- a homography, or a thin-plate spline through control points. From
there an overlay is

* rendered on a plane grid (`render`): RGBA, a tracing's white paper and grey block outlines made
  (semi-)transparent so the figures sit on the rock;
* drawn over a window of the planar orthomosaic, a hillshade of the planar DSM or its micro-relief
  (`background`): DSM minus a few-cm smoothing -- pecked carvings are millimetres deep, invisible in a
  hillshade at wall scale but standing out in the residual;
* draped over the surface in world coordinates (`drape`) for the tiled-model viewer.
"""
import json
from pathlib import Path

import cv2
import numpy as np

OVERLAY_FILE = "reference_overlays.json"


def load(folder) -> dict | None:
    p = Path(folder) / OVERLAY_FILE
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    data["folder"] = str(Path(folder))
    return data


def _image(folder, entry) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(Path(folder) / entry["file"]), np.uint8), cv2.IMREAD_COLOR)
    return img[:, :, ::-1].copy()                                            # RGB


class _Map:
    """image px <-> plane xy for one overlay."""

    def __init__(self, entry):
        self.entry = entry
        if entry["method"] == "homography":
            self.H = np.array(entry["H"], float)
            self.Hi = np.linalg.inv(self.H)
        else:
            from scipy.interpolate import RBFInterpolator

            P = np.array([p[:4] for p in entry["points"]], float)
            self.fwd = RBFInterpolator(P[:, :2], P[:, 2:], kernel="thin_plate_spline", smoothing=1e-3)
            # the inverse fitted to the forward map itself (dense samples over the image), so plane -> image ->
            # plane closes to well under a millimetre instead of two independent splines disagreeing by cm
            w, h = entry.get("_size", (P[:, 0].max() * 1.2, P[:, 1].max() * 1.2))
            gx, gy = np.meshgrid(np.linspace(0, w, 25), np.linspace(0, h, 25))
            img = np.column_stack([gx.ravel(), gy.ravel()])
            self.inv = RBFInterpolator(self.fwd(img), img, kernel="thin_plate_spline", smoothing=1e-6)

    def to_plane(self, xy):
        xy = np.asarray(xy, float)
        if self.entry["method"] == "homography":
            q = np.column_stack([xy, np.ones(len(xy))]) @ self.H.T
            return q[:, :2] / q[:, 2:3]
        return self.fwd(xy)

    def to_image(self, xy):
        xy = np.asarray(xy, float)
        if self.entry["method"] == "homography":
            q = np.column_stack([xy, np.ones(len(xy))]) @ self.Hi.T
            return q[:, :2] / q[:, 2:3]
        return self.inv(xy)


def _alpha(entry, rgb: np.ndarray) -> np.ndarray:
    if entry["kind"] != "tracing":
        return np.full(rgb.shape[:2], 255, np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    figure = ((s > 70) & (v > 40)) | (v < 90)
    outline = ~figure & (v < 205) & (s < 70)                                 # grey block outlines
    a = np.zeros(rgb.shape[:2], np.uint8)
    a[outline] = 110
    a[figure] = 235
    return a


def _with_size(folder, entry):
    if "_size" not in entry:
        h, w = _image(folder, entry).shape[:2]
        entry = {**entry, "_size": (w, h)}
    return entry


def footprint(folder, entry) -> tuple:
    """Plane bounds (x0, y0, x1, y1) of the image."""
    entry = _with_size(folder, entry)
    w, h = entry["_size"]
    edge = np.vstack([np.column_stack([np.linspace(0, w, 20), np.full(20, y)]) for y in (0, h)] +
                     [np.column_stack([np.full(20, x), np.linspace(0, h, 20)]) for x in (0, w)])
    P = _Map(entry).to_plane(edge)
    return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())


def render(folder, entry, bounds, res_m: float = 0.005) -> np.ndarray:
    """RGBA overlay on the plane grid `bounds` (x0, y0, x1, y1) at `res_m` (row 0 = top = y1)."""
    from scipy.ndimage import map_coordinates

    x0, y0, x1, y1 = bounds
    W, H = int(np.ceil((x1 - x0) / res_m)), int(np.ceil((y1 - y0) / res_m))
    img = _image(folder, entry)
    m = _Map(_with_size(folder, entry))
    # the inverse map on a coarse grid at every `step`-th output pixel, interpolated at the exact fractional
    # grid index of each pixel (a TPS on every pixel would be slow; cv2.resize would shift it by a cell)
    step = 8
    gx, gy = np.meshgrid(np.arange(0, W + step, step), np.arange(0, H + step, step))
    plane = np.column_stack([x0 + (gx.ravel() + 0.5) * res_m, y1 - (gy.ravel() + 0.5) * res_m])
    ij = m.to_image(plane).reshape(gy.shape + (2,))
    fy, fx = np.mgrid[0:H, 0:W].astype(np.float32) / step
    mx = map_coordinates(ij[:, :, 0], [fy, fx], order=1, mode="nearest").astype(np.float32)
    my = map_coordinates(ij[:, :, 1], [fy, fx], order=1, mode="nearest").astype(np.float32)
    rgb = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    a = cv2.remap(_alpha(entry, img), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.dstack([rgb, a])


def union_bounds(folder, entries, margin_m: float = 0.5) -> tuple:
    b = np.array([footprint(folder, e) for e in entries])
    return (float(b[:, 0].min() - margin_m), float(b[:, 1].min() - margin_m),
            float(b[:, 2].max() + margin_m), float(b[:, 3].max() + margin_m))


def _read_window(path, bounds, res_m, bands):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    x0, y0, x1, y1 = bounds
    W, H = int(np.ceil((x1 - x0) / res_m)), int(np.ceil((y1 - y0) / res_m))
    with rasterio.open(path) as ds:
        win = from_bounds(x0, y0, x1, y1, ds.transform)
        a = ds.read(bands, window=win, out_shape=(len(bands), H, W), resampling=Resampling.average,
                    boundless=True, fill_value=0, masked=False)
    return a


def background(folder, bounds, res_m: float = 0.005, kind: str = "ortho") -> np.ndarray:
    """RGB background on the plane grid: "ortho", "hillshade" (DSM) or "relief" (DSM micro-relief)."""
    from .planar import find_planar_pair

    dem, ortho = find_planar_pair(folder)
    if kind == "ortho":
        return np.moveaxis(_read_window(ortho, bounds, res_m, [1, 2, 3]), 0, -1).astype(np.uint8)
    d = _read_window(dem, bounds, res_m, [1])[0].astype(np.float32)
    bad = (d < -1000) | ~np.isfinite(d) | (d == 0)
    d = np.where(bad, np.nanmedian(np.where(bad, np.nan, d)), d).astype(np.float32)
    if kind == "relief":
        rel = d - cv2.GaussianBlur(d, (0, 0), 0.03 / res_m)                  # remove everything wider than ~3 cm
        lo, hi = np.percentile(rel[~bad], [2, 98])
        g = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 1.0), res_m)
        slope = np.arctan(np.hypot(gx, gy))
        g = np.zeros_like(d)
        for az in (315, 45, 225, 135):                                       # multi-directional hillshade
            a = np.radians(az)
            g += np.clip(np.cos(np.radians(40)) * np.cos(slope) + np.sin(np.radians(40)) * np.sin(slope) *
                         np.cos(a - np.arctan2(-gy, gx)), 0, 1) / 4
    g = np.where(bad, 0.5, g)
    return np.repeat((g * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def composite(base: np.ndarray, overlays: list, opacity: float = 0.8) -> np.ndarray:
    out = base.astype(np.float32)
    for o in overlays:
        a = o[:, :, 3:4].astype(np.float32) / 255 * opacity
        out = out * (1 - a) + o[:, :, :3].astype(np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def drape(folder, entry, T, bounds=None, step_m: float = 0.05, offset_m: float = 0.03):
    """Overlay footprint as a grid mesh on the planar DSM, in world coordinates (plane -> world `T`), lifted
    `offset_m` toward the viewer so it shows over the tiled model. Returns (xyz (n, 3), st (n, 2), triangles)."""
    from .planar import find_planar_pair

    b = bounds or footprint(folder, entry)
    dem, _ = find_planar_pair(folder)
    d = _read_window(dem, b, step_m, [1])[0].astype(np.float32)
    bad = (d < -1000) | ~np.isfinite(d) | (d == 0)
    if bad.all():
        return None
    d = np.where(bad, np.median(d[~bad]), d)
    H, W = d.shape
    x0, y0, x1, y1 = b
    xs = x0 + (np.arange(W) + 0.5) * step_m
    ys = y1 - (np.arange(H) + 0.5) * step_m
    gx, gy = np.meshgrid(xs, ys)
    L = np.column_stack([gx.ravel(), gy.ravel(), d.ravel() + offset_m])
    T = np.asarray(T, float)
    xyz = L @ T[:3, :3].T + T[:3, 3]
    st = np.column_stack([(gx.ravel() - x0) / (x1 - x0), (gy.ravel() - y0) / (y1 - y0)])   # t up, like the image
    idx = np.arange(H * W).reshape(H, W)
    a, b_, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.column_stack([np.r_[a, b_], np.r_[c, c], np.r_[b_, e]])
    keep = ~bad.ravel()[tris].any(axis=1)
    return xyz, st, tris[keep]
