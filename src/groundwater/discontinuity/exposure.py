"""Excavated surface vs joints: how each face lies to the excavated wall it was found on.

In a quarry the walls are cut by blasting and machines, not left by rock falls along joints, so the faces the
plane extraction finds are a mix of
* the excavated surface itself -- bench faces, pre-split / blasted faces -- and joints opened along it (blasting
  breaks along existing joints; the two are not told apart by orientation alone): steep and parallel to the wall;
* bench floors, berms and the floor of a blast: near horizontal;
* joints crossing the wall: they cannot be the excavated surface.
Exposure biases the first: a plane parallel to the wall shows as a wide face, one crossing it as a narrow strip
and a trace line. Iksan 2026-10 (LiDAR, E / S / W): on the east wall 79 % of the area of its main set lies within
15 deg of the wall itself.

The wall direction is local: the frame's horizontal normal (the digitized wall line), turned by the bends of the
wall front at the scale WALL_SMOOTH_M (the front: the WALL_FRONT_Q quantile of the depth toward the pit in each
0.5 m of chainage), so a bench face that follows a bend of the wall is still wall-parallel. Classes:

    parallel   steep (dip > FLAT_DIP_DEG) and the pole within PARALLEL_DEG of the local wall normal -- excavated
               surface, or a joint along it
    flat       dip <= FLAT_DIP_DEG -- bench floor / berm / blast floor, or a gently dipping joint
    cross      the rest -- crossing the wall
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARALLEL_DEG = 20.0       # as facade.frame.apparent_trace_direction: within this a plane shows as a face, not a line
FLAT_DIP_DEG = 20.0
WALL_SMOOTH_M = 5.0
WALL_FRONT_Q = 90
PROFILE_STEP_M = 0.5
CLASSES = {"parallel": "벽면 평행", "flat": "수평 (소단·바닥 가능)", "cross": "교차"}
CLASS_COLOURS = {"parallel": "#9ca3af", "flat": "#d97706", "cross": "#2563eb"}


def _pole(dip, dd):
    d, a = np.radians(np.asarray(dip, float)), np.radians(np.asarray(dd, float))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=-1)


def front_profile(u, w, smooth_m: float = WALL_SMOOTH_M, step: float = PROFILE_STEP_M):
    """(u grid, dw/du) of the wall front: the WALL_FRONT_Q quantile of w (toward the pit) per `step` of chainage,
    gaps filled, smoothed (Gaussian, sigma `smooth_m`)."""
    from scipy.ndimage import gaussian_filter1d

    u, w = np.asarray(u, float), np.asarray(w, float)
    ok = np.isfinite(u) & np.isfinite(w)
    u, w = u[ok], w[ok]
    if len(u) < 10:
        return np.array([0.0, 1.0]), np.zeros(2)
    g = np.arange(np.floor(u.min() / step) * step, u.max() + step, step)
    k = np.clip(((u - g[0]) / step).astype(int), 0, len(g) - 1)
    order = np.argsort(k, kind="stable")
    ks, starts = np.unique(k[order], return_index=True)
    front = np.full(len(g), np.nan)
    for kk, s, e in zip(ks, starts, np.r_[starts[1:], len(order)]):
        front[kk] = np.percentile(w[order[s:e]], WALL_FRONT_Q)
    have = np.isfinite(front)
    front = np.interp(np.arange(len(g)), np.flatnonzero(have), front[have])
    front = gaussian_filter1d(front, smooth_m / step, mode="nearest")
    return g, np.gradient(front, step)


def wall_normals(frame, u, profile=None) -> np.ndarray:
    """(n, 3) horizontal unit normals of the wall (toward the pit) at chainages u; with `profile` (front_profile)
    turned by the local bends of the wall front."""
    u = np.atleast_1d(np.asarray(u, float))
    grid = np.arange(np.floor(u.min()), np.ceil(u.max()) + 1.0, 0.5) if len(u) > 200 else None
    if grid is not None:                        # many points: the frame's axes on a 0.5 m grid, interpolated
        ax = np.array([np.r_[frame.axes_at(float(g))[0], frame.axes_at(float(g))[1]] for g in grid])
        A = np.column_stack([np.interp(u, grid, ax[:, i]) for i in range(6)])
        along, out = A[:, :3], A[:, 3:]
    else:
        ax = [frame.axes_at(float(x)) for x in u]
        along, out = np.array([a for a, _ in ax]), np.array([o for _, o in ax])
    if profile is not None:
        s = np.interp(u, profile[0], profile[1])        # dw/du: the front turns toward / away from the pit
        out = out - s[:, None] * along
    out[:, 2] = 0
    return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)


def wall_angle(normals, wall_n) -> np.ndarray:
    """Degrees between the plane poles and the (horizontal) wall normals, axial: 0 = parallel to the wall."""
    n = np.asarray(normals, float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    return np.degrees(np.arccos(np.clip(np.abs((n * wall_n).sum(-1)), 0, 1)))


def classify(dip, angle, parallel_deg: float = PARALLEL_DEG, flat_dip_deg: float = FLAT_DIP_DEG) -> np.ndarray:
    dip, angle = np.asarray(dip, float), np.asarray(angle, float)
    return np.where(dip <= flat_dip_deg, "flat", np.where(angle <= parallel_deg, "parallel", "cross"))


def _frame(folder: Path):
    from ..facade.frame import WallFrame

    v = json.loads((folder / "wall_frame.json").read_text(encoding="utf-8"))["polyline"]
    return WallFrame(np.asarray(v, float), fillet=True)


def depth_profile(folder: Path, smooth_m: float = WALL_SMOOTH_M):
    """front_profile from a saved analysis' facade depth raster (u, v -> w)."""
    import rasterio

    with rasterio.open(Path(folder) / "facade_depth.tif") as d:
        step = max(1, int(round(PROFILE_STEP_M / abs(d.transform.a) / 4)))
        z = d.read(1)[::step, ::step]
        T = d.transform
    u = T.c + (np.arange(z.shape[1]) * step + 0.5) * T.a
    uu = np.broadcast_to(u, z.shape)
    return front_profile(uu.ravel(), z.ravel(), smooth_m)


def classify_folder(folder, parallel_deg: float = PARALLEL_DEG, flat_dip_deg: float = FLAT_DIP_DEG,
                    smooth_m: float = WALL_SMOOTH_M) -> pd.DataFrame:
    """The faces of a saved analysis (discontinuity_patches.csv) with wall_az (azimuth of the local wall normal,
    toward the pit), wall_angle_deg and exposure (parallel / flat / cross)."""
    folder = Path(folder)
    p = pd.read_csv(folder / "discontinuity_patches.csv")
    frame = _frame(folder)
    try:
        prof = depth_profile(folder, smooth_m)
    except Exception:  # noqa: BLE001 -- no depth raster: the wall line alone
        prof = None
    wn = wall_normals(frame, p["chainage_m"].to_numpy(), prof)
    p["wall_az"] = np.degrees(np.arctan2(wn[:, 0], wn[:, 1])) % 360
    p["wall_angle_deg"] = wall_angle(_pole(p["dip"], p["dip_direction"]), wn)
    p["exposure"] = classify(p["dip"], p["wall_angle_deg"], parallel_deg, flat_dip_deg)
    return p


def point_mask(points_uvw, normals, frame, parallel_deg: float = PARALLEL_DEG, flat_dip_deg: float = FLAT_DIP_DEG,
               smooth_m: float = WALL_SMOOTH_M) -> np.ndarray:
    """True for points whose normal is parallel to the local wall (steep and within `parallel_deg`): left out of
    the set detection when the user asks (pipeline: set_detection.exclude_wall_parallel_deg)."""
    prof = front_profile(points_uvw[:, 0], points_uvw[:, 2], smooth_m)
    wn = wall_normals(frame, points_uvw[:, 0], prof)
    n = np.asarray(normals, float)
    dip = np.degrees(np.arccos(np.clip(np.abs(n[:, 2]) / np.linalg.norm(n, axis=1).clip(1e-12), 0, 1)))
    return (dip > flat_dip_deg) & (wall_angle(n, wn) <= parallel_deg)


def summary(P: pd.DataFrame, by=("set_id",)) -> pd.DataFrame:
    """Per group: faces, area, the area shares of the classes, the median roughness of each class."""
    rows = []
    for key, g in P.groupby(list(by)):
        a = g["area_m2"].sum()
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update({"faces": len(g), "area_m2": round(a, 1)})
        for c in CLASSES:
            m = g["exposure"] == c
            r[f"{c}_area_pct"] = round(100 * g.loc[m, "area_m2"].sum() / a, 1) if a else 0.0
            r[f"{c}_faces"] = int(m.sum())
            r[f"{c}_rough_mm"] = round(1000 * g.loc[m, "roughness_rms_m"].median(), 1) if m.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
