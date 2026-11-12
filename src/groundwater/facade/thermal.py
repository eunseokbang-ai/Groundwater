"""Thermal orthomosaic + DEM of a wall (Metashape planar export, e.g. "Current view" or the RGB model's plane) laid
over the RGB one.

The thermal model is aligned on its own images, so it sits a little off the RGB model (Iksan west wall 2026-10:
~27 cm vertically, varying along the wall). Its DEM is registered to the RGB DEM in map coordinates -- a rigid
ICP (planar.register, starting from the wall frame) and then a local correction: robust point-to-plane
translations in 20 m windows every 10 m along the wall, smoothed -- and stored as planar_to_world.json in the
thermal folder (source "icp_to_rgb_dem", "local_correction"). The colours are the camera's palette (not
radiometric temperatures): use them as relatively warm / cold against the surroundings.
"""
import json
from pathlib import Path

import numpy as np

from . import planar

THERMAL_WORDS = ("thermal", "therma", "열화상", "h20t", "tir")
WINDOW_M, STEP_M = 20.0, 10.0
TUKEY_M = 0.10


def _planar_folders(site_folder):
    """Folders (the site folder and its sub-folders) holding a planar DEM + orthomosaic pair."""
    site = Path(site_folder)
    if not site.is_dir():
        return []
    out = []
    for d in [site] + sorted(p for p in site.rglob("*") if p.is_dir() and "분석결과" not in p.parts):
        dem, ortho = planar.find_planar_pair(d)
        if dem is not None and ortho is not None:
            out.append(d)
    return out


def find_folders(site_folder) -> dict:
    """{"rgb": folder or None, "thermal": folder or None}: the planar pairs of a site, thermal by name."""
    rgb = thermal = None
    for d in _planar_folders(site_folder):
        dem, ortho = planar.find_planar_pair(d)
        name = (str(d.relative_to(site_folder)) + dem.name + ortho.name).lower()
        if any(w in name for w in THERMAL_WORDS):
            thermal = thermal or d
        else:
            rgb = rgb or d
    return {"rgb": rgb, "thermal": thermal}


def _world_points(folder, T, step_m):
    dem, _ = planar.find_planar_pair(folder)
    L = planar.local_points(dem, step_m)
    return L @ T[:3, :3].T + T[:3, 3]


def _fit_translation(X, Y, iters: int = 12):
    """t minimising the Tukey-weighted point-to-plane distance of Y + t to the surface X."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(X, dtype=np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    N = np.asarray(pc.normals)
    tree = cKDTree(X)
    t = np.zeros(3)
    w = np.zeros(0)
    for _ in range(iters):
        d, i = tree.query(Y + t, distance_upper_bound=0.6)
        ok = np.isfinite(d)
        if ok.sum() < 500:
            return None, 0
        n = N[i[ok]]
        r = np.einsum("ij,ij->i", n, Y[ok] + t - X[i[ok]])
        w = np.where(np.abs(r) < TUKEY_M, (1 - (r / TUKEY_M) ** 2) ** 2, 0.0)
        A = n * w[:, None]
        t -= np.linalg.solve(A.T @ n + 1e-6 * np.eye(3), A.T @ r)
    return t, int((w > 0).sum())


def register(rgb_folder, thermal_folder, frame, crs: str, progress=None) -> dict:
    """Register the thermal DEM to the RGB DEM (rigid ICP + local correction) and save planar_to_world.json in
    the thermal folder. The RGB pair must be placed already (its Metashape projection JSON or an ICP result)."""
    from scipy.spatial import cKDTree

    say = progress or (lambda s: None)
    ref = planar.load_registration(rgb_folder, crs)
    if ref is None:
        raise ValueError("RGB 평면 자료의 위치(평면 투영 JSON 또는 정합 결과)가 없습니다.")
    Tr = np.array(ref["matrix"])
    Wr = _world_points(rgb_folder, Tr, 0.05)
    dem_t, _ = planar.find_planar_pair(thermal_folder)
    say("강체 정합 (ICP)...")
    reg = planar.register(dem_t, Wr, frame, step_m=0.05)
    Tt = np.array(reg["matrix"])
    Wt = _world_points(thermal_folder, Tt, 0.05)
    say("벽을 따라 국부 보정...")
    ur, ut = frame.to_wall(Wr)[:, 0], frame.to_wall(Wt)[:, 0]
    knots = []
    for u0 in np.arange(np.floor(ut.min() / STEP_M) * STEP_M, ut.max(), STEP_M):
        mr = (ur >= u0 - (WINDOW_M - STEP_M) / 2) & (ur < u0 + (WINDOW_M + STEP_M) / 2)
        mt = (ut >= u0 - (WINDOW_M - STEP_M) / 2) & (ut < u0 + (WINDOW_M + STEP_M) / 2)
        if mr.sum() < 5000 or mt.sum() < 5000:
            continue
        X = Wr[mr][::2]
        c = X.mean(axis=0)
        t, n = _fit_translation(X - c, Wt[mt][::max(1, int(mt.sum()) // 150_000)] - c)
        if t is not None:
            knots.append((u0 + STEP_M / 2, *t))
    K = np.array(knots) if knots else np.zeros((0, 4))
    Ts = np.array([np.median(K[max(0, i - 1):i + 2, 1:4], axis=0) for i in range(len(K))]) if len(K) else K[:, 1:]
    if len(K):
        xy = frame.to_world(np.column_stack([K[:, 0], np.zeros(len(K)), np.zeros(len(K))]))[:, :2]
        reg["local_correction"] = {"note": "world shift added to the rigid placement, by chainage along the wall "
                                           "(robust point-to-plane fit to the RGB DEM, 20 m windows every 10 m, "
                                           "running median of 3)", "knots_xy": xy.round(3).tolist(),
                                   "t": Ts.round(4).tolist()}
    # fit after both steps, on a subsample: point-to-surface distance to the RGB DEM (along its normal -- the
    # distance to the nearest grid point would add the grid spacing and the bumps of trees and netting)
    import open3d as o3d

    X = Wr[::2]
    c = X.mean(axis=0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(X - c, dtype=np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    N = np.asarray(pc.normals)
    tree = cKDTree(X - c)
    sub = Wt[::max(1, len(Wt) // 200_000)]
    shift = planar._correction_by_chainage(frame, reg.get("local_correction"))
    if shift is not None:
        sub = sub + shift(frame.to_wall(sub)[:, 0])
    dist, i = tree.query(sub - c, distance_upper_bound=1.0)
    ok = np.isfinite(dist)
    d = np.abs(np.einsum("ij,ij->i", N[i[ok]], sub[ok] - c - (X - c)[i[ok]]))
    reg.update({"source": "icp_to_rgb_dem", "reference": str(Path(rgb_folder).name), "crs": crs,
                "fit_median_m": round(float(np.median(d)), 4), "fit_p75_m": round(float(np.percentile(d, 75)), 4)})
    planar.save_registration(thermal_folder, reg)
    return reg


def temperature_ortho(thermal_folder):
    """The temperature orthomosaic of a thermal folder (single-band float degC, named *temperature* / *온도*, made by
    scripts/metashape_swap_to_temperature.py on the same plane as the palette one), or None."""
    folder = Path(thermal_folder)
    if not folder.is_dir():
        return None
    hits = [f for f in sorted(folder.glob("*.tif")) if any(w in f.name.lower() for w in planar.VALUE_WORDS)]
    return hits[0] if hits else None


def load(thermal_folder, crs: str | None = None) -> dict | None:
    """The thermal registration saved by register(), or None (also when made for another crs / height datum)."""
    path = Path(thermal_folder) / planar.REG_FILE
    if not path.exists():
        return None
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("source") != "icp_to_rgb_dem":
        return None
    if crs and reg.get("crs") and reg["crs"] != crs:
        return None
    return reg
