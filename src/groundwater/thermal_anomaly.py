"""Thermal anomalies on the wall and how they line up with the joints.

With a temperature orthomosaic (degC, from the R-JPEG photos: scripts/dji_rjpeg_to_temperature.py and
metashape_swap_to_temperature.py) the values are temperatures. Without one, the palette mosaic's colour (iron:
dark purple -> magenta -> red -> orange -> yellow) is turned into a relative index (0 = coldest, 1 = warmest) by
its place along the palette -- a lookup built from the mosaic's own colours, ordered by brightness. Either way a
spot is judged against its surroundings only: a z-score against the masked mean / spread within a few metres
(the wall cools during an evening flight; the palette range drifted between shots). Only bare rock counts:
vegetation (facade photo), netting and flat ground (cleaning.surface_mask) are left out.

Which sign marks water depends on the time: groundwater (~13-16 degC all year) is warmer than the rock in the
evening, at night and in winter, cooler on a sunlit summer day (and a wet joint cools by evaporation). Shadows (overhangs, recesses, open joints) are cold too and much darker on the
photo: each anomaly gets the photo's brightness against its surroundings (z), and below SHADOW_Z it is marked as
a possible shadow (wet rock is darker as well, but less).
Each anomaly gets the share of its area within `near_m` of a mapped trace, the nearest trace family, the joint
set under it and the seepage grade there; the summary compares the near-trace share with that of all rock
(enrichment > 1 = the anomalies sit on joints more than chance).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .facade import planar
from .facade import thermal as TH

PALETTE_STEPS = 64
WINDOW_M = 5.0           # neighbourhood of the local mean / spread (Gaussian sigma = WINDOW_M / 2)
Z_MIN = 2.0
MIN_AREA_M2 = 0.25
NEAR_M = 0.3
SHADOW_Z = -1.5         # photo brightness z below which a cold patch may be a shadow
GRID_STEP = 3            # facade pixels per thermal-grid pixel (1.9 cm -> ~5.6 cm, the thermal mosaic's ~3.3 cm)


@dataclass
class ThermalAnomalies:
    u_min: float
    v_max: float
    res: float
    index: np.ndarray        # (H, W) degC (temperature mosaic) or relative index 0..1 (palette), NaN off it
    z: np.ndarray            # (H, W) local z-score, NaN off bare rock
    labels: np.ndarray       # (H, W) anomaly id (0 none)
    table: pd.DataFrame
    summary: dict
    params: dict


def palette_lut(rgb: np.ndarray, valid: np.ndarray, steps: int = PALETTE_STEPS) -> np.ndarray:
    """(steps, 3) palette colours, cold -> warm: the median colour of each brightness band of the mosaic."""
    c = rgb[valid].reshape(-1, 3).astype(np.float32)
    c = c[:: max(1, len(c) // 2_000_000)]
    lum = c @ np.array([0.299, 0.587, 0.114], np.float32)
    edges = np.percentile(lum, np.linspace(0, 100, steps + 1))
    out = []
    for i in range(steps):
        m = (lum >= edges[i]) & (lum <= edges[i + 1])
        out.append(np.median(c[m], axis=0) if m.any() else out[-1])
    return np.array(out)


def palette_index(rgb: np.ndarray, valid: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Relative index 0..1 of each colour: its nearest palette step."""
    from scipy.spatial import cKDTree

    out = np.full(valid.shape, np.nan, np.float32)
    _, i = cKDTree(lut).query(rgb[valid].reshape(-1, 3).astype(np.float32))
    out[valid] = i / (len(lut) - 1)
    return out


def _masked_mean_std(x: np.ndarray, ok: np.ndarray, sigma_px: float):
    import cv2

    w = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma_px)
    v = np.where(ok, x, 0).astype(np.float32)
    m = cv2.GaussianBlur(v, (0, 0), sigma_px) / np.maximum(w, 1e-6)
    m2 = cv2.GaussianBlur(v * v, (0, 0), sigma_px) / np.maximum(w, 1e-6)
    return m, np.sqrt(np.maximum(m2 - m * m, 1e-8)), w


def _trace_distance(result, shape, u_min, v_max, res) -> tuple[np.ndarray, np.ndarray]:
    """(distance to the nearest trace in m, family id of that nearest trace) on the grid."""
    import cv2

    fam_of = {id(t): f.family_id for f in result.trace_families for t in f.members}
    lines = np.zeros(shape, np.uint8)
    fam = np.full(shape, -1, np.int16)
    for t in result.traces:
        p = t.coords
        cols = ((p[:, 0] - u_min) / res).astype(int)
        rows = ((v_max - p[:, 1]) / res).astype(int)
        f = fam_of.get(id(t), -1)
        for k in range(len(p) - 1):
            cv2.line(lines, (cols[k], rows[k]), (cols[k + 1], rows[k + 1]), 1, 1)
            cv2.line(fam, (cols[k], rows[k]), (cols[k + 1], rows[k + 1]), int(f) + 1, 1)
    if not lines.any():
        return np.full(shape, np.inf, np.float32), np.full(shape, -1, np.int16)
    dist, lab = cv2.distanceTransformWithLabels((1 - lines).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    # the family of the nearest line pixel: labels index the zero (line) pixels in scan order
    ys, xs = np.nonzero(lines == 1)
    fam_line = fam[ys, xs] - 1
    order = np.full(lab.max() + 1, -1, np.int16)
    order[1:len(fam_line) + 1] = fam_line
    return dist * res, order[lab]


def detect(result, thermal_folder, crs: str, window_m: float = WINDOW_M, z_min: float = Z_MIN,
           min_area_m2: float = MIN_AREA_M2, near_m: float = NEAR_M, smap=None, step: int = GRID_STEP,
           exg_threshold: float = 0.06) -> ThermalAnomalies:
    import cv2

    from .discontinuity.cleaning import mask_codes
    from .discontinuity.vegetation import vegetation_mask

    reg = TH.load(thermal_folder, crs)
    if reg is None:
        raise ValueError("열화상 자료가 아직 RGB에 정합되지 않았습니다.")
    _, ortho = planar.find_planar_pair(thermal_folder)
    temp_path = TH.temperature_ortho(thermal_folder)
    fr = result.facade_rgb
    t = fr.transform
    res = abs(t.a) * step
    T = np.array(reg["matrix"])
    if temp_path is not None:                           # degC
        temp, ext, _ = planar.wall_image(result, temp_path, T, res=res, correction=reg.get("local_correction"),
                                         values=True)
        has = np.isfinite(temp) & (temp > -40) & (temp < 80)
        th = None
    else:
        th, ext, _ = planar.wall_image(result, ortho, T, res=res, correction=reg.get("local_correction"))
        has = ~np.all(th == 255, axis=-1)
        temp = None
    u_min, v_max = ext[0], ext[3]
    H, W = has.shape
    # bare rock on the same grid: the facade photo (vegetation) and the surface mask (netting, flat ground)
    uu = u_min + (np.arange(W) + 0.5) * res
    vv = v_max - (np.arange(H) + 0.5) * res
    fc = np.clip(((uu - t.c) / t.a).astype(int), 0, fr.array.shape[2] - 1)
    frr = np.clip(((vv - t.f) / t.e).astype(int), 0, fr.array.shape[1] - 1)
    photo = np.moveaxis(fr.array[:, frr][:, :, fc], 0, -1)
    depth_ok = np.isfinite(result.facade_depth.array[0][frr][:, fc])
    rock = has & depth_ok & ~vegetation_mask(photo, exg_threshold, scale=255.0)
    U, V = np.meshgrid(uu, vv)
    if getattr(result, "surface_mask", None) is not None:
        rock &= mask_codes(result.surface_mask, U.ravel(), V.ravel()).reshape(H, W) == 0
    if temp is not None:
        lut, idx, min_sd = None, np.where(has, temp, np.nan).astype(np.float32), 0.1     # degC
    else:
        lut = palette_lut(th, has)
        idx, min_sd = palette_index(th, has, lut), 0.02
    m, sd, w = _masked_mean_std(np.nan_to_num(idx), rock, window_m / 2 / res)
    z = np.where(rock & (w > 0.2), (idx - m) / np.maximum(sd, min_sd), np.nan)
    from .facade.region import in_core

    z[:, ~in_core(uu, result.config)] = np.nan          # overlap buffer: the neighbours' anomalies are theirs
    dT = np.where(np.isfinite(z), idx - m, np.nan)      # degC above / below the surroundings (index units without)
    lum = photo.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lm, lsd, _ = _masked_mean_std(lum, rock, window_m / 2 / res)
    lz = np.where(rock, (lum - lm) / np.maximum(lsd, 1.0), np.nan)

    dist, near_fam = _trace_distance(result, (H, W), u_min, v_max, res)
    sets = None
    if result.facade_sets is not None:
        sr, sc = result.facade_sets.world_to_pixel(U.ravel(), V.ravel())
        sr, sc = np.asarray(sr).reshape(H, W), np.asarray(sc).reshape(H, W)
        a = result.facade_sets.array[0]
        okp = (sr >= 0) & (sr < a.shape[0]) & (sc >= 0) & (sc < a.shape[1])
        sets = np.full((H, W), -1, np.int16)
        sets[okp] = a[sr[okp], sc[okp]]
    grade = None
    if smap is not None:
        gr = ((smap.v_max - vv) / smap.cell_m).astype(int)
        gc = ((uu - smap.u_min) / smap.cell_m).astype(int)
        R, C = np.meshgrid(gr, gc, indexing="ij")
        okg = (R >= 0) & (R < smap.classes.shape[0]) & (C >= 0) & (C < smap.classes.shape[1])
        grade = np.full((H, W), -1, np.int8)
        grade[okg] = smap.classes[R[okg], C[okg]]

    labels = np.zeros((H, W), np.int32)
    rows = []
    min_px = max(1, int(round(min_area_m2 / res ** 2)))
    zz = np.nan_to_num(z)
    nid = 0
    for kind, mask in (("차가움", zz <= -z_min), ("따뜻함", zz >= z_min)):
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_px:
                continue
            sel = lab == i
            nid += 1
            labels[sel] = nid if kind == "차가움" else -nid
            r_, c_ = np.nonzero(sel)
            fam = near_fam[sel][dist[sel] <= max(near_m, 1.0)]
            fam = fam[fam >= 0]
            st = sets[sel] if sets is not None else np.array([-1])
            st = st[st >= 0]
            g = grade[sel] if grade is not None else np.array([-1])
            rows.append({
                "id": nid, "종류": kind,
                "측점 (m)": f"{uu[c_.min()]:.1f}–{uu[c_.max()]:.1f}", "표고 (m)": f"{vv[r_.max()]:.1f}–{vv[r_.min()]:.1f}",
                "u": float(uu[c_].mean()), "v": float(vv[r_].mean()),
                "면적 (m²)": round(float(sel.sum() * res ** 2), 2), "z 평균": round(float(z[sel].mean()), 2),
                ("ΔT 평균 (°C)" if temp is not None else "지수 차이"): round(float(np.nanmean(dT[sel])), 2),
                "사진 밝기 z": round(float(np.nanmean(lz[sel])), 2), "그늘 의심": bool(np.nanmean(lz[sel]) < SHADOW_Z),
                f"절리 {near_m:g} m 이내 (%)": round(100 * float((dist[sel] <= near_m).mean()), 0),
                "가장 가까운 트레이스까지 (m)": round(float(dist[sel].min()), 2),
                "트레이스 방향군": (f"F{int(np.bincount(fam).argmax())}" if len(fam) else "-"),
                "절리군": (f"Set {int(np.bincount(st).argmax())}" if len(st) else "-"),
                "유출 가능성": ({2: "상", 1: "중", 0: "하"}.get(int(np.bincount(g[g >= 0]).argmax()), "-")
                           if (g >= 0).any() else "-"),
            })
    table = pd.DataFrame(rows)
    rock_px = int(np.isfinite(z).sum())
    near_all = float((dist[np.isfinite(z)] <= near_m).mean()) if rock_px else 0.0
    summary = {"rock_m2": round(rock_px * res ** 2, 1), "near_share_rock": round(near_all, 3)}
    for kind in ("차가움", "따뜻함"):
        sel = (labels > 0) if kind == "차가움" else (labels < 0)
        if sel.any():
            share = float((dist[sel] <= near_m).mean())
            summary[kind] = {"n": int((table["종류"] == kind).sum()), "area_m2": round(float(sel.sum() * res ** 2), 1),
                             "near_share": round(share, 3), "enrichment": round(share / max(near_all, 1e-6), 2)}
            ids = table.loc[(table["종류"] == kind) & ~table["그늘 의심"], "id"].to_numpy()
            lit = np.isin(np.abs(labels), ids) & sel
            if lit.any():                                  # the same without the possible shadows
                ls = float((dist[lit] <= near_m).mean())
                summary[kind].update({"n_not_shadow": int(len(ids)), "near_share_not_shadow": round(ls, 3),
                                      "enrichment_not_shadow": round(ls / max(near_all, 1e-6), 2)})
            if grade is not None:
                rg = grade[np.isfinite(z)]
                hi_all = float((rg == 2).mean()) if rg.size else 0.0
                hi = float((grade[sel] == 2).mean())
                summary[kind].update({"high_grade_share": round(hi, 3),
                                      "high_grade_enrichment": round(hi / max(hi_all, 1e-6), 2)})
    return ThermalAnomalies(u_min, v_max, res, idx, z, labels, table, summary,
                            {"window_m": window_m, "z_min": z_min, "min_area_m2": min_area_m2, "near_m": near_m,
                             "palette": None if lut is None else lut.round(1).tolist(),
                             "values": "temperature" if temp is not None else "palette index",
                             "source": str(temp_path or ortho)})
