"""From the joint network to seepage and grouting inputs (surface evidence only -- see the caveats below).

1. Seepage potential on the facade (`seepage_potential`): the facade is cut into `cell_m` cells and each
   cell gets four indicators, each scaled 0..1 over the rock cells by its 5-95 percentile:

   * P21        -- trace length per facade area (m/m2): how much jointing there is;
   * openness   -- mean depth of the open grooves (closing of the depth map minus the depth, as in trace
                   detection), an upper bound of the surface aperture, vegetation excluded;
   * connectivity -- log of the length of the connected trace network the cell's traces belong to
                   (traces touching within `touch_m` are one network): long networks reach further in;
   * wetness    -- share of green (moss, plants) pixels in a 10 cm strip along the traces: plants rooted
                   in joints mark where water comes out.

   Only rock cells are scored: at least 40 % non-vegetation data and at least `min_plane_cover` of the cell
   covered by joint-set surfaces (the set map). Dry, brown canopy is not green and would otherwise pass as
   rock -- with deep "grooves" between leaves -- but it holds no planar facets.
   The weighted sum is the score; cells in the top 15 % of the rock cells are "상", the next 35 % "중".
   Neighbouring "상" cells (8-connected, at least 2) form zones, numbered by their mean score (#1 = highest)
   and reported with chainage/elevation/world coordinates.

2. Grouting inputs (`grouting_inputs`), from the joint sets (3D sets + trace-only families with an
   orientation):

   * Snow's parallel-plate permeability tensor k = sum_k a_k^3 / (12 s_k) (I - n_k n_k^T). With no aperture
     data every set gets the same aperture, so only the principal directions and the anisotropy ratio are
     meaningful (absolute values need Lugeon tests);
   * hole orientation: for every trend/plunge the number of conductive joints met per metre,
     sum_k w_k |cos(angle to the pole)| / s_k (Terzaghi), w_k = the set's transmissivity ~ a_k^3 -- the
     best orientation overall and, when a curtain line is given, the best one inside the curtain plane;
   * hole spacing: probability that a joint of each set is cut by at least one hole of a row at spacing S,
     min(1, L_c / S) with L_c the part of the joint's trace on the curtain plane across the holes
     (persistence x |sin(trace, hole)|);
   * curtain check: per set the angle to the curtain plane (a set parallel to the curtain is hardly met by
     holes drilled in it and carries flow along it), its apparent dip on the curtain, and the end extension
     needed so that joints passing the curtain ends are cut (p90 persistence).

Caveats: everything is derived from the exposed face. Whether a joint is open at depth, its hydraulic
aperture and the head gradient come from boreholes (televiewer, Lugeon), ERT and water levels; the
seepage map ranks where to look and the grouting numbers are first estimates for the design team.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .discontinuity.vegetation import vegetation_mask

WEIGHTS = {"p21": 0.3, "openness": 0.25, "connectivity": 0.25, "wetness": 0.2}
CLASS_HIGH, CLASS_MID = 85.0, 50.0           # percentiles of the score over rock cells
WORK_RES = 0.02                              # m, grid for the openness / wetness rasters
MIN_CURTAIN_PLUNGE = 45                      # deg, shallowest curtain hole considered practical to drill and grout


@dataclass
class SeepageMap:
    cell_m: float
    u_min: float
    v_max: float
    layers: dict          # name -> (rows, cols) float arrays: p21, openness, connectivity, wetness, score
    rock: np.ndarray      # (rows, cols) bool, cells with enough rock to be scored
    classes: np.ndarray   # (rows, cols) int8: 2 = 상, 1 = 중, 0 = 하, -1 = not scored
    zones: pd.DataFrame


def _scale(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """0..1 over the rock cells: 5-95 percentile; for a sparse indicator (most cells exactly 0, e.g. plants
    along joints) 0 .. the 95th percentile of the non-zero values -- otherwise the 95th percentile is itself
    ~0 and any trace of the indicator jumps to 1."""
    if not mask.any():
        return np.zeros_like(x)
    v = x[mask]
    nz = v[v > 0]
    if len(nz) and len(nz) < 0.5 * len(v):
        lo, hi = 0.0, float(np.percentile(nz, 95))
    else:
        lo, hi = np.percentile(v, [5, 95])
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1) if hi > lo else np.zeros_like(x)


def _segments_touch(a0, a1, b0, b1, tol):
    """Do two 2D segments intersect or come within `tol`?"""
    def dist(p, q0, q1):
        d = q1 - q0
        t = np.clip(((p - q0) @ d) / max(d @ d, 1e-12), 0, 1)
        return np.linalg.norm(p - (q0 + t * d))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(b0, b1, a0), cross(b0, b1, a1)
    d3, d4 = cross(a0, a1, b0), cross(a0, a1, b1)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return min(dist(a0, b0, b1), dist(a1, b0, b1), dist(b0, a0, a1), dist(b1, a0, a1)) <= tol


def trace_networks(traces, touch_m: float = 0.1) -> np.ndarray:
    """Total length of the connected network each trace belongs to (union-find over touching traces)."""
    n = len(traces)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    p0 = np.array([t.coords[0] for t in traces])
    p1 = np.array([t.coords[1] for t in traces])
    lo, hi = np.minimum(p0, p1) - touch_m, np.maximum(p0, p1) + touch_m
    for i in range(n):
        cand = np.flatnonzero((lo[i + 1:, 0] <= hi[i, 0]) & (hi[i + 1:, 0] >= lo[i, 0]) &
                              (lo[i + 1:, 1] <= hi[i, 1]) & (hi[i + 1:, 1] >= lo[i, 1])) + i + 1
        for j in cand:
            if _segments_touch(p0[i], p1[i], p0[j], p1[j], touch_m):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    lengths = np.array([t.length_m for t in traces])
    total = pd.Series(lengths).groupby(roots).transform("sum").to_numpy()
    return total


def seepage_potential(result, cell_m: float = 1.0, weights: dict | None = None, recess_kernel_m: float = 0.15,
                      exg_threshold: float = 0.06, touch_m: float = 0.1, min_plane_cover: float = 0.15
                      ) -> SeepageMap | None:
    if result.facade_rgb is None:
        return None
    weights = {**WEIGHTS, **(weights or {})}
    t = result.facade_rgb.transform
    px = abs(t.a)
    rgb = np.moveaxis(result.facade_rgb.array, 0, -1)
    depth = result.facade_depth.array[0]
    H, W = depth.shape
    u_min, v_max = t.c, t.f
    rows, cols = int(np.ceil(H * px / cell_m)), int(np.ceil(W * px / cell_m))

    # work grid (2 cm): rock / vegetation masks and groove depth
    k = max(1.0, WORK_RES / px)
    wh, ww = max(1, int(H / k)), max(1, int(W / k))
    res = px * H / wh
    valid = np.isfinite(depth)
    veg = vegetation_mask(rgb, exg_threshold, scale=255.0) & valid
    num = cv2.resize(np.where(valid, depth, 0).astype(np.float32), (ww, wh), interpolation=cv2.INTER_AREA)
    den = cv2.resize(valid.astype(np.float32), (ww, wh), interpolation=cv2.INTER_AREA)
    vw = den > 0.5
    d = np.where(vw, num / np.maximum(den, 1e-6), 0).astype(np.float32)
    d = np.where(vw, d, np.median(d[vw]) if vw.any() else 0).astype(np.float32)
    vegw = cv2.resize(veg.astype(np.float32), (ww, wh), interpolation=cv2.INTER_AREA) > 0.5
    ks = max(5, int(round(recess_kernel_m / res)) | 1)
    recess = cv2.morphologyEx(d, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))) - d
    rock_w = vw & ~vegw
    recess = np.where(rock_w, np.clip(recess, 0, 0.3), 0)

    def to_cells(a, mask):
        """Mean of `a` over `mask` pixels per cell, and the mask fraction per cell."""
        cr = np.minimum((np.arange(wh) * res / cell_m).astype(int), rows - 1)
        cc = np.minimum((np.arange(ww) * res / cell_m).astype(int), cols - 1)
        idx = (cr[:, None] * cols + cc[None, :]).ravel()
        s = np.bincount(idx, weights=(a * mask).ravel(), minlength=rows * cols)
        n = np.bincount(idx, weights=mask.ravel().astype(float), minlength=rows * cols)
        tot = np.bincount(idx, minlength=rows * cols)
        return (s / np.maximum(n, 1)).reshape(rows, cols), (n / np.maximum(tot, 1)).reshape(rows, cols)

    openness, rock_frac = to_cells(recess, rock_w)
    rock = rock_frac >= 0.4
    from .facade.region import in_core

    rock &= in_core(u_min + (np.arange(cols) + 0.5) * cell_m, result.config)[None, :]    # the section core only
    plane_cover = np.zeros((rows, cols))
    if result.facade_sets is not None:
        fs = result.facade_sets
        a = fs.array[0]
        fres = abs(fs.transform.a)
        cr = np.clip(((np.arange(a.shape[0]) * fres + (v_max - fs.transform.f)) / cell_m).astype(int), 0, rows - 1)
        cc = np.clip(((np.arange(a.shape[1]) * fres + (fs.transform.c - u_min)) / cell_m).astype(int), 0, cols - 1)
        idx = (cr[:, None] * cols + cc[None, :]).ravel()
        cnt = np.bincount(idx, minlength=rows * cols)
        plane_cover = (np.bincount(idx, weights=(a >= 0).ravel(), minlength=rows * cols) / np.maximum(cnt, 1)).reshape(rows, cols)
        rock &= plane_cover >= min_plane_cover

    # traces: P21, network length, plants along them
    traces = [tr for tr in result.traces if tr.length_m > 0]
    p21 = np.zeros((rows, cols))
    conn = np.zeros((rows, cols))
    strip = np.zeros((wh, ww), np.uint8)
    if traces:
        net = trace_networks(traces, touch_m)
        for tr, nl in zip(traces, net):
            n_s = max(2, int(np.ceil(tr.length_m / 0.05)))
            pts = tr.coords[0] + np.linspace(0, 1, n_s)[:, None] * (tr.coords[1] - tr.coords[0])
            r = np.clip(((v_max - pts[:, 1]) / cell_m).astype(int), 0, rows - 1)
            c = np.clip(((pts[:, 0] - u_min) / cell_m).astype(int), 0, cols - 1)
            np.add.at(p21, (r, c), tr.length_m / n_s)
            conn[r, c] = np.maximum(conn[r, c], np.log10(1 + nl))
            a = ((pts[[0, -1], 0] - u_min) / res).astype(int), ((v_max - pts[[0, -1], 1]) / res).astype(int)
            cv2.line(strip, (int(a[0][0]), int(a[1][0])), (int(a[0][1]), int(a[1][1])), 1, max(1, int(round(0.10 / res))))
    p21 /= cell_m * cell_m
    strip_b = strip.astype(bool) & vw
    wet, _ = to_cells(vegw.astype(float), strip_b)

    layers = {"p21": p21, "openness": openness, "connectivity": conn, "wetness": wet}
    scaled = {n: _scale(a, rock) for n, a in layers.items()}
    score = sum(weights[n] * scaled[n] for n in scaled) / sum(weights.values())
    score = np.where(rock, score, np.nan)
    classes = np.full((rows, cols), -1, np.int8)
    if rock.any():
        hi, mid = np.percentile(score[rock], [CLASS_HIGH, CLASS_MID])
        classes[rock] = np.where(score[rock] >= hi, 2, np.where(score[rock] >= mid, 1, 0))
    layers = {**layers, "score": score}
    zones = _zones(result, classes, layers, cell_m, u_min, v_max)
    return SeepageMap(cell_m, u_min, v_max, layers, rock, classes, zones)


def _zones(result, classes, layers, cell_m, u_min, v_max) -> pd.DataFrame:
    n, lab = cv2.connectedComponents((classes == 2).astype(np.uint8), connectivity=8)
    rows = []
    depth = result.facade_depth
    fam_of = {id(t): f for f in result.trace_families for t in f.members}
    sets_r = result.facade_sets
    for z in range(1, n):
        rr, cc = np.nonzero(lab == z)
        if len(rr) < 2:
            continue
        u0, u1 = u_min + cc.min() * cell_m, u_min + (cc.max() + 1) * cell_m
        v1, v0 = v_max - rr.min() * cell_m, v_max - (rr.max() + 1) * cell_m
        uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
        dr, dc = depth.world_to_pixel(np.array([uc]), np.array([vc]))
        win = depth.array[0][max(dr[0] - 50, 0):dr[0] + 50, max(dc[0] - 50, 0):dc[0] + 50]
        w = float(np.nanmedian(win)) if np.isfinite(win).any() else 0.0
        x, y, zz = result.wall_frame.to_world(np.array([[uc, vc, w]]))[0]
        comp = {k: float(np.nanmean(layers[k][rr, cc])) for k in ("p21", "openness", "connectivity", "wetness")}
        fams = {}
        for tr in result.traces:
            m = 0.5 * (tr.coords[0] + tr.coords[1])
            if u0 <= m[0] <= u1 and v0 <= m[1] <= v1:
                f = fam_of.get(id(tr))
                key = "수동" if tr.source == "manual" else (f"F{f.family_id}" if f else "기타")
                fams[key] = fams.get(key, 0) + tr.length_m
        sets_here = ""
        if sets_r is not None:
            sr, sc = sets_r.world_to_pixel(np.array([u0, u1]), np.array([v1, v0]))
            blk = sets_r.array[0][max(min(sr), 0):max(sr) + 1, max(min(sc), 0):max(sc) + 1]
            ids, cnt = np.unique(blk[blk >= 0], return_counts=True)
            sets_here = ", ".join(f"Set {i}" for i in ids[np.argsort(-cnt)][:3])
        rows.append({"zone": len(rows) + 1, "area_m2": len(rr) * cell_m ** 2, "chainage_m": f"{u0:.0f}–{u1:.0f}",
                     "elevation_m": f"{v0:.0f}–{v1:.0f}", "score": round(float(np.nanmean(layers['score'][rr, cc])), 2),
                     "p21": round(comp["p21"], 2), "openness_cm": round(comp["openness"] * 100, 1),
                     "network_m": round(10 ** comp["connectivity"] - 1, 0), "plants_along_joints": round(comp["wetness"], 2),
                     "trace_families": ", ".join(f"{k} {v:.0f} m" for k, v in sorted(fams.items(), key=lambda kv: -kv[1])[:3]),
                     "sets_on_face": sets_here, "x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(zz), 2)})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df = df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    df["zone"] = np.arange(1, len(df) + 1)       # numbered by score: #1 is the highest everywhere it is shown
    return df


# --- grouting inputs ------------------------------------------------------------------------------------

def _pole(dip, dd):
    d, a = np.radians(dip), np.radians(dd)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def _line(trend, plunge):
    t, p = np.radians(trend), np.radians(plunge)
    return np.array([np.cos(p) * np.sin(t), np.cos(p) * np.cos(t), -np.sin(p)])


def _trend_plunge(v):
    v = v / np.linalg.norm(v)
    if v[2] > 0:
        v = -v
    return float(np.degrees(np.arctan2(v[0], v[1])) % 360), float(np.degrees(np.arcsin(-v[2])))


def joint_sets_for_flow(result) -> pd.DataFrame:
    """Sets entering the flow analysis: 3D sets and trace-only families with an estimated orientation."""
    rows = []
    for d in result.discontinuity_sets:
        pers = float(np.median([p.persistence_length for p in d.patches])) if d.patches else None
        face_parallel = False
        if d.patches:
            fp = [result.wall_frame.apparent_trace_direction(p.normal, result.wall_frame.to_wall(p.centroid[None])[0, 0]) is None
                  for p in d.patches]
            face_parallel = float(np.mean(fp)) > 0.5
        rows.append({"name": f"Set {d.set_id}", "dip": d.mean_dip, "dip_direction": d.mean_dip_direction,
                     "spacing_m": d.spacing_mean, "persistence_m": pers, "source": "3D 면",
                     "note": "벽면과 평행 (간격은 벽면 계단 간격일 수 있음)" if face_parallel else ""})
    for f in result.trace_families:
        if f.linked_set is None and f.dip is not None:
            rows.append({"name": f"트레이스 F{f.family_id}", "dip": f.dip, "dip_direction": f.dip_direction,
                         "spacing_m": f.apparent_spacing_m, "persistence_m": f.mean_length_m, "source": "입면 트레이스",
                         "note": f"방향 추정 신뢰도 {f.orientation_quality}, 간격은 입면 겉보기 값"})
    df = pd.DataFrame(rows)
    if len(df):
        df["spacing_m"] = df["spacing_m"].fillna(df["spacing_m"].median() if df["spacing_m"].notna().any() else 2.0)
        df["persistence_m"] = df["persistence_m"].fillna(1.0)
        df["aperture_rel"] = 1.0
    return df


def snow_tensor(sets: pd.DataFrame):
    """Permeability tensor (relative units), principal values / directions and each set's share."""
    k = np.zeros((3, 3))
    share = []
    for r in sets.itertuples():
        n = _pole(r.dip, r.dip_direction)
        kk = r.aperture_rel ** 3 / (12 * max(r.spacing_m, 0.05)) * (np.eye(3) - np.outer(n, n))
        k += kk
        share.append(np.trace(kk))
    vals, vecs = np.linalg.eigh(k)
    order = np.argsort(vals)[::-1]
    principal = [{"axis": i + 1, "relative_k": round(float(vals[j] / vals[order[0]]), 3),
                  **dict(zip(("trend", "plunge"), (round(x, 0) for x in _trend_plunge(vecs[:, j]))))}
                 for i, j in enumerate(order)]
    share = np.array(share) / max(np.sum(share), 1e-12)
    return k, pd.DataFrame(principal), share


def crossing_weights(sets: pd.DataFrame) -> np.ndarray:
    """Weight of one crossing of each set: its transmissivity ~ aperture^3 (normalised). Not the flow share:
    that already holds 1 / spacing, which hole_frequency applies through the crossing count."""
    a3 = np.array([float(r.aperture_rel) ** 3 for r in sets.itertuples()])
    return a3 / max(a3.sum(), 1e-12)


def _frequencies(sets: pd.DataFrame, weights, trend, plunge) -> np.ndarray:
    """hole_frequency for arrays of hole directions (degrees): sum over sets of w |n . d| / spacing."""
    t, p = np.radians(np.asarray(trend, float)), np.radians(np.asarray(plunge, float))
    d = np.stack([np.cos(p) * np.sin(t), np.cos(p) * np.cos(t), -np.sin(p)])            # (3, m), as _line
    poles = np.array([_pole(r.dip, r.dip_direction) for r in sets.itertuples()]).reshape(-1, 3)
    scale = np.asarray(weights, float) / np.maximum(sets["spacing_m"].to_numpy(float), 0.05)
    return (scale[:, None] * np.abs(poles @ d)).sum(axis=0)


def hole_frequency(sets: pd.DataFrame, weights, trend, plunge) -> float:
    return float(_frequencies(sets, weights, [trend], [plunge])[0])


def hole_orientations(sets: pd.DataFrame, weights, curtain_azimuth: float | None = None):
    """(grid DataFrame, best overall, best within the vertical curtain plane, vertical hole)."""
    plunge, trend = (a.ravel() for a in np.meshgrid(np.arange(0, 91, 5), np.arange(0, 360, 5), indexing="ij"))
    grid = pd.DataFrame({"trend": trend, "plunge": plunge, "joints_per_m": _frequencies(sets, weights, trend, plunge)})
    best = grid.loc[grid["joints_per_m"].idxmax()].to_dict()
    vertical = {"trend": 0, "plunge": 90, "joints_per_m": hole_frequency(sets, weights, 0, 90)}
    in_plane = None
    if curtain_azimuth is not None:
        cand = [(t % 360, p) for t in (curtain_azimuth, curtain_azimuth + 180) for p in range(MIN_CURTAIN_PLUNGE, 91, 5)]
        f = _frequencies(sets, weights, [c[0] for c in cand], [c[1] for c in cand])
        k = int(np.argmax(f))
        in_plane = {"trend": cand[k][0], "plunge": cand[k][1], "joints_per_m": float(f[k])}
    return grid, best, in_plane, vertical


def per_set_frequency(sets: pd.DataFrame, trend, plunge) -> list:
    d = _line(trend, plunge)
    out = []
    for r in sets.itertuples():
        c = abs(_pole(r.dip, r.dip_direction) @ d)
        out.append({"set": r.name, "angle_to_hole_deg": round(float(np.degrees(np.arcsin(min(1.0, c)))), 0),
                    "joints_per_10m": round(10 * c / max(r.spacing_m, 0.05), 1)})
    return out


def curtain_check(sets: pd.DataFrame, curtain_azimuth: float, spacings=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                  hole_plunge: float = 90.0):
    """Per set: angle to a vertical curtain plane, apparent dip on it, chance of being cut by a row of holes in
    the plane (`hole_plunge`, 90 = vertical) at spacing S along the curtain. A hole only cuts the part of the
    joint's trace across it: chance = persistence x |sin(trace, hole)| / (S x sin(plunge))."""
    cn = np.array([np.cos(np.radians(curtain_azimuth)), -np.sin(np.radians(curtain_azimuth)), 0.0])  # plane normal
    along = np.array([np.sin(np.radians(curtain_azimuth)), np.cos(np.radians(curtain_azimuth)), 0.0])
    rows = []
    for r in sets.itertuples():
        n = _pole(r.dip, r.dip_direction)
        angle = float(np.degrees(np.arccos(min(1.0, abs(n @ cn)))))         # angle between the planes
        line = np.cross(n, cn)
        app = None
        if np.linalg.norm(line) > 1e-6:
            line /= np.linalg.norm(line)
            app = float(np.degrees(np.arctan2(abs(line[2]), abs(line @ along))))
        # trace of a joint on the curtain plane ~ persistence; a set (near) parallel to the curtain is cut
        # by holes in that plane only where they happen to run along it -> no reliable coverage
        hole = _line(curtain_azimuth, hole_plunge)
        across = float(np.linalg.norm(np.cross(line, hole))) if app is not None else 0.0
        lc = r.persistence_m * across if angle >= 20 else 0.0
        S_eff = max(np.sin(np.radians(hole_plunge)), 0.2)            # perpendicular distance between holes / S
        row = {"set": r.name, "angle_to_curtain_deg": round(angle, 0),
               "apparent_dip_on_curtain_deg": None if app is None else round(app, 0)}
        for s in spacings:
            row[f"cut_at_{s:g}m"] = round(min(1.0, lc / (s * S_eff)), 2) if lc else 0.0
        row["remark"] = ("커튼과 거의 평행 — 커튼 면 안의 공으로는 잘 안 만남, 커튼을 따라 흐르는 경로" if angle < 20 else
                         "완경사 — 커튼 하부를 지나는 경로, 커튼 심도 확인" if (app is not None and app < 30) else "")
        rows.append(row)
    return pd.DataFrame(rows)


def grouting_inputs(result, curtain_azimuth: float | None = None) -> dict | None:
    sets = joint_sets_for_flow(result)
    if not len(sets):
        return None
    _, principal, share = snow_tensor(sets)
    sets = sets.assign(flow_share=np.round(share, 3))
    grid, best, in_plane, vertical = hole_orientations(sets, crossing_weights(sets), curtain_azimuth)
    out = {"sets": sets, "principal": principal, "grid": grid, "best": best, "in_plane": in_plane,
           "vertical": vertical, "per_set_best": pd.DataFrame(per_set_frequency(sets, best["trend"], best["plunge"])),
           "per_set_vertical": pd.DataFrame(per_set_frequency(sets, 0, 90)), "curtain_azimuth": curtain_azimuth}
    if curtain_azimuth is not None:
        out["curtain"] = curtain_check(sets, curtain_azimuth)
        conductive = sets[sets["flow_share"] >= 0.1]
        out["end_extension_m"] = round(float(np.percentile(conductive["persistence_m"], 90)), 1) if len(conductive) else None
    return out
