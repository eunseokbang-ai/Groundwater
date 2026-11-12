"""Kinematic (Markland-type) check of the joint sets against the local face orientation.

The face is not one plane: its dip and dip direction are fitted to the rock points in windows of
`window_m` chainage, so a bend or a leaning stretch gets its own test. For each window:

* planar sliding  -- a set dipping within `lateral_deg` of the face dip direction, less steeply than the
  face (daylights) and more steeply than the friction angle;
* wedge sliding   -- the intersection line of two sets plunges within 90 deg of the face dip direction,
  less steeply than the face's apparent dip along that trend and more steeply than the friction angle;
* flexural toppling -- a steep set dipping into the face (dip direction within `lateral_deg` of the
  opposite direction) with (90 - dip) + friction < face dip (Goodman & Bray).

kinematic_map applies the same three rules cell by cell (about 1 m): each cell gets its own face orientation
from the model's front surface, and only the sets observed near the cell are tested -- a map of where on the
wall each failure mode is kinematically possible.

Every set is tested in its measured orientation (lower-hemisphere dip / dip direction of its mean), as in the
usual Markland / Hoek & Bray practice. With `vertical_deg` (off by default) a set at least that steep is also
tested dipping the other way, for a sensitivity check of planar sliding and toppling (the wedge rule does not
depend on it: the intersection line is the same). It is off because the faces decide the side: on the Iksan walls
the faces of the near-vertical sets lean one way even when the set mean is within a degree of vertical (W 87/035:
no face area dipping into the face), and testing the other way adds toppling the faces do not show.

This is a screening of what is kinematically possible, not a stability factor: cohesion, water pressure,
persistence and block size are not considered.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


FRICTION_DEG = 30.0
LATERAL_DEG = 20.0
WINDOW_M = 5.0
VERTICAL_DEG = None      # e.g. 85: a set at least this steep also tested dipping the other way
MODES = ("평면파괴", "쐐기파괴", "전도파괴")


def dip_dipdir_to_pole(dip: float, dipdir: float) -> np.ndarray:
    d, a = np.radians(dip), np.radians(dipdir)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def _ang(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _ang_v(a, b):
    return np.abs((np.asarray(a, float) - b + 180.0) % 360.0 - 180.0)


def face_windows(points: np.ndarray, frame, window_m: float = WINDOW_M, min_points: int = 500,
                 band_m: float = 1.0) -> list[dict]:
    """Face orientation per chainage window.

    Dip direction: the outward normal of the wall line at the window (the wall line is drawn along the face).
    Dip: lean of the outer face envelope -- the 80th percentile of the offset w in each `band_m` elevation
    band, regressed on elevation. A plane fitted to all points is not used: over a 5 m window the face's
    steps and recesses (1-2 m deep) spread more than the window is long, and the fit then lies along the
    wall. An overhanging envelope is reported as dip 90 with `overhang` True.
    """
    uvw = frame.to_wall(points)
    out = []
    for u0 in np.arange(np.floor(uvw[:, 0].min()), uvw[:, 0].max(), window_m):
        m = (uvw[:, 0] >= u0) & (uvw[:, 0] < u0 + window_m)
        if m.sum() < min_points:
            continue
        v, w = uvw[m, 1], uvw[m, 2]
        bands = np.floor((v - v.min()) / band_m).astype(int)
        bv, bw = [], []
        for b in np.unique(bands):
            sel = bands == b
            if sel.sum() >= 20:
                bv.append(v[sel].mean())
                bw.append(np.percentile(w[sel], 80))
        if len(bv) < 3 or max(bv) - min(bv) < 3 * band_m:
            continue
        slope = np.polyfit(bv, bw, 1)[0]                 # dw/dv: < 0 leans back, > 0 overhangs
        lean = float(np.degrees(np.arctan(max(-slope, 0.0))))
        _, wall_n = frame.axes_at(u0 + window_m / 2)
        dd = float(np.degrees(np.arctan2(wall_n[0], wall_n[1])) % 360.0)
        out.append({"u0": float(u0), "u1": float(u0 + window_m), "face_dip": 90.0 - lean, "face_dip_direction": dd,
                    "overhang": bool(slope > 0.05), "height_m": float(max(bv) - min(bv)), "n_points": int(m.sum())})
    return out


def apparent_dip(dip: float, dipdir: float, trend: float) -> float:
    return float(np.degrees(np.arctan(np.tan(np.radians(dip)) * np.cos(np.radians(trend - dipdir)))))


def intersection(set_a: tuple, set_b: tuple):
    """(plunge, trend) of the intersection line of two planes given as (dip, dip direction)."""
    line = np.cross(dip_dipdir_to_pole(*set_a), dip_dipdir_to_pole(*set_b))
    if np.linalg.norm(line) < 1e-6:
        return None
    line /= np.linalg.norm(line)
    if line[2] > 0:
        line = -line                               # downward
    plunge = float(np.degrees(np.arcsin(-line[2])))
    trend = float(np.degrees(np.arctan2(line[0], line[1])) % 360.0)
    return plunge, trend


# --- the three rules, for scalars or arrays (face orientation per cell) -------------------------------------------

def _both_ways(rule, dip, dd, vertical_deg):
    """`rule(dip, dd)`, or for a plane at least `vertical_deg` steep also `rule(dip, dd + 180)` (None: one way)."""
    dip = np.asarray(dip, float)
    ok = rule(dip, np.asarray(dd, float))
    if vertical_deg is None:
        return ok
    return ok | ((dip >= vertical_deg) & rule(dip, np.asarray(dd, float) + 180.0))


def planar_possible(dip, dd, fdip, fdd, friction_deg: float, lateral_deg: float = LATERAL_DEG,
                    vertical_deg: float | None = VERTICAL_DEG):
    """Planar sliding: dips out of the face (within `lateral_deg`), less steeply than the face, more than friction.
    With `vertical_deg`, a plane at least that steep is also tested dipping the other way."""
    return _both_ways(lambda d, a: (_ang_v(a, fdd) <= lateral_deg) & (d > friction_deg) & (d < fdip), dip, dd,
                      vertical_deg)


def toppling_possible(dip, dd, fdip, fdd, friction_deg: float, lateral_deg: float = LATERAL_DEG,
                      vertical_deg: float | None = VERTICAL_DEG):
    """Flexural toppling (Goodman & Bray): dips into the face, (90 - dip) + friction < face dip. A plane at least
    `vertical_deg` steep is also tested dipping the other way."""
    return _both_ways(lambda d, a: (_ang_v(a, np.asarray(fdd, float) + 180.0) <= lateral_deg)
                      & ((90.0 - d) + friction_deg < fdip), dip, dd, vertical_deg)


def wedge_possible(plunge, trend, fdip, fdd, friction_deg: float):
    """Wedge sliding: the intersection line plunges out of the face (trend within 90 deg of its dip direction),
    less steeply than the face's apparent dip along that trend, more steeply than friction."""
    fdip = np.minimum(np.asarray(fdip, float), 89.99)
    app = np.degrees(np.arctan(np.tan(np.radians(fdip)) * np.cos(np.radians(np.asarray(trend, float) - fdd))))
    plunge = np.asarray(plunge, float)
    return (_ang_v(trend, fdd) < 90.0) & (plunge > friction_deg) & (plunge < app)


def kinematic_check(windows: list[dict], joints: dict, friction_deg: float = FRICTION_DEG,
                    lateral_deg: float = LATERAL_DEG, vertical_deg: float | None = VERTICAL_DEG) -> pd.DataFrame:
    """One row per window and possible failure mode. `joints`: name -> (dip, dip direction). A set at least
    `vertical_deg` steep is also tested dipping the other way; the detail then gives the orientation that fails."""
    rows = []
    names = list(joints)
    for w in windows:
        fdip, fdd = w["face_dip"], w["face_dip_direction"]
        base = {"u0": w["u0"], "u1": w["u1"], "face": f"{fdip:.0f}/{fdd:03.0f}"}
        for n in names:
            dip, dd = joints[n]
            for mode, ok in (("평면파괴", planar_possible), ("전도파괴", toppling_possible)):
                if ok(dip, dd, fdip, fdd, friction_deg, lateral_deg, vertical_deg):
                    flip = not ok(dip, dd, fdip, fdd, friction_deg, lateral_deg, None)   # only dipping the other way
                    shown = (dd + 180.0) % 360.0 if flip else dd
                    rows.append({**base, "mode": mode, "joints": n,
                                 "detail": f"{dip:.0f}/{shown:03.0f}" + (" (반대 방향)" if flip else "")})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                li = intersection(joints[names[i]], joints[names[j]])
                if li is None:
                    continue
                plunge, trend = li
                if wedge_possible(plunge, trend, fdip, fdd, friction_deg):
                    rows.append({**base, "mode": "쐐기파괴", "joints": f"{names[i]} × {names[j]}",
                                 "detail": f"교선 {plunge:.0f}→{trend:03.0f}"})
    return pd.DataFrame(rows, columns=["u0", "u1", "face", "mode", "joints", "detail"])


def representative_face(windows: list[dict]) -> tuple[float, float] | None:
    """(dip, dip direction) of the wall as a whole: median dip and height-weighted mean direction."""
    if not windows:
        return None
    dips = np.array([w["face_dip"] for w in windows])
    dds = np.radians([w["face_dip_direction"] for w in windows])
    wt = np.array([w.get("height_m", 1.0) for w in windows])
    dd = float(np.degrees(np.arctan2((np.sin(dds) * wt).sum(), (np.cos(dds) * wt).sum())) % 360.0)
    return float(np.median(dips)), dd


# --- the face cell by cell: where on the wall each failure mode is kinematically possible ------------------------

@dataclass
class KinematicMap:
    u_min: float
    v_max: float
    cell_m: float
    face_dip: np.ndarray            # (rows, cols) local face dip (90 where overhanging), NaN off the rock
    face_dd: np.ndarray
    overhang: np.ndarray            # bool
    valid: np.ndarray               # bool: rock face, not vegetation
    modes: dict                     # mode -> bool (rows, cols)
    joints: dict                    # mode -> (rows, cols) object array: the sets / pairs that make it possible
    params: dict


def _front_surface(result, cell_m: float = 0.1):
    """Front-most facade w on a `cell_m` grid (NaN where no data): (array, u0, v0 top, cell)."""
    d = result.facade_depth
    a = d.array[0].astype(np.float32)
    px = abs(d.transform.a)
    k = max(1, int(round(cell_m / px)))
    H, W = a.shape[0] // k, a.shape[1] // k
    with np.errstate(all="ignore"):
        front = np.nanmax(a[:H * k, :W * k].reshape(H, k, W, k), axis=(1, 3))
    return front, d.transform.c, d.transform.f, px * k


def local_faces(result, cell_m: float = 1.0, smooth_m: float = 3.0):
    """Local face orientation per `cell_m` cell: the front surface smoothed over `smooth_m` (block scale, not
    the small steps), its outward normal turned from wall into map coordinates.
    Returns (dip, dip direction, overhang, coverage, grid (u_min, v_max, cell))."""
    import cv2

    front, u0, v0, c = _front_surface(result)
    ok = np.isfinite(front)
    sig = max(1.0, smooth_m / 2.0 / c)
    num = cv2.GaussianBlur(np.where(ok, front, 0.0).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sig)
    with np.errstate(all="ignore"):
        sm = num / den
    dw_drow, dw_dcol = np.gradient(sm, c)
    dw_du, dw_dv = dw_dcol, -dw_drow                       # rows run down the elevation
    k = max(1, int(round(cell_m / c)))
    H, W = front.shape[0] // k, front.shape[1] // k
    sl = (slice(k // 2, H * k, k), slice(k // 2, W * k, k))   # the centre sample of each cell
    fu, fv = dw_du[sl][:H, :W], dw_dv[sl][:H, :W]
    cover = ok[:H * k, :W * k].reshape(H, k, W, k).mean(axis=(1, 3))
    cell = c * k
    uc = u0 + (np.arange(W) + 0.5) * cell
    _, tan, right = result.wall_frame._frame_at(uc)
    n_u, n_v, n_w = -fu, -fv, np.ones_like(fu)              # outward normal of w = f(u, v), wall coordinates
    nx = n_u * tan[None, :, 0] + n_w * right[None, :, 0]
    ny = n_u * tan[None, :, 1] + n_w * right[None, :, 1]
    nz = n_v
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    with np.errstate(all="ignore"):
        nx, ny, nz = nx / norm, ny / norm, nz / norm
    overhang = nz < 0                                       # the normal points down: the face leans out
    dip = np.where(overhang, 90.0, np.degrees(np.arccos(np.clip(np.abs(nz), 0, 1))))
    dd = np.degrees(np.arctan2(nx, ny)) % 360.0
    bad = ~np.isfinite(dip) | ~np.isfinite(dd) | (cover < 0.5)
    dip[bad] = np.nan
    dd[bad] = np.nan
    return dip, dd, overhang & ~bad, cover, (u0, v0, cell)


def joint_presence(result, names: list, shape, grid, near_m: float = 2.0) -> dict:
    """name -> bool (rows, cols): the set's faces (set map) or its traces lie within `near_m` of the cell."""
    import cv2

    from .trace_sets import family_sets

    u0, v0, cell = grid
    H, W = shape
    out = {}
    lab = result.facade_sets
    for name in names:
        m = np.zeros((H, W), np.uint8)
        sid = int(name.split()[1]) if name.startswith("Set ") else None
        if sid is not None and lab is not None:
            r, c = np.nonzero(lab.array[0] == sid)
            x, y = lab.pixel_to_world(r, c)
            rr = ((v0 - np.asarray(y)) / cell).astype(int)
            cc = ((np.asarray(x) - u0) / cell).astype(int)
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            m[rr[ok], cc[ok]] = 1
        for f in result.trace_families:
            linked = family_sets(f, result.wall_frame, result.discontinuity_sets)
            if (sid is not None and sid in linked) or name == f"트레이스 F{f.family_id}":
                for t in f.members:
                    (ua, va), (ub, vb) = t.coords
                    cv2.line(m, (int((ua - u0) / cell), int((v0 - va) / cell)),
                             (int((ub - u0) / cell), int((v0 - vb) / cell)), 1, 1)
        r = max(0, int(round(near_m / cell)))
        if r:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        out[name] = m.astype(bool)
    return out


def kinematic_map(result, joints: dict, friction_deg: float = FRICTION_DEG, lateral_deg: float = LATERAL_DEG,
                  cell_m: float = 1.0, smooth_m: float = 3.0, near_m: float | None = 2.0,
                  exclude_vegetation: bool = True, exg_threshold: float = 0.06,
                  vertical_deg: float | None = VERTICAL_DEG) -> KinematicMap:
    """Which failure modes are kinematically possible in each cell of the face: the cell's own face orientation
    (local_faces) against the joint sets -- with `near_m`, only the sets observed within that distance of the
    cell (faces of the set on the set map, or traces of its family); None: every set everywhere. Cells mostly
    covered by vegetation on the facade photo are left out (their surface is the foliage, not the rock). A set at
    least `vertical_deg` steep is also tested dipping the other way."""
    dip, dd, overhang, cover, grid = local_faces(result, cell_m, smooth_m)
    H, W = dip.shape
    valid = np.isfinite(dip)
    if exclude_vegetation and result.facade_rgb is not None:
        import cv2

        from .vegetation import vegetation_mask

        veg = vegetation_mask(np.moveaxis(result.facade_rgb.array, 0, -1), exg_threshold, scale=255.0)
        u0, v0, cell = grid
        t = result.facade_rgb.transform
        # the photo resampled onto this grid (same origin: both are facade rasters)
        k = cell / abs(t.a)
        h, w = int(H * k), int(W * k)
        share = cv2.resize(veg[:h, :w].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        valid &= share < 0.6
    names = list(joints)
    pres = joint_presence(result, names, (H, W), grid, near_m) if near_m is not None else \
        {n: np.ones((H, W), bool) for n in names}
    fd = np.where(valid, dip, 0.0)
    fdd = np.where(valid, dd, 0.0)
    modes = {m: np.zeros((H, W), bool) for m in MODES}
    which = {m: np.full((H, W), "", dtype=object) for m in MODES}

    def add(mode, mask, label):
        mask = mask & valid
        modes[mode] |= mask
        for r, c in zip(*np.nonzero(mask)):
            which[mode][r, c] = (which[mode][r, c] + ", " if which[mode][r, c] else "") + label

    for n in names:
        jd, jdd = joints[n]
        add("평면파괴", planar_possible(jd, jdd, fd, fdd, friction_deg, lateral_deg, vertical_deg) & pres[n], n)
        add("전도파괴", toppling_possible(jd, jdd, fd, fdd, friction_deg, lateral_deg, vertical_deg) & pres[n], n)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            li = intersection(joints[names[i]], joints[names[j]])
            if li is None:
                continue
            wg = wedge_possible(li[0], li[1], fd, fdd, friction_deg)
            add("쐐기파괴", wg & pres[names[i]] & pres[names[j]], f"{names[i]}×{names[j]}")
    u0, v0, cell = grid
    return KinematicMap(u0, v0, cell, dip, dd, overhang, valid, modes, which,
                        {"friction_deg": friction_deg, "lateral_deg": lateral_deg, "cell_m": cell, "smooth_m": smooth_m,
                         "near_m": near_m, "exclude_vegetation": exclude_vegetation, "vertical_deg": vertical_deg})
