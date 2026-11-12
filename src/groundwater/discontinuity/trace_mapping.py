"""2D discontinuity-trace extraction from the facade orthophoto, and a
cross-check of the 3D plane-based extraction against it.

Traces are found at a fixed ground scale (metres, not pixels), so the result does not change with the
facade pixel size. Six detectors are combined (ids in DETECTORS):

* depth recess: an open joint is a groove in the facade depth map (w, toward the viewer). A morphological
  closing with a `recess_kernel_m` disc minus the depth gives how far each pixel lies behind its
  surroundings; deeper than `min_recess_m` is a groove.
* dark line: a black-hat of the grey image with the same kernel finds narrow dark lines (shadowed or
  stained joints) that are not deep enough to show in the depth.
* depth step: where a joint face meets the wall surface the depth folds sharply (a block edge). Cells whose
  depth gradient exceeds `min_step_slope` (1 = 45 deg out of the facade plane) mark those edges.

* valleys (multi-scale): a Hessian line detector on the grey image and on the depth at 2, 5 and 10 cm
  (`valley_strength`), with hysteresis (`hysteresis`): wide or shadowed fissures and rounded grooves the
  fixed 15 cm kernel misses, followed through their weaker stretches from strong seeds.
* wide fissures: the same at 20, 35 and 50 cm, kept only where the grey image AND the depth both show the
  valley (an open fissure tens of cm wide is dark and recessed; a shadow or a dip alone is not), traced
  along its centre line.

The candidate mask (minus vegetation and the no-data border) is thinned to 1-px skeletons, cut at
junctions, and every branch is split into straight segments (`straightness_m`), so curved and irregular
joints are followed instead of needing one long straight Hough line. Segments shorter than `min_length_m`
are dropped. Brightness-edge detection (Canny + Hough) was used first; on a real wall (Ulsan) it mostly
traced surface texture -- rock texture has edges in every direction, a joint is a groove or a dark band.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..io.raster import Raster


@dataclass
class Trace2D:
    coords: np.ndarray  # (2, 2) endpoint map coordinates; on a facade raster: (chainage, elevation)
    pixel_coords: np.ndarray  # (2, 2) [[row1, col1], [row2, col2]]
    length_m: float
    group: int = -1  # segments of one skeleton branch (or one hand-drawn polyline) share a group
    source: str = "auto"  # "auto" (detected) or "manual" (drawn by an operator)
    xyz: np.ndarray | None = None  # (2, 3) world endpoints, when the facade depth is known


DETECTORS = {0: "사진 어두운 선", 1: "DEM 홈", 2: "DEM 단차", 3: "사진 골 (다중 크기)", 4: "DEM 골 (다중 크기)",
             5: "넓은 틈 (사진+DEM)"}
# a trace's detector is group // 1_000_000; manual traces have their own groups


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of a boolean mask to 1-px wide skeletons (vectorised)."""
    img = mask.astype(np.uint8).copy()
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                c = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            m = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            return img.astype(bool)


def _split_straight(rc: np.ndarray, tol: float, min_len: float, out: list, depth: int = 0,
                    max_gap: float = 3.0) -> None:
    """Recursively split pixel coordinates (N, 2) into straight runs; appends (start, end) in px.

    A run must be continuous: where its pixels, projected on their principal axis, leave a gap wider than
    `max_gap` px it is cut there first (otherwise pieces of a branching skeleton metres apart that happen to
    lie on one line -- an overhang lip and a step far below -- pass as one long trace across sound rock).
    A continuous run is straight when the RMS distance of its pixels from the axis is within `tol`;
    otherwise it is cut where it bends most (or at the middle) and both halves are tried again. Crossing
    and branching skeletons are separated by the same rule, so no junction bookkeeping is needed.
    """
    if len(rc) < 3 or depth > 40:
        return
    c = rc.mean(axis=0)
    _, _, vt = np.linalg.svd(rc - c, full_matrices=False)
    axis = vt[0]
    t = (rc - c) @ axis
    perp = (rc - c) @ np.array([-axis[1], axis[0]])
    length = t.max() - t.min()
    if length < min_len:
        return
    ts = np.sort(t)
    gaps = np.diff(ts)
    if gaps.max() > max_gap:
        g = int(np.argmax(gaps))
        cut = 0.5 * (ts[g] + ts[g + 1])
        _split_straight(rc[t <= cut], tol, min_len, out, depth + 1, max_gap)
        _split_straight(rc[t > cut], tol, min_len, out, depth + 1, max_gap)
        return
    if np.sqrt(np.mean(perp ** 2)) <= tol:
        # trim sparse tails (an isolated pixel or two) so the end points sit on the run itself
        lo, hi = np.percentile(t, [1, 99]) if len(t) > 50 else (t.min(), t.max())
        if hi - lo >= min_len:
            out.append((c + lo * axis, c + hi * axis))
        return
    cut = t[np.argmax(np.abs(perp))]
    if cut - t.min() < min_len / 2 or t.max() - cut < min_len / 2:
        cut = np.median(t)
    _split_straight(rc[t <= cut], tol, min_len, out, depth + 1, max_gap)
    _split_straight(rc[t > cut], tol, min_len, out, depth + 1, max_gap)


def skeleton_segments(skel: np.ndarray, min_len_px: float, tol_px: float) -> list:
    """Straight segments (group, start_rc, end_rc) following each connected skeleton."""
    _, lab = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    rows, cols = np.nonzero(lab)
    ids = lab[rows, cols]
    order = np.argsort(ids, kind="stable")
    rows, cols, ids = rows[order], cols[order], ids[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    segs = []
    for g, (r, c) in enumerate(zip(np.split(rows, cuts), np.split(cols, cuts))):
        if len(r) < min_len_px * 0.8:
            continue
        out = []
        _split_straight(np.column_stack([r, c]).astype(float), tol_px, min_len_px, out)
        segs += [(g, a, b) for a, b in out]
    return segs


def _drop_duplicates(segs: list, dist_px: float, max_angle_deg: float = 12.0) -> list:
    """Drop segments lying along a longer one (both ends within `dist_px` of it, similar direction)."""
    segs = sorted(segs, key=lambda s: -np.linalg.norm(s[2] - s[1]))
    kept = []
    cos_min = np.cos(np.radians(max_angle_deg))
    for g, a, b in segs:
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        dup = False
        for _, ka, kb in kept:
            kd = kb - ka
            kl = np.linalg.norm(kd)
            kd = kd / max(kl, 1e-9)
            if abs(d @ kd) < cos_min:
                continue
            ok = True
            for p in (a, b):
                t = np.clip((p - ka) @ kd, -dist_px, kl + dist_px)
                if np.linalg.norm(p - (ka + t * kd)) > dist_px:
                    ok = False
                    break
            if ok:
                dup = True
                break
        if not dup:
            kept.append((g, a, b))
    return kept


VALLEY_SCALES_M = (0.02, 0.05, 0.10)   # Gaussian scales of the multi-scale valley detector
WIDE_SCALES_M = (0.20, 0.35, 0.50)     # wide open / shadowed fissures: photo AND depth must agree


def valley_strength(img: np.ndarray, res: float, scales_m=VALLEY_SCALES_M) -> np.ndarray:
    """Multi-scale line-valley strength (Hessian): how much a pixel lies below its two sides across a
    narrow, elongated low -- a groove in the depth (w toward the viewer, so grooves are minima) or a dark
    line in a grey image. Scale-normalised (sigma^2 x curvature), so it is in the image's own units (m of
    depth, grey levels) whatever the scale: a 2 cm deep groove gives ~0.01-0.02 whether it is 3 or 30 cm
    wide. Only line-like lows count (curvature along the line < half the curvature across it), so pits,
    blobs and one-sided steps score low.
    """
    out = np.zeros(img.shape, np.float32)
    for s_m in scales_m:
        s = s_m / res
        if s < 0.7:
            continue
        g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) / 4.0      # 3x3 Sobel kernels carry a factor 4
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) / 4.0
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        root = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + root)                            # across the line (largest, > 0 at a low)
        l2 = 0.5 * (dxx + dyy - root)                            # along it
        line = (l1 > 0) & (np.abs(l2) < 0.5 * l1)
        out = np.maximum(out, np.where(line, s * s * (l1 - np.abs(l2)), 0))
    return out


def elongated(mask: np.ndarray, res: float, min_length_m: float = 1.0, min_ratio: float = 3.0) -> np.ndarray:
    """Keep connected regions that are long and thin (a fissure), not blotches (a stain, a shadowed hollow):
    length >= `min_length_m` and length / width >= `min_ratio`, from the region's principal axes."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    rows, cols = np.nonzero(lab)
    ids = lab[rows, cols]
    order = np.argsort(ids, kind="stable")
    rows, cols, ids = rows[order], cols[order], ids[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    for r_, c_, i in zip(np.split(rows, cuts), np.split(cols, cuts), ids[np.r_[0, cuts]] if len(ids) else []):
        if len(r_) < 10:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.vstack([r_, c_]).astype(float)))
        length, width = 4 * np.sqrt(max(ev[1], 0)), 4 * np.sqrt(max(ev[0], 1e-6))
        keep[i] = length * res >= min_length_m and length / width >= min_ratio
    return keep[lab]


def hysteresis(strength: np.ndarray, high: float, low: float, valid: np.ndarray) -> np.ndarray:
    """Pixels above `low` connected to at least one above `high` (Canny's rule): a crack is followed
    through its weaker stretches, isolated weak responses are dropped."""
    weak = (strength >= low) & valid
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    seeds = np.unique(lab[(strength >= high) & weak])
    keep = np.zeros(n, bool)
    keep[seeds[seeds > 0]] = True
    return keep[lab]


def _join_collinear(segs: list, gap_px: float, off_px: float, max_angle_deg: float = 12.0) -> list:
    """Join segments that continue one another -- an end within `gap_px` of another one's end, similar
    direction, all four ends within `off_px` of the joined line: a wavy crack cut into straight pieces (or seen
    by two detectors in turn) becomes one trace again. The joined trace runs between the two extreme ends
    along the line (so a piece lying inside the other never shortens it). Candidate pairs come from a k-d tree
    of the end points, in rounds until nothing joins (O(n log n) per round)."""
    segs = [(g, np.asarray(a, float), np.asarray(b, float)) for g, a, b in segs]
    cos_min = np.cos(np.radians(max_angle_deg))
    while len(segs) > 1:
        E = np.array([[a, b] for _, a, b in segs])                     # (n, 2, 2)
        pairs = cKDTree(E.reshape(-1, 2)).query_pairs(gap_px, output_type="ndarray")
        if not len(pairs):
            break
        si, sj = pairs[:, 0] // 2, pairs[:, 1] // 2
        keep = si != sj
        si, sj = si[keep], sj[keep]
        if not len(si):
            break
        d = E[:, 1] - E[:, 0]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        u = d / L[:, None]
        ok = np.abs(np.einsum("ij,ij->i", u[si], u[sj])) >= cos_min
        si, sj = si[ok], sj[ok]
        gap = np.linalg.norm(E[si][:, :, None] - E[sj][:, None], axis=3).min(axis=(1, 2))
        order = np.argsort(gap)
        used = np.zeros(len(segs), bool)
        merged = {}
        for k in order:
            a, b = int(si[k]), int(sj[k])
            if used[a] or used[b]:
                continue
            P = np.vstack([E[a], E[b]])                                  # 4 end points
            ax = u[a] if L[a] >= L[b] else u[b]
            t = (P - P[0]) @ ax
            p0, p1 = P[0] + t.min() * ax, P[0] + t.max() * ax
            line = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            off = np.abs((P - p0) @ np.array([-line[1], line[0]]))
            if off.max() > off_px:
                continue
            used[a] = used[b] = True
            merged[a] = (segs[a][0], p0, p1)
        if not merged:
            break
        segs = [merged.get(i, s_) for i, s_ in enumerate(segs) if i in merged or not used[i]]
    return segs


def _resample_mean(a: np.ndarray, valid: np.ndarray, k: float):
    """Area-average `a` over valid pixels to a grid k times coarser; (values, valid fraction > 0.5)."""
    if k <= 1.0:
        return a.astype(np.float32), valid
    h, w = max(1, int(a.shape[0] / k)), max(1, int(a.shape[1] / k))
    num = cv2.resize(np.where(valid, a, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    den = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    return num / np.maximum(den, 1e-6), den > 0.5


def extract_traces(
    raster: Raster,
    valid_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    resolution_m: float = 0.02,
    recess_kernel_m: float = 0.15,
    min_recess_m: float = 0.02,
    dark_contrast: float = 30.0,
    use_depth: bool = True,
    use_color: bool = True,
    use_steps: bool = True,
    min_step_slope: float = 1.0,
    min_length_m: float = 0.5,
    straightness_m: float = 0.03,
    use_valleys: bool = True,
    valley_depth_m: tuple = (0.08, 0.03),
    valley_dark: tuple = (28.0, 14.0),
    join_gap_m: float = 0.25,
    use_wide: bool = True,
    wide_dark: float = 20.0,
    wide_depth_m: float = 0.12,
    **_legacy,
) -> list[Trace2D]:
    """Candidate discontinuity traces on a facade raster (see module docstring).

    `depth` is the facade depth (w, metres) on the raster's grid; without it only dark lines are used.
    Detection runs at `resolution_m` (never finer than the raster); traces are returned in map units with
    `pixel_coords` in the input raster's pixels. Old Canny/Hough keyword arguments are accepted and ignored.
    """
    px = abs(raster.transform.a)
    res = max(px, resolution_m)
    k = res / px
    valid = np.ones(raster.array.shape[1:], bool) if valid_mask is None else valid_mask
    bands = raster.array[:3].astype(np.float32)
    gray_full = bands[0] if raster.array.shape[0] == 1 else bands.mean(axis=0)
    gray, v = _resample_mean(gray_full, valid, k)
    if not v.any():
        return []
    ks = max(5, int(round(recess_kernel_m / res)) | 1)          # at least 5 px wide or nothing fits inside
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    masks = []                                                  # (detector id, candidate mask); ids in DETECTORS
    if use_color:
        lo, hi = np.percentile(gray[v], [1, 99])
        g8 = np.clip((np.where(v, gray, np.median(gray[v])) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        g8 = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), 1.0)      # pixel noise is not a dark line
        masks.append((0, cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, se) >= dark_contrast))
        if use_valleys:
            masks.append((3, hysteresis(valley_strength(g8, res), *valley_dark, v)))
    if use_depth and depth is not None:
        d, dv = _resample_mean(depth, valid & np.isfinite(depth), k)
        d = np.where(dv, d, np.median(d[dv]) if dv.any() else 0).astype(np.float32)
        recess = cv2.morphologyEx(d, cv2.MORPH_CLOSE, se) - d   # w is toward the viewer: grooves are minima
        masks.append((1, dv & (recess >= min_recess_m)))
        if use_steps:
            ds = cv2.GaussianBlur(d, (0, 0), 1.0)
            grad = np.hypot(cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3)) / (8 * res)
            inner = cv2.erode(dv.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
            masks.append((2, inner & (grad >= min_step_slope)))
        if use_valleys:
            masks.append((4, hysteresis(valley_strength(d, res), *valley_depth_m, dv)))
        if use_wide and use_color:
            # a wide fissure (20-100 cm) is both a broad dark band and a broad groove; either alone is too
            # common (shadow under a ledge, a dip between blocks) -- both together, centre line by thinning
            both = np.minimum(valley_strength(g8, res, WIDE_SCALES_M) / wide_dark,
                              valley_strength(d, res, WIDE_SCALES_M) / wide_depth_m)
            masks.append((5, elongated(hysteresis(both, 1.0, 0.5, v & dv), res)))
    # edges of the data (DEM interpolation, the cut of the export) fold the depth: keep 10 cm away from them
    near_nodata = cv2.dilate((~v).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(2, int(np.ceil(0.10 / res)))) > 0
    min_px, tol_px = min_length_m / res, max(1.0, straightness_m / res)
    # each detector separately (a dark line beside a block edge would otherwise merge into one wide blob
    # whose skeleton is a loop), then collinear duplicates of the same joint are dropped
    segs = []
    for m_i, cand in masks:
        cand = cand & ~near_nodata
        # blobs that cannot hold a trace of the minimum length (speckle, small pits) go before thinning
        _, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        extent = np.hypot(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        keep = extent >= 0.8 * min_px
        keep[0] = False
        # the centre line of a fissure tens of cm wide wanders more than a crack's: straightness scaled to it
        tol_i = max(tol_px, 0.10 / res) if m_i == 5 else tol_px
        segs += [(m_i * 1_000_000 + g, a, b) for g, a, b in skeleton_segments(thin(keep[lab]), min_px, tol_i)]
    segs = _drop_duplicates(segs, 2.5 * tol_px + 1.0)
    segs = _join_collinear(segs, join_gap_m / res, max(0.05 / res, 1.5))
    segs = _drop_duplicates(segs, 2.5 * tol_px + 1.0)           # pieces left inside a joined trace

    traces = []
    for g, a, b in segs:
        rc = np.array([a, b]) * k + (k - 1) / 2            # analysis grid -> input raster pixels
        x, y = raster.pixel_to_world(rc[:, 0], rc[:, 1])
        coords = np.column_stack([x, y])
        traces.append(Trace2D(coords=coords, pixel_coords=rc, length_m=float(np.linalg.norm(coords[0] - coords[1])),
                              group=int(g)))
    return traces


def match_traces_to_patches(
    traces: list[Trace2D],
    patch_footprints: list[np.ndarray],
    expected_directions: list[np.ndarray | None],
    tolerance: float,
    max_angle_deg: float = 20.0,
) -> list[bool | None]:
    """Does an independent 2D trace corroborate each 3D discontinuity patch?

    A patch is confirmed when some trace passes within `tolerance` of its
    footprint (its points in map coordinates) AND runs within `max_angle_deg`
    of the patch's expected trace direction. Proximity alone is meaningless on a
    rock face, where edges are everywhere. Patches with no expected direction
    (planes parallel to the wall) get None: the check does not apply to them.
    """
    if not traces:
        return [None if d is None else False for d in expected_directions]

    step = tolerance / 2.0
    samples, owners, directions = [], [], []
    for k, trace in enumerate(traces):
        a, b = trace.coords
        n = int(np.ceil(np.linalg.norm(b - a) / step)) + 1
        samples.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
        owners.append(np.full(n, k))
        directions.append((b - a) / max(np.linalg.norm(b - a), 1e-12))
    tree = cKDTree(np.vstack(samples))
    owners = np.concatenate(owners)
    directions = np.array(directions)
    min_cos = np.cos(np.radians(max_angle_deg))

    confirmed: list[bool | None] = []
    for footprint, expected in zip(patch_footprints, expected_directions):
        if expected is None:
            confirmed.append(None)
            continue
        hits = tree.query_ball_point(footprint, r=tolerance)
        nearby = np.unique(owners[np.concatenate([np.asarray(h, dtype=int) for h in hits])]) if len(footprint) else []
        confirmed.append(bool(len(nearby)) and bool(np.any(np.abs(directions[nearby] @ expected) >= min_cos)))
    return confirmed
