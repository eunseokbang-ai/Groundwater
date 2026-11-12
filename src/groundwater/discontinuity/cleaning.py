"""Faces that are not joints of the wall, found after the extraction and left out of the joint sets.

* ground / water in front of the foot -- the flat ground, a water surface or a bench at the toe is a large,
  near-horizontal "face" and lands in the bedding set; on the Ulsan wall the river surface was the biggest
  Set 0 face (159 m2). Such a face lies at the foot of the wall and sticks out in front of the face above it;
  a bedding plane in the wall itself does not (it lies at or behind that face).
* faces on vegetation -- the colour filter removes green points, but dry leaves and branches are not green.
  A face whose surroundings on the facade photo are mostly green and that lies at the front surface (not on
  rock seen behind the foliage) is most likely the vegetation itself.
* ground under the trees -- a LiDAR scan as the main input sees through the canopy: the hillside floor above
  the wall gives near-horizontal "faces" behind the foliage (Iksan east wall 2026-10, LiDAR only: 62 % of the
  area of a 6/328 set lay under the canopy; photogrammetry, which sees only the canopy, 7 %). A face dipping
  GROUND_MAX_DIP or less, in mostly green surroundings and more than CANOPY_BEHIND_M behind the front surface,
  is ground (canopy_ground_faces); a steep face there is rock seen through the foliage and stays.
* rockfall netting -- a draped net hides the rock: the model's surface there is the net (or the net pressed
  on the rock), smooth and parallel to the face, and its cables are long straight lines. Netting has a
  cyan-grey tint (hue 160-215 deg; Iksan 2026-10: hue ~190, saturation ~0.10), bare rock is near grey
  (saturation ~0.04) and plants are yellow-green (hue 50-100). Where most of a ~1 m neighbourhood has that
  tint (surface_mask), the points are left out before the joint extraction and the traces are dropped.
* ground, benches and the slope top -- near-horizontal surfaces. On the facade they are the bands where the
  surface (front-most depth, smoothed) dips less than 35 deg (surface_mask: left out before the extraction,
  and traces there are dropped -- the edges of a cover mat at the foot read as long horizontal "joints").
  A truly level surface is seen edge-on from the front, so in 3D a near-horizontal face reaching 1.5 m or more
  back into the slope is ground or a bench (flat_faces); a sub-horizontal joint exposed on a steep wall is a
  narrow ledge or an overhang roof.
* tree crowns above the ground -- rock cannot stand in the air: a point more than ABOVE_GROUND_M above the highest
  ground of a bare-earth DTM within GROUND_RADIUS_M is vegetation, whatever its colour (above_ground, applied to
  the points before the extraction and to the LiDAR rear layer). Iksan 2026-10-30: above the wall top the LiDAR
  rear layer (the rearmost returns per facade cell) was the back of the tree crowns, and the near-horizontal sets
  were 56 % (east) and 36 % (west) such faces by area, ~10 m above the ground. The radius covers a DTM that ramps
  across a vertical cliff instead of following it (0.5 m DTM: foot to crest over ~8 m).

The sets are summarised again from the faces that remain (segmentation.summarise_set).
"""
import numpy as np

GROUND_MAX_DIP = 30.0        # deg: only near-horizontal faces can be ground / water
FOOT_BAND_M = 1.0            # within this height above the local foot of the wall
FOOT_BIN_M = 2.0             # chainage bins for the local foot (2nd percentile elevation)
FRONT_M = 0.5                # "in front of the face above": more than this in front of its envelope
MIN_FRONT_SHARE = 0.5        # ... for at least this share of the face's points
VEG_RADIUS_M = 0.5           # neighbourhood on the facade photo for the share of green
VEG_MIN_SHARE = 0.5
VEG_FRONT_TOL_M = 0.3        # a face within this of the front surface lies on what the photo shows
CANOPY_BEHIND_M = 0.5        # a face this far behind a green front surface is under the canopy
MASK_RES_M = 0.05            # grid of the netting / flat-surface mask (a facade of the analysis points)
NET_HUE_DEG = (160.0, 215.0)
NET_MIN_SAT = 0.09           # calibrated on the Iksan west wall: bare rock 0 %, netted areas ~60 % (trees and
NET_MIN_VALUE = 0.12         # outcrops showing through)
NET_WINDOW_M = 1.0
NET_MIN_SHARE = 0.45
FLAT_MAX_DIP = 35.0          # deg: surface dip on the smoothed facade depth below which it is ground
FLAT_SMOOTH_M = 0.25
FLAT_MIN_WIDTH_M = 0.15      # thinner flat bands (a joint's lip) are kept
FLAT_FACE_MAX_DIP = 30.0
FLAT_FACE_MIN_DEPTH_M = 1.5  # near-horizontal faces reaching this far back into the slope are ground / benches
MASK_NET, MASK_FLAT = 1, 2   # bits of the surface mask
ABOVE_GROUND_M = 2.0         # above the highest ground within GROUND_RADIUS_M: a tree crown, not rock
GROUND_RADIUS_M = 5.0
GROUND_AGREE_M = 1.0         # datum check: points within this of the DTM ...
GROUND_AGREE_MIN = 0.01      # ... must be at least this share of the cloud (else the DTM is not in its datum)


def ground_ceiling(dtm_path, bounds, radius: float = GROUND_RADIUS_M):
    """(ground, ceiling, transform): the DTM over `bounds` (x0, y0, x1, y1) + `radius` at its own resolution
    (NaN = no data) and the highest ground within `radius` of every cell."""
    import rasterio
    from rasterio.windows import from_bounds
    from scipy.ndimage import maximum_filter

    x0, y0, x1, y1 = bounds
    with rasterio.open(dtm_path) as d:
        win = from_bounds(x0 - radius, y0 - radius, x1 + radius, y1 + radius, d.transform)
        win = win.round_offsets().round_lengths()
        z = d.read(1, window=win, boundless=True, fill_value=d.nodata if d.nodata is not None else np.nan).astype(np.float32)
        t = d.window_transform(win)
        if d.nodata is not None:
            z[z == d.nodata] = np.nan
    r = max(1, int(round(radius / abs(t.a))))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ceil = maximum_filter(np.where(np.isfinite(z), z, -np.inf), footprint=(xx ** 2 + yy ** 2) <= r * r, mode="constant",
                          cval=-np.inf)
    ceil[~np.isfinite(ceil)] = np.nan
    return z, ceil, t


def _at(grid, t, x, y) -> np.ndarray:
    c = np.floor((np.asarray(x) - t.c) / t.a).astype(np.int64)
    r = np.floor((np.asarray(y) - t.f) / t.e).astype(np.int64)
    ok = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
    out = np.full(len(c), np.nan, np.float32)
    out[ok] = grid[r[ok], c[ok]]
    return out


def above_ground(xyz: np.ndarray, dtm_path, margin: float = ABOVE_GROUND_M, radius: float = GROUND_RADIUS_M,
                 check_datum: bool = True) -> tuple[np.ndarray, dict]:
    """(mask, info): points more than `margin` above the highest DTM ground within `radius` (False where the DTM has
    no data). The DTM must be in the points' height datum: when fewer than GROUND_AGREE_MIN of the points lie within
    GROUND_AGREE_M of it (`check_datum`), nothing is flagged and info["skipped"] says why."""
    xyz = np.asarray(xyz)
    if not len(xyz):
        return np.zeros(0, bool), {"removed": 0}
    lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    ground, ceil, t = ground_ceiling(dtm_path, (lo[0], lo[1], hi[0], hi[1]), radius)
    g = _at(ground, t, xyz[:, 0], xyz[:, 1])
    info = {"dtm": str(dtm_path), "margin_m": margin, "radius_m": radius}
    agree = float(np.mean(np.abs(xyz[:, 2] - g) <= GROUND_AGREE_M)) if np.isfinite(g).any() else 0.0
    info["points_on_dtm"] = round(agree, 4)
    if check_datum and agree < GROUND_AGREE_MIN:
        info["skipped"] = f"only {agree:.1%} of the points within {GROUND_AGREE_M} m of the DTM (height datum?)"
        return np.zeros(len(xyz), bool), info
    c = _at(ceil, t, xyz[:, 0], xyz[:, 1])
    with np.errstate(invalid="ignore"):
        out = np.isfinite(c) & (xyz[:, 2] > c + margin)
    info["removed"] = int(out.sum())
    return out, info


def ground_faces(uvw: np.ndarray, point_patch: np.ndarray, sets) -> set:
    """Ids of the near-horizontal faces at the foot that stick out in front of the face above them.
    uvw: wall coordinates of the analysis points, point_patch: their patch id (-1 none)."""
    if not len(uvw):
        return set()
    bins = np.floor(uvw[:, 0] / FOOT_BIN_M).astype(int)
    order = np.argsort(bins, kind="stable")
    keys, starts = np.unique(bins[order], return_index=True)
    foot = {int(k): float(np.percentile(uvw[order[s:e], 1], 2))
            for k, s, e in zip(keys, starts, np.r_[starts[1:], len(order)])}
    out = set()
    for d in sets:
        for p in d.patches:
            if p.dip > GROUND_MAX_DIP:
                continue
            q = uvw[point_patch == p.patch_id]
            if not len(q):
                continue
            cu, cv = float(np.median(q[:, 0])), float(np.median(q[:, 1]))
            f = foot.get(int(np.floor(cu / FOOT_BIN_M)))
            if f is None or cv - f > FOOT_BAND_M:
                continue
            # the face envelope 1-4 m above the foot, around this chainage (80th percentile of w = its front)
            win = (np.abs(uvw[:, 0] - cu) < 2.5) & (uvw[:, 1] > f + 1.0) & (uvw[:, 1] < f + 4.0)
            if win.sum() < 50:
                continue
            face = np.percentile(uvw[win, 2], 80)
            if (q[:, 2] > face + FRONT_M).mean() >= MIN_FRONT_SHARE:
                out.add(p.patch_id)
    return out


def vegetation_faces(sets, frame, facade_rgb, facade_depth, exg_threshold: float) -> set:
    """Ids of faces at the front surface in a mostly green neighbourhood of the facade photo."""
    import cv2

    from .vegetation import vegetation_mask

    if facade_rgb is None or facade_depth is None:
        return set()
    veg = vegetation_mask(np.moveaxis(facade_rgb.array, 0, -1), exg_threshold, scale=255.0)
    k = max(1, int(round(VEG_RADIUS_M / abs(facade_rgb.transform.a))))
    share = cv2.blur(veg.astype(np.float32), (2 * k + 1, 2 * k + 1))
    depth = facade_depth.array[0]
    out = set()
    for d in sets:
        for p in d.patches:
            u, v, w = frame.to_wall(p.centroid[None])[0]
            r, c = (int(x[0]) for x in facade_rgb.world_to_pixel(np.array([u]), np.array([v])))
            if not (0 <= r < share.shape[0] and 0 <= c < share.shape[1]) or share[r, c] <= VEG_MIN_SHARE:
                continue
            front = depth[r, c] if (0 <= r < depth.shape[0] and 0 <= c < depth.shape[1]) else np.nan
            if np.isfinite(front) and front - w < VEG_FRONT_TOL_M:
                out.add(p.patch_id)
    return out


def canopy_ground_faces(sets, frame, facade_rgb, facade_depth, exg_threshold: float) -> set:
    """Ids of near-horizontal faces under the canopy: ground under the trees seen by a LiDAR."""
    import cv2

    from .vegetation import vegetation_mask

    if facade_rgb is None or facade_depth is None:
        return set()
    veg = vegetation_mask(np.moveaxis(facade_rgb.array, 0, -1), exg_threshold, scale=255.0)
    k = max(1, int(round(VEG_RADIUS_M / abs(facade_rgb.transform.a))))
    share = cv2.blur(veg.astype(np.float32), (2 * k + 1, 2 * k + 1))
    depth = facade_depth.array[0]
    out = set()
    for d in sets:
        for p in d.patches:
            if p.dip > GROUND_MAX_DIP:
                continue
            u, v, w = frame.to_wall(p.centroid[None])[0]
            r, c = (int(x[0]) for x in facade_rgb.world_to_pixel(np.array([u]), np.array([v])))
            if not (0 <= r < share.shape[0] and 0 <= c < share.shape[1]) or share[r, c] <= VEG_MIN_SHARE:
                continue
            front = depth[r, c] if (0 <= r < depth.shape[0] and 0 <= c < depth.shape[1]) else np.nan
            if np.isfinite(front) and front - w > CANOPY_BEHIND_M:
                out.add(p.patch_id)
    return out


def netting_area(rgb: np.ndarray, valid: np.ndarray, res: float) -> np.ndarray:
    """Netted area of a facade image (H, W, 3 uint8): most of a NET_WINDOW_M neighbourhood has the net's tint."""
    import cv2

    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    tint = valid & (hue >= NET_HUE_DEG[0]) & (hue <= NET_HUE_DEG[1]) & (sat >= NET_MIN_SAT) & (val >= NET_MIN_VALUE)
    k = int(round(NET_WINDOW_M / res)) // 2 * 2 + 1
    share = cv2.blur(tint.astype(np.float32), (k, k)) / np.maximum(cv2.blur(valid.astype(np.float32), (k, k)), 1e-6)
    net = (valid & (share > NET_MIN_SHARE)).astype(np.uint8)
    net = cv2.morphologyEx(net, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))      # rock seen through the mesh
    return net.astype(bool) & valid


def flat_area(depth: np.ndarray, res: float) -> np.ndarray:
    """Near-horizontal surfaces of a facade depth raster (w = f(u, v), NaN off the model): the dip of the
    smoothed surface, arccos(|dw/dv| / |(-dw/du, -dw/dv, 1)|), below FLAT_MAX_DIP, in bands at least
    FLAT_MIN_WIDTH_M across."""
    import cv2

    valid = np.isfinite(depth)
    sig = max(0.5, FLAT_SMOOTH_M / res)
    num = cv2.GaussianBlur(np.where(valid, depth, 0.0).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sig)
    with np.errstate(all="ignore"):
        sm = num / den
    d_row, d_col = np.gradient(sm, res)
    fu, fv = d_col, -d_row                                   # rows run down the elevation
    with np.errstate(all="ignore"):
        dip = np.degrees(np.arccos(np.clip(np.abs(fv) / np.sqrt(fu ** 2 + fv ** 2 + 1.0), 0.0, 1.0)))
    flat = (valid & (dip < FLAT_MAX_DIP)).astype(np.uint8)
    k = max(1, int(round(FLAT_MIN_WIDTH_M / res)))
    return cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool) & valid


def surface_mask(uvw: np.ndarray, colors, netting: bool = True, flat: bool = True, res: float = MASK_RES_M):
    """Raster (uint8, wall coordinates) of where the model is not rock to analyse: MASK_NET where netting
    covers the face, MASK_FLAT on near-horizontal ground; from a facade of the analysis points (colours 0..1)."""
    from ..facade.orthophoto import FacadeAccumulator
    from ..io.raster import Raster

    if not len(uvw) or not (netting or flat):
        return None
    acc = FacadeAccumulator.covering(uvw, res, margin=res)
    acc.add(uvw, colors)
    rgb, depth = acc.finish(2)
    d = depth.array[0]
    valid = np.isfinite(d)
    m = np.zeros(d.shape, np.uint8)
    if netting and colors is not None:
        m[netting_area(np.moveaxis(rgb.array, 0, -1), valid, res)] |= MASK_NET
    if flat:
        m[flat_area(d, res)] |= MASK_FLAT
    return Raster(array=m[None], transform=depth.transform, crs=None, nodata=0)


def mask_codes(mask, u, v) -> np.ndarray:
    """Surface-mask bits at wall coordinates (0 outside the raster)."""
    a = mask.array[0]
    r, c = mask.world_to_pixel(np.asarray(u, float), np.asarray(v, float))
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.zeros(len(r), np.uint8)
    out[ok] = a[r[ok], c[ok]]
    return out


def flat_faces(uvw: np.ndarray, point_patch: np.ndarray, sets) -> set:
    """Ids of near-horizontal faces reaching FLAT_FACE_MIN_DEPTH_M or more back into the slope (ground, benches,
    the slope top seen edge-on from the front)."""
    out = set()
    if not len(uvw):
        return out
    order = np.argsort(point_patch, kind="stable")
    ids, starts = np.unique(point_patch[order], return_index=True)
    span = {int(i): order[s:e] for i, s, e in zip(ids, starts, np.r_[starts[1:], len(order)])}
    for d in sets:
        for p in d.patches:
            if p.dip > FLAT_FACE_MAX_DIP or p.patch_id not in span:
                continue
            w = uvw[span[p.patch_id], 2]
            if len(w) >= 10 and np.percentile(w, 95) - np.percentile(w, 5) >= FLAT_FACE_MIN_DEPTH_M:
                out.add(p.patch_id)
    return out


def drop_masked_traces(traces: list, mask) -> list:
    """Traces whose midpoint is not on netting or flat ground (hand-drawn traces are always kept)."""
    if mask is None or not traces:
        return traces
    mid = np.array([t.coords.mean(axis=0) for t in traces])
    bits = mask_codes(mask, mid[:, 0], mid[:, 1])
    return [t for t, b in zip(traces, bits) if t.source == "manual" or b == 0]


def drop_faces(sets, point_labels: np.ndarray, drop: set, normals: np.ndarray, link_distance: float) -> list:
    """The sets without the faces in `drop` (their points become unassigned in point_labels, in place),
    summarised again; a set left without faces is removed."""
    from .segmentation import summarise_set

    if not drop:
        return sets
    gone = np.isin(point_labels[:, 1], list(drop))
    point_labels[gone] = -1
    out = []
    for d in sets:
        patches = [p for p in d.patches if p.patch_id not in drop]
        if len(patches) == len(d.patches):
            out.append(d)
        elif patches:
            out.append(summarise_set(d.set_id, patches, normals, point_labels[:, 0], link_distance))
    return out
