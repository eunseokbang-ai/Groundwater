"""LiDAR (e.g. DJI Zenmuse L2, multi-return) behind rockfall netting and vegetation.

Photogrammetry sees the net and the canopy in front of the rock; a LiDAR pulse partly passes them and its last
returns trace the rock behind (Iksan west wall 2026-10: 76 % of the netted 0.5 m cells had returns more than 15 cm
behind the net; on bare rock LiDAR and photogrammetry agree to 3 cm). Where the net lies on the rock (within a few
cm) the two cannot be told apart, so there the surface is the rock with the net's texture on it.

rock_behind streams the LAS (every `keep_every`-th point), puts it in the analysis height datum (heights.py) and
on the photogrammetry (a robust point-to-plane translation fitted on bare rock), keeps the points in the cells of
`cells` (netted / vegetated, wall coordinates) and of those only the rear layer: per `cell_m` wall cell the points
within `layer_m` of the rearmost one -- seen from the pit nothing lies behind the rock. To keep the memory bounded,
the per-cell two rearmost depths are kept on the cell grid while streaming and points more than PREFILTER_M in front
of them are dropped on the way (the rear only moves back as points arrive, so a dropped point stays out; the margin
covers the alignment shift and the grid difference of the final rear_layer).

Density: behind a net more pulses find the rock, so the netted / vegetated cells read every point by default.
Without a planarity limit on the faces (segmentation.MAX_PLANE_RMS) that made the rear layer so continuous that the
region growing merged the netted rock into wall-sized rough "faces" (Iksan west wall 2026-10-24: one of 983 m2 at
0.62 m RMS); with the limit, every point gave 2,989 m2 of faces on the netting against 2,527 m2 from 1/8 (+18 %),
for ~15 min more reading (35 GB file).
"""
import warnings

import numpy as np

from . import heights

STEEP_MIN_DEG = 60.0     # large-scale dip of the rear surface below which it is the hillside under the trees
STEEP_SMOOTH_M = 1.0     # (Iksan: vegetated slope above the wall ~35-50 deg, the rock face 70-90 deg)
BAND_M = 0.5             # the local plane of the rear surface: only cells within this in depth
FACE_GAP_M = 4.0         # a face column continues up over gentle stretches (ledges) up to this high
FACE_SETBACK_M = 10.0    # ... and over steps back (benches) up to this deep
FACE_FAR_M = 20.0        # behind the face's deepest part: the hillside, the trunks, the scan's far edge
FACE_EDGE_M = 2.0        # above the span's top, the points at the face's depth there (within BAND_M)
KEEP_EVERY = 1           # see the module note on density
PREFILTER_M = 0.30       # streaming: kept while within this of the cell's second-rearmost depth so far
PRUNE_POINTS = 40_000_000
CELL_M = 0.05
LAYER_M = 0.06
TUKEY_M = 0.10


def _chunks(path, keep_every: int, chunk_size: int = 20_000_000):
    import laspy

    rng = np.random.default_rng(0)
    with laspy.open(str(path)) as f:
        datum = heights.las_datum(f.header)
        for ch in f.chunk_iterator(chunk_size):
            m = rng.random(len(ch)) < 1.0 / keep_every
            xyz = np.column_stack([np.asarray(ch.x)[m], np.asarray(ch.y)[m], np.asarray(ch.z)[m]])
            rgb = None
            if "red" in ch.point_format.dimension_names:
                rgb = np.column_stack([ch.red[m], ch.green[m], ch.blue[m]]).astype(np.float32) / 65535.0
            yield xyz, rgb, datum


def _cell_lookup(raster, u, v):
    """(value, flat cell index or -1) of the raster at wall coordinates."""
    a = raster.array[0]
    r, c = raster.world_to_pixel(u, v)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.zeros(len(u), a.dtype)
    out[ok] = a[r[ok], c[ok]]
    idx = np.full(len(u), -1, np.int64)
    idx[ok] = r[ok].astype(np.int64) * a.shape[1] + c[ok]
    return out, idx


def _update_rear(r1: np.ndarray, r2: np.ndarray, idx: np.ndarray, w: np.ndarray) -> None:
    """Running two smallest depths per cell (flat arrays r1 <= r2) with the points (cell idx, depth w) added."""
    if not len(idx):
        return
    o = np.lexsort((w, idx))
    i_s, w_s = idx[o], w[o]
    first = np.r_[True, i_s[1:] != i_s[:-1]]
    pos = np.flatnonzero(first)
    cell = i_s[pos]
    nxt = np.minimum(pos + 1, len(i_s) - 1)
    b = np.where((pos + 1 < len(i_s)) & (i_s[nxt] == cell), w_s[nxt], np.inf)
    v = np.sort(np.column_stack([r1[cell], r2[cell], w_s[pos], b]), axis=1)
    r1[cell], r2[cell] = v[:, 0], v[:, 1]


def _fit_translation(X, Y, iters: int = 12):
    """t minimising the Tukey-weighted point-to-plane distance of Y + t to the surface X."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    c = X.mean(axis=0)
    X, Y = X - c, Y - c
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(X, dtype=np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    N = np.asarray(pc.normals)
    tree = cKDTree(X)
    t = np.zeros(3)
    for _ in range(iters):
        d, i = tree.query(Y + t, distance_upper_bound=0.5)
        ok = np.isfinite(d)
        if ok.sum() < 1000:
            return np.zeros(3)
        n = N[i[ok]]
        r = np.einsum("ij,ij->i", n, Y[ok] + t - X[i[ok]])
        w = np.where(np.abs(r) < TUKEY_M, (1 - (r / TUKEY_M) ** 2) ** 2, 0.0)
        A = n * w[:, None]
        t -= np.linalg.solve(A.T @ n + 1e-6 * np.eye(3), A.T @ r)
    return t


def rear_layer(uvw: np.ndarray, cell_m: float = CELL_M, layer_m: float = LAYER_M) -> np.ndarray:
    """Points within `layer_m` of the rearmost (lowest w; w points to the pit) per wall cell; the second
    rearmost sets the layer so one stray point does not, and cells with fewer than 3 points are dropped."""
    key = (np.floor(uvw[:, 0] / cell_m).astype(np.int64) * 10_000_000
           + np.floor(uvw[:, 1] / cell_m).astype(np.int64))
    o = np.lexsort((uvw[:, 2], key))
    _, st, cnt = np.unique(key[o], return_index=True, return_counts=True)
    rear = uvw[o, 2][np.minimum(st + 1, st + cnt - 1)]
    sel = np.zeros(len(uvw), bool)
    sel[o] = (uvw[o, 2] <= np.repeat(rear, cnt) + layer_m) & np.repeat(cnt >= 3, cnt)
    return sel


def rock_behind(path, cfg: dict, frame, cells, reference: np.ndarray, region=None, keep_every: int = KEEP_EVERY,
                progress=None, dtm=None) -> dict:
    """{"xyz", "rgb", "shift", "n_read", "n_cells", "reference_on_face", ...}: LiDAR rock points in the cells where
    `cells` (a raster in wall coordinates, nonzero = use LiDAR) is set, in the analysis datum and aligned on
    `reference` (the base analysis points: photogrammetry, or the LiDAR itself thinned, on bare rock). Only the rear
    layer inside the rock face's span is kept (face_span); `reference_on_face` is the same test for the `reference`
    points (None without a rear layer) -- the base cloud's ground and trunks behind the wall top, which the scanner
    sees directly, are not rock either. With a bare-earth `dtm` (in the analysis datum), the rear layer above the
    ground is left out: above the wall top the rearmost return is the back of a tree crown (cleaning.above_ground)."""
    from .facade.region import is_active, region_mask

    say = progress or (lambda s: None)
    target = heights.target(cfg)
    crs = cfg.get("crs")
    keep, depth, cell, bare, colours, n_read, n_kept = [], [], [], [], [], 0, 0
    ncell = cells.array[0].size
    r1, r2 = np.full(ncell, np.inf, np.float32), np.full(ncell, np.inf, np.float32)

    def prune():
        nonlocal keep, depth, cell, colours
        X, D, C = np.vstack(keep), np.concatenate(depth), np.concatenate(cell)
        ok = D <= r2[C] + PREFILTER_M
        keep, depth, cell = [X[ok]], [D[ok]], [C[ok]]
        if colours:
            colours = [np.vstack(colours)[ok]]
        return int(ok.sum())

    poly = (region or {}).get("map_polygon")
    box = (np.min(poly, axis=0) - 1.0, np.max(poly, axis=0) + 1.0) if poly else None
    for xyz, rgb, datum in _chunks(path, keep_every):
        n_read += len(xyz)
        if box is not None:                    # a section of a larger scan: skip the rest before the wall frame
            m = np.all((xyz[:, :2] >= box[0]) & (xyz[:, :2] <= box[1]), axis=1)
            xyz, rgb = xyz[m], (rgb[m] if rgb is not None else None)
        if target == heights.ORTHOMETRIC and datum != heights.ORTHOMETRIC:
            xyz = heights.to_orthometric(xyz, crs)
        elif target != heights.ORTHOMETRIC and datum == heights.ORTHOMETRIC:
            xyz = xyz.copy()
            xyz[:, 2] += heights.geoid_n_map(xyz[:, 0], xyz[:, 1], crs)     # back to ellipsoidal
        uvw = frame.to_wall(xyz)
        inside = region_mask(uvw, region, xyz) if is_active(region) else np.ones(len(uvw), bool)
        flag, idx = _cell_lookup(cells, uvw[:, 0], uvw[:, 1])
        m = inside & (flag > 0)
        _update_rear(r1, r2, idx[m], uvw[m, 2].astype(np.float32))
        m &= uvw[:, 2] <= r2[np.maximum(idx, 0)] + PREFILTER_M
        keep.append(xyz[m])
        depth.append(uvw[m, 2].astype(np.float32))
        cell.append(idx[m])
        if rgb is not None:
            colours.append(rgb[m])
        n_kept += int(m.sum())
        if n_kept > PRUNE_POINTS:
            n_kept = prune()
        b = inside & (flag == 0)
        bare.append(xyz[b][:: max(1, int(b.sum()) // 20_000 + 1)])
        say(f"라이다 읽는 중: {n_read * keep_every / 1e6:,.0f}백만 점")
    if keep:
        prune()
    K = np.column_stack([np.vstack(keep), np.zeros((sum(len(k) for k in keep), 3))]) if keep else np.zeros((0, 6))
    del depth, cell
    rgb = np.vstack(colours) if colours else None
    # align on the photogrammetry where both see bare rock
    B = np.vstack(bare) if bare else np.zeros((0, 3))
    ref = reference[:: max(1, len(reference) // 400_000)]
    shift = _fit_translation(ref, B[:: max(1, len(B) // 200_000)]) if len(B) > 5000 and len(ref) > 5000 else np.zeros(3)
    if len(K):
        K[:, :3] += shift
        K[:, 3:] = frame.to_wall(K[:, :3])
    sel = rear_layer(K[:, 3:]) if len(K) else np.zeros(0, bool)
    n_rear = int(sel.sum())
    n_ledge, on_face = 0, None
    if sel.any():                          # the rock face with its ledges, not the ground / trunks above or the floor
        bare_rock = reference[:: max(1, len(reference) // 4_000_000)]   # the bare rock beside: the face's height
        ctx = frame.to_wall(bare_rock) if len(bare_rock) else None
        span = face_span(K[sel, 3:], ctx)
        rock, steep = span(K[sel, 3:])
        idx = np.flatnonzero(sel)
        sel[idx[~rock]] = False
        n_ledge = int((rock & ~steep).sum())
        on_face, _ = span(frame.to_wall(reference))
    n_steep = int(sel.sum())
    if dtm and sel.any():                  # not in the air: the back of the tree crowns above the wall top
        from .discontinuity.cleaning import above_ground

        idx = np.flatnonzero(sel)
        air, _ = above_ground(K[idx, :3], dtm, check_datum=False)
        sel[idx[air]] = False
    return {"xyz": K[sel, :3], "rgb": rgb[sel] if rgb is not None and len(K) else None, "shift": shift,
            "n_read": n_read, "n_cells": int((cells.array[0] > 0).sum()), "n_rear": n_rear,
            "n_outside_face_dropped": n_rear - n_steep, "n_gentle_kept_on_face": n_ledge,
            "n_above_ground_dropped": n_steep - int(sel.sum()), "reference_on_face": on_face}


def _dip_grid(uvw: np.ndarray, res: float, smooth_m: float, band_m: float, origin=None):
    """(dip per cell (NaN = too little data), flat cell index per point, (H, W), mean w per cell) of the surface
    w(u, v) on a `res` grid (mean w per cell): a Gaussian-weighted local plane fit (sigma `smooth_m`, out to 2 sigma)
    over the neighbouring cells within `band_m` in depth of the cell -- so the ground far behind the wall top or the
    floor in front of the toe, at the same u, v, does not tilt the wall's plane. Where too few cells lie within the
    band to span the window (a surface stepping back more than `band_m` from cell to cell: near-horizontal ground),
    the plain fit over all neighbours."""
    u0, vt = origin if origin is not None else (uvw[:, 0].min(), uvw[:, 1].max())
    cu = np.floor((uvw[:, 0] - u0) / res).astype(np.int64)
    cv = np.floor((vt - uvw[:, 1]) / res).astype(np.int64)
    H, W = int(cv.max()) + 1, int(cu.max()) + 1
    key = cv * W + cu
    n = np.bincount(key, minlength=H * W).reshape(H, W)
    z = (np.bincount(key, weights=uvw[:, 2], minlength=H * W).reshape(H, W) / np.maximum(n, 1))
    ok = n > 0
    sig = max(0.5, smooth_m / res)
    r = int(np.ceil(2 * sig))
    P = np.pad(np.where(ok, z, np.nan), r, constant_values=np.nan)
    names = ("1", "u", "v", "uu", "vv", "uv", "z", "uz", "vz")
    gated, plain = ({k: np.zeros((H, W)) for k in names} for _ in range(2))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            g = np.exp(-(dx * dx + dy * dy) / (2 * sig * sig))
            dz = P[r + dy:r + dy + H, r + dx:r + dx + W] - z
            du, dv = dx * res, -dy * res
            for m, wgt in ((gated, np.where(np.abs(dz) <= band_m, g, 0.0)),     # NaN (no data) compares False
                           (plain, np.where(np.isfinite(dz), g, 0.0))):
                dzz = np.nan_to_num(dz)
                m["1"] += wgt
                m["u"] += wgt * du
                m["v"] += wgt * dv
                m["uu"] += wgt * du * du
                m["vv"] += wgt * dv * dv
                m["uv"] += wgt * du * dv
                m["z"] += wgt * dzz
                m["uz"] += wgt * du * dzz
                m["vz"] += wgt * dv * dzz

    def fit(m, good):
        A = np.stack([np.stack([m["uu"], m["uv"], m["u"]], -1), np.stack([m["uv"], m["vv"], m["v"]], -1),
                      np.stack([m["u"], m["v"], m["1"]], -1)], -2)
        rhs = np.stack([m["uz"], m["vz"], m["z"]], -1)
        coef = np.full((H, W, 3), np.nan)
        coef[good] = np.linalg.solve(A[good] + 1e-9 * np.eye(3), rhs[good][..., None])[..., 0]
        return coef

    def spread(m, k1, k2):
        with np.errstate(all="ignore"):
            return m[k2] / np.maximum(m["1"], 1e-9) - (m[k1] / np.maximum(m["1"], 1e-9)) ** 2

    # the gated fit where the cells within the band spread over the window in u and v; else (a surface stepping
    # back more than the band from cell to cell: near-horizontal ground) the plain fit
    min_var = (2 * res) ** 2
    use_g = ok & (gated["1"] > 3.0) & (spread(gated, "u", "uu") >= min_var) & (spread(gated, "v", "vv") >= min_var)
    use_p = ok & ~use_g & (plain["1"] > 1e-3)
    coef = np.where(use_g[..., None], fit(gated, use_g), fit(plain, use_p))
    fu, fv = coef[..., 0], coef[..., 1]
    with np.errstate(all="ignore"):
        dip = np.degrees(np.arccos(np.clip(np.abs(fv) / np.sqrt(fu ** 2 + fv ** 2 + 1.0), 0, 1)))
    return dip, key, (H, W), np.where(ok, z, np.nan)


def steep_surface(uvw: np.ndarray, res: float = 0.25, smooth_m: float = STEEP_SMOOTH_M,
                  min_dip: float = STEEP_MIN_DEG, band_m: float = BAND_M) -> np.ndarray:
    """Points whose surface, at the scale of `smooth_m`, dips `min_dip` or more (_dip_grid; dip =
    arccos(|dw/dv| / |(-dw/du, -dw/dv, 1)|) as in cleaning.flat_area). Too little data: steep."""
    dip, key, _, _ = _dip_grid(uvw, res, smooth_m, band_m)
    return np.nan_to_num(dip.ravel()[key], nan=90.0) >= min_dip


def face_span(uvw: np.ndarray, context: np.ndarray | None = None, res: float = 0.25,
              smooth_m: float = STEEP_SMOOTH_M, min_dip: float = STEEP_MIN_DEG, band_m: float = BAND_M,
              gap_m: float = FACE_GAP_M, setback_m: float = FACE_SETBACK_M, far_m: float = FACE_FAR_M,
              edge_m: float = FACE_EDGE_M, column_m: float = 1.0, passes: int = 2):
    """The rock face's span on the wall, from a rear layer `uvw` (wall coordinates) and `context` (more points of
    the same wall, as they are -- e.g. the bare rock beside the netting -- of which the rear layer per `res` cell is
    taken; only in the cells `uvw` does not cover), as a test for any points: returns f(q) -> (inside, steep) per
    point of q (wall coordinates).

    A wall rises from the floor at 70-90 deg with gentle steps and ledges on the way, and at its top turns into the
    gentle original ground surface (with the trees). The face's surface is the rear layer (seen from the pit nothing
    lies behind the rock; in front lie the netting, vegetation and, at the bend of a U, the side walls). Per
    `column_m` wide column of the wall, the face runs from its lowest steep
    (>= `min_dip`) surface up to the highest one reached without a gap of more than `gap_m` in height or a step
    back of more than `setback_m` in depth (a bench; not the hillside far behind the wall top). Where the wall top
    is a ridge, the scan's far edge and the trunks behind it lie at the same wall u, v as the top metres of the
    face, tens of metres behind it (Iksan east wall: from 6 m below the crest), and the rear layer there follows
    them; so the candidates more than `far_m` behind the face's deepest part in their column are left out and
    the span found again (`passes`; well past a bench's `setback_m`, so the walk's limit is not met by what stays
    of something just behind it). The local plane fit reaches 2 * `smooth_m` out, so at the crest and the toe it
    takes in the gentle ground beyond and the last metres of the face do not read steep (Iksan east wall: the top
    2-3 m); there the points up to `edge_m` above the span at the face's depth (within `band_m` of the steep
    surface at the span's top) are inside too. Inside: in the span by its cell and not more than `far_m` behind the
    face's deepest part in its column -- the face with its gentle parts (steps, inclined recesses); outside: what
    lies above the span (the original ground, the trunks and the scan's far edge behind it, steep or not), below
    its toe (the floor), far behind the face, or in a column with no steep surface. `steep`: the surface's
    large-scale dip in the point's cell >= `min_dip` (a cell without a dip counts as steep)."""
    ctx = context if context is not None and len(context) else None
    both = uvw if ctx is None else np.vstack([uvw, ctx])
    origin = (both[:, 0].min(), both[:, 1].max())
    del both
    cell = lambda q: (np.floor((origin[1] - q[:, 1]) / res).astype(np.int64) * 10_000_000  # noqa: E731
                      + np.floor((q[:, 0] - origin[0]) / res).astype(np.int64))
    k = max(1, int(round(column_m / res)))
    gap = max(1, int(round(gap_m / res)))
    for it in range(passes):
        allp = uvw
        if ctx is not None:                       # the context's rear layer, only in the cells uvw does not cover
            c = ctx[rear_layer(ctx, cell_m=res)]
            allp = np.vstack([uvw, c[~np.isin(cell(c), cell(uvw))]])
        dip, _, (H, W), wgrid = _dip_grid(allp, res, smooth_m, band_m, origin)
        steep_cell = dip >= min_dip                               # NaN: False (not counted for the span)
        # the face's height span per column (rows: 0 = top)
        nb = (W + k - 1) // k
        S = np.zeros((H, nb * k), bool)
        S[:, :W] = steep_cell
        Wd = np.full((H, nb * k), np.nan)
        Wd[:, :W] = np.where(steep_cell, wgrid, np.nan)
        S = S.reshape(H, nb, k).any(axis=2)
        with warnings.catch_warnings():                           # all-NaN rows: no steep cell there
            warnings.simplefilter("ignore", RuntimeWarning)
            Wd = np.nanmedian(Wd.reshape(H, nb, k), axis=2)          # depth of the steep surface per row, column
        inside = np.zeros((H, nb), bool)
        deep = np.full(nb, np.nan)                                # the face's deepest part per column
        tops = np.full(nb, -1)                                    # the span's top row
        for c in range(nb):
            rows = np.flatnonzero(S[:, c])
            if not len(rows):
                continue
            bottom = top = rows[-1]
            for rr in rows[-2::-1]:                               # upward
                if top - rr > gap or abs(Wd[rr, c] - Wd[top, c]) > setback_m:
                    break
                top = rr
            inside[top:bottom + 1, c] = True
            deep[c] = np.nanmin(Wd[top:bottom + 1, c])
            tops[c] = top
        if it == passes - 1:
            break
        def far(q):                                               # NaN (no face in the column): False
            cb = np.clip(np.floor((q[:, 0] - origin[0]) / res).astype(np.int64) // k, 0, nb - 1)
            return q[:, 2] < deep[cb] - far_m

        fu = far(uvw)
        fc = far(ctx) if ctx is not None else np.zeros(0, bool)
        if not fu.any() and not fc.any():
            break
        uvw = uvw[~fu]
        ctx = ctx[~fc] if ctx is not None else None

    def test(q):
        col = np.floor((q[:, 0] - origin[0]) / res).astype(np.int64)
        row = np.floor((origin[1] - q[:, 1]) / res).astype(np.int64)
        ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        rin, cin = row[ok], col[ok]
        on, steep = np.zeros(len(q), bool), np.ones(len(q), bool)
        cb = cin // k
        top = tops[cb]
        w = q[ok, 2]
        past = ((top >= 0) & (rin < top) & (rin >= top - int(round(edge_m / res)))      # just above the crest row
                & (np.abs(w - Wd[np.maximum(top, 0), cb]) <= band_m))
        on[ok] = (inside[rin, cb] | past) & ~(w < deep[cb] - far_m)
        steep[ok] = np.nan_to_num(dip[rin, cin], nan=90.0) >= min_dip
        return on, steep

    return test


def rock_surface(uvw: np.ndarray, context: np.ndarray | None = None, **kw) -> tuple[np.ndarray, np.ndarray]:
    """(rock, ledge) per point of a rear layer `uvw`: the points inside the rock face's span (face_span; `kw` its
    parameters) and, of those, the gentle ones (`ledge`: steps, inclined recesses); not what lies above the face
    or below its toe -- the gentle ground, and the tree trunks and the far edge of the scan behind the wall top,
    which seen from the pit look like a vertical surface."""
    rock, steep = face_span(uvw, context, **kw)(uvw)
    return rock, rock & ~steep


def lidar_cells(smask, plants_uv, all_uv, flat_bit: int, net_bit: int, veg_share: float = 0.5, grow_m: float = 0.25):
    """Raster (uint8, the surface mask's grid) of the cells to fill from LiDAR: netting, or mostly vegetation in
    the photogrammetry (`plants_uv` / `all_uv`: wall u, v of the removed green points / of all points), grown by
    `grow_m`; flat ground left out."""
    import cv2

    from .io.raster import Raster

    a = smask.array[0]
    H, W = a.shape

    def counts(uv):
        r, c = smask.world_to_pixel(uv[:, 0], uv[:, 1])
        r, c = np.asarray(r), np.asarray(c)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        return np.bincount(r[ok] * W + c[ok], minlength=H * W).reshape(H, W).astype(np.float32)

    veg = counts(plants_uv) > veg_share * np.maximum(counts(all_uv), 1)
    use = ((a & net_bit) > 0) | veg
    k = max(1, int(round(grow_m / abs(smask.transform.a)))) * 2 + 1
    use = cv2.dilate(use.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    use &= (a & flat_bit) == 0
    return Raster(array=use.astype(np.uint8)[None], transform=smask.transform, crs=None, nodata=0)
