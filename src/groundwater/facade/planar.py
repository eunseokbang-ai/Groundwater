"""Wall-parallel ("planar projection") orthomosaic + DEM, e.g. Metashape's planar export.

These rasters live in the projection plane's own coordinates: x along the plane, y up the plane, DEM value
= distance along the plane normal. The GeoTIFF does not say where that plane is in the world, so it is
registered once to the project point cloud (rigid transform, ICP) and the transform is stored next to the
rasters (planar_to_world.json). When Metashape's own projection is exported next to the rasters
(scripts/metashape_export_planar_projection.py -> *_planar_projection.json: the 4x4 matrix from geocentric
EPSG:4978 to the plane) that exact definition is used instead of ICP. With it the rasters become a textured 3D surface of the face (for the
grouting scene) and, later, a sharper facade for trace detection (2-3 mm straight from the photos instead of
rasterised points).
"""
import json
from pathlib import Path

import numpy as np

REG_FILE = "planar_to_world.json"


VALUE_WORDS = ("temperature", "온도")        # single-band mosaics of a measured value, next to the DEM


def find_planar_pair(folder: str | Path):
    """(dem, ortho) GeoTIFFs in a folder whose CRS is not a projected map CRS (local/geocentric plane)."""
    import rasterio

    folder = Path(folder)
    if not folder.is_dir():
        return None, None
    dem = ortho = None
    for f in sorted(folder.glob("*.tif")):
        if any(w in f.name.lower() for w in VALUE_WORDS):
            continue                                     # a single-band value mosaic (temperature), not the DEM
        try:
            with rasterio.open(f) as ds:
                planar = ds.crs is None or not ds.crs.is_projected
                if not planar:
                    continue
                if ds.count == 1 and dem is None:
                    dem = f
                elif ds.count >= 3 and ortho is None:
                    ortho = f
        except Exception:  # noqa: BLE001
            continue
    return dem, ortho


def _grid(dem_path, step_m: float):
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(dem_path) as ds:
        k = max(1, int(round(step_m / abs(ds.transform.a))))
        h, w = max(1, ds.height // k), max(1, ds.width // k)
        d = ds.read(1, out_shape=(h, w), resampling=Resampling.nearest).astype(np.float64)
        t = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        nod = ds.nodata
    ok = np.isfinite(d) & (d > -1000)
    if nod is not None:
        ok &= d != nod
    return d, ok, t


def local_points(dem_path, step_m: float = 0.05) -> np.ndarray:
    import rasterio

    d, ok, t = _grid(dem_path, step_m)
    rr, cc = np.nonzero(ok)
    x, y = rasterio.transform.xy(t, rr, cc)
    return np.column_stack([x, y, d[ok]])


def register(dem_path, world_points: np.ndarray, frame, step_m: float = 0.05) -> dict:
    """Rigid local -> world transform by ICP, starting from the wall frame (plane x along the wall, y up,
    normal out of the face; the four sign combinations are tried and the best kept)."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    L = local_points(dem_path, step_m)
    L = L[::max(1, len(L) // 400_000)]
    origin = np.median(world_points, axis=0)
    W = world_points[::max(1, len(world_points) // 500_000)] - origin
    pw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(W, dtype=np.float64)))
    pw.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    pl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(L, dtype=np.float64)))
    along, nrm = frame.axes_at(frame.length / 2)
    tree = cKDTree(W)
    best = None
    # only proper rotations (det +1): the two other sign pairs are mirror images, which ICP would keep
    d0 = float(np.sign(np.linalg.det(np.column_stack([along, [0.0, 0.0, 1.0], nrm])))) or 1.0
    for sx in (1, -1):
        for sd in (sx * d0,):
            R = np.column_stack([sx * along, [0.0, 0.0, 1.0], sd * nrm])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.median(W, axis=0) - np.median(L @ R.T, axis=0)
            for thr in (3.0, 1.0, 0.3, 0.1, 0.05):
                reg = o3d.pipelines.registration.registration_icp(
                    pl, pw, thr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
                T = reg.transformation
            dist, _ = tree.query(L @ T[:3, :3].T + T[:3, 3])
            med = float(np.median(dist))
            if best is None or med < best["median_m"]:
                best = {"median_m": med, "p75_m": float(np.percentile(dist, 75)), "T": T.copy()}
    T = best["T"]
    T[:3, 3] += origin
    return {"matrix": T.tolist(), "median_residual_m": round(best["median_m"], 4),
            "p75_residual_m": round(best["p75_m"], 4), "dem": Path(dem_path).name}


def save_registration(folder, reg: dict) -> Path:
    path = Path(folder) / REG_FILE
    path.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    return path


def load_registration(folder, crs: str | None = None) -> dict | None:
    """The plane -> map transform: from Metashape's exported projection when present (exact) and a map CRS
    is given, else the stored ICP registration."""
    folder = Path(folder)
    if crs:
        exact = registration_from_metashape(folder, crs)
        if exact is not None:
            return exact
    path = folder / REG_FILE
    if not path.exists():
        return None
    reg = json.loads(path.read_text(encoding="utf-8"))
    # an ICP result is in the height datum of the cloud it was fitted to: not reused for another one
    if crs and reg.get("crs") and reg.get("source") != "metashape" and reg["crs"] != crs:
        return None
    return reg


def registration_from_metashape(folder, crs: str, step_m: float = 0.2) -> dict | None:
    """Exact plane -> `crs` transform from a *_planar_projection.json (Metashape OrthoProjection matrix,
    geocentric -> plane). Plane points are taken to geocentric, to lon/lat/ellipsoidal height and to the map
    projection; over a wall a rigid 4x4 reproduces that to well under a millimetre, so it is stored in the same
    form as the ICP result (cached in planar_to_world.json with source "metashape")."""
    from rasterio.warp import transform

    folder = Path(folder)
    files = sorted(folder.glob("*_planar_projection.json"))
    if not files:
        return None
    cache = folder / REG_FILE
    src = files[0]
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if (c.get("source") == "metashape" and c.get("crs") == crs and c.get("rigid_fit_max_error_m", 1.0) < 0.01
                and c.get("projection_file_mtime") == int(src.stat().st_mtime)):
            return c
    pj = json.loads(src.read_text(encoding="utf-8"))
    # the plane products: a chunk can also hold top-down (identity-matrix) DEMs listed as "Planar"
    prods = [q for q in pj.get("products", []) if q.get("matrix") and "Planar" in str(q.get("projection_type"))
             and not np.allclose(np.array(q["matrix"], float), np.eye(4))]
    dem, _ = find_planar_pair(folder)
    if not prods or dem is None:
        return None
    import rasterio

    with rasterio.open(dem) as ds:                          # the product this DEM was exported from: its resolution
        px = abs(ds.transform.a)
    pick = min(prods, key=lambda q: (q["kind"] != "dem", abs(float(q.get("resolution") or 0) - px)))
    M = np.array(pick["matrix"], float)
    L = local_points(dem, step_m)
    L = L[::max(1, len(L) // 50_000)]
    ecef = (L - M[:3, 3]) @ M[:3, :3]                      # plane = R ecef + t
    from .. import heights

    lon, lat, h = transform("EPSG:4978", "EPSG:4979", *(ecef[:, i].tolist() for i in range(3)))
    x, y = transform("EPSG:4326", heights.base_crs(crs), lon, lat)
    h = np.asarray(h, float)
    if heights.is_orthometric_crs(crs):                  # ellipsoidal -> orthometric (해발고도)
        h = h - heights.geoid_n(lon, lat)
    W = np.column_stack([x, y, h])
    cl, cw = L.mean(axis=0), W.mean(axis=0)
    U, _, Vt = np.linalg.svd((L - cl).T @ (W - cw))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    t = cw - R @ cl
    err = np.linalg.norm(L @ R.T + t - W, axis=1)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    reg = {"matrix": T.tolist(), "source": "metashape", "crs": crs, "projection_file": src.name,
           "projection_file_mtime": int(src.stat().st_mtime), "rigid_fit_max_error_m": round(float(err.max()), 5),
           "median_residual_m": 0.0, "dem": dem.name}
    cache.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    return reg


def _correction_by_chainage(frame, correction: dict | None):
    """f(u) -> (N, 3) world shifts from a local correction {knots_xy: [[x, y], ...], t: [[dx, dy, dz], ...]}
    (knots on the wall, so the correction outlives an edit of the wall line), or None."""
    if not correction or not correction.get("knots_xy"):
        return None
    xy = np.asarray(correction["knots_xy"], float)
    t = np.asarray(correction["t"], float)
    ku = frame.to_wall(np.column_stack([xy, np.zeros(len(xy))]))[:, 0]
    o = np.argsort(ku)
    ku, t = ku[o], t[o]
    return lambda u: np.column_stack([np.interp(np.asarray(u, float), ku, t[:, j]) for j in range(3)])


def textured_surface(dem_path, ortho_path, T, step_m: float = 0.08):
    """(vertices world (N, 3), faces (M, 3), colours (N, 3) uint8) of the DEM as a grid mesh."""
    import rasterio
    from rasterio.enums import Resampling

    d, ok, t = _grid(dem_path, step_m)
    h, w = d.shape
    rr, cc = np.mgrid[0:h, 0:w]
    x, y = rasterio.transform.xy(t, rr.ravel(), cc.ravel())
    L = np.column_stack([x, y, d.ravel()])
    T = np.asarray(T)
    V = L @ T[:3, :3].T + T[:3, 3]
    col = np.full((h * w, 3), 128, np.uint8)
    if ortho_path:
        with rasterio.open(ortho_path) as ds:
            img = ds.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        col = np.moveaxis(img, 0, -1).reshape(-1, 3)
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    okf = ok.ravel()
    t1 = np.column_stack([a.ravel(), c.ravel(), b.ravel()])
    t2 = np.column_stack([b.ravel(), c.ravel(), e.ravel()])
    faces = np.vstack([t1, t2])
    faces = faces[okf[faces].all(axis=1)]
    # drop long thin triangles across depth jumps (overhang edges, tree crowns)
    edge = np.max([np.linalg.norm(V[faces[:, i]] - V[faces[:, (i + 1) % 3]], axis=1) for i in range(3)], axis=0)
    faces = faces[edge < 4 * step_m]
    used = np.unique(faces)
    remap = np.full(h * w, -1)
    remap[used] = np.arange(len(used))
    return V[used], remap[faces], col[used]


# --- the planar rasters as the facade source ------------------------------------------------------------

def _to_local(T, world):
    T = np.asarray(T)
    return (world - T[:3, 3]) @ T[:3, :3]          # world = R @ local + t  ->  local = R^T (world - t)


def _to_world(T, local):
    T = np.asarray(T)
    return local @ T[:3, :3].T + T[:3, 3]


def facade_rgb_from_planar(result, ortho_path, T, chunk_rows: int = 400):
    """Re-colour the facade image (wall frame u, v) from the planar orthomosaic: every facade pixel with a
    depth is lifted to 3D, taken into the plane and sampled there. Pixels outside the ortho keep their
    point-cloud colour. Returns (Raster, share of pixels re-coloured)."""
    import rasterio

    from ..io.raster import Raster

    fr, fd = result.facade_rgb, result.facade_depth.array[0]
    tf = fr.transform
    out = fr.array.copy()
    from rasterio.enums import Resampling

    with rasterio.open(ortho_path) as ds:
        # read at about the facade pixel size (averaged), not the full 2-3 mm mosaic (~0.8 GB) to be sampled
        k = max(1.0, abs(tf.a) / abs(ds.transform.a))
        h2, w2 = max(1, int(ds.height / k)), max(1, int(ds.width / k))
        img = ds.read([1, 2, 3], out_shape=(3, h2, w2), resampling=Resampling.average)
        alpha = ds.read(4, out_shape=(h2, w2), resampling=Resampling.nearest) if ds.count >= 4 else None
        inv = ~(ds.transform * ds.transform.scale(ds.width / w2, ds.height / h2))
    H, W = fd.shape
    cols = np.arange(W)
    done = 0
    for r0 in range(0, H, chunk_rows):
        rows = np.arange(r0, min(r0 + chunk_rows, H))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        w = fd[r0:r0 + len(rows)]
        ok = np.isfinite(w)
        if not ok.any():
            continue
        u = tf.c + (cc[ok] + 0.5) * tf.a
        v = tf.f + (rr[ok] + 0.5) * tf.e
        loc = _to_local(T, result.wall_frame.to_world(np.column_stack([u, v, w[ok]])))
        px, py = inv * (loc[:, 0], loc[:, 1])
        px, py = np.floor(px).astype(int), np.floor(py).astype(int)
        inside = (px >= 0) & (px < img.shape[2]) & (py >= 0) & (py < img.shape[1])
        if alpha is not None:
            inside[inside] &= alpha[py[inside], px[inside]] > 0
        tr, tc = rr[ok][inside], cc[ok][inside]
        out[:, tr, tc] = img[:, py[inside], px[inside]]
        done += int(inside.sum())
    return Raster(array=out, transform=tf, crs=None, nodata=None), done / max(1, int(np.isfinite(fd).sum()))


def wall_image(result, ortho_path, T, u_range=None, v_range=None, max_px: int = 8000, res: float | None = None,
               correction: dict | None = None, values: bool = False):
    """The planar orthomosaic itself, resampled onto the wall frame (chainage x elevation) for display, as fine as
    it is (2-3 mm) or as `max_px` on the longer side allows (or exactly `res` m per pixel: two mosaics on one
    grid): each wall pixel is lifted to 3D with the facade depth (gaps take the nearest depth, so the mosaic is
    not cut where the cloud is thin), taken into the plane and sampled. `correction`: a local placement
    correction of this mosaic (facade.thermal: world shifts by chainage, added to the rigid T). Returns (RGB
    uint8, top row = highest elevation, white where the mosaic has no data; extent [u0, u1, v0, v1]; pixel size).
    `values`: a single-band mosaic of a measured value (temperature): float32 (H, W), NaN where it has no data."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from scipy import ndimage

    fd = result.facade_depth
    d = fd.array[0]
    tf = fd.transform
    U0, U1 = tf.c, tf.c + d.shape[1] * tf.a
    V1, V0 = tf.f, tf.f + d.shape[0] * tf.e
    u0, u1 = (max(U0, u_range[0]), min(U1, u_range[1])) if u_range else (U0, U1)
    v0, v1 = (max(V0, v_range[0]), min(V1, v_range[1])) if v_range else (V0, V1)
    shift = _correction_by_chainage(result.wall_frame, correction)

    def world_of(uvw):                                   # where this mosaic placed what is at uvw on the wall
        X = result.wall_frame.to_world(uvw)
        return X if shift is None else X - shift(uvw[:, 0])

    with rasterio.open(ortho_path) as ds:
        native = abs(ds.transform.a)
        res = res or max(native, (u1 - u0) / max_px, (v1 - v0) / max_px)
        W, H = max(1, int((u1 - u0) / res)), max(1, int((v1 - v0) / res))
        # facade depth for every wall pixel (nearest facade pixel; gaps filled with the nearest depth)
        valid = np.isfinite(d)
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        dfill = d[idx[0], idx[1]]
        uu = u0 + (np.arange(W) + 0.5) * res
        vv = v1 - (np.arange(H) + 0.5) * res
        fc = np.clip(((uu - tf.c) / tf.a).astype(int), 0, d.shape[1] - 1)
        fr = np.clip(((vv - tf.f) / tf.e).astype(int), 0, d.shape[0] - 1)
        # the mosaic window covering the wall area, read at the display pixel size
        corners = []
        for vr in (fr[0], fr[-1], fr[len(fr) // 2]):
            for uc_, cc in ((uu[0], fc[0]), (uu[-1], fc[-1]), (uu[len(uu) // 2], fc[len(fc) // 2])):
                corners.append([uc_, tf.f + (vr + 0.5) * tf.e, dfill[vr, cc]])
        loc = _to_local(T, world_of(np.array(corners)))
        inv = ~ds.transform
        px, py = inv * (loc[:, 0], loc[:, 1])
        m = 0.1 * max(np.ptp(px), np.ptp(py)) + 50
        c0, c1 = int(max(0, px.min() - m)), int(min(ds.width, px.max() + m))
        r0, r1 = int(max(0, py.min() - m)), int(min(ds.height, py.max() + m))
        out = np.full((H, W), np.nan, np.float32) if values else np.full((H, W, 3), 255, np.uint8)
        if c1 <= c0 or r1 <= r0:
            return out, [u0, u1, v0, v1], res
        k = max(1.0, res / native)
        h2, w2 = max(1, int((r1 - r0) / k)), max(1, int((c1 - c0) / k))
        win = Window(c0, r0, c1 - c0, r1 - r0)
        if values:                                       # no-data cells stay out of the averaging
            band = ds.read(1, window=win, out_shape=(h2, w2), resampling=Resampling.average, masked=True)
            img = band.filled(np.nan).astype(np.float32)[None]
            img[~np.isfinite(img) | (img < -1000)] = np.nan
            if ds.count >= 2:                            # Metashape's coverage band: 0 = no data (value 0 there)
                cover = ds.read(2, window=win, out_shape=(h2, w2), resampling=Resampling.nearest)
                img[0][cover <= 0] = np.nan
            alpha = None
        else:
            img = ds.read([1, 2, 3], window=win, out_shape=(3, h2, w2), resampling=Resampling.average)
            alpha = ds.read(4, window=win, out_shape=(h2, w2), resampling=Resampling.nearest) if ds.count >= 4 else None
        wt = ds.window_transform(win)
        inv2 = ~(wt * wt.scale((c1 - c0) / w2, (r1 - r0) / h2))
    for a in range(0, H, 500):                           # rows in blocks: tens of millions of pixels at most
        rows = np.arange(a, min(a + 500, H))
        rr, cc = np.meshgrid(rows, np.arange(W), indexing="ij")
        w = dfill[fr[rr], fc[cc]]
        loc = _to_local(T, world_of(np.column_stack([uu[cc].ravel(), vv[rr].ravel(), w.ravel()])))
        qx, qy = inv2 * (loc[:, 0], loc[:, 1])
        qx, qy = np.floor(qx).astype(int), np.floor(qy).astype(int)
        ok = (qx >= 0) & (qx < img.shape[2]) & (qy >= 0) & (qy < img.shape[1])
        if alpha is not None:
            ok[ok] &= alpha[qy[ok], qx[ok]] > 0
        if values:
            blk = out[a:a + len(rows)].reshape(-1)
            blk[ok] = img[0, qy[ok], qx[ok]]
            out[a:a + len(rows)] = blk.reshape(len(rows), W)
            continue
        blk = out[a:a + len(rows)].reshape(-1, 3)
        blk[ok] = np.moveaxis(img[:, qy[ok], qx[ok]], 0, -1)
        out[a:a + len(rows)] = blk.reshape(len(rows), W, 3)
    return out, [u0, u1, v0, v1], res


def detect_traces_planar(result, dem_path, ortho_path, T, resolution_m: float = 0.01, exg_threshold: float = 0.06,
                         veg_on: bool = True, **trace_params):
    """Traces found on the planar orthomosaic + DEM (sharper than the point-cloud facade), returned in the
    wall frame (u, v) with world end points, ready for families / cross-check like any facade trace."""
    import rasterio
    from rasterio.enums import Resampling

    from ..discontinuity.trace_mapping import extract_traces
    from ..discontinuity.vegetation import vegetation_mask
    from ..io.raster import Raster

    with rasterio.open(dem_path) as ds:
        k = max(1.0, resolution_m / abs(ds.transform.a))
        h, w = int(ds.height / k), int(ds.width / k)
        d = ds.read(1, out_shape=(h, w), resampling=Resampling.average, masked=True).filled(np.nan).astype(np.float32)
        t = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
    d[d < -1000] = np.nan
    with rasterio.open(ortho_path) as ds:
        bands = [1, 2, 3, 4] if ds.count >= 4 else [1, 2, 3]
        img = ds.read(bands, out_shape=(len(bands), h, w), resampling=Resampling.average)
    valid = np.isfinite(d)
    if len(bands) == 4:
        valid &= img[3] > 0
    if veg_on:
        valid &= ~vegetation_mask(np.moveaxis(img[:3], 0, -1), exg_threshold, scale=255.0)
    raster = Raster(array=img[:3], transform=t, crs=None, nodata=None)
    params = {k_: v for k_, v in trace_params.items() if k_ != "resolution_m"}
    traces = extract_traces(raster, valid_mask=valid, depth=d, resolution_m=resolution_m, **params)

    frame, fr = result.wall_frame, result.facade_rgb
    kept = []
    for tr in traces:
        ends = []
        for x, y in tr.coords:
            r_, c_ = rasterio.transform.rowcol(t, x, y)
            win = d[max(r_ - 2, 0):r_ + 3, max(c_ - 2, 0):c_ + 3]
            if not np.isfinite(win).any():
                break
            ends.append([x, y, float(np.nanmedian(win))])
        if len(ends) != 2:
            continue
        xyz = _to_world(T, np.array(ends))
        uvw = frame.to_wall(xyz)
        tr.coords = uvw[:, :2]
        tr.length_m = float(np.linalg.norm(tr.coords[1] - tr.coords[0]))
        tr.xyz = xyz
        if fr is not None:
            rows, cols = fr.world_to_pixel(tr.coords[:, 0], tr.coords[:, 1])
            tr.pixel_coords = np.column_stack([rows, cols]).astype(float)
        kept.append(tr)
    return kept
