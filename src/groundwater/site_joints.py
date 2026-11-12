"""Site-wide joint model: the joint faces of every section of a site (each section's saved analysis) as discs in
one 3D space, so how the discontinuities may run through the rock mass -- under the pit floor, behind and between
the walls -- can be looked at apart from the cliffs that exposed them.

* faces: the observed faces (area >= min_area), with their site-wide set (viz.site3d.site_sets: a set seen in two
  sections is one set) -- a disc of the face's own size (radius sqrt(area / pi)) at its centroid, normal to it;
* extension: the disc radius times `factor` -- an assumption, not an observation: exposed joint faces are cut
  by the excavation, joints persist further than they show (ISRM: persistence is the least known parameter);
* continuity: faces of one set that lie on one plane (normals within LINK_DEG, centroids within LINK_M of the
  common plane, within LINK_MAX_M of its largest face) -- possibly in different sections, on opposite walls --
  form a candidate continuous discontinuity, whose extent is that of its members. This is evidence the
  observation itself gives (a single large face also shows persistence, but only its own);
* sections: where the discs cut a horizontal plane (plan at elevation z) or a vertical plane along a line A -> B,
  with the ground (a bare-earth DTM: rock and soil below it, air above) for context.

Heights: orthometric (the sections' results are; KNGeoid18).
"""
from pathlib import Path

import numpy as np
import pandas as pd

LINK_DEG = 8.0
LINK_M = 1.0
LINK_MAX_M = 120.0
DISC_SEG = 32


def _pole(dip, dd):
    d, a = np.radians(np.asarray(dip, float)), np.radians(np.asarray(dd, float))
    return np.stack([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)], axis=-1)


def load_faces(site, sections=None, min_area: float = 5.0, kinds=None, variant: str = "") -> pd.DataFrame:
    """The faces of the sections' results with their site-wide set: section, set_id, group (site set index),
    group_label, x, y, z, nx, ny, nz, dip, dd, area, r0 (radius of a disc of that area), persistence, exposure
    (discontinuity.exposure: parallel to the excavated wall / flat / cross); only the `kinds` of exposure given
    (all by default). The sets are grouped on all faces either way. `variant`: viz.site3d.section_results."""
    from .discontinuity.exposure import classify_folder
    from .viz.site3d import section_results, site_sets

    chosen = [(n, d) for n, d in section_results(site, variant) if sections is None or n in sections]
    rows, sets_by = [], {}
    for name, d in chosen:
        p = classify_folder(d)
        s = pd.read_csv(d / "discontinuity_sets.csv")
        s["area"] = s["set_id"].map(p.groupby("set_id")["area_m2"].sum()).fillna(0)
        sets_by[name] = s
        p = p[(p["area_m2"] >= min_area) & (p["exposure"].isin(kinds) if kinds else True)]
        rows.append(pd.DataFrame({"section": name, "set_id": p["set_id"], "patch_id": p["patch_id"],
                                  "x": p["centroid_x"], "y": p["centroid_y"], "z": p["centroid_z"],
                                  "dip": p["dip"], "dd": p["dip_direction"], "area": p["area_m2"],
                                  "persistence": p["persistence_length_m"], "exposure": p["exposure"]}))
    if not rows:
        return pd.DataFrame()
    F = pd.concat(rows, ignore_index=True)
    g = site_sets(sets_by)
    F["group"] = [g.get((s, int(i)), (-1, ""))[0] for s, i in zip(F["section"], F["set_id"])]
    F["group_label"] = [g.get((s, int(i)), (-1, ""))[1] for s, i in zip(F["section"], F["set_id"])]
    n = _pole(F["dip"], F["dd"])
    F["nx"], F["ny"], F["nz"] = n[:, 0], n[:, 1], n[:, 2]
    F["r0"] = np.sqrt(F["area"] / np.pi)
    return F[F["group"] >= 0].reset_index(drop=True)


def continuity(F: pd.DataFrame, link_deg: float = LINK_DEG, link_m: float = LINK_M,
               link_max_m: float = LINK_MAX_M) -> pd.DataFrame:
    """Candidate continuous discontinuities: groups (>= 2 faces) of faces of one set lying on one plane. Seeded
    by the largest faces: a seed's plane takes every unassigned face of its set within `link_deg`, `link_m` of the
    plane and `link_max_m` of the seed, is refitted to them (twice) -- every member is near the common plane, no
    chain of small steps drifts across a stepped wall. One row per group: id, group, group_label, members (face
    indices), sections, n, x, y, z (centre), nx, ny, nz, dip, dd, span_m (largest distance between members),
    area_m2 (observed), rms_m (members' distance from the plane). Also writes F['plane'] (-1 = alone)."""
    F["plane"] = -1
    if len(F) < 2:
        return pd.DataFrame()
    P = F[["x", "y", "z"]].to_numpy()
    N = F[["nx", "ny", "nz"]].to_numpy()
    A = F["area"].to_numpy()
    G = F["group"].to_numpy()
    cos_t = np.cos(np.radians(link_deg))
    free = np.ones(len(F), bool)
    rows = []
    for seed in np.argsort(-A):
        if not free[seed]:
            continue
        c, n = P[seed], N[seed]
        m = np.array([seed])
        for _ in range(3):
            cand = free & (G == G[seed])
            near = cand & (np.abs(N @ n) >= cos_t) & (np.abs((P - c) @ n) <= link_m) &                 (np.linalg.norm(P - P[seed], axis=1) <= link_max_m)
            m = np.flatnonzero(near)
            if len(m) < 2:
                break
            w = A[m]
            c = (P[m] * w[:, None]).sum(axis=0) / w.sum()
            nn = (N[m] * np.sign(N[m] @ n)[:, None] * w[:, None]).sum(axis=0)
            n = nn / np.linalg.norm(nn)
        if len(m) < 2:
            continue
        free[m] = False
        if n[2] < 0:
            n = -n
        span = float(np.max(np.linalg.norm(P[m][:, None] - P[m][None], axis=2)))
        rows.append({"id": len(rows), "group": int(G[seed]), "group_label": F["group_label"].iloc[seed],
                     "members": m.tolist(), "sections": "·".join(sorted({x.split("_")[0] for x in F["section"].iloc[m]})),
                     "n": len(m), "x": c[0], "y": c[1], "z": c[2], "nx": n[0], "ny": n[1], "nz": n[2],
                     "dip": round(float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), 1),
                     "dd": round(float(np.degrees(np.arctan2(n[0], n[1])) % 360), 1), "span_m": round(span, 1),
                     "area_m2": round(float(A[m].sum()), 1),
                     "rms_m": round(float(np.sqrt(np.mean(((P[m] - c) @ n) ** 2))), 2)})
    C = pd.DataFrame(rows).sort_values("span_m", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    for k, r in C.iterrows():
        F.loc[r.members, "plane"] = k
    if len(C):
        C["id"] = np.arange(len(C))
    return C


def _basis(n):
    n = np.asarray(n, float)
    a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def disc(c, n, r, seg: int = DISC_SEG) -> np.ndarray:
    """(seg, 3) ring of a disc."""
    a, b = _basis(n)
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    return np.asarray(c, float) + r * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)


def plane_patch(F: pd.DataFrame, plane: pd.Series, factor: float = 1.0, seg: int = DISC_SEG) -> np.ndarray:
    """Outline (convex hull in the plane) of a candidate continuous discontinuity: the members' discs projected
    onto the common plane, each radius times `factor`."""
    from scipy.spatial import ConvexHull

    c = np.array([plane.x, plane.y, plane.z])
    n = np.array([plane.nx, plane.ny, plane.nz])
    a, b = _basis(n)
    pts = []
    for i in plane.members:
        q = np.array([F.x[i], F.y[i], F.z[i]])
        q = q - ((q - c) @ n) * n
        pts.append(disc(q, n, F.r0[i] * factor, seg))
    Q = np.vstack(pts) - c
    uv = np.column_stack([Q @ a, Q @ b])
    h = uv[ConvexHull(uv).vertices]
    return c + h[:, :1] * a + h[:, 1:] * b


def cut(centres, normals, radii, p0, m) -> np.ndarray:
    """(n, 2, 3) chords where discs (centres, unit normals, radii) cut the plane through p0 with unit normal m;
    NaN rows for discs it misses (or parallel)."""
    C, Nn, R = np.asarray(centres, float), np.asarray(normals, float), np.asarray(radii, float)
    m = np.asarray(m, float)
    d = np.cross(Nn, m)
    sin = np.linalg.norm(d, axis=1)
    out = np.full((len(C), 2, 3), np.nan)
    ok = sin > 1e-3
    d[ok] /= sin[ok, None]
    u = m - (Nn @ m)[:, None] * Nn                      # m within each disc's plane, |u| = sin
    delta = (C - p0) @ m                                # disc centre from the cutting plane
    q = C - (delta / np.maximum(sin, 1e-9) ** 2)[:, None] * u
    h = np.abs(delta) / np.maximum(sin, 1e-9)
    ok &= h < R
    half = np.sqrt(np.maximum(R ** 2 - h ** 2, 0))
    out[ok, 0] = q[ok] - half[ok, None] * d[ok]
    out[ok, 1] = q[ok] + half[ok, None] * d[ok]
    return out


def find_ground(site) -> Path | None:
    """A bare-earth DTM of the site (dtm / dem in the name, in the site folder or the repository's DEM(5m)),
    the finest first."""
    import rasterio

    from .site_folder import ROOT

    cands = [f for f in Path(site).glob("*.tif") if any(w in f.name.lower() for w in ("dtm", "dem"))]
    cands += sorted((ROOT / "DEM(5m)").glob("dem*.tif")) if (ROOT / "DEM(5m)").is_dir() else []
    best = []
    for f in cands:
        try:
            with rasterio.open(f) as d:
                if d.count == 1 and d.crs is not None and d.crs.is_projected:
                    best.append((abs(d.transform.a), f))
        except Exception:  # noqa: BLE001
            continue
    return min(best)[1] if best else None


def ground_grid(path, bounds, cell: float = 2.0):
    """(z (rows north first), transform) of the DTM over `bounds` (x0, y0, x1, y1) on a `cell` grid."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    x0, y0, x1, y1 = bounds
    W, H = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    t = Affine(cell, 0, x0, 0, -cell, y1)
    z = np.full((H, W), np.nan, np.float32)
    with rasterio.open(path) as d:
        reproject(rasterio.band(d, 1), z, dst_transform=t, dst_crs=d.crs, resampling=Resampling.average,
                  src_nodata=d.nodata, dst_nodata=np.nan)
    return z, t


def ground_at(z, t, x, y) -> np.ndarray:
    c = np.floor((np.asarray(x) - t.c) / t.a).astype(int)
    r = np.floor((np.asarray(y) - t.f) / t.e).astype(int)
    ok = (r >= 0) & (r < z.shape[0]) & (c >= 0) & (c < z.shape[1])
    out = np.full(len(c), np.nan)
    out[ok] = z[r[ok], c[ok]]
    return out
