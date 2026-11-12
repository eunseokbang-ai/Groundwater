"""Photo-textured tiled model (Cesium 3D Tiles, e.g. a Metashape "tiled model" export) as the base of the
virtual grouting scene.

A tiled model keeps the full photo texture (Ulsan: 2.7 mm/px over the whole 3D surface, also on overhangs and
side faces the wall-parallel orthophoto stretches) in a level-of-detail tree, so a browser streams only what
is on screen -- Plotly cannot texture a mesh at all. Its geometry is simplified (Ulsan leaves: 8.5 cm edges vs
3 cm in the OBJ), so the analysis keeps using the point cloud / mesh; the tiled model is for looking.

* `TileArchive` reads a .3tz (zip: tileset.json + glb tiles) or a folder in place, for the local web server;
* `scene_json` writes the holes, the joint planes and the cut-away frame in geocentric coordinates (EPSG:4978)
  for the CesiumJS page (wall_digitizer/tiles3d.html). Map heights are taken as ellipsoidal heights, as in
  the Metashape export (checked: tiles and analysis cloud agree to ~1 cm).
"""
import json
import threading
import zipfile
from pathlib import Path

import numpy as np

SCENE_FILE = "grouting_scene_tiles.json"
_archives: dict = {}
_lock = threading.Lock()


def find_tileset(path) -> Path | None:
    """A .3tz archive or a folder holding tileset.json (the path itself or inside it)."""
    p = Path(str(path).strip().strip('"'))
    if p.is_file() and p.suffix.lower() == ".3tz":
        return p
    if p.is_file() and p.name == "tileset.json":
        return p.parent
    if p.is_dir():
        if (p / "tileset.json").exists():
            return p
        tz = sorted(p.glob("*.3tz"), key=lambda f: -f.stat().st_size)
        if tz:
            return tz[0]
    return None


class TileArchive:
    """Files of a tileset, from a .3tz zip or a folder."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None

    def read(self, rel: str) -> bytes | None:
        rel = rel.lstrip("/")
        if self.zip is not None:
            try:
                return self.zip.read(rel)
            except KeyError:
                return None
        f = (self.path / rel).resolve()
        return f.read_bytes() if self.path.resolve() in f.parents and f.is_file() else None


def archive(path) -> TileArchive:
    key = str(Path(path).resolve())
    with _lock:
        if key not in _archives:
            _archives[key] = TileArchive(Path(path))
        return _archives[key]


def _to_ecef(xyz: np.ndarray, crs: str) -> np.ndarray:
    from rasterio.warp import transform

    from .. import heights

    xyz = np.asarray(xyz, float)
    lon, lat = transform(heights.base_crs(crs), "EPSG:4326", xyz[:, 0].tolist(), xyz[:, 1].tolist())
    h = xyz[:, 2]
    if heights.is_orthometric_crs(crs):                  # orthometric -> ellipsoidal for the geocentric scene
        h = h + heights.geoid_n(lon, lat)
    X, Y, Zc = transform("EPSG:4979", "EPSG:4978", lon, lat, np.asarray(h, float).tolist())
    return np.column_stack([X, Y, Zc])


SKY_MARGIN_M = 15.0      # the sky cut: this far above the highest real surface (LiDAR top, trees included)


def sky_clip(crs: str, xy, height: float, lo: float, hi: float) -> dict:
    """A horizontal cut for the CesiumJS page: everything of the tiled model above `height` (the analysis height
    datum) is not drawn -- floating noise in the sky (wrong depths against the sky; Iksan whole-site model 2026-10:
    a blob at 250-272 m over a 186 m hill). Sent as a point on the plane at `height`, the local up vector and
    the slider range."""
    x, y = float(xy[0]), float(xy[1])
    p0 = _to_ecef(np.array([[x, y, height]]), crs)[0]
    p1 = _to_ecef(np.array([[x, y, height + 1.0]]), crs)[0]
    up = (p1 - p0) / np.linalg.norm(p1 - p0)
    return {"p": p0.round(3).tolist(), "up": up.round(9).tolist(), "h": round(float(height), 1),
            "min": round(float(lo), 0), "max": round(float(hi), 0)}


def ground_top(site=None, points=None) -> float | None:
    """The highest real surface: the top of a LiDAR scan of the site (its section index, trees included), else
    the highest of `points`."""
    if site is not None:
        from .. import lidar_section as LS
        from ..site_folder import scan

        tops = []
        for f in scan(site)["lidar"]:
            idx = LS.load_index(f)
            if idx is not None:
                tops.append(float(np.nanmax(idx["plan_z"])))
        if tops:
            return max(tops)
    if points is not None and len(points):
        return float(np.percentile(np.asarray(points)[:, 2], 99.9))
    return None


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def scene_json(result, holes, sets, tiles_path, joints=None, cov=None, title: str = "",
               joint_shade: str = "seepage", smap=None, overlay_folder=None) -> dict:
    """Everything the CesiumJS page draws, in ECEF metres."""
    from .grouting3d import _set_color
    from .joints3d import joint_groups

    from .. import heights

    crs = heights.reg_crs(result.config) or "EPSG:5187"
    frame = result.wall_frame
    names = list(dict.fromkeys(r.name for r in sets.itertuples()))

    def colour_of(name):
        return "#9ca3af" if name not in names else _set_color(name, names.index(name))

    out = {"title": title, "tiles": str(tiles_path), "crs": crs, "holes": [], "joint_groups": [], "missed": []}
    if holes:
        ends = np.array([[*h["collar"], *(h["collar"] + h["length"] * h["dir"])] for h in holes])
        e0, e1 = _to_ecef(ends[:, :3], crs), _to_ecef(ends[:, 3:], crs)
        out_pts = [h["collar"] + h["breakout_m"] * h["dir"] if h.get("breakout_m") is not None else None for h in holes]
        for i, h in enumerate(holes):
            hits = cov["per_hole"][i] if cov else []
            label = (f"공 {i + 1}: {h['joint']} 겨냥, 길이 {h['length']:.1f} m" if "target" in h
                     else f"공 {i + 1}: 측점 {h['u']:.1f} m, 길이 {h['length']:.0f} m")
            if cov:
                label += f", 지나는 절리 {len(hits)}개"
            rec = {"a": e0[i].round(3).tolist(), "b": e1[i].round(3).tolist(), "label": label}
            if out_pts[i] is not None:
                rec["out"] = _to_ecef(out_pts[i][None], crs)[0].round(3).tolist()
            if hits:
                P = np.array([h["collar"] + t * h["dir"] for _, t in hits])
                rec["hits"] = _to_ecef(P, crs).round(3).ravel().tolist()
            out["holes"].append(rec)
    if joints:
        for g in joint_groups(joints, colour_of, joint_shade):
            polys, cols = [], []
            for k, i in enumerate(g["idx"]):
                j = joints[i]
                ring = j.poly if j.source == "trace" else j.poly[1:]       # outline in order (fan centre first for traces)
                polys.append(_to_ecef(ring, crs).round(3).ravel().tolist())
                cols.append(_hex(g["rgb"][k]))
            out["joint_groups"].append({"label": g["label"], "kind": g["kind"], "colour": colour_of(g["name"]),
                                        "polys": polys, "colours": cols})
        if cov is not None:
            for i, (hi, hit) in enumerate(zip(cov["high"], cov["hit"])):
                if hi and not hit:
                    j = joints[i]
                    ring = np.vstack([j.poly, j.poly[:1]]) if j.source == "trace" else np.vstack([j.poly[1:], j.poly[1:2]])
                    out["missed"].append(_to_ecef(ring, crs).round(3).ravel().tolist())
            out["coverage"] = {"high": cov["n_high"], "high_hit": cov["n_high_hit"]}
    # the mapped joint traces on the face, so the holes can be checked against the joints they aim at
    tr = [t for t in result.traces if t.xyz is not None]
    if tr:
        fam_of = {id(t): f.family_id for f in result.trace_families for t in f.members}
        E = _to_ecef(np.vstack([t.xyz for t in tr]), crs).round(3)
        high = []
        if smap is not None:
            for t in tr:
                um, vm = 0.5 * (t.coords[0] + t.coords[1])
                r_ = int((smap.v_max - vm) / smap.cell_m)
                c_ = int((um - smap.u_min) / smap.cell_m)
                ok = 0 <= r_ < smap.classes.shape[0] and 0 <= c_ < smap.classes.shape[1]
                high.append(bool(ok and smap.classes[r_, c_] == 2))
        else:
            high = [False] * len(tr)
        out["traces"] = [{"p": E[2 * i:2 * i + 2].ravel().tolist(), "high": high[i],
                          "family": fam_of.get(id(t)), "manual": t.source == "manual"} for i, t in enumerate(tr)]
    out["overlays"] = reference_overlays(result, overlay_folder) if overlay_folder else []
    # the wall frame every metre: chainage, a point on the wall line and the horizontal tangent (for cut planes)
    pts = np.asarray(result.point_cloud.points)
    uvw = frame.to_wall(pts[::max(1, len(pts) // 200_000)])
    u_lo, u_hi = np.percentile(uvw[:, 0], [0.5, 99.5])
    v_mid = float(np.median(uvw[:, 1]))
    us = np.arange(np.floor(u_lo), np.ceil(u_hi) + 1, 1.0)
    P = frame.to_world(np.column_stack([us, np.full(len(us), v_mid), np.zeros(len(us))]))
    E = _to_ecef(P, crs)
    # the wall's own horizontal tangent at every metre (cut planes on an arc are normal to the wall there,
    # not to the chord between the two cut chainages)
    along = np.array([frame.axes_at(float(u))[0] for u in us])
    Ta = _to_ecef(P + along, crs) - E
    Ta /= np.linalg.norm(Ta, axis=1, keepdims=True)
    out["wall"] = {"u": us.tolist(), "p": E.round(3).ravel().tolist(), "t": Ta.round(6).ravel().tolist()}
    # camera: in front of the middle of the wall, looking at it
    um = 0.5 * (u_lo + u_hi)
    _, outward = frame.axes_at(um)
    centre = frame.to_world(np.array([[um, v_mid, 0.0]]))[0]
    eye = centre + outward * 0.9 * (u_hi - u_lo) + np.array([0, 0, 0.15 * (u_hi - u_lo)])
    out["camera"] = {"eye": _to_ecef(eye[None], crs)[0].round(3).tolist(),
                     "target": _to_ecef(centre[None], crs)[0].round(3).tolist()}
    site = (result.config.get("paths") or {}).get("output_dir")
    top = ground_top(Path(site).parent if site else None, pts)
    if top is not None:                    # the wall's own top (+40 m for the trees) when no LiDAR scan is indexed
        top = top if ground_top(Path(site).parent if site else None) is not None else top + 40.0
        out["sky"] = sky_clip(crs, centre[:2], top + SKY_MARGIN_M, top - 20, top + 150)
    return out


def reference_overlays(result, folder, max_px: int = 2048, step_m: float = 0.05, offset_m: float = 0.03) -> list:
    """Reference images of a planar folder (reference_overlays.json) draped on the surface, in ECEF, with the
    image as a PNG data URI -- for the CesiumJS page."""
    import base64

    import cv2

    from ..facade import planar
    from ..facade import reference_overlays as ro

    data = ro.load(folder) if folder else None
    from .. import heights

    crs = heights.reg_crs(result.config) or "EPSG:5187"
    reg = planar.load_registration(folder, crs) if data else None
    if not data or reg is None:
        return []
    out = []
    for e in data["overlays"]:
        b = ro.footprint(folder, e)
        res = max(0.004, max(b[2] - b[0], b[3] - b[1]) / max_px)
        rgba = ro.render(folder, e, b, res)
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        mesh = ro.drape(folder, e, reg["matrix"], b, step_m, offset_m)
        if not ok or mesh is None:
            continue
        xyz, st, tris = mesh
        # camera straight in front of the image: along the plane normal (the planar DSM's +z, toward the viewer)
        T = np.asarray(reg["matrix"], float)
        centre = xyz.mean(axis=0)
        n = T[:3, 2] / np.linalg.norm(T[:3, 2])
        eye = centre + n * 1.2 * max(b[2] - b[0], b[3] - b[1])
        cam = _to_ecef(np.vstack([eye, centre]), crs).round(3)
        out.append({"name": e["name"], "kind": e["kind"], "eye": cam[0].tolist(), "target": cam[1].tolist(),
                    "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
                    "positions": _to_ecef(xyz, crs).round(3).ravel().tolist(),
                    "st": st.astype(np.float32).round(5).ravel().tolist(), "indices": tris.ravel().tolist()})
    return out


def write_scene(path, scene: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(scene, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path
