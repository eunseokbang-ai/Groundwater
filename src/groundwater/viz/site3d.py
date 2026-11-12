"""All sections of a site on the photo-textured tiled model of the whole site (CesiumJS page wall_digitizer/
tiles3d.html): the joint faces of every section, coloured by site-wide joint set (a set seen in two sections gets
one colour), the mapped traces and the seepage-potential cells, with a label per section.

Built from what each analysis saved (분석결과: discontinuity_sets / _patches.csv, classified_point_cloud.ply +
point_labels.npy, traces.csv + facade_depth.tif, seepage_cells.csv, wall_frame.json, analysis_config.yaml), so no
0.5 GB result has to be loaded. Heights: the results are orthometric (KNGeoid18), the tiled model ellipsoidal --
tiles3d._to_ecef adds the geoid height (Iksan 2026-10: whole-site Metashape model vs LiDAR, N = 24.5 m as expected).
"""
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_DEG = 20.0          # sets of different sections within this (poles, axial) are one site-wide set
FACE_OFFSET_M = 0.03      # faces drawn this far in front of the rock so the model texture does not hide them
MAX_RING = 24
SITE_COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
SEEP_COLOURS = {2: "#dc2626", 1: "#f59e0b"}
KIN_DRAPE_M = 1.0         # failure-mode zones (1 m cells) draped on a mesh this coarse in the site scene (0.1 m: 2 GB)
KIN_OFFSET_M = 0.3        # ... this far off the face, so the coarse mesh stays in front of the rock


def section_results(site, variant: str = "") -> list:
    """[(label, result folder)] of a site: each section's 분석결과 (구간/<name>/), then the site folder's own.
    `variant`: the suffix of another result folder of the same sections, e.g. "_벽면평행제외" for 분석결과_벽면평행제외
    (a run with other options); the labels stay the section names."""
    from .. import site_folder as S

    site = Path(site)
    out = []
    for name in S.sections(site):
        d = S.section_dir(site, name) / (S.OUTPUT_DIRNAME + variant)
        if (d / "discontinuity_patches.csv").exists():
            out.append((name, d))
    d = site / (S.OUTPUT_DIRNAME + variant)
    if (d / "discontinuity_patches.csv").exists():
        out.append(("현장 폴더", d))
    return out


def result_variants(site) -> list:
    """The suffixes of the result folders of a site's sections (and the site folder): "" (분석결과) first, then
    every 분석결과_<suffix> holding a saved analysis -- by how many sections have it, then by name."""
    from .. import site_folder as S

    site = Path(site)
    dirs = [S.section_dir(site, n) for n in S.sections(site)] + [site]
    count = {}
    for base in dirs:
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith(S.OUTPUT_DIRNAME) and (d / "discontinuity_patches.csv").exists():
                v = d.name[len(S.OUTPUT_DIRNAME):]
                count[v] = count.get(v, 0) + 1
    rest = sorted((v for v in count if v), key=lambda v: (-count[v], v))
    return ([""] if "" in count else []) + rest


def _pole(dip, dd):
    d, a = np.radians(dip), np.radians(dd)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def site_sets(sets_by_section: dict) -> dict:
    """{(section, set id): (site group index, label)}: sets of all sections grouped by orientation, largest
    first (a group's orientation is that of its largest set; the label names the sections it shows in)."""
    rows = sorted(((sec, int(r.set_id), float(r.mean_dip), float(r.mean_dip_direction), float(r.area))
                   for sec, df in sets_by_section.items() for r in df.itertuples()), key=lambda t: -t[4])
    groups, members, out = [], [], {}
    for sec, sid, dip, dd, _ in rows:
        p = _pole(dip, dd)
        k = next((i for i, (q, _) in enumerate(groups) if abs(p @ q) >= np.cos(np.radians(GROUP_DEG))), None)
        if k is None:
            groups.append((p, f"{dip:.0f}°/{dd:03.0f}°"))
            members.append([])
            k = len(groups) - 1
        if sec not in members[k]:
            members[k].append(sec)
        out[(sec, sid)] = k
    # named by orientation and the sections it shows in (e.g. "81°/265° · E·S·W"), not by a number
    names = [f"{o} · {'·'.join(s.split('_')[0] for s in m)}" for (_, o), m in zip(groups, members)]
    return {key: (k, names[k]) for key, k in out.items()}


def _frame(folder: Path):
    import json

    from ..facade.frame import WallFrame

    v = json.loads((folder / "wall_frame.json").read_text(encoding="utf-8"))["polyline"]
    return WallFrame(np.asarray(v, float), fillet=True)


def _crs(folder: Path) -> str:
    import yaml

    from .. import heights

    cfg = yaml.safe_load((folder / "analysis_config.yaml").read_text(encoding="utf-8"))
    return heights.reg_crs(cfg) or cfg.get("crs") or "EPSG:5186"


def face_rings(folder: Path, frame, min_area: float = 0.0) -> list:
    """[(set id, ring xyz)] per joint face: the convex hull of its points in its own plane (at most MAX_RING
    vertices), moved FACE_OFFSET_M out of the rock."""
    import open3d as o3d
    from scipy.spatial import ConvexHull

    patches = pd.read_csv(folder / "discontinuity_patches.csv")
    patches = patches[patches["area_m2"] >= min_area]
    pts = np.asarray(o3d.io.read_point_cloud(str(folder / "classified_point_cloud.ply")).points)
    pid = np.load(folder / "point_labels.npy")[:, 1]
    keep = np.isin(pid, patches["patch_id"].to_numpy())
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(pid[idx], kind="stable")]
    ids, starts = np.unique(pid[order], return_index=True)
    span = {int(i): order[s:e] for i, s, e in zip(ids, starts, np.r_[starts[1:], len(order)])}
    rng = np.random.default_rng(0)
    out = []
    for r in patches.itertuples():
        sel = span.get(int(r.patch_id))
        if sel is None or len(sel) < 3:
            continue
        if len(sel) > 3000:
            sel = rng.choice(sel, 3000, replace=False)
        P = pts[sel]
        c = np.array([r.centroid_x, r.centroid_y, r.centroid_z])
        n = _pole(r.dip, r.dip_direction)
        out_n = frame.axes_at(float(r.chainage_m))[1]           # horizontal, toward the pit
        s = np.sign(n @ out_n) if abs(n @ out_n) > 0.2 else 1.0
        a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        q = np.column_stack([(P - c) @ a, (P - c) @ b])
        try:
            hull = ConvexHull(q)
        except Exception:  # noqa: BLE001 -- degenerate (collinear) face
            continue
        h = q[hull.vertices]
        if len(h) > MAX_RING:
            h = h[np.linspace(0, len(h) - 1, MAX_RING).astype(int)]
        ring = c + h[:, :1] * a + h[:, 1:] * b + s * FACE_OFFSET_M * n
        out.append((int(r.set_id), ring))
    return out


def _depth_lookup(folder: Path):
    """w(u, v) from the facade depth raster (median of 5 x 5 pixels), NaN off the model."""
    import rasterio

    with rasterio.open(folder / "facade_depth.tif") as ds:
        d, T = ds.read(1), ds.transform
    H, W = d.shape

    def at(u, v):
        c = np.floor((np.asarray(u) - T.c) / T.a).astype(int)
        r = np.floor((np.asarray(v) - T.f) / T.e).astype(int)
        out = np.full(len(c), np.nan)
        for i, (rr, cc) in enumerate(zip(r, c)):
            win = d[max(0, rr - 2):rr + 3, max(0, cc - 2):cc + 3]
            win = win[np.isfinite(win)]
            if len(win):
                out[i] = float(np.median(win))
        return out
    return at


def trace_segments(folder: Path, frame, depth) -> np.ndarray:
    """(n, 2, 3) end points of the mapped traces on the surface."""
    t = pd.read_csv(folder / "traces.csv")
    if not len(t):
        return np.zeros((0, 2, 3))
    u = np.r_[t["u1"].to_numpy(), t["u2"].to_numpy()]
    v = np.r_[t["v1"].to_numpy(), t["v2"].to_numpy()]
    w = depth(u, v) + FACE_OFFSET_M
    xyz = frame.to_world(np.column_stack([u, v, w]))
    n = len(t)
    seg = np.stack([xyz[:n], xyz[n:]], axis=1)
    return seg[np.isfinite(seg).all(axis=(1, 2))]


def seepage_quads(folder: Path, frame, depth, classes=(2, 1)) -> dict:
    """{class: [quad xyz]} of the seepage-potential cells (1 m) on the surface."""
    f = folder / "seepage_cells.csv"
    if not f.exists():
        return {}
    c = pd.read_csv(f)
    c = c[c["class"].isin(classes)]
    if not len(c):
        return {}
    u, v = c["chainage_m"].to_numpy(), c["elevation_m"].to_numpy()
    w = depth(u, v) + 2 * FACE_OFFSET_M
    h = 0.5
    out = {}
    for k in classes:
        m = (c["class"].to_numpy() == k) & np.isfinite(w)
        corners = np.stack([np.column_stack([u[m] + du, v[m] + dv, w[m]]) for du, dv in ((-h, -h), (h, -h), (h, h), (-h, h))], 1)
        xyz = frame.to_world(corners.reshape(-1, 3)).reshape(-1, 4, 3)
        out[k] = list(xyz)
    return out


def _open_side(A: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Horizontal unit vector from `centre` toward the pit's open side: the middle of the widest angle around
    the centre with no faces (`A`: face centroids) -- for a U, its open end."""
    ang = np.sort(np.arctan2(A[:, 1] - centre[1], A[:, 0] - centre[0]))
    if len(ang) < 2:
        return np.array([0.0, -1.0, 0.0])
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    i = int(np.argmax(gaps))
    mid = ang[i] + gaps[i] / 2
    return np.array([np.cos(mid), np.sin(mid), 0.0])


def kinematic_overlays(folder: Path, label: str, modes=None, show: bool = True) -> list:
    """The failure-mode zones of a section's saved analysis (kinematics.kinematic_map with the dashboard's
    defaults: the run's friction angle, +-20 deg lateral limit, 1 m cells, wall smoothed over 3 m, the sets seen
    within 2 m, vegetated cells left out) draped on its face: one overlay per mode, named with the section."""
    from ..discontinuity import kinematics as K
    from ..pipeline import load_results
    from .kinematic_view import tiles_overlays

    r = load_results(folder)
    joints = {f"Set {d.set_id}": (d.mean_dip, d.mean_dip_direction) for d in r.discontinuity_sets}
    for f in r.trace_families:
        if f.linked_set is None and f.dip is not None:
            joints[f"트레이스 F{f.family_id}"] = (f.dip, f.dip_direction)
    phi = float(r.config.get("kinematics", {}).get("friction_deg", K.FRICTION_DEG))
    veg = (r.config.get("vegetation") or {}).get("exg_threshold", 0.06)
    kmap = K.kinematic_map(r, joints, phi, K.LATERAL_DEG, 1.0, 3.0, 2.0, True, veg)
    out = tiles_overlays(r, kmap, modes or K.MODES, KIN_DRAPE_M, KIN_OFFSET_M)
    for o in out:
        o["name"] = f"{label} · {o['name'].replace('파괴 가능 구역: ', '')}"
        o["show"] = show
    return out


def site_scene(site, tiles_path, sections=None, traces: bool = True, seepage: bool = True, progress=None,
               variant: str = "", kinematics: bool = False) -> dict:
    """The CesiumJS scene of `sections` (labels from section_results; all by default) on the tiled model."""
    from .tiles3d import _to_ecef

    say = progress or (lambda s: None)
    chosen = [(n, d) for n, d in section_results(site, variant) if sections is None or n in sections]
    if not chosen:
        raise ValueError("No section results")
    sets_by = {n: pd.read_csv(d / "discontinuity_sets.csv").assign(
        area=lambda df, d=d: df["set_id"].map(pd.read_csv(d / "discontinuity_patches.csv").groupby("set_id")["area_m2"].sum()).fillna(0))
        for n, d in chosen}
    groups = site_sets(sets_by)
    n_groups = max(k for k, _ in groups.values()) + 1
    labels = {k: lab for k, lab in groups.values()}
    face_groups = [{"label": labels[k], "kind": "obs", "colour": SITE_COLOURS[k % len(SITE_COLOURS)], "polys": [], "colours": []}
                   for k in range(n_groups)]
    seep_groups = {k: {"label": "유출 가능성 " + {2: "상 (상위 15%)", 1: "중 (다음 35%)"}[k], "kind": "obs" if k == 2 else "opt",
                       "colour": SEEP_COLOURS[k], "polys": [], "colours": []} for k in (2, 1)}
    scene = {"title": "현장 전체 — 구간별 분석 결과", "tiles": str(tiles_path), "holes": [], "missed": [],
             "intro": "현장 전체 실사 타일 모델 위의 구간별 절리면(같은 방향의 절리군은 같은 색), 트레이스, 유출 가능성 칸. "
                      "드래그 회전, 휠 확대, 오른쪽 드래그 이동, 더블클릭 = 그 점을 회전 중심으로.",
             "joint_groups": [], "traces": [], "labels": [], "overlays": []}
    allc, crs = [], None
    for name, d in chosen:
        say(f"{name}: 절리면 불러오는 중...")
        crs = _crs(d)
        frame = _frame(d)
        for sid, ring in face_rings(d, frame):
            k = groups.get((name, sid), (None,))[0]
            if k is None:
                continue
            face_groups[k]["polys"].append(_to_ecef(ring, crs).round(3).ravel().tolist())
            face_groups[k]["colours"].append(face_groups[k]["colour"])
        depth = _depth_lookup(d)
        if traces:
            seg = trace_segments(d, frame, depth)
            if len(seg):
                E = _to_ecef(seg.reshape(-1, 3), crs).round(3).reshape(-1, 6)
                scene["traces"] += [{"p": e.tolist(), "high": False} for e in E]
        if seepage:
            for k, quads in seepage_quads(d, frame, depth).items():
                for q in quads:
                    seep_groups[k]["polys"].append(_to_ecef(q, crs).round(3).ravel().tolist())
                    seep_groups[k]["colours"].append(SEEP_COLOURS[k])
        p = pd.read_csv(d / "discontinuity_patches.csv")
        top = np.array([p["centroid_x"].median(), p["centroid_y"].median(), np.percentile(p["centroid_z"], 97) + 8])
        allc.append(p[["centroid_x", "centroid_y", "centroid_z"]].to_numpy())
        scene["labels"].append({"text": name, "p": _to_ecef(top[None], crs)[0].round(3).tolist()})
        if kinematics:
            say(f"{name}: 파괴 가능 구역 계산 중...")
            scene["overlays"] += kinematic_overlays(d, name.split("_")[0])
    scene["joint_groups"] = [g for g in face_groups if g["polys"]] + [g for g in seep_groups.values() if g["polys"]]
    scene["crs"] = crs
    # camera: from the open side of the pit (the walls face into it), above, looking at its middle
    A = np.vstack(allc)
    centre = np.median(A, axis=0)
    R = float(np.ptp(A[:, :2], axis=0).max())
    mids = np.array([np.median(a, axis=0) for a in allc])
    pit = mids.mean(axis=0)                       # inside the U: the middle of the sections
    side = _open_side(A, pit)
    eye = centre + side * 0.8 * R + np.array([0, 0, 1.3 * R])
    scene["camera"] = {"eye": _to_ecef(eye[None], crs)[0].round(3).tolist(),
                       "target": _to_ecef(centre[None], crs)[0].round(3).tolist()}
    scene["wall"] = None
    from .tiles3d import SKY_MARGIN_M, ground_top, sky_clip

    top = ground_top(site, A)
    if top is not None:
        scene["sky"] = sky_clip(crs, centre[:2], top + SKY_MARGIN_M, top - 20, top + 150)
    return scene
