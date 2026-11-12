"""More figures and numbers for the full version of the 2nd interim deck (make_round2_full_pptx.py): the west wall's
50°/046° faces of the re-run without wall-parallel normals (where they lie, sections along their dip direction
through the LiDAR, the 3D views saved from the 3D page), the per-wall results of the re-runs (pole density,
failure-zone maps, Markland panels) and the face span on the base cloud (before / after).
Run make_round2_update_figures.py and make_round2_full_figures.py first.

    .venv\\Scripts\\python.exe docs\\figures\\make_round2_more_figures.py [--no-sections | --sections-only]

Writes docs/figures/round2/more/*.png and more.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from groundwater import report  # noqa: E402
from groundwater import site_joints as SJ  # noqa: E402
from groundwater.discontinuity import kinematics as K  # noqa: E402
from groundwater.pipeline import load_results  # noqa: E402
from groundwater.viz import kinematic_view as KV  # noqa: E402
from groundwater.viz import lidar_section_view as LV  # noqa: E402
from groundwater.viz import static_map  # noqa: E402

import make_round2_full_figures as FF  # noqa: E402
import make_round2_update_figures as UF  # noqa: E402

SITE = REPO / "Filed_Data" / "IcSan20261021"
OUT = HERE / "round2" / "more"
SECS = {"E": "E_동측벽", "S": "S_남측끝", "W": "W_서측벽"}
VARIANT = "_벽면평행제외"
BEFORE = "_이전_뒤층만_벽면범위"          # the runs before the face span was applied to the base cloud
SNAPS = SITE / "3D화면저장"
LAS = SITE / "U자전체자료" / "IcSan20261024_LiDAR_L2_whole.las"
DIP, DD, TOL, HALF = 50.0, 46.0, 10.0, 0.10
SET_COLOUR = "#16a34a"                   # the 50°/046° faces, green as on the 3D page
SEC_COLOUR = "#facc15"                   # ... in the sections yellow (the sets' colours there include green)
# sections through the west wall's 50°/046° faces (plan x, y of a large face; chainage on the wall)
CUTS = [("①", 203172.6, 384802.0), ("②", 203169.3, 384817.2), ("③", 203148.9, 384834.3),
        ("④", 203125.6, 384857.3), ("⑤", 203105.0, 384879.8), ("⑥", 203070.8, 384919.0)]
BACK, FRONT = 40.0, 35.0


def pole(dip, dd):
    d, a = np.radians(dip), np.radians(dd)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def the_set(r):
    """The re-run's set nearest to 50°/046°."""
    ref = pole(DIP, DD)
    return min(r.discontinuity_sets,
               key=lambda d: np.degrees(np.arccos(min(1.0, abs(float(pole(d.mean_dip, d.mean_dip_direction) @ ref))))))


def plan_figure(r, s, path):
    """Plan of the west wall over the 10/23 orthophoto: all faces of the re-run grey, the 50°/046° set's faces green
    (by area), the section lines ①-⑥."""
    import matplotlib.patheffects as pe

    plt = report._setup_mpl()
    P = [p for d in r.discontinuity_sets for p in d.patches]
    xy = np.array([p.centroid[:2] for p in P])
    c = np.array([p.centroid[:2] for p in s.patches])
    a = np.array([p.area for p in s.patches])
    lo, hi = xy.min(0) - 25, xy.max(0) + 25
    ext = [lo[0], hi[0], lo[1], hi[1]]
    img, e = LV.ortho_image(LV.find_orthos(SITE)[0], ext, 1600)
    fig, ax = plt.subplots(figsize=(8.2, 8.2 * (hi[1] - lo[1]) / (hi[0] - lo[0])))
    ax.imshow(img, extent=e, origin="upper", alpha=0.75)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#6b7280", alpha=0.35, lw=0, label="다른 절리면")
    ax.scatter(c[:, 0], c[:, 1], s=4 + 2.5 * a, c=SET_COLOUR, alpha=0.85, lw=0.3, ec="white",
               label=f"{DIP:.0f}°/{DD:03.0f}° 면 (크기 = 면적)")
    d = np.array([np.sin(np.radians(DD)), np.cos(np.radians(DD))])
    for lab, x, y in CUTS:
        A, B = np.array([x, y]) - BACK * d, np.array([x, y]) + FRONT * d
        ax.plot([A[0], B[0]], [A[1], B[1]], color="#dc2626", lw=2.2)
        ax.annotate(lab, B, fontsize=17, fontweight="bold", color="#dc2626", ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=9)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.set_title(f"서측 벽 · {DIP:.0f}°/{DD:03.0f}° 면과 단면 위치 (경사방향 {DD:03.0f}°, 붉은 선)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def section(r, res_dir, lab, cx, cy, path):
    """Section along 046 through (cx, cy): the LiDAR returns and the rock layer as the section tool shows them, the
    analysis points of the joint sets, the ones facing 50°/046° (within 10°) in yellow, a 50° line through them."""
    import open3d as o3d
    import pandas as pd
    import plotly.graph_objects as go

    from groundwater import lidar_section as LS

    d = np.array([np.sin(np.radians(DD)), np.cos(np.radians(DD))])          # along the dip direction: hill -> pit
    A, B = np.array([cx, cy]) - BACK * d, np.array([cx, cy]) + FRONT * d
    slab = LS.read_slab(LAS, LS.load_index(LAS), A, B)
    kinds = LS.return_kind(slab["ret"], slab["nret"])
    _, folder = LS.owning_section(SITE, (A + B) / 2)
    frame = LS.section_frame(folder, slab["xyz"][:: max(1, len(slab["xyz"]) // 200_000)])
    pc = o3d.io.read_point_cloud(str(res_dir / "classified_point_cloud.ply"))
    P, N = np.asarray(pc.points), np.asarray(pc.normals)
    labels = np.load(res_dir / "point_labels.npy")
    cls = LS.classify(slab["xyz"], frame, SJ.find_ground(SITE), P)
    facing = np.degrees(np.arccos(np.clip(np.abs(N @ pole(DIP, DD)), 0, 1))) <= TOL
    u = (B - A) / np.hypot(*(B - A))
    rel = P[:, :2] - A
    ss, dd = rel @ u, rel @ np.array([-u[1], u[0]])
    m = (np.abs(dd) <= HALF) & (ss >= -1) & (ss <= slab["length"] + 1)
    sid = labels[:, 0]
    in_set = facing & (sid == s_id(r))
    lab_pts = np.where(in_set, 99, sid)
    t = pd.read_csv(res_dir / "discontinuity_sets.csv")
    names = {int(x.set_id): f"Set {int(x.set_id)} {x.mean_dip:.0f}°/{x.mean_dip_direction:03.0f}°" for x in t.itertuples()}
    names[99] = f"{DIP:.0f}°/{DD:03.0f}° 절리군의 점 (±{TOL:.0f}°)"
    title = f"단면 {lab} · 경사방향 {DD:03.0f}°로 자름 (언덕 → 구덩이) · 두께 ±{HALF * 100:.0f} cm"
    fig = LV.section_figure(slab, kinds, cls, HALF, (ss[m], P[m, 2], lab_pts[m]), names, title, True)
    for tr in fig.data:                                                       # the set's points: yellow, larger
        if tr.name and names[99] in tr.name:
            tr.marker.color, tr.marker.size = SEC_COLOUR, 9
            tr.marker.line = dict(color="#111827", width=1.2)
    cm = m & in_set
    out = {"label": lab, "x": cx, "y": cy, "pts": int(m.sum()), "set_pts": int(cm.sum())}
    if cm.any():
        sc, zc = float(np.median(ss[cm])), float(np.median(P[cm, 2]))
        L = 8.0
        dp = np.radians(DIP)
        fig.add_trace(go.Scatter(x=[sc - L, sc + L], y=[zc + L * np.tan(dp), zc - L * np.tan(dp)], mode="lines",
                                 line=dict(color="#dc2626", width=3, dash="dash"), name=f"{DIP:.0f}° 기준선 (이 단면의 진경사)"))
        out.update(z_lo=round(float(P[cm, 2].min()), 1), z_hi=round(float(P[cm, 2].max()), 1))
        nb = (r.config.get("lidar_rock") or {}).get("base_points")
        if nb:
            out["base_pct"] = round(100 * float((np.flatnonzero(cm) < nb).mean()))
    # the wall: the rock layer and the analysis points with a margin (the slab runs far into the pit and the hill)
    keep = (np.abs(slab["d"]) <= HALF) & cls["rear"]
    xs = np.r_[slab["s"][keep], ss[m]]
    zs = np.r_[slab["xyz"][keep, 2], P[m, 2]]
    if len(xs):
        fig.update_xaxes(range=[max(-1.0, float(np.percentile(xs, 1)) - 8), min(slab["length"] + 1, float(np.percentile(xs, 99)) + 8)])
        fig.update_yaxes(range=[float(zs.min()) - 2, float(np.percentile(zs, 99.5)) + 4])
    fig.update_layout(title=None)
    path.write_bytes(static_map.preview_image(fig, width_px=1000, height_px=780, legend=False, show_title=False,
                                              font_scale=1.5))
    st = LV.slice_stats(slab, kinds, cls, HALF)
    out.update(outside=st.get("벽면 범위 밖 제외"), rock=st.get("암반으로 남김"), total=st.get("전체"))
    print("section", out, flush=True)
    return out


_SID = {}


def s_id(r):
    if id(r) not in _SID:
        _SID[id(r)] = the_set(r).set_id
    return _SID[id(r)]


def behind_front(r, lim=10.0):
    """Area (m²) of the joint faces more than `lim` behind the front surface at their place."""
    front, u0, v0, cell = K._front_surface(r, 0.5)
    F = [p for d in r.discontinuity_sets for p in d.patches]
    if not F:
        return 0
    uvw = r.wall_frame.to_wall(np.array([p.centroid for p in F]))
    c = np.floor((uvw[:, 0] - u0) / cell).astype(int)
    rr = np.floor((v0 - uvw[:, 1]) / cell).astype(int)
    ok = (rr >= 0) & (rr < front.shape[0]) & (c >= 0) & (c < front.shape[1])
    behind = np.full(len(F), np.nan)
    behind[ok] = front[rr[ok], c[ok]] - uvw[ok, 2]
    area = np.array([p.area for p in F])
    return round(float(area[np.nan_to_num(behind) > lim].sum()))


def run_numbers(r):
    L = r.config.get("lidar_rock") or {}
    return {"faces": len([p for d in r.discontinuity_sets for p in d.patches]),
            "area": round(sum(p.area for d in r.discontinuity_sets for p in d.patches)),
            "sets": UF.sets_of(r), "kin": UF.kin_of(r), "behind10": behind_front(r),
            "base_points": L.get("base_points"), "base_dropped": L.get("base_points_outside_face_dropped", 0),
            "rear_dropped": L.get("rear_points_outside_face_dropped", 0)}


def per_wall(r, k, tag):
    """Pole density, failure-zone map, Markland panels and the bar chart of a run."""
    fam = r.trace_families
    FF.png(report.fig_stereonet(r, fam, {"method": "exponential_kamb", "sigma": 3, "weighted": False, "poles": True,
                                         "cmap": "Greys"}), OUT / f"density{tag}_{k}.png")
    FF.png(report.fig_kinematics(r), OUT / f"kinbar{tag}_{k}.png")
    joints = FF.joints_of(r)
    phi = float(r.config.get("kinematics", {}).get("friction_deg", K.FRICTION_DEG))
    veg = (r.config.get("vegetation") or {}).get("exg_threshold", 0.06)
    kmap = K.kinematic_map(r, joints, phi, K.LATERAL_DEG, 1.0, 3.0, 2.0, True, veg)
    fig = KV.map_figure(r, kmap, list(K.MODES))
    fig.update_layout(title=None)
    (OUT / f"kinmap{tag}_{k}.png").write_bytes(static_map.preview_image(fig, width_px=1800, max_height_px=800))
    face = K.representative_face(r.face_windows)
    st_png, counts = KV.stereonet_panels(r, face, phi, K.LATERAL_DEG)
    (OUT / f"markland{tag}_{k}.png").write_bytes(st_png)
    FF.png(report.fig_trace_map(r, fam), OUT / f"traces{tag}_{k}.png")
    return {"kinmap": KV.summary(kmap), "face": [round(face[0]), round(face[1]) % 360],
            "markland": {m: {kk: int(v) for kk, v in c.items() if isinstance(v, (int, np.integer))} for m, c in counts.items()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / "more.json"
    data = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}

    # the west wall's 50°/046° faces in the re-run
    wx = SITE / "구간" / SECS["W"] / ("분석결과" + VARIANT)
    r = load_results(wx)
    s = the_set(r)
    a = np.array([p.area for p in s.patches])
    uvw = r.wall_frame.to_wall(np.array([p.centroid for p in s.patches]))
    data["set50"] = {"dip": round(s.mean_dip), "dd": round(s.mean_dip_direction) % 360, "faces": len(s.patches),
                     "area": round(float(a.sum())), "u": np.percentile(uvw[:, 0], [5, 50, 95]).round().tolist(),
                     "z": np.percentile([p.centroid[2] for p in s.patches], [5, 50, 95]).round().tolist(),
                     "big": int((a >= 10).sum()), "median_area": round(float(np.median(a)), 1)}
    nb = (r.config.get("lidar_rock") or {}).get("base_points")
    if nb:
        pid = r.point_labels[:, 1]
        ids = np.array([p.patch_id for p in s.patches])
        inside = np.isin(pid, ids)
        data["set50"]["base_pct"] = round(100 * float((np.flatnonzero(inside) < nb).mean()))
    # the wall face of the 5 m windows (benches included) against the set
    fd = np.array([[w["face_dip"], w["face_dip_direction"]] for w in r.face_windows], float)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.array([pole(*x) for x in fd]) @ pole(DIP, DD)), 0, 1)))
    data["set50"]["face"] = {"dip": np.percentile(fd[:, 0], [10, 50, 90]).round().tolist(),
                             "dd": np.percentile(fd[:, 1], [10, 50, 90]).round().tolist(), "windows": len(fd),
                             "within10": round(100 * float((ang <= 10).mean())), "within20": round(100 * float((ang <= 20).mean())),
                             "steeper": round(100 * float((fd[:, 0] > DIP).mean()))}
    plan_figure(r, s, OUT / "set50_plan.png")
    if "--no-sections" not in sys.argv:
        data["sections50"] = [section(r, wx, lab, x, y, OUT / f"set50_section_{i + 1}.png")
                              for i, (lab, x, y) in enumerate(CUTS)]
    print("set50", data["set50"], flush=True)

    if "--sections-only" in sys.argv:
        js.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return
    # per-wall figures of both runs, before / after the face span on the base cloud
    data["walls"] = {}
    for k, sec in SECS.items():
        w = {}
        for tag, sub in (("", "분석결과"), ("_x", "분석결과" + VARIANT)):
            rr = r if (k == "W" and tag == "_x") else load_results(SITE / "구간" / sec / sub)
            w["now" + tag] = run_numbers(rr)
            w["fig" + tag] = per_wall(rr, k, tag)
            old = SITE / "구간" / sec / BEFORE / sub
            if old.exists():
                w["before" + tag] = run_numbers(load_results(old))
        data["walls"][k] = w
        print(k, {t: (w[t]["faces"], w[t]["kin"]) for t in w if t.startswith(("now", "before"))}, flush=True)

    # views saved from the 3D page
    for name, stem in (("서측벽_50-046면만", "벽면평행제외_벽면평행제외_서측벽_50-046면만"),
                       ("서측벽_50-046면만_측면", "벽면평행제외_벽면평행제외_서측벽_50-046면만_측면"),
                       ("3_U자_북쪽에서_절리군", "3_U자_북쪽에서_절리군"), ("4_U자_북쪽에서_파괴형태", "4_U자_북쪽에서_파괴형태")):
        src = SNAPS / f"현장전체_절리_파괴형태_3D장면_{stem}.png"
        if src.exists():
            shutil.copy(src, OUT / f"tiles_{name}.png")
        else:
            print("missing view", src)
    js.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
