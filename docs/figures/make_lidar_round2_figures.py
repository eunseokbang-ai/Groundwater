"""Figures of the 2nd interim deck (Iksan U-shaped quarry wall, LiDAR-only joint analysis, 2026-10): one per
section (E / S / W) -- facade with the joint-set overlay, stereonet -- plus the LiDAR rear-layer cross-section,
the planarity limit before / after and the kinematic summary. Written to docs/figures/round2/.

    .venv\\Scripts\\python.exe docs\\figures\\make_lidar_round2_figures.py

Reads the saved results (Filed_Data/IcSan20261021/구간/<section>/분석결과 and 분석결과 for photogrammetry W).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from groundwater import report  # noqa: E402
from groundwater.pipeline import load_results  # noqa: E402

SITE = REPO / "Filed_Data" / "IcSan20261021"
OUT = HERE / "round2"
SECTIONS = {"E": "구간/E_동측벽/분석결과", "S": "구간/S_남측끝/분석결과", "W": "구간/W_서측벽/분석결과",
            "W_photo": "분석결과"}
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else None        # folder with lidar_windows.pkl (rear-layer figure)


def facade_sets(result, path, max_px=2600):
    """Facade photo with the set overlay, buffer strips greyed, chainage / elevation axes."""
    plt = report._setup_mpl()
    rgb, ext = report.facade_preview(result, max_px)
    ov = report.set_overlay(result, rgb.shape[:2]).astype(float) / 255
    img = rgb.astype(float) / 255 * (1 - ov[..., 3:4]) + ov[..., :3] * ov[..., 3:4]
    core = (result.config.get("section") or {}).get("core_chainage")
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    fig, ax = plt.subplots(figsize=(16, max(4.0, 16 * aspect)))
    ax.imshow(img, extent=ext)
    if core:
        for x0, x1 in ((ext[0], core[0]), (core[1], ext[1])):
            if x1 > x0:
                ax.axvspan(x0, x1, color="#6b7280", alpha=0.55, lw=0)
                ax.text((x0 + x1) / 2, ext[2] + 2, "겹침", ha="center", va="bottom", fontsize=12, color="white", weight="bold")
    for d in result.discontinuity_sets:
        ax.plot([], [], "s", color=report.SET_COLORS[d.set_id % len(report.SET_COLORS)], ms=14,
                label=f"Set {d.set_id}  {d.mean_dip:.0f}°/{d.mean_dip_direction:03.0f}°")
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9)
    ax.set_xlabel("측점 (m)", fontsize=14)
    ax.set_ylabel("표고 (m, 해발)", fontsize=14)
    ax.tick_params(labelsize=12)
    path.write_bytes(report._png(fig, dpi=120))


def stereonet(result, path):
    path.write_bytes(report.fig_stereonet(result, result.trace_families))


def rear_layer_section(scratch, path):
    """Netted window: single / first / last returns in a 10 cm slice, and the rear layer the program keeps."""
    from groundwater.lidar import rear_layer

    d = pickle.load(open(scratch / "lidar_windows.pkl", "rb"))
    plt = report._setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, key, u0, title in ((axs[0], "net", 162.0, "낙석방지망 구간"), (axs[1], "bare", 162.0, "맨 암반 구간")):
        a = d[key]                               # x y z u v w return_number number_of_returns intensity r g b
        s = a[np.abs(a[:, 3] - u0) < 0.05]
        single, multi = s[:, 7] == 1, s[:, 7] > 1
        first, last = multi & (s[:, 6] == 1), multi & (s[:, 6] == s[:, 7])
        ax.scatter(s[single, 5], s[single, 4], s=2, c="#9ca3af", label="단일 반사")
        ax.scatter(s[first, 5], s[first, 4], s=2, c="#2563eb", label="첫 반사 (망·잎)")
        ax.scatter(s[last, 5], s[last, 4], s=2, c="#dc2626", label="마지막 반사")
        keep = rear_layer(s[:, 3:6], cell_m=0.05, layer_m=0.06)
        r = s[keep]
        o = np.argsort(r[:, 4])
        ax.plot(r[o, 5], r[o, 4], "-", color="#111827", lw=1.6, label="프로그램이 남기는 뒤쪽 층 (암반)")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("벽면에서의 거리 w (m, + = 굴착지 쪽)", fontsize=13)
        ax.set_ylabel("표고 (m)", fontsize=13)
        ax.legend(loc="upper right" if key == "net" else "lower left", fontsize=11, markerscale=5)
        ax.set_aspect("equal", adjustable="datalim")
    path.write_bytes(report._png(fig, dpi=130))


def planarity(before_dir, after_dir, path, window=((10, 48), (66, 101))):
    """The faces in one netted window without and with the planarity limit, each face in its own colour."""
    plt = report._setup_mpl()
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, name, title in ((axs[0], before_dir, "평면성 한계 없음"), (axs[1], after_dir, "평면성 한계 (RMS 10 cm)")):
        r = load_results(SITE / name)
        pts = np.asarray(r.point_cloud.points)
        uvw = r.wall_frame.to_wall(pts)
        (u0, u1), (v0, v1) = window
        m = (uvw[:, 0] >= u0) & (uvw[:, 0] <= u1) & (uvw[:, 1] >= v0) & (uvw[:, 1] <= v1)
        lab = r.point_labels[m, 1]
        q = uvw[m]
        ax.scatter(q[lab < 0, 0], q[lab < 0, 1], s=0.3, c="#e5e7eb")
        ids = np.unique(lab[lab >= 0])
        col = {i: rng.random(3) * 0.8 + 0.1 for i in ids}
        c = np.array([col[i] for i in lab[lab >= 0]]) if len(ids) else np.zeros((0, 3))
        ax.scatter(q[lab >= 0, 0], q[lab >= 0, 1], s=0.3, c=c)
        big = max((p.area for d in r.discontinuity_sets for p in d.patches
                   if u0 <= r.wall_frame.to_wall(p.centroid[None])[0, 0] <= u1), default=0)
        ax.set_title(f"{title}: 면 {len(ids)}개, 가장 큰 면 {big:.0f} m²", fontsize=15)
        ax.set_xlabel("측점 (m)", fontsize=12)
        ax.set_ylabel("표고 (m)", fontsize=12)
        ax.set_aspect("equal")
    path.write_bytes(report._png(fig, dpi=130))


def kinematics(results, path):
    """Share of the 5 m face windows where each failure mode is kinematically possible, per section."""
    plt = report._setup_mpl()
    modes = ["평면파괴", "쐐기파괴", "전도파괴"]
    colors = {"평면파괴": "#d62728", "쐐기파괴": "#ff7f0e", "전도파괴": "#9467bd"}
    names = list(results)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.26
    for i, m in enumerate(modes):
        vals = []
        for n in names:
            r = results[n]
            k, nw = r.kinematics, max(1, len(r.face_windows))
            vals.append(100 * (k[k["mode"] == m]["u0"].nunique() if k is not None and len(k) else 0) / nw)
        xs = np.arange(len(names)) + (i - 1) * width
        ax.bar(xs, vals, width, color=colors[m], label=m)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=12)
    ax.set_xticks(range(len(names)), names, fontsize=14)
    ax.set_ylabel("가능한 구간 비율 (%)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    path.write_bytes(report._png(fig, dpi=130))


def summary(r) -> dict:
    """The numbers the deck quotes, from a result."""
    c = r.config
    faces = [p for d in r.discontinuity_sets for p in d.patches]
    area = sum(p.area for p in faces) or 1.0
    k, nw = r.kinematics, max(1, len(r.face_windows))
    kin = {m: round(100 * (k[k["mode"] == m]["u0"].nunique() if k is not None and len(k) else 0) / nw)
           for m in ("평면파괴", "쐐기파괴", "전도파괴")}
    return {
        "core_chainage": (c.get("section") or {}).get("core_chainage"),
        "faces": len(faces), "area_m2": round(area), "traces": len(r.traces), "windows": len(r.face_windows),
        "sets": [{"id": d.set_id, "dip": round(d.mean_dip), "dd": round(d.mean_dip_direction), "faces": len(d.patches),
                  "area": round(sum(p.area for p in d.patches)), "share": round(100 * sum(p.area for p in d.patches) / area),
                  "k": round(d.fisher_k), "spacing": None if d.spacing_mean is None else round(d.spacing_mean, 2)}
                 for d in r.discontinuity_sets],
        "candidates": [{"dip": round(x["dip"]), "dd": round(x["dip_direction"]), "contrast": x["contrast"]}
                       for x in (c.get("set_detection") or {}).get("candidates", [])],
        "families": [{"id": f.family_id, "n": f.n_traces, "linked": f.linked_set,
                      "dip": None if f.dip is None else round(f.dip), "dd": None if f.dip_direction is None else round(f.dip_direction)}
                     for f in r.trace_families],
        "kinematics": kin, "removed": c.get("faces_removed"),
        "lidar_points_added": (c.get("lidar_rock") or {}).get("points_added"),
        "voxel_m": (c.get("auto_scale_applied") or {}).get("voxel_size"),
        "facade_area_m2": c.get("facade_area_m2"), "load": c.get("load_thinning"),
    }


def compare_sets(a, b) -> list:
    """Each set of `a` with the nearest set of `b` and the angle between their poles."""
    from groundwater.discontinuity.clustering import plane_normal

    out = []
    for d in a.discontinuity_sets:
        pa = plane_normal(d.mean_dip, d.mean_dip_direction)
        ang = [(np.degrees(np.arccos(min(1.0, abs(pa @ plane_normal(e.mean_dip, e.mean_dip_direction))))), e)
               for e in b.discontinuity_sets]
        g, e = min(ang, key=lambda t: t[0]) if ang else (None, None)
        out.append({"a": f"{d.mean_dip:.0f}/{d.mean_dip_direction:03.0f}",
                    "b": None if e is None else f"{e.mean_dip:.0f}/{e.mean_dip_direction:03.0f}",
                    "angle": None if g is None else round(float(g), 1)})
    return out


def main():
    OUT.mkdir(exist_ok=True)
    loaded = {}
    for key, rel in SECTIONS.items():
        if not (SITE / rel / "result.pkl").exists():
            print("missing", rel)
            continue
        r = load_results(SITE / rel)
        loaded[key] = r
        facade_sets(r, OUT / f"facade_{key}.png")
        stereonet(r, OUT / f"stereo_{key}.png")
        print("section", key, "done", flush=True)
    kin = {k: loaded[k] for k in ("E", "S", "W") if k in loaded}
    if kin:
        kinematics({{"E": "E 동측 벽", "S": "S 남측 끝", "W": "W 서측 벽"}[k]: v for k, v in kin.items()}, OUT / "kinematics.png")
    if (SITE / "분석결과_겹침_라이다전체밀도").exists() and (SITE / "분석결과_평면성_라이다전체").exists():
        planarity("분석결과_겹침_라이다전체밀도", "분석결과_평면성_라이다전체", OUT / "planarity.png")
    if SCRATCH and (SCRATCH / "lidar_windows.pkl").exists():
        rear_layer_section(SCRATCH, OUT / "rear_layer.png")
    import json

    data = {k: summary(r) for k, r in loaded.items()}
    if "W" in loaded and "W_photo" in loaded:
        data["W_compare"] = compare_sets(loaded["W_photo"], loaded["W"])
    (OUT / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
