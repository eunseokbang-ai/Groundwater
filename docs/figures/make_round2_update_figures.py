"""Figures and numbers for the update of the 2nd interim deck (make_round2_update_pptx.py): the excavated-surface
check (discontinuity/exposure.py), the re-runs with the wall-parallel normals left out (구간/*/분석결과_벽면평행제외),
the stability check of both, the site-wide underground joint model and the views of the tiled model saved from the
3D page (Filed_Data/.../3D화면저장).

    .venv\\Scripts\\python.exe docs\\figures\\make_round2_update_figures.py

Writes docs/figures/round2/update/*.png and update.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from groundwater import report  # noqa: E402
from groundwater import site_joints as SJ  # noqa: E402
from groundwater.discontinuity import exposure as X  # noqa: E402
from groundwater.pipeline import load_results  # noqa: E402
from groundwater.viz import exposure_view as EV  # noqa: E402
from groundwater.viz import lidar_section_view as LV  # noqa: E402
from groundwater.viz import site_joints_view as V  # noqa: E402
from groundwater.viz import static_map  # noqa: E402

SITE = REPO / "Filed_Data" / "IcSan20261021"
OUT = HERE / "round2" / "update"
SECS = ["E_동측벽", "S_남측끝", "W_서측벽"]
VARIANT = "_벽면평행제외"
NAMES = {"E": "E 동측 벽", "S": "S 남측 끝", "W": "W 서측 벽"}
MODES = ["평면파괴", "쐐기파괴", "전도파괴"]
SNAPS = SITE / "3D화면저장"


def sets_of(r):
    faces = [p for d in r.discontinuity_sets for p in d.patches]
    area = sum(p.area for p in faces) or 1.0
    return [{"dip": round(d.mean_dip), "dd": round(d.mean_dip_direction) % 360, "faces": len(d.patches),
             "share": round(100 * sum(p.area for p in d.patches) / area)} for d in r.discontinuity_sets]


def kin_of(r):
    k, nw = r.kinematics, max(1, len(r.face_windows))
    return {m: round(100 * (k[k["mode"] == m]["u0"].nunique() if k is not None and len(k) else 0) / nw) for m in MODES}


def families_of(r):
    return [{"dip": round(f.dip), "dd": round(f.dip_direction) % 360, "n": f.n_traces,
             "trace_only": f.linked_set is None} for f in r.trace_families if f.dip is not None]


def kin_compare(data, path):
    plt = report._setup_mpl()
    colors = {"평면파괴": "#c026d3", "쐐기파괴": "#f59e0b", "전도파괴": "#06b6d4"}    # as on the tiled model
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, k in zip(axes, "ESW"):
        for i, m in enumerate(MODES):
            a, b = data[k]["kin"][m], data[k]["kin_x"][m]
            ax.bar(i - 0.19, a, 0.36, color=colors[m], alpha=0.95)
            ax.bar(i + 0.19, b, 0.36, color=colors[m], alpha=0.45, hatch="//", edgecolor="white")
            ax.text(i - 0.19, a + 1.5, f"{a}", ha="center", fontsize=11)
            ax.text(i + 0.19, b + 1.5, f"{b}", ha="center", fontsize=11, color="#4b5563")
        ax.set_xticks(range(3), [m.replace("파괴", "") for m in MODES], fontsize=13)
        ax.set_title(NAMES[k], fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("가능한 5 m 구간 비율 (%)", fontsize=12)
    from matplotlib.patches import Patch

    fig.legend([Patch(color="#6b7280"), Patch(facecolor="#6b7280", alpha=0.45, hatch="//", edgecolor="white")],
               ["기본 분석", "벽면 평행 제외 분석"], loc="upper center", ncol=2, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.write_bytes(report._png(fig, dpi=130))


def exposure_bar(P, path):
    """Stacked bars: each section's sets by how their faces lie to the wall (area)."""
    plt = report._setup_mpl()
    S = X.summary(P, by=("section", "set_label"))
    labels = [f"{s.split('_')[0]}  {lab}" for s, lab in zip(S["section"], S["set_label"])]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    left = np.zeros(len(S))
    for c in EV.ORDER:
        v = S["area_m2"].to_numpy() * S[f"{c}_area_pct"].to_numpy() / 100
        ax.barh(range(len(S)), v, left=left, color=X.CLASS_COLOURS[c], label=X.CLASSES[c], height=0.62)
        for i, (l, w, pct) in enumerate(zip(left, v, S[f"{c}_area_pct"])):
            if pct >= 9 and w >= 450:
                ax.text(l + w / 2, i, f"{pct:.0f}%", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        left += v
    ax.set_yticks(range(len(S)), labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("절리면 면적 (m²)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=12, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    path.write_bytes(report._png(fig, dpi=130))


def crop_sky(src, dst, top=0.08):
    from PIL import Image

    im = Image.open(src)
    w, h = im.size
    im.crop((0, int(h * top), w, h)).save(dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for s in SECS:
        k = s[0]
        a = load_results(SITE / "구간" / s / "분석결과")
        b = load_results(SITE / "구간" / s / ("분석결과" + VARIANT))
        data[k] = {"sets": sets_of(a), "sets_x": sets_of(b), "kin": kin_of(a), "kin_x": kin_of(b),
                   "fam": families_of(a), "fam_x": families_of(b),
                   "area": round(sum(p.area for d in a.discontinuity_sets for p in d.patches)),
                   "area_x": round(sum(p.area for d in b.discontinuity_sets for p in d.patches)),
                   "windows": len(a.face_windows)}
        print(s, data[k]["sets"], data[k]["sets_x"], data[k]["kin"], data[k]["kin_x"], flush=True)
    kin_compare(data, OUT / "kin_compare.png")

    # the excavated-surface check of the normal runs
    P = EV.load_sections(SITE, SECS)
    tot = P["area_m2"].sum()
    data["exposure"] = {c: round(100 * P.loc[P["exposure"] == c, "area_m2"].sum() / tot) for c in X.CLASSES}
    data["exposure_by"] = {s[0]: {c: round(100 * g.loc[g["exposure"] == c, "area_m2"].sum() / g["area_m2"].sum())
                                  for c in X.CLASSES} for s, g in P.groupby("section")}
    Px = EV.load_sections(SITE, SECS, variant=VARIANT)
    data["exposure_x"] = {c: round(100 * Px.loc[Px["exposure"] == c, "area_m2"].sum() / Px["area_m2"].sum())
                          for c in X.CLASSES}
    data["faces_n"], data["faces_n_x"] = len(P), len(Px)
    (OUT / "exposure_stereo.png").write_bytes(EV.stereonets_png(P))
    exposure_bar(P, OUT / "exposure_bar.png")
    Q = P[P["section"] == "E_동측벽"]
    (OUT / "exposure_elev_E.png").write_bytes(static_map.preview_image(EV.elevation_figure(Q, Q["folder"].iloc[0], 560),
                                                                       width_px=1800, max_height_px=760))
    print("exposure", data["exposure"], data["exposure_by"], data["exposure_x"], flush=True)

    # the site-wide underground joint model: normal, normal without wall-parallel faces, the re-runs
    sj = {}
    for tag, kinds, var in (("기본", None, ""), ("기본_교차만", ("cross",), ""), ("벽면평행제외", None, VARIANT)):
        F = SJ.load_faces(SITE, SECS, 5.0, kinds, var)
        C = SJ.continuity(F)
        sj[tag] = {"faces": len(F), "planes": len(C), "multi": int(C["sections"].str.contains("·").sum()) if len(C) else 0}
    data["site_joints"] = sj
    F = SJ.load_faces(SITE, SECS, 5.0, None, VARIANT)
    C = SJ.continuity(F)
    lo, hi = F[["x", "y"]].min().to_numpy() - 60, F[["x", "y"]].max().to_numpy() + 60
    ground = SJ.ground_grid(SJ.find_ground(SITE), (lo[0], lo[1], hi[0], hi[1]), 2.0)
    bg = LV.ortho_image(LV.find_orthos(SITE)[0], [lo[0], hi[0], lo[1], hi[1]], 1600)
    fig = V.ortho_map(F, C, 3.0, ground, bg, "planes", True, True, 20.0, 8.0)
    fig.update_layout(title=None, showlegend=False, margin=dict(l=60, r=10, t=10, b=45))       # legend on the slide
    fig.for_each_trace(lambda t: t.type != "image" and t.update(showlegend=False))
    (OUT / "sj_ortho.png").write_bytes(static_map.preview_image(fig, width_px=1150, max_height_px=1150))
    fig = V.section_figure(F, C, 3.0, (203080, 384870), (203290, 384880), ground, planes_full=True, full_min_span=20.0)
    fig.update_layout(title=None, showlegend=False, margin=dict(l=60, r=10, t=10, b=45))
    fig.for_each_trace(lambda t: t.type != "image" and t.update(showlegend=False))
    fig.update_xaxes(title=dict(text="서 → 동 거리 (m)", font=dict(size=18)), tickfont=dict(size=16))
    fig.update_yaxes(title=dict(text="표고 (m)", font=dict(size=18)), tickfont=dict(size=16))
    (OUT / "sj_section.png").write_bytes(static_map.preview_image(fig, width_px=1500, max_height_px=760))
    print("site joints", sj, flush=True)
    # legends: the site-wide sets and their colours (the 3D scenes and the underground figures colour by these)
    data["legend_sj"] = [[lab, V._colour(g)] for g, lab in sorted({(int(g), lab) for g, lab in zip(F["group"], F["group_label"])})]
    for var in ("", VARIANT):
        p = SITE / f"현장전체_절리_파괴형태_3D장면{var}.json"
        if p.exists():
            sc = json.loads(p.read_text(encoding="utf-8"))
            data["legend" + var] = [[g["label"], g["colour"]] for g in sc["joint_groups"] if not g["label"].startswith("유출")]

    # views saved from the 3D page
    for name in ("기본_U자_절리군", "기본_U자_파괴형태", "기본_서측벽_절리군", "기본_서측벽_파괴형태",
                 "벽면평행제외_U자_절리군", "벽면평행제외_U자_파괴형태", "벽면평행제외_서측벽_절리군", "벽면평행제외_서측벽_파괴형태"):
        stem = "현장전체_절리_파괴형태_3D장면" + (VARIANT if name.startswith("벽면평행제외") else "")
        src = SNAPS / f"{stem}_{name}.png"
        if src.exists():
            crop_sky(src, OUT / f"tiles_{name}.png", 0.0 if "서측벽" in name else 0.06)
        else:
            print("missing view", src)
    (OUT / "update.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
