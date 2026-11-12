"""Figures and numbers for the full version of the 2nd interim deck (make_round2_full_pptx.py): method illustrations
and per-wall results that the short deck leaves out -- pole-density stereonets, trace maps, failure-zone maps and
Markland panels, structural seepage potential, the underground plan section, the 10/23 DSM layers, the 3D views.
Run make_round2_update_figures.py first (the comparison numbers).

    .venv\\Scripts\\python.exe docs\\figures\\make_round2_full_figures.py

Writes docs/figures/round2/full/*.png and full.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from groundwater import hydro, report  # noqa: E402
from groundwater import site_joints as SJ  # noqa: E402
from groundwater.discontinuity import kinematics as K  # noqa: E402
from groundwater.pipeline import load_results  # noqa: E402
from groundwater.viz import kinematic_view as KV  # noqa: E402
from groundwater.viz import lidar_section_view as LV  # noqa: E402
from groundwater.viz import site_joints_view as V  # noqa: E402
from groundwater.viz import static_map  # noqa: E402

SITE = REPO / "Filed_Data" / "IcSan20261021"
OUT = HERE / "round2" / "full"
SECS = {"E": "E_동측벽", "S": "S_남측끝", "W": "W_서측벽"}
VARIANT = "_벽면평행제외"
TERRAIN = REPO / "webapp" / "ortho_history" / "terrain"
SNAPS = SITE / "3D화면저장"


def joints_of(r):
    j = {f"Set {d.set_id}": (d.mean_dip, d.mean_dip_direction) for d in r.discontinuity_sets}
    for f in r.trace_families:
        if f.linked_set is None and f.dip is not None:
            j[f"트레이스 F{f.family_id}"] = (f.dip, f.dip_direction)
    return j


def numbers(r) -> dict:
    c = r.config
    faces = [p for d in r.discontinuity_sets for p in d.patches]
    return {"faces": len(faces), "area_m2": round(sum(p.area for p in faces)), "traces": len(r.traces),
            "trace_m": round(sum(t.length for t in r.traces)) if r.traces and hasattr(r.traces[0], "length") else None,
            "families": [{"angle": round(f.apparent_angle), "n": f.n_traces, "linked": f.linked_set,
                          "dip": None if f.dip is None else round(f.dip),
                          "dd": None if f.dip_direction is None else round(f.dip_direction) % 360}
                         for f in r.trace_families],
            "removed": c.get("faces_removed"), "auto": c.get("auto_scale_applied"),
            "lidar": {k: v for k, v in (c.get("lidar_rock") or {}).items() if k != "file"},
            "candidates": [{"dip": round(x["dip"]), "dd": round(x["dip_direction"]), "contrast": round(x["contrast"], 1)}
                           for x in (c.get("set_detection") or {}).get("candidates", [])],
            "fixed": (c.get("set_detection") or {}).get("fixed_sets"),
            "core": (c.get("section") or {}).get("core_chainage"),
            "windows": len(r.face_windows), "length_m": round(r.wall_frame.length),
            "above_ground": c.get("above_ground"), "gentle": gentle_on_ground(r)}


def gentle_on_ground(r, dip_max=30.0, near=1.0):
    """The gentle faces (dip <= 30°) and the share of their area within 1 m of the DTM right under them: bench
    floors and the quarry floor rather than joints in the wall."""
    from groundwater.discontinuity.cleaning import _at, ground_ceiling

    faces = [p for d in r.discontinuity_sets for p in d.patches if p.dip <= dip_max]
    dtm = SJ.find_ground(SITE)
    if not faces or dtm is None:
        return None
    xyz = np.array([p.centroid for p in faces])
    area = np.array([p.area for p in faces])
    z, _, t = ground_ceiling(dtm, (*xyz[:, :2].min(0), *xyz[:, :2].max(0)), 0.0)
    h = xyz[:, 2] - _at(z, t, xyz[:, 0], xyz[:, 1])
    on = np.abs(h) <= near
    return {"faces": len(faces), "area_m2": round(float(area.sum())), "on_ground_pct": round(100 * float(area[on].sum() / area.sum())),
            "above_3m_pct": round(100 * float(area[h > 3].sum() / area.sum()))}


def png(fig_bytes, path):
    if fig_bytes:
        path.write_bytes(fig_bytes)
        return True
    return False


def photo_cover(path, step=4):
    """West wall, photogrammetry only (the 10/21 facade image of the photogrammetry run, drawn from the RGB cloud
    before the LiDAR fill): netting and vegetation, where the photo shows no rock. Returns the areas (core)."""
    from groundwater.discontinuity.cleaning import netting_area
    from groundwater.discontinuity.vegetation import vegetation_mask

    r = load_results(SITE / "분석결과")
    R = r.facade_rgb
    rgb = np.moveaxis(R.array, 0, -1)
    res = abs(R.transform.a)
    valid = np.isfinite(r.facade_depth.array[0])
    net = netting_area(rgb, valid, res)
    veg = vegetation_mask(rgb, scale=255) & valid & ~net
    u = R.transform.c + (np.arange(valid.shape[1]) + 0.5) * R.transform.a
    core = r.config["section"]["core_chainage"]
    c = (u >= core[0]) & (u <= core[1])
    A = res * res
    out = {"rock_m2": round(float((valid & ~net & ~veg)[:, c].sum() * A)), "net_m2": round(float(net[:, c].sum() * A)),
           "veg_m2": round(float(veg[:, c].sum() * A)), "total_m2": round(float(valid[:, c].sum() * A))}
    out["net_pct"] = round(100 * out["net_m2"] / out["total_m2"])
    out["veg_pct"] = round(100 * out["veg_m2"] / out["total_m2"])

    # draw: the photo, netting tinted orange, vegetation green, outside the core greyed
    sl = (slice(None, None, step), slice(None, None, step))
    img = rgb[sl].astype(float) / 255
    for m, col in ((net[sl], (0.976, 0.451, 0.086)), (veg[sl], (0.086, 0.639, 0.290))):
        img[m] = img[m] * 0.35 + np.array(col) * 0.65
    img[~valid[sl]] = 1.0
    top = R.transform.f
    ext = [u[0] - res / 2, u[-1] + res / 2, top + valid.shape[0] * R.transform.e, top]
    rows = np.flatnonzero(valid[:, c].any(axis=1))                  # crop the empty sky / ground rows
    ylim = (top + (rows[-1] + 1) * R.transform.e, top + rows[0] * R.transform.e)
    plt = report._setup_mpl()
    from matplotlib.patches import Patch

    aspect = (ylim[1] - ylim[0]) / (core[1] - core[0])
    fig, ax = plt.subplots(figsize=(16, max(4.0, 16 * aspect)))
    ax.imshow(img, extent=ext)
    for x0, x1 in ((ext[0], core[0]), (core[1], ext[1])):
        if x1 > x0:
            ax.axvspan(x0, x1, color="#6b7280", alpha=0.55, lw=0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(*ylim)
    ax.legend([Patch(color="#f97316"), Patch(color="#16a34a")],
              [f"낙석방지망 {out['net_pct']}%", f"식생 {out['veg_pct']}%"], loc="upper right", fontsize=15, framealpha=0.9)
    ax.set_xlabel("측점 (m)", fontsize=14)
    ax.set_ylabel("표고 (m, 해발)", fontsize=14)
    ax.tick_params(labelsize=12)
    path.write_bytes(report._png(fig, dpi=120))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--photo" in sys.argv:                      # only the photogrammetry-coverage figure, into full.json
        js = OUT / "full.json"
        data = json.loads(js.read_text(encoding="utf-8"))
        data["photo_cover"] = photo_cover(OUT / "photo_cover_W.png")
        js.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(data["photo_cover"])
        return
    data = {}
    for k, s in SECS.items():
        r = load_results(SITE / "구간" / s / "분석결과")
        n = numbers(r)
        fam = r.trace_families
        png(report.fig_trace_map(r, fam), OUT / f"traces_{k}.png")
        png(report.fig_stereonet(r, fam, {"method": "exponential_kamb", "sigma": 3, "weighted": False, "poles": True,
                                          "cmap": "Greys"}), OUT / f"density_{k}.png")
        png(report.fig_kinematics(r), OUT / f"kinbar_{k}.png")
        # the failure-zone map with the dashboard's defaults
        joints = joints_of(r)
        phi = float(r.config.get("kinematics", {}).get("friction_deg", K.FRICTION_DEG))
        veg = (r.config.get("vegetation") or {}).get("exg_threshold", 0.06)
        kmap = K.kinematic_map(r, joints, phi, K.LATERAL_DEG, 1.0, 3.0, 2.0, True, veg)
        fig = KV.map_figure(r, kmap, list(K.MODES))
        fig.update_layout(title=None)
        (OUT / f"kinmap_{k}.png").write_bytes(static_map.preview_image(fig, width_px=1800, max_height_px=800))
        n["kinmap"] = KV.summary(kmap)
        n["kinmap_cells"] = int(kmap.valid.sum())
        face = K.representative_face(r.face_windows)
        st_png, counts = KV.stereonet_panels(r, face, phi, K.LATERAL_DEG)
        (OUT / f"markland_{k}.png").write_bytes(st_png)
        n["face"] = [round(face[0]), round(face[1]) % 360]
        n["markland"] = {m: {kk: int(v) for kk, v in c.items() if isinstance(v, (int, np.integer))} for m, c in counts.items()}
        # structural seepage potential
        smap = hydro.seepage_potential(r)
        png(report.fig_seepage(r, smap), OUT / f"seepage_{k}.png")
        if smap is not None:
            cl = smap.classes[smap.valid] if hasattr(smap, "valid") else smap.classes.ravel()
            n["seepage"] = {"cells": int(np.size(cl)), "high": int(np.sum(cl == 2)), "mid": int(np.sum(cl == 1)),
                            "zones": int(len(smap.zones)) if getattr(smap, "zones", None) is not None else None}
        data[k] = n
        print(k, {x: n[x] for x in ("faces", "traces", "removed", "seepage")}, flush=True)

    # underground: plan section at 60 m (the re-runs without wall-parallel normals)
    F = SJ.load_faces(SITE, list(SECS.values()), 5.0, None, VARIANT)
    C = SJ.continuity(F)
    lo, hi = F[["x", "y"]].min().to_numpy() - 60, F[["x", "y"]].max().to_numpy() + 60
    ground = SJ.ground_grid(SJ.find_ground(SITE), (lo[0], lo[1], hi[0], hi[1]), 2.0)
    bg = LV.ortho_image(LV.find_orthos(SITE)[0], [lo[0], hi[0], lo[1], hi[1]], 1600)
    for z0 in (60.0, 80.0):
        fig = V.plan_figure(F, C, 3.0, z0, ground, bg)
        fig.update_layout(title=None, showlegend=False)
        fig.for_each_trace(lambda t: t.type != "image" and t.update(showlegend=False))
        (OUT / f"sj_plan_{int(z0)}.png").write_bytes(static_map.preview_image(fig, width_px=1150, max_height_px=1150))
    data["groups"] = sorted({(int(g), lab) for g, lab in zip(F["group"], F["group_label"])})

    # the 10/23 DSM layers of the orthophoto viewer
    from PIL import Image

    for name in ("dsm_relief", "dsm_change"):
        src = TERRAIN / f"{name}.webp"
        if src.exists():
            im = Image.open(src)
            if name == "dsm_change":                       # the change over a grey shaded relief
                base = Image.open(TERRAIN / "dsm_relief.webp").convert("L").convert("RGBA").resize(im.size)
                im = Image.alpha_composite(base, im.convert("RGBA"))
            im.thumbnail((1600, 1600))
            im.convert("RGB").save(OUT / f"{name}.png")
    meta = TERRAIN / "dsm_meta.json"
    if meta.exists():
        data["dsm"] = json.loads(meta.read_text(encoding="utf-8"))

    # the LiDAR section saved from the section tool, the top views of the tiled model
    sec = sorted((SITE / "구간" / "E_동측벽" / "분석결과" / "단면확인").glob("*.png"))
    if sec:
        shutil.copy(sec[0], OUT / "lidar_section.png")
    for name in ("1_절리군_전체", "2_파괴형태_전체"):
        src = SNAPS / f"현장전체_절리_파괴형태_3D장면_{name}.png"
        if src.exists():
            shutil.copy(src, OUT / f"tiles_{name}.png")
    (OUT / "full.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
