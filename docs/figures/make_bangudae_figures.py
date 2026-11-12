"""Large-type figures from the real Ulsan (Bangudae petroglyph cliff) analysis for the workflow slides: the
program's rehearsal on real data before Iksan. Writes PNGs to data/bangudae/ (make_workflow_slides.py reads
them; re-run this after a new analysis).

    .venv\\Scripts\\python.exe docs\\figures\\make_bangudae_figures.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "data" / "bangudae"
RESULT = REPO / "Filed_Data" / "Ulsan" / "분석결과"
sys.path.insert(0, str(REPO / "src"))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    import mplstereonet  # noqa: F401  (registers the stereonet projection)

    from groundwater import hydro
    from groundwater.pipeline import load_results
    from groundwater.report import FAMILY_COLORS, SET_COLORS
    from groundwater.viz import grouting3d as g
    from groundwater.viz import joints3d as J
    from groundwater.viz import section

    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    OUT.mkdir(parents=True, exist_ok=True)
    r = load_results(str(RESULT))
    sets_df = hydro.joint_sets_for_flow(r)

    def save(fig, name, **kw):
        fig.savefig(OUT / f"{name}.png", dpi=150, facecolor="white", bbox_inches="tight", **kw)
        plt.close(fig)
        print("wrote", OUT / f"{name}.png")

    # facade image (shared by the maps)
    fr = r.facade_rgb
    t = fr.transform
    img = np.moveaxis(fr.array, 0, -1).copy()
    img[~np.isfinite(r.facade_depth.array[0])] = 255

    def facade_crop(u0, u1, v0, v1, k=None):
        c0, c1 = int((u0 - t.c) / t.a), int((u1 - t.c) / t.a)
        r0, r1 = int((t.f - v1) / -t.e), int((t.f - v0) / -t.e)
        crop = img[max(r0, 0):r1, max(c0, 0):c1]
        k = k or max(1, int(np.ceil(crop.shape[1] / 3000)))
        return crop[::k, ::k], [u0, u1, v0, v1]

    # 1. stereonets: the poles, and their density map (the dashboard's density option: Kamb, exponential
    # smoothing, sigma 3) -- drawn alike (same frame, labels and set planes) to stand side by side
    from matplotlib.colors import ListedColormap

    def stereo(name, draw):
        fig = plt.figure(figsize=(5.2, 5.2))
        ax = fig.add_subplot(111, projection="stereonet")
        out = draw(ax)
        for d in r.discontinuity_sets:
            ax.plane(d.mean_dip_direction - 90, d.mean_dip, color=SET_COLORS[d.set_id % len(SET_COLORS)], lw=2)
        ax.grid(True)
        ax.set_azimuth_ticks([0, 90, 180, 270], labels=["N", "E", "S", "W"], fontsize=17)
        ax.tick_params(labelsize=17)
        fig.savefig(OUT / f"{name}.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print("wrote", OUT / f"{name}.png")
        return out

    def poles(ax):
        for d in r.discontinuity_sets:
            ax.pole([p.dip_direction - 90 for p in d.patches], [p.dip for p in d.patches], "o", ms=3.2,
                    color=SET_COLORS[d.set_id % len(SET_COLORS)])

    pts = np.array([(p.dip_direction - 90, p.dip) for d in r.discontinuity_sets for p in d.patches])

    def density(ax):
        cols = plt.get_cmap("YlOrBr")(np.linspace(0.0, 1.0, 16))
        cols[0] = (1.0, 1.0, 1.0, 1.0)                      # near-uniform density left white
        kw = dict(measurement="poles", method="exponential_kamb", sigma=3.0, gridsize=180)
        cf = ax.density_contourf(pts[:, 0], pts[:, 1], levels=16, cmap=ListedColormap(cols), **kw)
        ax.density_contour(pts[:, 0], pts[:, 1], levels=8, colors="#374151", linewidths=0.5, alpha=0.5, **kw)
        return float(np.nanmax(cf.levels))

    stereo("stereonet", poles)
    peak = stereo("stereonet_density", density)
    (OUT / "stereonet_density.json").write_text(f'{{"max_sigma": {peak:.0f}}}', encoding="utf-8")

    # 2. trace map (panel + the wide fissure) ----------------------------------------------------------
    # a family linked to a 3D set takes that set's colour (as in the stereonet, the set map and the sections)
    def fam_colour(f):
        if f.linked_set is not None:
            return SET_COLORS[f.linked_set % len(SET_COLORS)]
        return FAMILY_COLORS[f.family_id % len(FAMILY_COLORS)]

    crop, ext = facade_crop(28, 58, 76.5, 92)
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.imshow(crop, extent=ext)
    fam = {id(m): f for f in r.trace_families for m in f.members}
    for tr in r.traces:
        f = fam.get(id(tr))
        col = fam_colour(f) if f else "#d1d5db"
        ax.plot(tr.coords[:, 0], tr.coords[:, 1], color=col, lw=2.6 if f else 1.6)
    from groundwater.discontinuity.trace_sets import family_sets, family_tag

    hand = [Line2D([], [], color=fam_colour(f), lw=5,
                   label=f"F{f.family_id} {f.apparent_angle:.0f}° · {f.n_traces}개 → "
                         + family_tag(family_sets(f, r.wall_frame, r.discontinuity_sets)))
            for f in r.trace_families] + [Line2D([], [], color="#d1d5db", lw=5, label="방향군 밖")]
    ax.legend(handles=hand, loc="lower left", fontsize=17, framealpha=0.9)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("측점 (m)", fontsize=18)
    ax.set_ylabel("표고 (m)", fontsize=18)
    ax.tick_params(labelsize=16)
    save(fig, "traces")

    # 3. seepage potential over the whole wall ----------------------------------------------------------
    sm = hydro.seepage_potential(r)
    crop, ext = facade_crop(2, 76, 76, 106)
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.imshow(crop, extent=ext)
    rows, cols = sm.classes.shape
    over = np.zeros((rows, cols, 4))
    over[sm.classes == 2] = [0.86, 0.15, 0.15, 0.62]
    over[sm.classes == 1] = [0.98, 0.62, 0.10, 0.38]
    ax.imshow(over, extent=[sm.u_min, sm.u_min + cols * sm.cell_m, sm.v_max - rows * sm.cell_m, sm.v_max],
              interpolation="nearest")
    for z in sm.zones.head(5).itertuples():
        u0, u1 = (float(x) for x in z.chainage_m.split("–"))
        v0, v1 = (float(x) for x in z.elevation_m.split("–"))
        ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec="white", lw=2.6))
        ax.text(u0, v1 + 0.25, f"#{z.zone}", color="white", fontsize=17, fontweight="bold",
                bbox=dict(fc="#111827", ec="none", pad=1.5))
    ax.legend(handles=[Patch(color=(0.86, 0.15, 0.15, 0.75), label="상 (상위 15%)"),
                       Patch(color=(0.98, 0.62, 0.10, 0.6), label="중 (다음 35%)")], loc="upper right", fontsize=17)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("측점 (m)", fontsize=18)
    ax.set_ylabel("표고 (m)", fontsize=18)
    ax.tick_params(labelsize=16)
    save(fig, "seepage")
    import json

    top = sm.zones.head(5)
    uu = [float(x) for c in top.chainage_m for x in c.split("–")]
    vv = [float(x) for c in top.elevation_m for x in c.split("–")]
    (OUT / "seepage.json").write_text(json.dumps({"zones": len(sm.zones), "top5_chainage": [min(uu), max(uu)],
                                                  "top5_elevation": [min(vv), max(vv)]}), encoding="utf-8")

    # 3a. the four indicators one by one over the wall-parallel orthomosaic itself (the dashboard's seepage map
    # view: background = 고해상도 정사영상, 표시할 값 = one indicator; low values fade out) -------------------------
    indicator_maps(r, sm, plt)

    # 3b. failure modes: the stereographic test for the wall, and where on the wall each mode is possible
    # (the dashboard's 안정성 검토), the map in the same frame as the seepage map ------------------------------
    import json

    from groundwater.discontinuity import kinematics as K
    from groundwater.viz import kinematic_view as KV

    kjoints = {f"Set {d.set_id}": (d.mean_dip, d.mean_dip_direction) for d in r.discontinuity_sets}
    for f in r.trace_families:
        if f.linked_set is None and f.dip is not None:
            kjoints[f"트레이스 F{f.family_id}"] = (f.dip, f.dip_direction)
    phi = float(r.config.get("kinematics", {}).get("friction_deg", K.FRICTION_DEG))
    kface = K.representative_face(r.face_windows)
    png, kcounts = KV.stereonet_panels(r, kface, phi, slide=True)
    (OUT / "kinematic_stereonet.png").write_bytes(png)
    print("wrote", OUT / "kinematic_stereonet.png")
    km = K.kinematic_map(r, kjoints, phi, exg_threshold=(r.config.get("vegetation") or {}).get("exg_threshold", 0.06))
    crop, ext = facade_crop(2, 76, 76, 106)
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.imshow(crop, extent=ext)
    H, W = km.valid.shape
    ax.imshow(KV.overlay_rgba(km, gap=False), extent=[km.u_min, km.u_min + W * km.cell_m, km.v_max - H * km.cell_m, km.v_max],
              interpolation="nearest")
    ax.legend(handles=[Patch(color=KV.COUNT_COLOURS[n], label=f"{n}가지 가능") for n in (1, 2, 3)],
              title="평면·쐐기·전도 중", title_fontsize=15, loc="upper right", fontsize=16, framealpha=0.9)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("측점 (m)", fontsize=18)
    ax.set_ylabel("표고 (m)", fontsize=18)
    ax.tick_params(labelsize=16)
    save(fig, "kinematic_map")
    rock = int(km.valid.sum())
    (OUT / "kinematic.json").write_text(json.dumps({
        "face": [round(kface[0]), round(kface[1])], "friction": phi,
        "stereonet": {m: round(c["share"] * 100) for m, c in kcounts.items()},
        "map_share": {m: round(100 * km.modes[m].sum() / max(rock, 1)) for m in K.MODES}}, ensure_ascii=False), encoding="utf-8")

    # 4. grouting: network holes on the facade + a cross-section ----------------------------------------
    rock = J.RockTest(r)
    obs = J.observed(r, sm, depth_factor=2.0, max_depth=8.0, rock=rock)
    holes = g.network_layout(r, sm, obs, sets_df, rock=rock)
    cov = J.coverage(holes, obs)
    us = np.array([h["u"] for h in holes])
    crop, ext = facade_crop(max(2, us.min() - 4), us.max() + 4, 76, 98)
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.imshow(crop, extent=ext)
    over = np.zeros((rows, cols, 4))
    over[sm.classes == 2] = [0.86, 0.15, 0.15, 0.35]
    ax.imshow(over, extent=[sm.u_min, sm.u_min + cols * sm.cell_m, sm.v_max - rows * sm.cell_m, sm.v_max],
              interpolation="nearest")
    for h in holes:
        e = r.wall_frame.to_wall(np.array([h["collar"], h["collar"] + h["length"] * h["dir"]]))
        ax.plot(e[:, 0], e[:, 1], color="#7f1d1d", lw=3.2, solid_capstyle="round")
        ax.scatter(e[0, 0], e[0, 1], s=90, c="#fde047", edgecolors="black", linewidths=1.3, zorder=3)
    ax.legend(handles=[Patch(color=(0.86, 0.15, 0.15, 0.6), label="유출 가능성 '상'"),
                       Line2D([], [], color="#7f1d1d", lw=4, marker="o", mfc="#fde047", mec="black", ms=11,
                              label=f"표적 천공 {len(holes)}공 (입구 · 벽면 투영)")], loc="lower left", fontsize=16,
              framealpha=0.9)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("측점 (m)", fontsize=18)
    ax.set_ylabel("표고 (m)", fontsize=18)
    ax.tick_params(labelsize=16)
    save(fig, "grouting_plan")

    names = list(dict.fromkeys(sets_df["name"]))
    colour = lambda n: "#9ca3af" if n not in names else g._set_color(n, names.index(n))  # noqa: E731
    u0 = float(us[np.argmax([(np.abs(us - u) < 1.5).sum() for u in us])])
    # the dashboard's own section drawing (viz/static_map), larger type for the slide
    sf = section.section_figure(r, u0, holes, obs, cov, colour, band=1.5, depth_shown=10.0)
    sf.update_layout(title=f"단면 측점 {u0:.0f} m").update_xaxes(title="거리 w (m, − 암반 안쪽)")
    still(sf, "grouting_section", 460, 394, font_scale=1.5, legend=False)
    print(f"network: {len(holes)} holes, {sum(h['length'] for h in holes):.0f} m, high joints {cov['n_high_hit']}/{cov['n_high']}")

    # 5. main joint planes through the rock block (block diagram) + a cross-section ----------------------------
    major_figures(r, sets_df, rock, colour, save, plt)


INDICATORS = ["p21", "openness", "connectivity", "wetness"]


def indicator_maps(r, sm, plt):
    """seepage_ind_<name>.png: each indicator of the seepage score over the wall-parallel orthomosaic (planar.wall_image),
    the colours of the dashboard (seepage_view: low values nearly see-through), the five best zones boxed."""
    import json

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    from groundwater import hydro
    from groundwater.facade import planar
    from groundwater.viz import seepage_view as SV

    folder = (r.config.get("planar") or {}).get("folder")
    _, ortho = planar.find_planar_pair(folder)
    reg = planar.load_registration(folder, r.config.get("crs"))
    bg, ext, res = planar.wall_image(r, ortho, reg["matrix"], (2.0, 76.0), (76.0, 106.0), max_px=3200)
    rows, cols = sm.classes.shape
    cext = [sm.u_min, sm.u_min + cols * sm.cell_m, sm.v_max - rows * sm.cell_m, sm.v_max]
    info = {"ortho_mm": round(res * 1000, 1), "weights": hydro.WEIGHTS, "range": {}}
    for name in INDICATORS:
        label, unit, cmap, _ = SV.LAYERS[name]
        vals = SV.layer_values(sm, name)
        lo, hi = SV._range(vals)
        fig, ax = plt.subplots(figsize=(7.6, 3.05))
        ax.imshow(bg, extent=ext)
        ax.imshow(SV._rgba(sm, name, alpha=215), extent=cext, interpolation="nearest")
        for z in sm.zones.head(5).itertuples():
            u0, u1 = (float(x) for x in z.chainage_m.split("–"))
            v0, v1 = (float(x) for x in z.elevation_m.split("–"))
            ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec="white", lw=1.4))
            ax.text(u0, v1 + 0.3, f"#{z.zone}", color="white", fontsize=9, fontweight="bold",
                    bbox=dict(fc="#7f1d1d", ec="none", pad=1.0))
        # the colour key as drawn: the colour map with its fade, over the mid-grey of a rock face
        t = np.linspace(0, 1, 256)
        cm = plt.get_cmap(cmap)(0.12 + 0.88 * t)
        a = SV._alpha(t)[:, None] * 215 / 255
        key = LinearSegmentedColormap.from_list("k", np.clip(cm[:, :3] * a + 0.62 * (1 - a), 0, 1))
        sc = plt.cm.ScalarMappable(cmap=key, norm=plt.Normalize(lo, hi))
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.012)
        cb.set_label(unit, fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax.set_title(f"{label}  ·  가중 {hydro.WEIGHTS[name]:.2f}", fontsize=13, fontweight="bold", loc="left")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.tick_params(labelsize=10)
        ax.set_xlabel("측점 (m)", fontsize=11, labelpad=1)
        ax.set_ylabel("표고 (m)", fontsize=11)
        fig.savefig(OUT / f"seepage_ind_{name}.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        info["range"][name] = [round(lo, 2), round(hi, 2)]
        print("wrote", OUT / f"seepage_ind_{name}.png")
    (OUT / "seepage_indicators.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


MAJOR_WINDOW = (24.0, 40.0)        # chainage window of the block diagram (the dashboard's default window)
MAJOR_DEPTH = 8.0                  # block depth behind the face (m)
MAJOR_MAX, MAJOR_MIN_LEN, MAJOR_SEP = 4, 10.0, 1.5   # parallel planes per set: at most 4, each with >= 10 m of evidence,
                                                     # >= 1.5 m apart (the dashboard's sliders)


def still(fig, name, w, h, **kw):
    """A Plotly section as the dashboard draws it (viz/static_map), `w` x `h` slide pixels at 2x."""
    from groundwater.viz import static_map

    data = static_map.preview_image(fig, width_px=w, height_px=h, **kw)
    path = OUT / f"{name}.png"
    if data[:3] == b"\xff\xd8\xff":                 # a photo map comes back as JPEG
        from PIL import Image
        import io

        Image.open(io.BytesIO(data)).save(path)
    else:
        path.write_bytes(data)
    print("wrote", path)


SEC_V = 88.0                       # slide 13 sections: plan elevation (m), wall-parallel depth (m); cross section at
SEC_D = 2.0                        # the middle of the window
SEC_PANELS = {"a": 364, "b": 290, "c": 250}       # image size (slide px) of the three sections, sized to their shapes
SEC_PANEL_H = 404                                 # (make_workflow_slides.s_bang_structure lays them out)


def major_figures(r, sets_df, rock, colour, save, plt):
    """The main joint plane of each set through a rock block behind the face (setmap3d.major_model, as in the
    dashboard's large-scale view), drawn as a block diagram, a cross-section through it and the plane list."""
    import json

    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    from groundwater.viz import section
    from groundwater.viz import setmap3d as S

    prof = section._profile_points(r)
    v_lo, v_hi = (float(x) for x in np.percentile(prof[:, 1], [1, 99]))
    planes = S.major_planes(r, sets_df, MAJOR_WINDOW, per_set=MAJOR_MAX, min_length_m=MAJOR_MIN_LEN, min_sep_m=MAJOR_SEP)
    m = S.major_model(r, rock, planes, MAJOR_WINDOW, (v_lo, v_hi), MAJOR_DEPTH, cells=90)
    u0, u1, v0, v1, back, w_front = m["box"]
    # true plan: map coordinates turned so the wall direction at the middle of the window runs along x (flat planes
    # stay flat where the wall line bends; the dashboard's default 3D coordinates)
    disp = S.Disp(r.wall_frame, 0.5 * (u0 + u1))
    facing = {pl["name"]: abs(float(pl["normal"] @ m["outward"])) > np.cos(np.radians(30)) for pl in m["planes"]}

    fig = plt.figure(figsize=(9.5, 9.0))
    ax = fig.add_subplot(projection="3d")
    V, F, own = m["mesh"]["V"], m["mesh"]["F"], m["mesh"]["owner"]
    D = disp(V)
    for k, j in enumerate(m["joints"]):
        tri = D[F[own == k]]
        ax.add_collection3d(Poly3DCollection(tri, facecolor=colour(j.name), edgecolor="none",
                                             alpha=0.22 if facing.get(j.name) else 0.5))
    # where each plane meets the face, its traces on the block's sides / top / back, the intersections
    for name, sg in m["lines"]["face"]:
        if len(sg):
            ax.add_collection3d(Line3DCollection([disp(s_) for s_ in sg], colors=colour(name), linewidths=2.2))
    for name, sg in m["lines"]["sides"]:
        if len(sg):
            ax.add_collection3d(Line3DCollection([disp(s_) for s_ in sg], colors=colour(name), linewidths=1.6))
    for run in m["intersections"]:
        q = disp(run)
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color="black", lw=3.2)
    c8 = [(u0, v0, w_front), (u1, v0, w_front), (u1, v0, back), (u0, v0, back),
          (u0, v1, w_front), (u1, v1, w_front), (u1, v1, back), (u0, v1, back)]
    box_pts = []
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        q = disp.polyline([c8[i], c8[j]])          # the block follows the wall: its edges are drawn in true plan
        box_pts.append(q)
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color="#6b7280", lw=1.0, ls="--")
    B = np.vstack(box_pts)
    (x0, y0, z0), (x1, y1, z1) = B.min(axis=0), B.max(axis=0)
    for (p, t_), pl in zip(m["labels"], m["planes"]):
        q = disp(p[None])[0]
        if pl["rank"]:                              # only each set's best plane is labelled; its parallels share its colour
            continue
        ax.text(q[0], q[1], q[2] + 0.6, t_.replace("  ", " "), fontsize=14, fontweight="bold", color="#111827",
                bbox=dict(fc="white", ec="#9ca3af", alpha=0.9, pad=1.5), zorder=1e6)   # over the planes
    # where sections A (horizontal) and C (cross) cut the block, as the dashed boxes of the dashboard's 3D view
    u_s = 0.5 * (u0 + u1)
    for tag, corners, at in (("A", [(u0, SEC_V, w_front), (u1, SEC_V, w_front), (u1, SEC_V, back), (u0, SEC_V, back)], 1),
                             ("C", [(u_s, v0, w_front), (u_s, v0, back), (u_s, v1, back), (u_s, v1, w_front)], 3)):
        q = np.vstack([disp.polyline([corners[k], corners[(k + 1) % 4]]) for k in range(4)])
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color=S.REF_LINE, lw=1.8, ls=(0, (5, 3)))
        c = disp(np.array([corners[at]]))[0]
        ax.text(c[0], c[1], c[2], f" {tag} ", fontsize=14, fontweight="bold", color="white",
                bbox=dict(fc=S.REF_COLOURS[tag], ec="none", pad=2.5, boxstyle="round,pad=0.25"))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=24, azim=-122)
    ax.set_xlabel("벽 방향 거리 (m, 직선)", fontsize=14, labelpad=10)
    ax.set_ylabel("깊이 (m, 직선)", fontsize=14, labelpad=10)
    ax.set_zlabel("표고 (m)", fontsize=14, labelpad=8)
    ax.tick_params(labelsize=11)
    save(fig, "major_planes")

    # sections A (horizontal), B (wall-parallel) and C (cross) through the block, as the dashboard's section cards
    # draws them -- each showing where the other two cut it
    prof_s = {"uvw": prof}
    A = S.plan_section(prof_s, rock, m["joints"], m["mesh"], colour, SEC_V, MAJOR_WINDOW, depth_shown=MAJOR_DEPTH,
                       frame=r.wall_frame, refs={"u": u_s, "d": SEC_D})
    B = S.parallel_section(r, rock, m["joints"], m["mesh"], colour, SEC_D, refs={"v": SEC_V, "u": u_s},
                           u_range=(u0, u1))
    B.update_yaxes(range=[v0, v1])
    C = S.cross_section(prof_s, m["joints"], m["mesh"], colour, u_s, depth_shown=MAJOR_DEPTH,
                        refs={"v": SEC_V, "d": SEC_D}, rock=rock)
    C.update_yaxes(range=[v0, v1])
    # short titles for the slide (the dashboard's are explanations)
    A.update_layout(title=f"A 수평 단면 · 표고 {SEC_V:.0f} m").update_xaxes(title="벽 방향 거리 (m)").update_yaxes(title="깊이 (m)")
    B.update_layout(title=f"B 벽면 평행 · 깊이 {SEC_D:.0f} m").update_xaxes(title="측점 (m)").update_yaxes(title="표고 (m)")
    C.update_layout(title=f"C 횡단면 · 측점 {u_s:.0f} m").update_xaxes(title="깊이 (m)").update_yaxes(title="표고 (m)")
    import re

    def short(m_):
        return f"</b> {float(m_.group(1)):g} m"

    for f_ in (A, B, C):                  # badges as "C 32 m": the slide's panels are narrow
        for an in f_.layout.annotations:
            an.text = re.sub(r"</b> (?:측점|표고|깊이) ([0-9.]+) m", short, an.text)
    kw = dict(font_scale=1.2, legend=False)
    still(A, "major_sec_a", SEC_PANELS["a"], SEC_PANEL_H, **kw)
    still(B, "major_sec_b", SEC_PANELS["b"], SEC_PANEL_H, **kw)
    still(C, "major_sec_c", SEC_PANELS["c"], SEC_PANEL_H, **kw)

    rows = [{"name": f"{p['name']}-{S.plane_tag(p['rank'])}", "orient": f"{p['dip']:.0f}°/{p['dd']:03.0f}°",
             "length_m": round(p["length_m"], 1), "pieces": p["pieces"]} for p in m["planes"]]
    (OUT / "major_planes.json").write_text(json.dumps({"window": MAJOR_WINDOW, "depth": MAJOR_DEPTH, "planes": rows,
                                                       "max_per_set": MAJOR_MAX, "min_length_m": MAJOR_MIN_LEN,
                                                       "min_sep_m": MAJOR_SEP,
                                                       "intersections": len(m["intersections"]),
                                                       "sections": {"A_elevation": SEC_V, "B_depth": SEC_D,
                                                                    "C_chainage": u_s}},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print("major planes:", rows)


if __name__ == "__main__":
    main()
