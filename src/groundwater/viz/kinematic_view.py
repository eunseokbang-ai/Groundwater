"""Kinematic analysis views: the stereographic test (Markland) for a face orientation, and the map of where on
the wall planar, wedge and toppling failure are kinematically possible (discontinuity.kinematics) -- on the
facade photo, the 3D point cloud and the tiled model.
"""
import io

import numpy as np

from ..discontinuity import kinematics as K

MODE_COLOURS = {"평면파괴": "#c026d3", "쐐기파괴": "#f59e0b", "전도파괴": "#0ea5e9"}   # not the joint-set colours
MODE_SHORT = {"평면파괴": "평면", "쐐기파괴": "쐐기", "전도파괴": "전도"}


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


# --- the stereographic test --------------------------------------------------------------------------------------

def _zone(ax, plunges, trends, colour, alpha=0.22):
    """Fill the region bounded by the (plunge, trend) outline on a stereonet (lines, lower hemisphere)."""
    import mplstereonet

    lon, lat = mplstereonet.line(np.asarray(plunges, float), np.asarray(trends, float))
    ax.fill(lon, lat, color=colour, alpha=alpha, lw=0, zorder=1)
    ax.plot(np.r_[lon, lon[:1]], np.r_[lat, lat[:1]], color=colour, lw=1.2, zorder=2)


def _sector(p_lo, p_hi, t0, t1, n=60):
    t = np.linspace(t0, t1, n)
    return (np.r_[np.full(n, p_hi), np.linspace(p_hi, p_lo, 8), np.full(n, p_lo), np.linspace(p_lo, p_hi, 8)],
            np.r_[t, np.full(8, t1), t[::-1], np.full(8, t0)])


def domain_face(windows: list[dict], u_range) -> tuple[float, float] | None:
    """Face orientation of a chainage range (a structural domain): the 5 m windows it overlaps, each weighted by
    its overlap -- median dip, mean dip direction (kinematics.representative_face)."""
    sel = []
    for w in windows:
        ov = min(w["u1"], u_range[1]) - max(w["u0"], u_range[0])
        if ov > 0:
            sel.append({**w, "height_m": w.get("height_m", 1.0) * ov})
    return K.representative_face(sel)


def critical(result, face: tuple, friction_deg: float = K.FRICTION_DEG, lateral_deg: float = K.LATERAL_DEG,
             u_range=None, max_pairs: int = 4000) -> dict:
    """The stereographic test without the drawing: the joint faces (poles; with `u_range`, only the faces whose
    centre lies in that chainage range), the intersection lines of a random sample of pairs of faces of different
    sets, and which of them fall in each mode's critical zone for the face (dip, dip direction)."""
    fdip, fdd = face
    phi, lat = friction_deg, lateral_deg
    faces = [(d.set_id, p) for d in result.discontinuity_sets for p in d.patches]
    if u_range is not None and faces:
        u = result.wall_frame.to_wall(np.array([p.centroid for _, p in faces]))[:, 0]
        faces = [f for f, x in zip(faces, u) if u_range[0] <= x < u_range[1]]
    dip = np.array([p.dip for _, p in faces], float)
    dd = np.array([p.dip_direction for _, p in faces], float)
    sid = np.array([s for s, _ in faces], int)
    planar = np.asarray(K.planar_possible(dip, dd, fdip, fdd, phi, lat), bool)
    topple = np.asarray(K.toppling_possible(dip, dd, fdip, fdd, phi, lat), bool)
    rng = np.random.default_rng(0)
    pairs = []
    if len(faces) > 1 and len(np.unique(sid)) > 1:
        a = rng.integers(0, len(faces), max_pairs * 3)
        b = rng.integers(0, len(faces), max_pairs * 3)
        keep = sid[a] != sid[b]
        pairs = list(zip(a[keep][:max_pairs], b[keep][:max_pairs]))
    lines = [K.intersection((dip[i], dd[i]), (dip[j], dd[j])) for i, j in pairs]
    lines = np.array([x for x in lines if x is not None]) if pairs else np.zeros((0, 2))
    wedge = np.asarray(K.wedge_possible(lines[:, 0], lines[:, 1], fdip, fdd, phi), bool) if len(lines)         else np.zeros(0, bool)
    counts = {}
    for mode, crit in (("평면파괴", planar), ("쐐기파괴", wedge), ("전도파괴", topple)):
        counts[mode] = {"critical": int(crit.sum()), "total": len(crit), "share": crit.sum() / max(len(crit), 1)}
    return {"dip": dip, "dd": dd, "sid": sid, "planar": planar, "topple": topple, "lines": lines, "wedge": wedge,
            "counts": counts}


def domain_table(result, friction_deg: float = K.FRICTION_DEG, lateral_deg: float = K.LATERAL_DEG,
                 max_pairs: int = 1500) -> list[dict]:
    """The stereographic test strip by strip: each 5 m window's own face against the joint faces in it."""
    rows = []
    for w in result.face_windows:
        face = (w["face_dip"], w["face_dip_direction"])
        c = critical(result, face, friction_deg, lateral_deg, (w["u0"], w["u1"]), max_pairs)
        n = c["counts"]
        rows.append({"측점 (m)": f"{w['u0']:.0f}–{w['u1']:.0f}",
                     "벽면": f"{face[0]:.0f}°/{face[1]:03.0f}°" + (" 오버행" if w.get("overhang") else ""),
                     "절리면 수": n["평면파괴"]["total"],
                     "평면 %": round(100 * n["평면파괴"]["share"]), "쐐기 %": round(100 * n["쐐기파괴"]["share"]),
                     "전도 %": round(100 * n["전도파괴"]["share"])})
    return rows


def stereonet_panels(result, face: tuple, friction_deg: float = K.FRICTION_DEG, lateral_deg: float = K.LATERAL_DEG,
                     title: str = "", max_pairs: int = 4000, slide: bool = False, u_range=None) -> tuple[bytes, dict]:
    """Three stereonets (equal area, lower hemisphere) for the face (dip, dip direction): planar sliding and
    flexural toppling on the poles of the joint faces, wedge sliding on the intersection lines of faces of
    different sets. The critical zone of each mode is shaded; returns (PNG, counts). `u_range`: only the joint
    faces in that chainage range (a structural domain with its own face). `slide`: large type, short titles and
    no captions (for a presentation panel)."""
    from ..report import SET_COLORS, _setup_mpl

    plt = _setup_mpl()
    import mplstereonet  # noqa: F401

    fdip, fdd = face
    phi, lat = friction_deg, lateral_deg
    c = critical(result, face, phi, lat, u_range, max_pairs)
    dip, dd, sid, planar, topple, lines, wedge = (c[k] for k in ("dip", "dd", "sid", "planar", "topple", "lines",
                                                                  "wedge"))

    fig = plt.figure(figsize=(13.0, 4.9) if slide else (15.5, 5.9))
    axes = [fig.add_subplot(1, 3, i + 1, projection="stereonet") for i in range(3)]
    counts = {}
    for ax, mode in zip(axes, K.MODES):
        ax.plane(fdd - 90, fdip, color="#111827", lw=2.4, zorder=5)               # the face
        ax.grid(True, alpha=0.35)
        if slide:                                            # side by side: only N, the nets' E / W would touch
            ax.set_azimuth_ticks([0], labels=["N"], fontsize=15)
        else:
            ax.set_azimuth_ticks([0, 90, 180, 270], labels=["N", "E", "S", "W"], fontsize=11)
        col = MODE_COLOURS[mode]
        if mode == "쐐기파괴":
            ax.cone(90, 0, 90 - phi, facecolor="none", edgecolor="#6b7280", lw=1.3, ls="--", zorder=4)
            if np.tan(np.radians(phi)) < np.tan(np.radians(min(fdip, 89.9))):
                half = np.degrees(np.arccos(np.tan(np.radians(phi)) / np.tan(np.radians(min(fdip, 89.9)))))
                t = np.linspace(fdd - half, fdd + half, 90)
                app = np.degrees(np.arctan(np.tan(np.radians(min(fdip, 89.9))) * np.cos(np.radians(t - fdd))))
                _zone(ax, np.r_[np.full(len(t), phi), app[::-1]], np.r_[t, t[::-1]], col)
            for d in result.discontinuity_sets:
                ax.plane(d.mean_dip_direction - 90, d.mean_dip, color=SET_COLORS[d.set_id % len(SET_COLORS)], lw=1.6,
                         alpha=0.8)
            if len(lines):
                ax.line(lines[~wedge, 0], lines[~wedge, 1], "o", color="#9ca3af", ms=1.6, alpha=0.5, zorder=3)
                ax.line(lines[wedge, 0], lines[wedge, 1], "o", color=col, ms=2.6, mec="#111827", mew=0.3, zorder=6)
            n_crit, n_all = int(wedge.sum()), len(lines)
            what = "교선"
        else:
            ax.cone(90, 0, phi, facecolor="none", edgecolor="#6b7280", lw=1.3, ls="--", zorder=4)   # friction circle
            if mode == "평면파괴" and fdip > phi:
                _zone(ax, *_sector(90 - min(fdip, 90), 90 - phi, fdd + 180 - lat, fdd + 180 + lat), col)
            if mode == "전도파괴" and fdip - phi > 0:
                _zone(ax, *_sector(0, fdip - phi, fdd - lat, fdd + lat), col)
            crit = planar if mode == "평면파괴" else topple
            for s in np.unique(sid):
                m = sid == s
                c = SET_COLORS[int(s) % len(SET_COLORS)]
                ax.pole(dd[m & ~crit] - 90, dip[m & ~crit], "o", color=c, ms=2.2, alpha=0.45, zorder=3)
                ax.pole(dd[m & crit] - 90, dip[m & crit], "o", color=c, ms=4.2, mec="#111827", mew=0.6, zorder=6)
            n_crit, n_all = int(crit.sum()), len(crit)
            what = "극점"
        share = n_crit / max(n_all, 1)
        counts[mode] = {"critical": n_crit, "total": n_all, "share": share}
        if slide:
            ax.set_title(f"{mode} {share:.0%}", fontsize=21, pad=44, color=col, fontweight="bold")
        else:
            ax.set_title(f"{mode}  —  임계 {what} {n_crit}/{n_all} ({share:.0%})", fontsize=12.5, pad=22, color="#111827")
    if slide:
        fig.subplots_adjust(wspace=0.32)
    if not slide:
        where = f"측점 {u_range[0]:g}–{u_range[1]:g} m · 절리면 {len(dip)}개 · " if u_range is not None else ""
        fig.suptitle(title or f"{where}벽면 {fdip:.0f}°/{fdd:03.0f}° · 마찰각 {phi:g}° · 측방 한계 ±{lat:g}°",
                     fontsize=13, y=0.995)
        fig.text(0.5, 0.015, "검은 대원 = 벽면 · 점선 = 마찰각 · 색 영역 = 해당 파괴가 운동학적으로 가능한 범위 · "
                 "테두리 점 = 그 범위에 든 극점/교선 (색 = 절리군)", ha="center", fontsize=10, color="#475569")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue(), counts


# --- the map: cells coloured by failure mode ---------------------------------------------------------------------

COUNT_COLOURS = {1: "#c4b5fd", 2: "#8b5cf6", 3: "#4c1d95"}    # how many modes a cell has (all modes shown)


def overlay_rgba(kmap, modes=K.MODES, px: int = 12, alpha: int = 170, gap: bool = True) -> np.ndarray:
    """RGBA image of the map, `px` pixels per cell (top row = highest elevation), a 1 px gap between cells.
    One mode: its cells in its colour. Several: each cell coloured by how many of them are possible there."""
    H, W = kmap.valid.shape
    out = np.zeros((H * px, W * px, 4), np.uint8)
    shown = [m for m in K.MODES if m in modes]
    if not shown:
        return out
    count = np.sum([kmap.modes[m] for m in shown], axis=0)
    block = np.ones((px, px), bool)
    if gap:
        block[-1, :] = block[:, -1] = False
    for n in range(1, len(shown) + 1):
        mask = np.kron(count == n, block).astype(bool)
        colour = MODE_COLOURS[shown[0]] if len(shown) == 1 else COUNT_COLOURS[n]
        out[mask, :3] = _rgb(colour)
        out[mask, 3] = alpha
    return out


def summary(kmap) -> list[dict]:
    """Area (m2) and share of the rock face per mode, with the joints that cause it."""
    cell2 = kmap.cell_m ** 2
    rock = int(kmap.valid.sum())
    rows = []
    for m in K.MODES:
        mask = kmap.modes[m]
        joints = {}
        for s in kmap.joints[m][mask]:
            for j in s.split(", "):
                joints[j] = joints.get(j, 0) + 1
        rows.append({"파괴 형태": m, "면적 (m²)": round(mask.sum() * cell2, 1), "암면 대비": f"{mask.sum() / max(rock, 1):.1%}",
                     "원인 절리군": ", ".join(f"{k} ({v * cell2:.0f} m²)" for k, v in sorted(joints.items(), key=lambda kv: -kv[1]))})
    return rows


def map_figure(result, kmap, modes=K.MODES):
    """Plotly figure: the facade photo with the failure-mode cells over it (and their details on hover)."""
    import base64

    import cv2
    import plotly.graph_objects as go

    from .. import report

    rgb, ext = report.facade_preview(result, 3000)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb[::-1]), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    dx = (ext[1] - ext[0]) / rgb.shape[1]
    dy = (ext[3] - ext[2]) / rgb.shape[0]
    fig = go.Figure(go.Image(source="data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(),
                             x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy, hoverinfo="skip"))
    ov = overlay_rgba(kmap, modes)
    ok, png = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(ov[::-1]), cv2.COLOR_RGBA2BGRA))
    H, W = kmap.valid.shape
    px = ov.shape[0] / H
    d = kmap.cell_m / px
    fig.add_trace(go.Image(source="data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
                           x0=kmap.u_min + d / 2, dx=d, y0=kmap.v_max - H * kmap.cell_m + d / 2, dy=d, hoverinfo="skip"))
    shown = [m for m in K.MODES if m in modes]
    legend = ([(shown[0], MODE_COLOURS[shown[0]])] if len(shown) == 1 else
              [(f"{n}가지 가능", COUNT_COLOURS[n]) for n in range(1, len(shown) + 1)])
    for name, colour in legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", size=13, color=colour)))
    # hover: one invisible marker per flagged cell
    rr, cc = np.nonzero(np.any([kmap.modes[m] for m in modes], axis=0)) if modes else (np.zeros(0, int),) * 2
    if len(rr):
        txt = []
        for r, c in zip(rr, cc):
            parts = [f"{MODE_SHORT[m]}: {kmap.joints[m][r, c]}" for m in modes if kmap.modes[m][r, c]]
            txt.append(f"벽면 {kmap.face_dip[r, c]:.0f}°/{kmap.face_dd[r, c]:03.0f}°"
                       + (" (오버행)" if kmap.overhang[r, c] else "") + "<br>" + "<br>".join(parts))
        fig.add_trace(go.Scatter(x=kmap.u_min + (cc + 0.5) * kmap.cell_m, y=kmap.v_max - (rr + 0.5) * kmap.cell_m,
                                 mode="markers", marker=dict(size=8, opacity=0), text=txt, hoverinfo="text",
                                 showlegend=False))
    what = shown[0] if len(shown) == 1 else " · ".join(MODE_SHORT[m] for m in shown) + " 중 가능한 형태 수"
    fig.update_layout(title=f"파괴 가능 구역 — {what} (셀마다 그 자리 벽면 방향 × 근처 절리군)", legend=dict(orientation="h"))
    fig.update_xaxes(title="측점 (m)")
    fig.update_yaxes(title="표고 (m)", autorange=True)
    return fig


def scene_figure(result, kmap, modes=K.MODES, npts: int = 400_000, opacity: float = 0.6, cloud_opacity: float = 1.0):
    """Plotly 3D figure (map coordinates): the point cloud in photo colours with the failure-mode cells draped on
    its front surface, one layer per mode (toggled in the legend)."""
    import plotly.graph_objects as go

    from . import setmap3d

    pts = np.asarray(result.point_cloud.points)
    idx = np.random.default_rng(0).choice(len(pts), min(npts, len(pts)), replace=False)
    P = pts[idx]
    origin = np.array([np.median(P[:, 0]), np.median(P[:, 1]), 0.0])
    col = np.full((len(P), 3), 150, np.uint8)
    if result.facade_rgb is not None:
        uv = result.wall_frame.to_wall(P)
        r_, c_ = result.facade_rgb.world_to_pixel(uv[:, 0], uv[:, 1])
        ok = (r_ >= 0) & (r_ < result.facade_rgb.array.shape[1]) & (c_ >= 0) & (c_ < result.facade_rgb.array.shape[2])
        col[ok] = np.moveaxis(result.facade_rgb.array[:, r_[ok], c_[ok]], 0, -1)
    traces = [dict(type="scatter3d", x=(P[:, 0] - origin[0]).astype(np.float32), y=(P[:, 1] - origin[1]).astype(np.float32),
                   z=P[:, 2].astype(np.float32), mode="markers", opacity=cloud_opacity, name="암벽 (점군)",
                   marker=dict(size=1.6, color=["#%02x%02x%02x" % tuple(c) for c in col]), hoverinfo="skip")]
    layers = [(m, kmap.modes[m], MODE_COLOURS[m]) for m in K.MODES if m in modes]
    traces += setmap3d.cell_drape_traces(result, kmap.u_min, kmap.v_max, kmap.cell_m, layers, origin, opacity)
    fig = go.Figure(traces, skip_invalid=False, _validate=False)
    _, outward = result.wall_frame.axes_at(result.wall_frame.length / 2)
    fig.update_layout(scene=dict(aspectmode="data", camera=dict(eye=dict(x=float(outward[0]) * 1.6,
                                                                         y=float(outward[1]) * 1.6, z=0.4)),
                                 xaxis_title="동 X (m)", yaxis_title="북 Y (m)", zaxis_title="표고 Z (m)"),
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig


def tiles_overlays(result, kmap, modes=K.MODES, step_m: float = 0.1, offset_m: float = 0.14) -> list:
    """One draped image per failure mode for the tiled-model viewer (tiles3d.scene_json overlays), on a drape
    mesh of `step_m` lifted `offset_m` (+3 cm per mode) off the face."""
    from . import setmap3d

    out = []
    k = max(1, int(round(kmap.cell_m / 0.05)))
    for i, m in enumerate(mm for mm in K.MODES if mm in modes):
        rgba = np.zeros(kmap.valid.shape + (4,), np.uint8)
        rgba[kmap.modes[m], :3] = _rgb(MODE_COLOURS[m])
        rgba[kmap.modes[m], 3] = 170
        rgba = np.repeat(np.repeat(rgba, k, axis=0), k, axis=1)
        ov = setmap3d._image_overlay(result, rgba, kmap.u_min, kmap.v_max, kmap.cell_m / k, kmap.cell_m / k,
                                     f"파괴 가능 구역: {m}", "kinematic", step_m, offset_m + 0.03 * i,
                                     colors=[MODE_COLOURS[m]])
        if ov:
            out.append(ov)
    return out
