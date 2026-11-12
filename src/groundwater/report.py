"""Figures and a self-contained HTML report of one analysis run (분석결과/분석보고서.html).

The same figures are shown in the dashboard tabs, so what is on screen and what is handed in match.
"""
import base64
import datetime
import io
from pathlib import Path

import numpy as np
import pandas as pd

SET_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]
FAMILY_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


def family_colour(f) -> str:
    """A trace family in the colour of the 3D set it is linked to (as the stereonet, set map and sections);
    a trace-only family keeps a family colour."""
    if f.linked_set is not None:
        return SET_COLORS[f.linked_set % len(SET_COLORS)]
    return FAMILY_COLORS[f.family_id % len(FAMILY_COLORS)]
PREVIEW_MAX_PX = 6000
REPORT_NAME = "분석보고서.html"


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in have] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _png(fig, dpi=110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def facade_preview(result, max_px: int = PREVIEW_MAX_PX):
    """(RGB uint8 image, extent [u_min, u_max, v_min, v_max]) of the facade, at most `max_px` wide/high."""
    import cv2

    rgb = np.moveaxis(result.facade_rgb.array, 0, -1).copy()
    rgb[~np.isfinite(result.facade_depth.array[0])] = 0
    h, w = rgb.shape[:2]
    k = max(1.0, max(h, w) / max_px)
    if k > 1:
        rgb = cv2.resize(rgb, (max(1, round(w / k)), max(1, round(h / k))), interpolation=cv2.INTER_AREA)
    t = result.facade_rgb.transform
    return rgb, [t.c, t.c + w * t.a, t.f + h * t.e, t.f]


def set_overlay(result, shape) -> np.ndarray:
    """RGBA (uint8) set-colour overlay resampled to `shape` (rows, cols) of the facade preview."""
    import cv2

    out = np.zeros(shape + (4,), np.uint8)
    if result.facade_sets is None:
        return out
    lab = cv2.resize(result.facade_sets.array[0].astype(np.int16), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    for s in range(int(lab.max()) + 1):
        m = lab == s
        out[m, :3] = _hex_rgb(SET_COLORS[s % len(SET_COLORS)])
        out[m, 3] = 150
    return out


def fig_set_map(result) -> bytes | None:
    if result.facade_rgb is None:
        return None
    plt = _setup_mpl()
    rgb, ext = facade_preview(result, 2400)
    ov = set_overlay(result, rgb.shape[:2]).astype(float) / 255
    blend = rgb.astype(float) / 255 * (1 - ov[..., 3:4]) + ov[..., :3] * ov[..., 3:4]
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    fig, ax = plt.subplots(figsize=(14, max(3.5, 14 * aspect + 1)))
    ax.imshow(blend, extent=ext)
    for d in result.discontinuity_sets:
        ax.plot([], [], "s", color=SET_COLORS[d.set_id % len(SET_COLORS)], ms=10,
                label=f"Set {d.set_id}  {d.mean_dip:.0f}°/{d.mean_dip_direction:03.0f}°")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlabel("측점 (m)")
    ax.set_ylabel("표고 (m)")
    ax.set_title("절리군 위치 지도 (입면, 색 = 그 자리 암면이 속한 절리군)")
    return _png(fig)


def fig_trace_map(result, families) -> bytes | None:
    if result.facade_rgb is None:
        return None
    plt = _setup_mpl()
    rgb, ext = facade_preview(result, 2400)
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    fig, ax = plt.subplots(figsize=(14, max(3.5, 14 * aspect + 1)))
    ax.imshow(rgb, extent=ext, alpha=0.75)
    from .discontinuity.trace_sets import family_sets, family_tag

    fam_of = {}
    for f in families:
        for t in f.members:
            fam_of[id(t)] = f
    for t in result.traces:
        f = fam_of.get(id(t))
        c = "0.6" if f is None else family_colour(f)
        ax.plot(t.coords[:, 0], t.coords[:, 1], color=c, lw=2.2 if t.source == "manual" else 1.3,
                ls="--" if t.source == "manual" else "-")
    for f in families:
        tag = family_tag(family_sets(f, result.wall_frame, result.discontinuity_sets))
        ax.plot([], [], "-", color=family_colour(f), lw=3,
                label=f"F{f.family_id}: {f.apparent_angle:.0f}° · {f.total_length_m:.0f} m · {tag}")
    n_out = sum(1 for t in result.traces if id(t) not in fam_of)
    if n_out:
        ax.plot([], [], "-", color="0.6", lw=3, label=f"방향군 밖 ({n_out}개)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlabel("측점 (m)")
    ax.set_ylabel("표고 (m)")
    ax.set_title("입면 트레이스와 방향군 (점선 = 수동 입력)")
    return _png(fig)


def fig_trace_angles(result, families) -> bytes | None:
    if not result.traces:
        return None
    plt = _setup_mpl()
    from .discontinuity.trace_sets import apparent_angle, family_sets, family_tag
    ang = np.array([apparent_angle(t) for t in result.traces])
    w = np.array([t.length_m for t in result.traces])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.hist(ang, bins=60, range=(0, 180), weights=w, color="0.65")
    for f in families:
        ax.axvline(f.apparent_angle, color=family_colour(f), lw=2.5,
                   label=f"F{f.family_id} {f.apparent_angle:.0f}° ({family_tag(family_sets(f, result.wall_frame, result.discontinuity_sets))})")
    ax.set_xlim(0, 180)
    ax.set_xlabel("입면 트레이스 각도 (°, 0 = 수평 →, 90 = 수직)")
    ax.set_ylabel("트레이스 길이 합 (m)")
    ax.legend(fontsize=9)
    return _png(fig)


DENSITY_CMAPS = {"흰색→갈색": "YlOrBr", "흰색→파랑": "Blues", "viridis": "viridis"}     # name -> matplotlib colormap
DENSITY_UNITS = {"exponential_kamb": "극점 밀도 (균등 분포 대비 σ, Kamb 지수 평활)",
                 "schmidt": "극점 밀도 (1 % 면적당 %, Schmidt)"}


def fig_stereonet(result, families, density: dict | None = None) -> bytes:
    """Poles (size = area), set mean planes, trace-only families and the face per window.

    density {"method": "exponential_kamb" | "schmidt", "sigma": smoothing (Kamb), "weighted": by face area,
    "poles": also draw the poles, "cmap": a matplotlib colormap}: a smoothly contoured pole-density map under them."""
    plt = _setup_mpl()
    import mplstereonet  # noqa: F401
    fig = plt.figure(figsize=(6.2, 6.9 if density else 6.2))
    ax = fig.add_subplot(111, projection="stereonet")
    if density:
        pts = [(p.dip_direction - 90, p.dip, p.area) for d in result.discontinuity_sets for p in d.patches]
        if len(pts) >= 3:
            s_, d_, a_ = (np.array(v, float) for v in zip(*pts))
            method = density.get("method", "exponential_kamb")
            kw = dict(measurement="poles", method=method, gridsize=180)
            if method != "schmidt":
                kw["sigma"] = float(density.get("sigma", 3.0))
            if density.get("weighted", False):
                kw["weights"] = np.clip(a_, 1e-6, None)
            from matplotlib.colors import ListedColormap

            cmap = density.get("cmap", "YlOrBr")
            cols = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, 16))
            cols[0] = (1.0, 1.0, 1.0, 1.0)                     # the lowest band (near uniform) left white
            cf = ax.density_contourf(s_, d_, levels=16, cmap=ListedColormap(cols), **kw)
            ax.density_contour(s_, d_, levels=8, colors="white" if cmap == "viridis" else "#374151", linewidths=0.45,
                               alpha=0.5, **kw)
            cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.045, pad=0.07)
            cb.set_label(DENSITY_UNITS.get(method, "극점 밀도") + (" · 면적 가중" if density.get("weighted", False) else ""),
                         fontsize=8)
            cb.ax.tick_params(labelsize=7)
    show_poles = not density or density.get("poles", False)
    det = (getattr(result, "config", None) or {}).get("set_detection") or {}
    pinned = set(det.get("fixed_set_ids") or [])
    for d in result.discontinuity_sets:
        c = SET_COLORS[d.set_id % len(SET_COLORS)]
        if show_poles:
            for p in d.patches:
                ms = 2 + 5 * min(p.area, 2) / 2
                if density:                       # over a density map: small, outlined so they stay readable
                    ax.pole(p.dip_direction - 90, p.dip, "o", color=c, ms=ms * 0.7, alpha=0.8, mec="white", mew=0.4)
                else:
                    ax.pole(p.dip_direction - 90, p.dip, "o", color=c, ms=ms, alpha=0.45)
        ax.plane(d.mean_dip_direction - 90, d.mean_dip, color=c, lw=2,
                 label=f"Set {d.set_id} {d.mean_dip:.0f}/{d.mean_dip_direction:03.0f}" + (" (직접 지정)" if d.set_id in pinned else ""))
    for k, cand in enumerate(det.get("candidates") or []):        # peaks that did not become sets
        ax.plane(cand["dip_direction"] - 90, cand["dip"], color="#6b7280", lw=1.3, ls=":")
        ax.pole(cand["dip_direction"] - 90, cand["dip"], "D", ms=8, mfc="none", mec="#374151", mew=1.4,
                label=f"후보 {cand['dip']:.0f}/{cand['dip_direction']:03.0f} (밀도 {cand['contrast']:.1f}배)")
    for f in families:
        if f.linked_set is None and f.dip is not None:
            ax.plane(f.dip_direction - 90, f.dip, color=FAMILY_COLORS[f.family_id % len(FAMILY_COLORS)], lw=2, ls="--",
                     label=f"트레이스 F{f.family_id} {f.dip:.0f}/{f.dip_direction:03.0f} (추정)")
    for w in result.face_windows:
        ax.pole(w["face_dip_direction"] - 90, w["face_dip"], "k*", ms=9)
    ax.plot([], [], "k*", ms=9, label="벽면 (구간별)")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    return _png(fig)


def fig_kinematics(result) -> bytes | None:
    k = result.kinematics
    if k is None or not len(result.face_windows):
        return None
    plt = _setup_mpl()
    modes = ["평면파괴", "쐐기파괴", "전도파괴"]
    colors = {"평면파괴": "#d62728", "쐐기파괴": "#ff7f0e", "전도파괴": "#9467bd"}
    fig, ax = plt.subplots(figsize=(12, 2.8))
    for i, m in enumerate(modes):
        for w in result.face_windows:
            hit = k[(k["mode"] == m) & (k["u0"] == w["u0"])] if len(k) else []
            ax.barh(i, w["u1"] - w["u0"], left=w["u0"], color=colors[m] if len(hit) else "#eeeeee",
                    edgecolor="white")
    ax.set_yticks(range(len(modes)), modes)
    ax.set_xlabel("측점 (m)")
    ax.set_title("운동학적으로 가능한 파괴 형태 (색 = 해당 구간에서 가능)")
    return _png(fig)


def _img(png: bytes | None) -> str:
    if not png:
        return "<p class='muted'>(자료 없음)</p>"
    return f"<img src='data:image/png;base64,{base64.b64encode(png).decode()}'>"


def _hydro(result):
    from . import hydro

    smap = hydro.seepage_potential(result)
    g = hydro.grouting_inputs(result, result.config.get("curtain", {}).get("azimuth") if result.config.get("curtain") else None)
    return smap, g


def build_report(result, output_dir: str | Path) -> Path:
    """Write 분석보고서.html (figures embedded) into `output_dir` and return its path."""
    from .discontinuity.trace_sets import families_table

    cfg = result.config
    fams = result.trace_families
    sets = pd.DataFrame([{
        "절리군": f"Set {d.set_id}", "경사/경사방향": f"{d.mean_dip:.0f}° / {d.mean_dip_direction:03.0f}°",
        "Fisher k": round(d.fisher_k, 1) if np.isfinite(d.fisher_k) else None, "절리면 수": len(d.patches),
        "간격 (m, 측선)": None if d.spacing_mean is None else round(d.spacing_mean, 2),
        "연장성 중앙값 (m)": round(float(np.median([p.persistence_length for p in d.patches])), 2) if d.patches else None,
        "면적 합 (m²)": round(sum(p.area for p in d.patches), 0),
    } for d in result.discontinuity_sets])
    ft = families_table(fams, result.wall_frame, result.discontinuity_sets)
    kin = result.kinematics if result.kinematics is not None else pd.DataFrame()
    kin_sum = (kin.groupby(["mode", "joints"]).size().reset_index(name="구간 수") if len(kin) else pd.DataFrame())
    applied = cfg.get("auto_scale_applied", {})
    notes = []
    for d in result.discontinuity_sets:
        n = [p for p in d.patches]
        if n and np.mean([result.wall_frame.apparent_trace_direction(p.normal, result.wall_frame.to_wall(p.centroid[None])[0, 0]) is None for p in n]) > 0.5:
            notes.append(f"Set {d.set_id}는 벽면과 거의 평행합니다 — 노출 벽면 자체를 이루는 절리이며, 간격은 벽면이 계단식으로 물러나는 간격에 가깝습니다.")
    for f in fams:
        if f.linked_set is None:
            if f.dip is not None:
                notes.append(f"트레이스 방향군 F{f.family_id} ({f.apparent_angle:.0f}°)는 3차원 면으로 잡히지 않은 절리군입니다 — 추정 방향 {f.dip:.0f}°/{f.dip_direction:03.0f}° (신뢰도 {f.orientation_quality}).")
            else:
                notes.append(f"트레이스 방향군 F{f.family_id} ({f.apparent_angle:.0f}°)는 3차원 면으로 잡히지 않았고, 벽면이 평평해 3차원 방향은 정할 수 없습니다 (겉보기 각만).")
    if cfg.get("field_measurements") is None and result.field_comparison is None:
        notes.append("현장 실측 자료가 없어 방향 정확도를 검증하지 않았습니다. 절리군별 10~20개 클리노미터 측정을 권장합니다.")

    css = ("body{font-family:'Malgun Gothic',sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1f2937}"
           "h1{font-size:24px}h2{font-size:18px;border-bottom:2px solid #e5e7eb;padding-bottom:4px;margin-top:28px}"
           "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #d1d5db;padding:4px 8px}"
           "th{background:#f3f4f6}img{max-width:100%}.muted{color:#6b7280}li{margin:4px 0}")
    html = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>불연속면 분석 보고서</title>"
            f"<style>{css}</style></head><body>",
            f"<h1>불연속면 분석 보고서 — {cfg.get('site_name', '')}</h1>",
            f"<p class='muted'>작성 {datetime.datetime.now():%Y-%m-%d %H:%M} · 점군 {Path(cfg['paths'].get('point_cloud') or '').name} · "
            f"벽 길이 {result.wall_frame.length:.1f} m · 분석 점 {len(result.point_labels):,}개</p>",
            "<h2>1. 해석 요점</h2><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>",
            "<h2>2. 절리군 (3차원 면)</h2>", sets.to_html(index=False), _img(fig_stereonet(result, fams)),
            "<h2>3. 절리군 위치 지도</h2>", _img(fig_set_map(result)),
            "<h2>4. 입면 트레이스 방향군</h2>", ft.to_html(index=False) if len(ft) else "<p>없음</p>",
            _img(fig_trace_map(result, fams)), _img(fig_trace_angles(result, fams)),
            "<h2>5. 운동학적 안정성 검토 (선별)</h2>",
            f"<p class='muted'>마찰각 {cfg.get('kinematics', {}).get('friction_deg', 30)}°, 벽면 방향은 {cfg.get('kinematics', {}).get('window_m', 5)} m 구간별로 벽면 기준선의 법선 방향과 점군 외곽(80 백분위) 기울기에서 구함. "
            "가능 여부만 표시하며 안전율이 아닙니다.</p>",
            kin_sum.to_html(index=False) if len(kin_sum) else "<p>가능한 파괴 형태 없음</p>", _img(fig_kinematics(result)),
            *seepage_section(result, *_hydro(result)),
            "<h2>8. 설정</h2>",
            f"<p class='muted'>자동 크기: {applied} · 식생 제외 {cfg.get('vegetation_removed_fraction')} · "
            f"최소 절리면 면적 {cfg.get('patches', {}).get('min_patch_area')} m² · 분석 범위 {cfg.get('region')}</p>",
            "</body></html>"]
    path = Path(output_dir) / REPORT_NAME
    path.write_text("\n".join(html), encoding="utf-8")
    return path


# --- seepage potential and grouting inputs (groundwater.hydro) -------------------------------------------

def fig_seepage(result, smap) -> bytes | None:
    if smap is None or result.facade_rgb is None:
        return None
    plt = _setup_mpl()
    import matplotlib.patches as mpatches
    rgb, ext = facade_preview(result, 2400)
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    fig, ax = plt.subplots(figsize=(14, max(3.5, 14 * aspect + 1)))
    ax.imshow(rgb, extent=ext, alpha=0.8)
    rows, cols = smap.classes.shape
    ov = np.zeros((rows, cols, 4))
    ov[smap.classes == 2] = (0.86, 0.15, 0.15, 0.55)
    ov[smap.classes == 1] = (0.98, 0.75, 0.14, 0.35)
    ax.imshow(ov, extent=[smap.u_min, smap.u_min + cols * smap.cell_m, smap.v_max - rows * smap.cell_m, smap.v_max],
              interpolation="nearest")
    for z in smap.zones.head(10).itertuples():
        u0, u1 = map(float, z.chainage_m.split("–"))
        v0, v1 = map(float, z.elevation_m.split("–"))
        ax.add_patch(mpatches.Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec="white", lw=1.5))
        ax.text(u0, v1 + 0.2, f"#{z.zone}", color="white", fontsize=9, weight="bold",
                bbox=dict(fc="#b91c1c", ec="none", pad=1.5))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.legend(handles=[mpatches.Patch(color=(0.86, 0.15, 0.15, 0.7), label="상 (상위 15 %)"),
                       mpatches.Patch(color=(0.98, 0.75, 0.14, 0.6), label="중 (다음 35 %)")],
              loc="upper right", fontsize=9)
    ax.set_xlabel("측점 (m)")
    ax.set_ylabel("표고 (m)")
    ax.set_title("지하수 유출 가능성 (표면 절리망 기준, 상위 10개 구역 번호)")
    return _png(fig)


def fig_hole_orientation(g) -> bytes | None:
    if not g:
        return None
    plt = _setup_mpl()
    import mplstereonet
    grid = g["grid"]
    fig = plt.figure(figsize=(6.4, 6.2))
    ax = fig.add_subplot(111, projection="stereonet")
    lon, lat = mplstereonet.line(grid["plunge"].to_numpy(), grid["trend"].to_numpy())
    sc = ax.scatter(lon, lat, c=grid["joints_per_m"], cmap="viridis", s=18)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="공 1 m당 만나는 (투수 가중) 절리 수")
    b = g["best"]
    ax.line(b["plunge"], b["trend"], "r*", ms=16, label=f"최적 {b['trend']:.0f}°/{b['plunge']:.0f}°")
    ax.line(90, 0, "ws", ms=9, mec="k", label="연직공")
    if g.get("in_plane"):
        p = g["in_plane"]
        ax.line(p["plunge"], p["trend"], "m^", ms=11, label=f"커튼 면 안 최적 {p['trend']:.0f}°/{p['plunge']:.0f}°")
    for r in g["principal"].itertuples():
        ax.line(r.plunge, r.trend, "kx", ms=9)
    ax.plot([], [], "kx", label="투수 주축 (1, 2, 3)")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1.0), fontsize=8)
    ax.set_title("그라우팅 공 방향별 절리 관통 수 (하반구, 공 방향 = 선)", fontsize=10)
    return _png(fig)


def seepage_section(result, smap, g) -> list:
    """HTML blocks for the report."""
    parts = ["<h2>6. 지하수 유출 가능성 (표면 절리망 기준)</h2>",
             "<p class='muted'>1 m 격자마다 절리 밀도(P21), 열린 홈 깊이, 연결된 절리망 길이, 절리 옆 식생을 0~1로 환산해 가중 합산. "
             "실제 유출 확인은 열화상·관측 자료로 해야 합니다.</p>"]
    if smap is not None:
        parts += [_img(fig_seepage(result, smap)), smap.zones.head(15).to_html(index=False)]
    if g:
        parts += ["<h2>7. 그라우팅 설계 입력 (초기값)</h2>",
                  "<p class='muted'>Snow 평행판 모델, 절리군 개구폭은 모두 같다고 가정(상대값). 간격·연장성은 드론 표면 값이며 "
                  "시추·루전 시험으로 보정해야 합니다.</p>",
                  g["sets"].round(2).to_html(index=False), "<h3>투수 주축</h3>", g["principal"].to_html(index=False),
                  _img(fig_hole_orientation(g)),
                  "<h3>최적 방향 공이 만나는 절리 (10 m당)</h3>", g["per_set_best"].to_html(index=False),
                  "<h3>연직공이 만나는 절리 (10 m당)</h3>", g["per_set_vertical"].to_html(index=False)]
        if "curtain" in g:
            parts += [f"<h3>차수 커튼 검토 (커튼 방위 {g['curtain_azimuth']:.0f}°)</h3>", g["curtain"].to_html(index=False),
                      f"<p>커튼 끝 연장 권장(전도성 절리군 연장성 p90): {g.get('end_extension_m')} m 이상 — 드론 연장성은 하한값입니다.</p>"]
    return parts
