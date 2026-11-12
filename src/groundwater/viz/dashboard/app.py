"""Streamlit dashboard for the Phase 1 discontinuity-extraction pipeline.

Run with:
    streamlit run src/groundwater/viz/dashboard/app.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running via `streamlit run` without an editable install.
_SRC_ROOT = Path(__file__).resolve().parents[2].parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
# Streamlit reloads an edited module only if it lies in this script's folder or on PYTHONPATH: without this, a
# change to groundwater/viz/*.py (or any module outside dashboard/) is ignored until the server restarts, while
# app.py itself is reloaded -- the two then disagree (e.g. a call with a changed signature fails).
_pp = os.environ.get("PYTHONPATH", "")
if str(_SRC_ROOT) not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(_SRC_ROOT), _pp) if p)


@st.cache_resource(show_spinner=False)
def _module_times() -> dict:
    """File time of every program module as it was when loaded, kept for the life of the server process."""
    return {"new": True, "seen": {}}


def _program_modules() -> list:
    return [n for n in list(sys.modules) if n == "groundwater" or n.startswith("groundwater.")]


def _refresh_program_modules():
    """Drop the program modules (groundwater.*) from memory when any of their files changed since they were
    loaded, so this run imports them afresh. Streamlit's own watcher misses a change made while no browser
    session was watching that file (each session watches only from its own first run), and then keeps the old
    module until the server restarts -- app.py itself is always rerun, so the two disagree (a missing function,
    a changed signature). All of them go together: they import from each other."""
    state = _module_times()
    stale = state["new"]                       # first run of this check in a server that may have run for hours
    state["new"] = False
    for name, t in list(state["seen"].items()):
        mod = sys.modules.get(name)
        if mod is None:                        # Streamlit's watcher already dropped it: it loads afresh anyway
            del state["seen"][name]
            continue
        f = getattr(mod, "__file__", None)
        try:
            if f and os.path.getmtime(f) != t:
                stale = True
                break
        except OSError:
            pass
    if stale:
        for name in _program_modules():
            sys.modules.pop(name, None)
        state["seen"].clear()


def _note_program_modules():
    """Record the file time of every program module loaded so far (end of each run)."""
    seen = _module_times()["seen"]
    for name in _program_modules():
        if name not in seen:
            f = getattr(sys.modules[name], "__file__", None)
            try:
                if f:
                    seen[name] = os.path.getmtime(f)
            except OSError:
                pass


_refresh_program_modules()

from groundwater import hydro, report, site_folder  # noqa: E402
from groundwater.discontinuity.trace_sets import families_table, family_tag  # noqa: E402
from groundwater.facade.region import core_chainage  # noqa: E402
from groundwater.pipeline import RESULT_PICKLE, load_config, load_results, run_pipeline, save_results  # noqa: E402
from groundwater.validation.field_compare import summarize_errors  # noqa: E402
from groundwater.viz.stereonet import plot_stereonet  # noqa: E402

st.set_page_config(page_title="암벽 불연속면 분석", page_icon=":material/landscape:", layout="wide")

MAP_HEIGHT = 880                       # px; the map views get as much of the screen as they can use
POPUP_TITLES = {"trace_map": "입면 트레이스 지도", "seep_map": "유출 가능성 지도", "thermal_anom": "열화상 이상 부분 × 절리", "kin_map": "파괴 가능 구역 지도", "kin_3d": "파괴 가능 구역 (3D)", "set_map": "절리군 위치 지도", "overlay": "참고 이미지 겹쳐 보기",
                "section": "그라우팅 단면", "sm_plan": "수평 단면 (x-y)", "sm_cross": "횡단면 (y-z)",
                "sm_par": "벽면 평행 단면 (x-z)", "v3d_main": "점군 3D", "sm_cloud": "점군 3D — 절리군 색",
                "sm_interior": "암반 내부 절리면 (3D)", "lidar_section": "라이다 단면", "sj_plan": "지중 절리 수평 단면", "sj_ortho": "정사영상 위 절리", "ex_plan": "굴착면 구분 — 평면", "sj_section": "지중 절리 수직 단면", "g3d": "그라우팅 3차원 가상 시공"}


def _reg_crs(result):
    """The result's map CRS with its height datum (heights.reg_crs): '+KNGeoid18' for orthometric results."""
    from groundwater import heights

    return heights.reg_crs(result.config)


@st.cache_resource(show_spinner=False)
def _view_server():
    """The local server of the popup windows (viz/popup.py), one per dashboard process."""
    from groundwater.viz import popup

    return popup.ViewServer()


def _popup_button(url: str, name: str, label: str):
    from groundwater.viz import popup

    st.html(popup.open_button_html(url, f"gw_{name}", label), unsafe_allow_javascript=True)


def _figure_window(fig, name: str, height: int, label: str | None = None, **preview):
    """A 2D figure as a still image on the page, interactive (wheel zoom, drag pan, hover) in its own popup
    window: inside the page a large Plotly map takes the wheel from the page and keeps redrawing, so scrolling
    past it stalls (viz/popup.py)."""
    import hashlib

    from groundwater.viz import popup, static_map

    fig.update_layout(dragmode="pan", height=height)
    sig = hashlib.sha1(fig.to_json().encode("utf-8")).hexdigest()
    title = POPUP_TITLES.get(name, name)
    url = _session_cache(f"_pop_{name}", sig, lambda: _view_server().publish(name, popup.plotly_page(fig, title)))
    _popup_button(url, name, label or f"새 창에서 크게 보기 — {title} (휠 확대·축소, 드래그 이동, 점 정보)")
    opts = preview or {"max_height_px": height}
    png = _session_cache(f"_png_{name}", (sig, tuple(sorted(opts.items()))),
                         lambda: static_map.preview_image(fig, **opts))
    st.image(png, width="stretch")


def _map_chart(fig, view: str):
    """A map (image in chainage / elevation) at 1:1 scale, see _figure_window. The axis ranges, not the plot
    frame, keep the 1:1 scale in the window (Plotly's constrain="domain" made the frame jump at every zoom)."""
    fig.update_layout(margin=dict(l=55, r=10, t=36, b=45))
    fig.update_xaxes(constrain="range")
    fig.update_yaxes(constrain="range", scaleanchor="x", scaleratio=1)
    result = st.session_state.get("result")
    core = core_chainage(result.config) if result is not None and view != "section" else None
    if core and result.facade_depth is not None:   # the overlap buffer: processed, owned by the neighbouring section
        f, (h, w) = result.facade_depth.transform, result.facade_depth.array.shape[1:]
        u0, u1, v0, v1 = f.c, f.c + abs(f.a) * w, f.f - abs(f.e) * h, f.f
        for x0, x1 in ((u0, core[0]), (core[1], u1)):
            if x1 > x0:                       # a filled trace, not a shape: the page's still image draws traces
                fig.add_scatter(x=[x0, x1, x1, x0, x0], y=[v0, v0, v1, v1, v0], fill="toself", mode="lines",
                                fillcolor="rgba(128,128,128,0.55)", line=dict(width=0), hoverinfo="skip",
                                showlegend=False, name="겹침 구간")
                fig.add_annotation(x=(x0 + x1) / 2, y=v1, text="겹침<br>(이웃 구간)", showarrow=False, yanchor="top",
                                   font=dict(size=11, color="white"), bgcolor="rgba(80,80,80,0.8)")
    _figure_window(fig, view, MAP_HEIGHT)


VIEW_ICONS = {"3D": "view_in_ar", "절리군 지도": "map", "스테레오넷": "explore", "트레이스": "timeline",
              "유출 가능성": "water_drop", "열화상": "thermostat", "그라우팅 입력": "construction", "안정성 검토": "warning", "라이다 단면": "straighten", "지중 절리 (현장)": "landscape", "굴착면 구분": "construction",
              "표": "table_chart", "현장 검증": "fact_check", "보고서": "description", "암각화 (참고 이미지)": "image"}

MAX_3D_POINTS = 80_000


@st.cache_resource(max_entries=2, show_spinner=False)
def _load_results_cached(path: str, mtime: float):
    """A saved result, shared by all browser tabs / reloads until the file changes (it is ~0.5 GB)."""
    return load_results(path)


@st.cache_resource(max_entries=1, show_spinner=False)
def _site_mesh_cached(path: str, mtime: float, result_key, hide_veg: bool, _result):
    from groundwater.viz import grouting3d

    return grouting3d.load_site_mesh(path, _result, hide_vegetation=hide_veg)


def _result_key():
    """Identity of the result on screen (path + file time, or the run time), for caches."""
    return st.session_state.get("result_key", ("none",))


def _flow_sets(result):
    """hydro.joint_sets_for_flow for the result on screen (a copy: callers may add columns), made once."""
    return _session_cache("_flow_sets", _result_key(), lambda: hydro.joint_sets_for_flow(result)).copy()


def _session_cache(name: str, key, make):
    """Value cached in the session under `name` while `key` stays the same."""
    c = st.session_state.get(name)
    if c is not None and c[0] == key:
        return c[1]
    v = make()
    st.session_state[name] = (key, v)
    return v


def _fragment(fn):
    """st.fragment: a widget inside reruns only this part of the page (a 3D view with its own controls), not the
    whole app -- the map and the other open 3D views are not rebuilt and sent again. The remembered options are
    saved at the end of such a partial run too (the full run saves them in main)."""
    import functools

    @functools.wraps(fn)
    def body(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            if st.session_state.get("_input_mode") == "현장 폴더":
                _store_settings()

    return st.fragment(body)


def _downsample_for_display(points: np.ndarray, colors: np.ndarray, max_points: int):
    """The same sample every rerun (a new random one each time would redraw the chart and reset the camera)."""
    if len(points) <= max_points:
        return points, colors
    idx = np.sort(np.random.default_rng(0).choice(len(points), max_points, replace=False))
    return points[idx], colors[idx]


def _view3d(fig, key: str, height: int):
    """A 3D figure with Cesium-style mouse controls (left drag pan, right drag / wheel zoom, middle or Ctrl + left
    drag rotate), see viz/view3d.py."""
    from groundwater.viz import view3d

    if fig.layout.title and fig.layout.title.text:
        st.markdown(f"**{fig.layout.title.text}**")
    html, sig = view3d.page_for(fig, key)
    title = POPUP_TITLES.get(key.split("_major")[0].removesuffix("_plan"), "3D")
    url = _session_cache(f"_pop3d_{key}", sig, lambda: _view_server().publish(
        key, html.replace("<head>", f"<head><title>{title}</title>", 1)))
    _popup_button(url, key, "새 창에서 크게 보기 (창 전체 · 휠로 바로 확대·축소)")
    view3d.show(fig, key, height)


def render_3d_view(result):
    _site_tiles_scene()
    # the same figure object every rerun of this result: the 3D page is then built once and not sent again
    fig = _session_cache("_3d_fig", _result_key(), lambda: _main_3d_figure(result))
    _view3d(fig, "v3d_main", 700)


def _variant_picker(site: Path, key: str, col=None) -> str:
    """Which result folder of the sections to use (viz.site3d.result_variants): 분석결과 = '기본', else the folder
    suffix (e.g. '벽면평행제외'); shown only when there is more than one."""
    from groundwater.viz import site3d

    vs = site3d.result_variants(site) if site.is_dir() else []
    if len(vs) <= 1:
        return vs[0] if vs else ""
    labels = {v: (v.lstrip("_") or "기본") for v in vs}
    st.session_state[key] = st.session_state.get(key, _saved(key, vs[0]))
    if st.session_state[key] not in vs:
        st.session_state[key] = vs[0]
    return (col or st).selectbox("결과 종류", vs, format_func=labels.get, key=key,
                                 help="구간마다의 결과 폴더: '기본' = 분석결과, 그 밖 = 분석결과_<이름> (다른 설정으로 돌린 결과, "
                                      "예: 벽면평행제외 = 절벽면과 평행한 면을 절리군 찾기에서 뺀 분석).")


def _site_tiles_scene():
    """Every section's results on the photo-textured tiled model of the whole site, in a browser tab."""
    from groundwater.viz import site3d, tiles3d

    site = Path(str(st.session_state.get("site_folder", "")).strip().strip('"'))
    secs = site3d.section_results(site) if site.is_dir() else []
    if not secs:
        return
    with st.expander(":material/public: 현장 전체 실사 3D — 타일 모델 위에 모든 구간의 결과", expanded=False):
        st.caption("현장 전체 타일 모델(Metashape .3tz, 사진 텍스처)을 배경으로 구간마다의 절리면을 올립니다. 같은 방향의 "
                   "절리군은 구간이 달라도 같은 색(방향 ±20°로 묶음)이고, 이름에 그 절리군이 나오는 구간을 적습니다. 트레이스와 "
                   "유출 가능성 칸(1 m)도 함께 올릴 수 있습니다. 타일 모델은 분석에는 쓰지 않고 보기용입니다 "
                   "(인터넷 연결 필요: 3D 엔진 CesiumJS).")
        if not str(st.session_state.get("site3d_tiles") or _saved("site3d_tiles", "")).strip():
            f = tiles3d.find_tileset(site)                    # the largest .3tz of the site folder
            st.session_state["site3d_tiles"] = str(f) if f else ""
        tp = st.text_input("현장 전체 타일 모델 (.3tz 또는 tileset.json 폴더)", **_w("site3d_tiles", ""),
                           help="비워 두면 현장 폴더에서 가장 큰 .3tz를 고릅니다.")
        tiles = tiles3d.find_tileset(tp) if tp.strip() else None
        if tp.strip() and tiles is None:
            st.warning("타일 모델을 찾을 수 없습니다.")
        variant = _variant_picker(site, "site3d_variant")
        secs = site3d.section_results(site, variant)
        names = [n for n, _ in secs]
        default = [n for n in names if n != "현장 폴더"] or names
        c = st.columns([3, 1, 1, 1.3], vertical_alignment="bottom")
        pick = c[0].multiselect("구간 결과", names, default=default, key="site3d_secs",
                                help="'현장 폴더'는 현장 폴더 자체의 분석결과(서측 사진측량)입니다. 서측 라이다 결과와 같은 벽이라 "
                                     "둘 다 고르면 겹칩니다.")
        tr = c[1].toggle("트레이스", value=True, key="site3d_tr")
        sp = c[2].toggle("유출 가능성", value=True, key="site3d_sp")
        kin = c[3].toggle("파괴 가능 구역", value=False, key="site3d_kin",
                          help="'안정성 검토' 화면의 파괴 가능 구역 지도(기본 설정: 마찰각 = 분석 설정, 측방 ±20°, 1 m 셀, "
                               "벽면 평활 3 m, 근처 2 m 절리군, 식생 셀 제외)를 구간마다 계산해 벽 표면에 붙입니다. "
                               "구간마다 30초 안팎 더 걸립니다.")
        if st.button("현장 전체 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not pick):
            with st.spinner("구간 결과를 모으는 중 (구간마다 10초 안팎)..."):
                scene = site3d.site_scene(site, tiles, pick, traces=tr, seepage=sp, variant=variant, kinematics=kin)
                path = tiles3d.write_scene(site / "현장전체_3D장면.json", scene)
            url = _open_tiles_viewer(path)
            st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}` "
                        f"(절리면 {sum(len(g['polys']) for g in scene['joint_groups'] if not g['label'].startswith('유출')):,}개)")


def _main_3d_figure(result):
    display_points, display_set_ids = _downsample_for_display(np.asarray(result.point_cloud.points),
                                                              result.point_labels[:, 0], MAX_3D_POINTS)
    fig = go.Figure()
    unassigned = display_set_ids == -1
    if unassigned.any():
        p = display_points[unassigned]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=1.5, color="lightgray"), name="미분류"))
    for sid in sorted(s for s in set(display_set_ids.tolist()) if s != -1):
        p = display_points[display_set_ids == sid]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=2, color=report.SET_COLORS[sid % len(report.SET_COLORS)]),
                                   name=f"Set {sid}"))
    _, outward = result.wall_frame.axes_at(result.wall_frame.length / 2)
    eye = dict(x=float(outward[0]) * 1.6, y=float(outward[1]) * 1.6, z=0.5)      # in front of the wall, a little above
    fig.update_layout(
        scene=dict(aspectmode="data", camera=dict(eye=eye), xaxis_title="동 X (m)", yaxis_title="북 Y (m)",
                   zaxis_title="표고 Z (m)"),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig


TRACE_STATES = [
    (True, "lime", "입면 트레이스로 확인된 절리면"),
    (False, "orange", "맞는 트레이스 없음"),
    (None, "deepskyblue", "벽면과 평행 (확인 불가)"),
]


MAX_FACADE_DISPLAY_PX = 3000      # longer side of the facade image sent to the browser
MAX_DISPLAY_TRACES = 5000         # longest traces drawn; all of them are still used in the analysis


def _facade_display(result):
    """Facade orthophoto as a JPEG data URI, downscaled to MAX_FACADE_DISPLAY_PX, and its pixel size (m)."""
    import base64
    import io

    import cv2
    from PIL import Image

    img = np.moveaxis(result.facade_rgb.array, 0, -1).copy()
    img[~np.isfinite(result.facade_depth.array[0])] = 255
    h, w = img.shape[:2]
    k = max(1.0, max(h, w) / MAX_FACADE_DISPLAY_PX)
    if k > 1:
        img = cv2.resize(img, (max(1, round(w / k)), max(1, round(h / k))), interpolation=cv2.INTER_AREA)
    # Plotly places the first row of a `source` image at y0 and goes up with a positive dy (a negative dy is
    # ignored), so the image is sent bottom row first and anchored at the lowest elevation
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(buf, "JPEG", quality=85)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, abs(result.facade_rgb.transform.a) * w / img.shape[1], abs(result.facade_rgb.transform.e) * h / img.shape[0]


def _family_sets(result) -> dict:
    """family id -> the sets it may belong to (trace_sets.family_sets), once per result."""
    from groundwater.discontinuity.trace_sets import family_sets

    return _session_cache("_fam_sets", _result_key(), lambda: {
        f.family_id: family_sets(f, result.wall_frame, result.discontinuity_sets) for f in result.trace_families})


def render_trace_map(result):
    if result.facade_rgb is None:
        st.info("점군에 색이 없어 입면 영상과 트레이스 교차검증을 건너뛰었습니다.")
        return

    # the full-resolution image and tens of thousands of separate line traces exceed Streamlit's message
    # limit on a real wall: send a downscaled JPEG and all traces as one NaN-separated line
    src, dx, dy = _session_cache("_facade_disp", _result_key(), lambda: _facade_display(result))
    t = result.facade_rgb.transform
    v_bottom = t.f + result.facade_rgb.array.shape[1] * t.e
    fig = go.Figure(go.Image(source=src, x0=t.c + dx / 2, dx=dx, y0=v_bottom + dy / 2, dy=dy, hoverinfo="skip"))
    traces = sorted(result.traces, key=lambda tr: tr.length_m, reverse=True)[:MAX_DISPLAY_TRACES]
    fam_of = {id(t): f for f in result.trace_families for t in f.members}
    groups = {}
    for tr in traces:
        f = fam_of.get(id(tr))
        key = ("manual", None) if tr.source == "manual" else ("family", f.family_id if f else None)
        groups.setdefault(key, []).append(tr)
    for (kind, fid), trs in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        seg = np.full((len(trs), 3, 2), np.nan)
        seg[:, :2] = np.array([tr.coords for tr in trs])
        xy = seg.reshape(-1, 2)
        if kind == "manual":
            name, color, width = "수동 트레이스", "#facc15", 3
        elif fid is None:
            name, color, width = "방향군 밖", "#9ca3af", 1.2
        else:
            f = next(f for f in result.trace_families if f.family_id == fid)
            tag = family_tag(_family_sets(result)[fid])
            name, color, width = f"F{fid} {f.apparent_angle:.0f}° ({tag})", report.family_colour(f), 2
        fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color=color, width=width),
                                   name=name, hoverinfo="skip", connectgaps=False))

    patches = [p for d in result.discontinuity_sets for p in d.patches]
    if patches:
        uv = result.wall_frame.to_wall(np.array([p.centroid for p in patches]))
        for state, color, label in TRACE_STATES:
            idx = [i for i, p in enumerate(patches) if result.trace_confirmed.get(p.patch_id) is state]
            if not idx:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=uv[idx, 0], y=uv[idx, 1], mode="markers", name=label,
                    marker=dict(color=color, size=9, line=dict(color="black", width=1)),
                    text=[f"Set {patches[i].set_id} / patch {patches[i].patch_id}<br>"
                          f"{patches[i].dip:.0f}/{patches[i].dip_direction:03.0f}" for i in idx],
                    hovertemplate="%{text}<br>측점 %{x:.2f} m<br>표고 %{y:.2f} m<extra></extra>",
                )
            )

    fig.update_xaxes(title="측점 (m)")
    fig.update_yaxes(title="표고 (m)", autorange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"))
    _map_chart(fig, "trace_map")
    st.caption(
        f"벽면을 따라 펼친 입면 영상 (측점 {result.wall_frame.length:.1f} m). 트레이스 {len(result.traces)}개 — 색 = 방향군, "
        "회색 = 방향군 밖, 노랑 = 수동"
        + (f" (긴 것 {MAX_DISPLAY_TRACES}개만 표시)" if len(result.traces) > MAX_DISPLAY_TRACES else "")
        + ". 점 = 3차원 절리면 (트레이스 교차검증 결과별 색)."
    )


SETMAP_INTERIOR_SAMPLE = 250_000     # cloud points behind the interior view (face context)
SECTION_PANEL_H = 500               # px, the interior view's plan-section card (full width)
SECTION_ROW_H = 600                 # px, its wall-parallel and cross-section cards (side by side)
# the two side-by-side section images at one height on any screen: the columns' width ratio shifts with the
# screen (fixed card padding), so the cross section's image box follows the wall-parallel one (fitted, not stretched)
SAME_HEIGHT_JS = """<script>
(() => {
  const sync = () => {
    const b = document.querySelector('.st-key-sm_card_b img'), c = document.querySelector('.st-key-sm_card_c img');
    if (!b || !c) return;
    const h = Math.round(b.getBoundingClientRect().height);
    if (h > 0 && c.style.height !== h + 'px') {
      Object.assign(c.style, { height: h + 'px', width: '100%', objectFit: 'contain' });
    }
  };
  if (!window.__gwSameHeight) {
    window.__gwSameHeight = true;
    window.addEventListener('resize', sync);
    setInterval(sync, 700);                    // images are replaced when a section changes
  }
  sync();
})();
</script>"""


def _jpeg_uri_bottom_first(img) -> str:
    """JPEG data URI of an image for a Plotly Image trace anchored at its bottom row (see _facade_display)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _setmap_display(result):
    from groundwater.viz import setmap3d

    img = setmap3d.setmap_blend(result, MAX_FACADE_DISPLAY_PX)
    t = result.facade_rgb.transform
    H, W = result.facade_rgb.array.shape[1:]
    return _jpeg_uri_bottom_first(img), abs(t.a) * W / img.shape[1], abs(t.e) * H / img.shape[0]


def _range_state(key, lo, hi, default):
    """A remembered (lo, hi) range slider state, reset when the saved value is not a valid range."""
    v = st.session_state.get(key, _saved(key, None))
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        v = default
    st.session_state[key] = tuple(float(min(max(x, lo), hi)) for x in v)
    return key


def _section_chart(fig, key: str, height: int = 620):
    """A 2D section, see _figure_window."""
    _figure_window(fig, key, height)


def _sm_disp(result, u_range):
    """The 3D views' coordinates (setmap3d.Disp) as chosen in the set map view."""
    from groundwater.viz import setmap3d

    if st.session_state.get("sm_coords", _saved("sm_coords", "실제 평면 (직선)")).startswith("실제"):
        return setmap3d.Disp(result.wall_frame, 0.5 * (u_range[0] + u_range[1]))
    return setmap3d.UNROLLED


def _sm_colour_of(sets):
    from groundwater.viz import grouting3d

    names = list(dict.fromkeys(sets["name"])) if len(sets) else []
    return lambda name: "#9ca3af" if name not in names else grouting3d._set_color(name, names.index(name))


def render_setmap(result):
    """Joint-set map: on the orthophoto (default), then -- on request, one below the other -- on the point cloud,
    on the tiled model and continued into the rock with sections."""
    from groundwater.viz import setmap3d

    if result.facade_sets is None or result.facade_rgb is None:
        st.info("색 정보가 없어 입면 지도를 만들 수 없습니다.")
        return
    src, dx, dy = _session_cache("_sm_disp", _result_key(), lambda: _setmap_display(result))
    t = result.facade_rgb.transform
    v_bottom = t.f + result.facade_rgb.array.shape[1] * t.e
    fig = go.Figure(go.Image(source=src, x0=t.c + dx / 2, dx=dx, y0=v_bottom + dy / 2, dy=dy, hoverinfo="skip"))
    for d in result.discontinuity_sets:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Set {d.set_id}  {d.mean_dip:.0f}°/{d.mean_dip_direction:03.0f}°",
                                 marker=dict(symbol="square", size=13, color=setmap3d.set_colour(d.set_id))))
    fig.update_xaxes(title="측점 (m)")
    fig.update_yaxes(title="표고 (m)", autorange=True)
    fig.update_layout(title="절리군 위치 지도 — 정사영상 (색 = 그 자리 암면이 속한 절리군)", legend=dict(orientation="h"))
    _map_chart(fig, "set_map")
    st.caption("입면 위에 각 자리의 암면이 어느 절리군에 속하는지 색으로 표시. 색이 없는 곳은 식생·거친 면·작은 조각. "
               "확대·이동은 '새 창에서 크게 보기'로 엽니다.")

    st.markdown("**다른 모델에서 보기** — 켜면 아래에 차례로 열립니다")
    row = st.container(horizontal=True)
    show_cloud = row.toggle(":material/scatter_plot: 점군 3D", **_w("sm_cloud", False),
                            help="분석 점군에서 절리군에 속한 점을 그 색으로, 나머지는 정사영상 색으로. 돌리기·확대 가능")
    show_tiles = row.toggle(":material/view_in_ar: 타일 모델", **_w("sm_tiles", False),
                            help="절리군 지도를 실사 타일 모델 표면에 붙여 새 탭에서 봅니다")
    show_int = row.toggle(":material/layers: 암반 내부 절리면 (3D · 단면)", **_w("sm_interior", False),
                          help="노출된 암면의 절리면을 그 방향 그대로 암반 속으로 연장해 3차원과 단면으로 봅니다")
    sets = _flow_sets(result)
    if show_cloud or show_int:
        coord_modes = ["실제 평면 (직선)", "측점 기준 (벽 따라 펼침)"]
        st.radio("3D·수평 단면 좌표", coord_modes, horizontal=True, **_w("sm_coords", coord_modes[0]),
                 help="실제 평면: 지도 좌표를 보는 구간 가운데의 벽 방향으로 돌려 그립니다 (평면인 절리는 평면, 직선). "
                      "측점 기준: 벽 기준선을 따라 펼친 좌표 — 기준선이 꺾이는 곳(원호)에서는 축이 함께 돌아 평면인 "
                      "절리도 휘어 보입니다. 측점·깊이를 정확히 읽을 때 씁니다.")
    if show_cloud:
        st.divider()
        _setmap_cloud(result)
    if show_tiles:
        st.divider()
        _setmap_tiles(result, sets)
    if show_int:
        st.divider()
        _setmap_interior(result, sets)


@_fragment
def _setmap_cloud(result):
    from groundwater.viz import setmap3d

    st.subheader(":material/scatter_plot: 점군 3D — 절리군 색")
    lo, hi = (float(np.floor(x)) for x in _chainage_range(result))
    c = st.columns([2, 3, 1.4], vertical_alignment="bottom")
    npts = c[0].select_slider("표시 점 수", [50_000, 100_000, 150_000, 250_000, 400_000], **_w("sm_npts", 150_000),
                              format_func=lambda n: f"{n // 1000}천", help="많을수록 촘촘하지만 화면이 느려집니다")
    key = _range_state("sm_cloud_range", lo, hi + 1, (lo, hi + 1))
    rng = c[1].slider("표시 측점 구간 (m)", lo, hi + 1, step=0.5, key=key)
    other = c[2].checkbox("절리군 밖 점도", **_w("sm_other", True), help="식생·거친 면 등 (정사영상 색)")
    sample = _session_cache("_sm_sample", (_result_key(), npts), lambda: setmap3d.cloud_sample(result, npts))
    disp = _sm_disp(result, rng)
    fig = _session_cache("_sm_cloudfig", (_result_key(), npts, rng, other, disp.plan),
                         lambda: setmap3d.cloud_figure(sample, result.discontinuity_sets, rng, other, disp=disp))
    _view3d(fig, "sm_cloud", 820)
    st.caption(("실제 평면 좌표: 보는 구간 가운데의 벽 방향이 x, 그 안쪽이 y (직선 거리), z = 표고. " if disp.plan else
                "벽면 좌표로 펼친 점군: x = 측점(벽 따라), y = 깊이(+ 암반 안쪽), z = 표고. ")
               + "마우스는 세슘(타일 모델 화면)과 같습니다. 왼쪽 범례에서 절리군을 켜고 끕니다.")


def _sm_joint_opts():
    """Options of the interior view (also used by the tiled-model scene when the interior view is closed)."""
    g = lambda k, d: st.session_state.get(k, _saved(k, d))  # noqa: E731
    return {"depth_factor": g("sm_jdepth", 2.0), "max_depth": g("sm_jmax", 5.0), "dfn": g("sm_dfn", False),
            "dfn_depth": g("sm_dfn_depth", 5.0), "traces": g("sm_traces", True)}


def _sm_joints(result, sets, opts):
    """Joint planes continued into the rock (and their wall-coordinate mesh), cached per option set."""
    from groundwater.viz import joints3d, setmap3d

    def make():
        rock = joints3d.RockTest(result)
        joints = joints3d.observed(result, None, opts["depth_factor"], opts["max_depth"], rock=rock)
        if not opts["traces"]:
            joints = [j for j in joints if j.source != "trace"]
        n_obs = len(joints)
        if opts["dfn"]:
            joints += joints3d.statistical(result, sets, opts["dfn_depth"], rock=rock)[0]
        return {"rock": rock, "joints": joints, "mesh": setmap3d.joint_mesh(result, joints), "n_obs": n_obs}

    key = (_result_key(), tuple(sorted(opts.items())),
           tuple(map(tuple, sets[["name", "spacing_m", "persistence_m"]].values.tolist())))
    return _session_cache("_sm_joints", key, make)


@_fragment
def _setmap_tiles(result, sets):
    from groundwater.viz import setmap3d, tiles3d

    st.subheader(":material/view_in_ar: 타일 모델 — 실사 모델에 절리군 지도 붙이기")
    st.caption("절리군 지도를 타일 모델 표면에 이미지로 붙여 새 탭에서 엽니다. 불투명도·표면 잘라내기·이미지 켜기/끄기는 "
               "그 화면에서 조절합니다 (인터넷 연결 필요: 3D 엔진 CesiumJS).")
    tiles = _tiles_path_input(st.session_state.get("mesh_path", ""), st.session_state.get("planar_dir", ""), "_sm")
    with_joints = st.checkbox("암반 내부 절리면도 함께 (아래 '암반 내부 절리면' 설정)", **_w("sm_tiles_joints", True))
    out = st.session_state.get("output_dir")
    if st.button("타일 모델 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not out,
                 key="sm_tiles_open"):
        with st.spinner("장면을 만드는 중..."):
            J = _sm_joints(result, sets, _sm_joint_opts())
            scene = tiles3d.scene_json(result, [], sets, tiles, J["joints"] if with_joints else None, None,
                                       title="절리군 위치 지도 — 타일 모델", joint_shade="none",
                                       overlay_folder=_reference_folder(result))
            ov = setmap3d.tiles_overlay(result)
            scene["overlays"] = ([ov] if ov else []) + scene["overlays"]
            path = tiles3d.write_scene(Path(out) / "setmap_scene_tiles.json", scene)
        url = _open_tiles_viewer(path)
        st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}`")


@_fragment
def _setmap_interior(result, sets):
    from groundwater.viz import section, setmap3d

    st.subheader(":material/layers: 암반 내부 절리면 — 노출된 암면에서 안쪽으로 연장")
    modes = ["주요 절리면 구조 (거시)", "개별 절리 연장 (상세)"]
    mode = st.radio("표시 방식", modes, horizontal=True, **_w("sm_mode", modes[0]),
                    help="거시: 절리군마다 가장 길게 이어진 절리면 몇 개만 골라 암반 블록을 관통하게 그려, 큰 절리면이 "
                         "어떻게 이어지고 서로 만나는지(교선) 봅니다. 상세: 관측된 절리 하나하나를 그 크기만큼 연장합니다.")
    major = mode == modes[0]
    if major:
        st.caption("절리군마다, 같은 평면 위에 놓인 노출면과 트레이스를 모아(절리군 평균 방향, 법선 방향 위치가 간격의 1/3 "
                   "안) 합친 길이가 가장 긴 평면을 **주요 절리면**으로 골라 암반 블록 끝까지 연장합니다. 검은 선은 주요 "
                   "절리면끼리 만나는 교선, 굵은 색 선은 그 절리면이 암면에 드러나는 선, 블록 옆·윗·뒷면의 색 선은 블록을 "
                   "자른 단면에 나타나는 절리 흔적입니다. 위치는 관측, 연장은 해석(가정)입니다.")
    else:
        st.caption("노출된 암면에서 찾은 절리면(입면 트레이스와 3차원 노출면)을 각자의 면 방향 그대로 암반 속으로 연장한 "
                   "추정입니다. 위치와 방향은 관측값, 안쪽으로 이어지는 크기만 가정(트레이스·노출면 크기 × 배율, 최대값 "
                   "제한)이며 암면 앞으로 나오는 부분은 잘라냅니다. 벽면과 거의 평행한 노출면은 연장해도 암면 자체이므로 빼고, "
                   "그런 절리군은 **통계 절리망**(간격·연장성으로 만든 가상 절리, 위치 임의)으로 안쪽 분포를 볼 수 있습니다.")
    c = st.columns(4)
    if major:
        per_set = c[0].slider("절리군마다 최대 평면 수", 1, 10, **_w("sm_major_max", 4),
                              help="한 절리군은 몇 m 간격으로 나란한 여러 평면입니다. 근거가 긴 순서로 이 수까지 그립니다 "
                                   "(1 = 가장 길게 이어진 면 하나, 가장 단순한 거시 구조).")
        min_len = c[1].slider("평면마다 근거 길이 최소 (m)", 0.0, 60.0, step=5.0, **_w("sm_major_minlen", 20.0),
                              help="그 평면 위에 놓인 노출면 연장성 + 트레이스 길이의 합. 짧은 근거의 평면은 그리지 않습니다.")
        min_sep = c[1].slider("나란한 평면 사이 최소 간격 (m)", 0.5, 5.0, step=0.5, **_w("sm_major_sep", 1.5),
                              help="이보다 가까운 평면은 같은 면의 굴곡으로 보고 하나만 그립니다.")
        extent_lbl = c[0].radio("연장 범위", ["블록 전체", "관측 범위 + 3 m"], horizontal=True,
                                **_w("sm_major_ext", "블록 전체"),
                                help="블록 전체: 주요 절리면을 블록 끝까지 (거시 구조). 관측 범위: 그 절리면의 노출면·"
                                     "트레이스가 있는 범위에서 3 m만 더")
        bdepth = c[1].slider("블록 깊이 (암면 뒤, m)", 2.0, 20.0, step=1.0, **_w("sm_major_depth", 8.0))
        jmax = bdepth
    else:
        jdepth = c[0].slider("연장 크기 = 트레이스·노출면 크기 ×", 0.5, 3.0, step=0.25, **_w("sm_jdepth", 2.0),
                             help="절리는 대략 길이만큼 깊다고 봅니다. 드론으로 본 길이는 하한값이라 키워 볼 만합니다.")
        jmax = c[0].slider("최대 연장 (m)", 1.0, 15.0, step=0.5, **_w("sm_jmax", 5.0))
        traces = c[1].checkbox("입면 트레이스도 연장", **_w("sm_traces", True),
                               help="끄면 3차원 노출면(절리군 지도의 색 면)만 연장합니다")
        dfn = c[1].checkbox("통계 절리망 함께 (위치 임의)", **_w("sm_dfn", False))
        dfn_depth = c[1].slider("통계 절리망 두께 (m)", 1.0, 15.0, step=0.5, disabled=not dfn, **_w("sm_dfn_depth", 5.0))
    jop = c[2].slider("절리면 불투명도", 0.1, 1.0, step=0.05, **_w("sm_jop", 0.75))
    face_modes = {"점군": "points", "표면 (반투명)": "surface", "끄기": "none"}
    face = face_modes[c[2].radio("암면 표시", list(face_modes), horizontal=True, **_w("sm_face_mode", "점군"),
                                 help="점군: 분석 점군을 정사영상 색으로 촘촘하게 (점 수를 늘릴수록 형상이 또렷하고 화면은 "
                                      "무거움). 표면: 정사영상 색을 입힌 암면 표면을 반투명하게 (가벼움)")]
    face_op = c[2].slider("암면 불투명도", 0.05, 1.0, step=0.05, **_w("sm_face_op", 0.35), disabled=face == "none",
                          help="낮출수록 암면 뒤의 절리면이 잘 보입니다")
    npts = c[2].select_slider("암면 점 수 (표시 구간 안)", [100_000, 250_000, 500_000, 1_000_000, 2_000_000],
                              **_w("sm_face_pts", 500_000), format_func=lambda n: f"{n // 10000}만",
                              disabled=face != "points")
    lo, hi = (float(np.floor(x)) for x in _chainage_range(result))
    mid, span = 0.5 * (lo + hi), hi - lo
    key = _range_state("sm_range", lo, hi + 1, (round(mid - 0.15 * span), round(mid + 0.15 * span)))
    rng = c[3].slider("표시 측점 구간 (m)", lo, hi + 1, step=0.5, key=key,
                      help="좁힐수록 안쪽이 잘 보입니다. 단면도 이 구간을 그립니다. 거시 보기는 넓게 잡아도 가볍습니다.")
    sample = _session_cache("_sm_isample", _result_key(), lambda: setmap3d.cloud_sample(result, SETMAP_INTERIOR_SAMPLE))
    colour_of = _sm_colour_of(sets)
    v_lo, v_hi = (round(float(x), 1) for x in np.percentile(sample["uvw"][:, 1], [1, 99]))
    if major:
        rock = _sm_joints(result, sets, _sm_joint_opts())["rock"]
        mkey = (_result_key(), tuple(rng), per_set, min_len, min_sep, extent_lbl, bdepth,
                tuple(map(tuple, sets[["name", "dip", "dip_direction", "spacing_m"]].values.tolist())))

        def make_major():
            pl = setmap3d.major_planes(result, sets, rng, per_set, min_length_m=min_len, min_sep_m=min_sep)
            return setmap3d.major_model(result, rock, pl, rng, (v_lo, v_hi), bdepth,
                                        "block" if extent_lbl == "블록 전체" else "observed")

        model = _session_cache("_sm_major", mkey, make_major)
        if model is None:
            st.info("이 구간에서 주요 절리면을 찾지 못했습니다. 측점 구간을 넓혀 보세요.")
            return
        J = {"rock": rock, "joints": model["joints"], "mesh": model["mesh"]}
        rows = [{"주요 절리면": f"{p['name']}-{setmap3d.plane_tag(p['rank'])}",
                 "경사/경사방향": f"{p['dip']:.0f}°/{p['dd']:03.0f}°",
                 "근거 길이 (m)": round(p["length_m"], 1), "노출면·트레이스 수": p["pieces"]} for p in model["planes"]]
        st.caption(f"주요 절리면 {len(model['planes'])}개 · 교선 {len(model['intersections'])}개 (근거 길이 = 그 평면 위 "
                   "노출면 연장성 + 트레이스 길이의 합)")
        with st.expander("주요 절리면 목록", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        opts = {"depth_factor": jdepth, "max_depth": jmax, "dfn": dfn, "dfn_depth": dfn_depth, "traces": traces}
        with st.spinner("절리면을 암반 속으로 연장하는 중..."):
            J = _sm_joints(result, sets, opts)
        n_tr = sum(j.source == "trace" for j in J["joints"])
        n_fc = sum(j.source == "facet" for j in J["joints"])
        n_dfn = len(J["joints"]) - J["n_obs"]
        st.caption(f"관측 연장 {J['n_obs']}개 (트레이스 {n_tr} · 노출면 {n_fc})" + (f" · 통계 절리망 {n_dfn}개" if dfn else ""))

    # the section cards (each with its position slider) sit below the 3D view but are made first: the 3D view
    # shows where the sections cut
    view_box = st.container()
    st.session_state["sm_sec_v"] = float(min(max(st.session_state.get("sm_sec_v", _saved("sm_sec_v", 0.5 * (v_lo + v_hi))), v_lo), v_hi))
    st.session_state["sm_sec_u"] = float(min(max(st.session_state.get("sm_sec_u", _saved("sm_sec_u", 0.5 * (rng[0] + rng[1]))), lo), hi + 1))
    st.markdown("##### :material/splitscreen: 단면으로 보기")
    st.caption("세 단면 모두 1:1 축척. 색 선 = 그 면을 지나는 절리면 (실선: 관측 연장·주요 절리면, 점선: 통계 절리망) · "
               "회색 점 = 암면 점군 단면 · 갈색 = 암반 · 검은 파선과 A·B·C 표시 = 다른 단면이 이 단면을 자르는 곳. "
               "확대·이동·점 정보는 각 그림의 '새 창에서 크게 보기'로.")
    # laid out by shape: the long, flat plan section across the page; the wall-parallel (~1.7:1) and the cross
    # section (~0.8:1) side by side at one height
    card_a = st.container(border=True)
    row = st.columns([2, 1], gap="medium")
    card_b, card_c = row[0].container(border=True, key="sm_card_b"), row[1].container(border=True, key="sm_card_c")
    with card_a:
        st.markdown("**:blue-badge[A] 수평 단면** :gray[x-y · 위에서 내려다본 면]")
        v0 = st.slider("표고 z (m)", v_lo, v_hi, step=0.25, key="sm_sec_v")
    with card_b:
        st.markdown("**:green-badge[B] 벽면 평행 단면** :gray[x-z · 암면을 그 깊이만큼 벗겨낸 모습]")
        dpar = st.slider("암면에서 깊이 (m)", 0.0, 10.0, step=0.25, **_w("sm_sec_d", 1.0))
    with card_c:
        st.markdown("**:orange-badge[C] 횡단면** :gray[y-z · 벽에 직각]")
        u0 = st.slider("측점 x (m)", lo, hi + 1, step=0.25, key="sm_sec_u")
    planes = {"v": v0, "u": u0, "depth_shown": jmax}

    surface = dense = None
    if face == "surface":
        surface = _session_cache("_sm_surface", (_result_key(), tuple(rng)), lambda: setmap3d.face_surface(result, rng))
    elif face == "points":                    # the ~3 M point section sample, in photo colours
        dense = _session_cache("_sm_dense", _result_key(), lambda: setmap3d.dense_face_sample(result))
    disp = _sm_disp(result, rng)
    # the same figure object while nothing it shows changes: the 3D page is then not rebuilt (~1 s) and re-sent
    fkey = (mkey if major else (_result_key(), tuple(sorted(opts.items()))),
            tuple(map(tuple, sets[["name", "dip", "dip_direction"]].values.tolist())), tuple(rng), jop, face, face_op,
            npts, v0, u0, jmax, disp.plan)
    if major:
        fig = _session_cache("_sm_ifig", fkey, lambda: setmap3d.major_figure(
            dense or sample, model, colour_of, jop, face, face_op, npts, surface=surface, planes=planes, disp=disp))
    else:
        fig = _session_cache("_sm_ifig", fkey, lambda: setmap3d.interior_figure(
            dense or sample, J["joints"], J["mesh"], colour_of, rng, jop, face, face_op, npts, surface=surface,
            planes=planes, disp=disp))
    with view_box:
        _view3d(fig, ("sm_interior_major" if major else "sm_interior") + ("_plan" if disp.plan else ""), 820)
        st.caption("x = 측점, y = 깊이(+ 암반 안쪽), z = 표고. 점선 사각형 = 아래 단면 A·C의 위치. 왼쪽 범례에서 항목을 "
                   "켜고 끌 수 있습니다. 마우스는 세슘과 같습니다 (가운데 드래그 또는 Ctrl + 왼쪽 드래그 = 회전, 더블클릭 = "
                   "회전 중심 지정, Space = 모델 중심으로).")

    # panel sizes = their width on a wide screen, so the fonts match; B and C have the same height
    prof = {"uvw": section._profile_points(result)}
    label = "새 창에서 크게 보기"
    with card_a:
        _figure_window(setmap3d.plan_section(prof, J["rock"], J["joints"], J["mesh"], colour_of, v0, rng, depth_shown=jmax,
                                             frame=result.wall_frame if disp.plan else None, refs={"u": u0, "d": dpar}),
                       "sm_plan", SECTION_PANEL_H, label, width_px=1470, height_px=SECTION_PANEL_H, show_title=False)
        st.caption(f"표고 {v0:.2f} m — 아래가 암면, 위로 갈수록 암반 안쪽. "
                   + ("실제 평면: 구간 가운데의 벽 방향을 가로로 놓은 직선 거리." if disp.plan else "측점 기준: 벽 기준선을 따라 펼친 좌표."))
    with card_b:
        _figure_window(setmap3d.parallel_section(result, J["rock"], J["joints"], J["mesh"], colour_of, dpar,
                                                 refs={"v": v0, "u": u0}),
                       "sm_par", SECTION_ROW_H, label, width_px=960, height_px=SECTION_ROW_H, show_title=False)
        st.caption(f"암면에서 {dpar:.2f} m 안쪽 (벽 전체). 회색 가는 선 = 암면(깊이 0)의 트레이스 — 경사진 절리가 깊이에 "
                   "따라 어디로 옮겨 가는지 비교.")
    with card_c:
        _figure_window(setmap3d.cross_section(prof, J["joints"], J["mesh"], colour_of, u0, depth_shown=jmax,
                                              refs={"v": v0, "d": dpar}, rock=J["rock"]),
                       "sm_cross", SECTION_ROW_H, label, width_px=465, height_px=SECTION_ROW_H, show_title=False)
        st.caption(f"측점 {u0:.2f} m — 왼쪽이 암면, 오른쪽으로 암반 안쪽.")
    st.html(SAME_HEIGHT_JS, unsafe_allow_javascript=True)


def render_tables(result):
    sets_rows = [
        {
            "set_id": d.set_id,
            "mean_dip": round(d.mean_dip, 1),
            "mean_dip_direction": round(d.mean_dip_direction, 1),
            "fisher_k": round(d.fisher_k, 1) if np.isfinite(d.fisher_k) else None,
            "n_points": d.n_points,
            "n_patches": len(d.patches),
            "spacing_mean_m": round(d.spacing_mean, 3) if d.spacing_mean else None,
        }
        for d in result.discontinuity_sets
    ]
    sets_df = pd.DataFrame(sets_rows)
    st.subheader("절리군")
    st.dataframe(sets_df, width="stretch")

    patch_rows = []
    for d in result.discontinuity_sets:
        for p in d.patches:
            u, v, _ = result.wall_frame.to_wall(p.centroid)[0]
            patch_rows.append(
                {
                    "set_id": p.set_id,
                    "patch_id": p.patch_id,
                    "chainage_m": round(u, 2),
                    "elevation_m": round(v, 2),
                    "trace_confirmed_2d": result.trace_confirmed.get(p.patch_id),
                    "n_points": p.n_points,
                    "dip": round(p.dip, 1),
                    "dip_direction": round(p.dip_direction, 1),
                    "roughness_rms_m": round(p.roughness_rms, 4),
                    "persistence_length_m": round(p.persistence_length, 2),
                    "area_m2": round(p.area, 3),
                }
            )
    patches_df = pd.DataFrame(patch_rows)
    st.subheader("개별 절리면")
    st.dataframe(patches_df, width="stretch")
    st.download_button(
        "절리면 목록 CSV 내려받기", patches_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="discontinuity_patches.csv",
    )


FIELD_TEMPLATE = ("id,dip,dip_direction,x,y,z,note\n"
                  "J01,78,292,,,,예) 경사 판상 절리 (Set 1 추정)\n"
                  "J02,8,297,,,,예) 층리\n")


def render_field_help():
    st.info("현장 실측 자료가 없어 방향 정확도를 검증하지 않았습니다.")
    st.markdown(
        "**권장 측정** — 클리노미터(또는 스마트폰 앱)로 절리면의 **경사/경사방향**을 잽니다.\n"
        "- 절리군마다 **10~20개**, 벽면 여러 구간에 고르게\n"
        "- **벽면을 가로지르는 수직 절리**(입면에 선으로만 보이는 것)를 꼭 포함 — 드론 3차원 분석이 가장 약한 방향입니다\n"
        "- 좌표(x, y, z)는 있으면 위치별 비교에 쓰고, 없어도 됩니다 (EPSG 좌표계는 점군과 같게)\n"
        "- 파일을 현장 폴더에 넣으면 다음 분석부터 자동으로 비교합니다")
    st.download_button("실측 입력 양식 내려받기 (CSV)", FIELD_TEMPLATE.encode("utf-8-sig"), file_name="field_joints.csv")
    folder = st.session_state.get("site_folder")
    up = st.file_uploader("작성한 실측 CSV를 현장 폴더에 넣기", type=["csv"])
    if up is not None and folder and Path(folder).is_dir():
        dest = Path(folder) / "field_joints.csv"
        if st.session_state.get("_field_upload") != up.file_id:      # once per uploaded file, not every rerun
            dest.write_bytes(up.getvalue())
            st.session_state["_field_upload"] = up.file_id
        st.success(f"저장했습니다: {dest} — 분석을 다시 실행하면 비교 결과가 나옵니다.")


def render_validation(result):
    if result.field_comparison is None:
        render_field_help()
        return
    summary = summarize_errors(result.field_comparison)
    cols = st.columns(4)
    cols[0].metric("현장 실측 수", summary["n"])
    cols[1].metric("평균 각도 오차", f"{summary['mean_error_deg']:.1f}°" if summary["mean_error_deg"] is not None else "-")
    cols[2].metric("RMSE", f"{summary['rmse_deg']:.1f}°" if summary["rmse_deg"] is not None else "-")
    cols[3].metric("최대 오차", f"{summary['max_error_deg']:.1f}°" if summary["max_error_deg"] is not None else "-")
    st.dataframe(result.field_comparison, width="stretch")


DEMO_FOLDER = str(_SRC_ROOT.parent / "data" / "raw" / "demo_site")
NONE = "(사용 안 함)"

# Settings remembered between sessions: the last site folder and, per folder, the dashboard options.
# GENERAL options carry over to a folder opened for the first time; SITE options (files, analysis limits,
# planar / mesh data) belong to one site only.
SETTINGS_FILE = Path.home() / ".groundwater" / "dashboard_settings.json"
GENERAL_KEYS = ["adv_auto", "adv_veg", "adv_exg", "adv_min_area", "adv_trace_len", "adv_friction",
                "seep_w_p21", "seep_w_openness", "seep_w_connectivity", "seep_w_wetness", "seep_cell",
                "g3d_layout2", "nw_cls", "nw_length", "nw_max", "nw_gap", "nw_stat", "tg_cls", "tg_minlen", "tg_spacing", "tg_depth", "tg_angle", "tg_extra",
                "g3d_spacing", "g3d_length", "g3d_offset", "g3d_hideveg2", "g3d_opacity", "g3d_wall_mode", "g3d_wall_pts", "g3d_seep", "g3d_seep_cls", "g3d_seep_op", "adv_clean_ground", "adv_clean_veg", "adv_clean_net", "adv_clean_flat", "adv_lidar", "adv_overlap", "adv_fixed_sets", "adv_excl_wall", "site3d_tiles", "ls_file", "ls_half", "ls_cls", "ls_an", "ls_mode", "ls_bg", "sj_area", "sj_factor", "sj_full", "sj_full_span", "sj_sym", "sj_disc_lines", "sj_sym_m", "sj_kinds", "ex_par", "ex_flat", "site3d_variant", "ex_variant", "sj_variant", "seep_bg", "seep_layer", "seep_op", "seep_tiles", "th_op", "th_tiles", "ta_win", "ta_z", "ta_area", "ta_near", "kin_phi", "kin_lat", "kin_scope", "kin_range", "kin_cell", "kin_smooth", "kin_local", "kin_noveg",
                "kin_modes", "kin_3d", "kin_tiles", "kin_3d_pts", "kin_3d_op", "kin_3d_cop", "st_mode", "st_method", "st_sigma", "st_cmap", "st_weighted", "g3d_joints", "g3d_dfn",
                "g3d_jdepth", "g3d_jmax", "g3d_dfn_depth", "g3d_jopacity", "g3d_jshade", "g3d_pred", "sec_band",
                "sm_cloud", "sm_tiles", "sm_interior", "sm_npts", "sm_other", "sm_jdepth", "sm_jmax", "sm_traces",
                "sm_dfn", "sm_dfn_depth", "sm_jop", "sm_coords", "sm_mode", "sm_major_n", "sm_major_max", "sm_major_minlen", "sm_major_sep", "sm_major_ext", "sm_major_depth", "sm_face_mode", "sm_face_op", "sm_face_pts", "sm_sec_d", "sm_tiles_joints"]
SITE_KEYS = ["pick_cloud", "pick_mesh", "pick_field", "adv_u_lo", "adv_u_hi", "adv_v_lo", "adv_v_hi",
             "adv_use_poly", "g3d_rows", "g3d_cut_on", "g3d_cut_range", "sec_u", "planar_dir", "mesh_path",
             "tiles_path", "sm_cloud_range", "sm_range", "sm_sec_v", "sm_sec_u", "th_rgb_dir", "th_dir", "th_range"]
_PLANAR_WIDGETS = ("adv_planar", "planar_dir_input", "planar_redetect_dir")
_LINKED_WIDGETS = {"planar_dir": list(_PLANAR_WIDGETS), "mesh_path": ["mesh_path_input"]}
_RESULT_STATE = ("result", "result_key", "output_dir", "loaded_from", "_figs", "_seep", "_seep_key", "_seep_default",
                 "_g3d", "_g3d_key", "_g3d_raw", "_joints_cache", "_ref_cache", "_sec_cache", "_3d_sample", "_3d_fig",
                 "_facade_disp", "_sm_disp", "_sm_sample", "_sm_cloudfig", "_sm_joints", "_sm_isample", "_sm_surface", "_sm_dense", "_sm_major",
                 "_sm_ifig", "_flow_sets", "_fam_sets",
                 "_kin_stereo", "_kin_map", "_kin_3dfig", "_kin_domains", "_th_imgs", "_th_folders", "_pop_thermal", "_ta", "_ta_smap", "_seep_ortho", "_seep_facade")


def _planar_changed(src):
    v = str(st.session_state.get(src, "")).strip().strip('"')
    st.session_state["planar_dir"] = v
    for k in _PLANAR_WIDGETS:
        if k != src:
            st.session_state[k] = v


def _planar_input(label, key, **kw) -> str:
    """The planar-data folder field; the sidebar, trace and grouting fields all edit the same value."""
    st.session_state.setdefault("planar_dir", _saved("planar_dir", ""))
    st.session_state[key] = st.session_state.get(key, st.session_state["planar_dir"])     # see _w
    return st.text_input(label, key=key, on_change=_planar_changed, args=(key,), **kw).strip().strip('"')


def _settings_file() -> dict:
    import json

    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _folder_id(folder) -> str:
    try:
        return str(Path(str(folder).strip().strip('"')).resolve())
    except OSError:
        return str(folder)


def _saved(key, default):
    """Remembered value of an option for the current folder (else the last general value, else `default`)."""
    return st.session_state.get("_saved_values", {}).get(key, default)


def _w(key, default):
    """Keyword args for a remembered widget: its state starts from the saved value.

    The value is assigned again in every run, just before the widget: a value put into Session State in an earlier
    run, before the widget existed (a restored setting in a view opened later), is kept by the server but the
    widget shows its own default instead (Streamlit 1.62)."""
    st.session_state[key] = st.session_state.get(key, _saved(key, default))
    return {"key": key}


def _restore_session():
    """Once per browser session: reopen the last site folder."""
    if st.session_state.get("_settings_restored"):
        return
    st.session_state["_settings_restored"] = True
    last = _settings_file().get("last_folder")
    if last and Path(last).is_dir() and "site_folder" not in st.session_state:
        st.session_state["site_folder"] = last


def _site_key(folder, section=None) -> str:
    """Settings key of a site folder, or of one section of it (each section keeps its own options)."""
    return _folder_id(folder) + (f"::{section}" if section else "")


def _apply_folder_settings(folder, section=None):
    """When the site folder or section changes (and on start), load its options into the widget states."""
    fid = _site_key(folder, section)
    prev = st.session_state.get("_settings_folder")
    if prev == fid:
        return
    if prev is not None:                       # another site: the result on screen belongs to the old one
        for k in _RESULT_STATE:
            st.session_state.pop(k, None)
    st.session_state["_settings_folder"] = fid
    data = _settings_file()
    values = {k: v for k, v in (data.get("general") or {}).items() if k in GENERAL_KEYS}
    values.update(data.get("folders", {}).get(fid) or {})
    st.session_state["_saved_values"] = values
    for k in GENERAL_KEYS + SITE_KEYS:
        st.session_state.pop(k, None)
        for linked in _LINKED_WIDGETS.get(k, []):
            st.session_state.pop(linked, None)
        if k in values:
            st.session_state[k] = values[k]


def _store_settings():
    """Write the current options, merged with the saved ones (widgets not drawn this run keep their value)."""
    import json

    folder = st.session_state.get("site_folder")
    if not folder or not Path(str(folder).strip().strip('"')).is_dir():
        return
    fid = _site_key(folder, st.session_state.get("_section"))
    vals = dict(st.session_state.get("_saved_values", {}))
    for k in GENERAL_KEYS + SITE_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            vals[k] = list(v) if isinstance(v, tuple) else v
    st.session_state["_saved_values"] = vals
    data = _settings_file()
    last_sec = {**(data.get("last_section") or {}), _folder_id(folder): st.session_state.get("_section")}
    new = {**data, "last_folder": _folder_id(folder), "last_section": last_sec,
           "general": {**(data.get("general") or {}), **{k: vals[k] for k in GENERAL_KEYS if k in vals}},
           "folders": {**(data.get("folders") or {}), fid: {k: vals[k] for k in SITE_KEYS if k in vals}}}
    if new != data:
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass


def _browse_folder():
    """Native folder dialog (the dashboard runs on this PC, so tkinter can open it)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askdirectory(initialdir=st.session_state.get("site_folder") or None,
                                       title="현장 자료 폴더 선택")
        root.destroy()
    except Exception as e:  # noqa: BLE001
        st.session_state["browse_error"] = str(e)
        return
    if path:
        st.session_state["site_folder"] = str(Path(path))


def _pick(label, files, folder, help_text=None, default_first=True, key=None):
    names = [str(f.relative_to(folder)) for f in files]
    options = names + [NONE] if default_first else [NONE] + names
    if key:
        if st.session_state.get(key, _saved(key, options[0])) not in options:
            st.session_state.pop(key, None)              # a file chosen last time is gone
            st.session_state[key] = options[0]
        st.session_state[key] = st.session_state.get(key, _saved(key, options[0]))     # see _w
    choice = st.selectbox(label, options, help=help_text, key=key)
    return None if choice == NONE else Path(folder) / choice


def _open_digitizer(ortho: str, folder: Path, open_page: bool = True):
    """Start the wall-line digitizer server (once) and open its page for this orthomosaic and site folder."""
    import os
    import subprocess
    import time
    import webbrowser

    from groundwater.viz.wall_digitizer.server import DEFAULT_PORT, is_running, page_url

    if not is_running(DEFAULT_PORT):
        env = {**os.environ, "PYTHONPATH": str(_SRC_ROOT)}
        subprocess.Popen([sys.executable, "-m", "groundwater.viz.wall_digitizer", "--no-browser"], env=env,
                         cwd=str(_SRC_ROOT.parent), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(40):
            if is_running(DEFAULT_PORT):
                break
            time.sleep(0.25)
    url = page_url(DEFAULT_PORT, ortho, folder)
    if open_page:
        webbrowser.open(url)
    return url


def _open_tiles_viewer(scene_path) -> str:
    import webbrowser

    from groundwater.viz.wall_digitizer.server import DEFAULT_PORT, is_running, tiles3d_url

    if not is_running(DEFAULT_PORT):
        _open_digitizer("", Path("."), open_page=False)
    url = tiles3d_url(DEFAULT_PORT, scene_path)
    webbrowser.open(url)
    return url


def _open_facade_tool(result_dir: str, site: str | None):
    import webbrowser

    from groundwater.viz.wall_digitizer.server import DEFAULT_PORT, facade_url, is_running

    if not is_running(DEFAULT_PORT):
        _open_digitizer("", Path(site or "."), open_page=False)
    url = facade_url(DEFAULT_PORT, result_dir, site)
    webbrowser.open(url)
    return url


def wall_line_tool(folder: Path, found: dict):
    """Button that opens the click-to-digitize page for wall_line.csv on the drone orthomosaic."""
    with st.expander("벽면 기준선 그리기 (정사영상에서 클릭)", expanded=found["wall_line"] is None):
        st.caption("드론 정사영상 위에서 암사면 하단을 따라 클릭해 wall_line.csv를 만듭니다. "
                   "저장 위치는 위의 현장 자료 폴더입니다.")
        ortho = st.text_input("정사영상 (GeoTIFF 파일 또는 들어 있는 폴더)", value=str(folder),
                              help="폴더를 넣으면 그 안에서 가장 큰 컬러 GeoTIFF를 찾습니다 (DTM·DSM 제외).")
        if st.button("정사영상 열기", width="stretch"):
            url = _open_digitizer(ortho.strip().strip('"'), folder)
            st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 저장한 뒤 이 화면을 새로 고치면 반영됩니다.")


def _section_picker(folder: Path):
    """The section of the site to analyse / view (구간/<name>/), or None for the folder itself."""
    secs = site_folder.sections(folder)
    if not secs:
        st.session_state["_section"] = None
        return None
    whole = "(현장 폴더 자체)"
    opts = secs + [whole]
    key = "site_section"
    if st.session_state.get(key) not in opts:
        last = (_settings_file().get("last_section") or {}).get(_folder_id(folder))
        st.session_state[key] = last if last in opts else opts[0]
    pick = st.selectbox("구간", opts, key=key,
                        help="U자 벽처럼 나눈 구간마다 기준선·구간 다각형·분석 결과가 따로 있습니다 (현장 폴더/구간/<이름>/). "
                             "점군·라이다·열화상 등 자료는 현장 폴더의 것을 함께 씁니다.")
    st.session_state["_section"] = None if pick == whole else pick
    return st.session_state["_section"]


def sidebar_folder():
    """Site-folder input: choose a folder, check the detected files, run. Returns a config or None."""
    _restore_session()
    st.session_state.setdefault("site_folder", DEMO_FOLDER)
    st.text_input("현장 자료 폴더", key="site_folder", help="마지막으로 쓴 폴더와 설정을 다음 실행 때 그대로 불러옵니다.")
    st.button("폴더 선택…", on_click=_browse_folder, width="stretch")
    if st.session_state.pop("browse_error", None):
        st.warning("폴더 선택 창을 열 수 없습니다. 경로를 직접 붙여 넣으세요.")
    folder = Path(st.session_state["site_folder"].strip().strip('"'))
    if not folder.is_dir():
        st.error("폴더를 찾을 수 없습니다.")
        return None
    section = _section_picker(folder)
    _apply_folder_settings(folder, section)
    found = site_folder.scan(folder)
    sdir = site_folder.section_dir(folder, section)
    if section:
        wl = sdir / site_folder.WALL_LINE_NAME
        found["wall_line"] = wl if wl.exists() else None
    for d in (sdir / site_folder.OUTPUT_DIRNAME, sdir):
        if (d / RESULT_PICKLE).exists():
            st.session_state["prev_result_dir"] = str(d)
            break
    else:
        st.session_state.pop("prev_result_dir", None)
    wall_line_tool(folder, found)
    if not found["point_cloud"] and not found["mesh"] and not found["lidar"]:
        st.error("점군(.las .laz .ply .pcd) 또는 메쉬(.obj) 파일이 없습니다.")
        return None

    st.markdown(":material/inventory_2: **찾은 자료**")
    from groundwater.facade.region import SECTION_FILE, load_section

    sec_poly = load_section(sdir / SECTION_FILE)
    clouds = site_folder.main_cloud_options(found, sec_poly)
    cloud = (_pick("주 점군 (사진측량 또는 라이다)", clouds, folder,
                   "이 구간을 덮는 점군이 먼저 나옵니다 (사진측량 → 라이다). 라이다만 있으면 라이다로 분석합니다: "
                   "구간 범위의 점만 읽고, 망·식생 뒤는 같은 라이다의 뒤쪽 층으로 보강합니다.", key="pick_cloud")
             if clouds else None)
    cloud_is_lidar = cloud is not None and cloud in found["lidar"]
    mesh = (_pick("메쉬 (점군이 없을 때만 사용)", found["mesh"], folder, default_first=cloud is None, key="pick_mesh")
            if found["mesh"] else None)
    if cloud is not None:
        info = site_folder.cloud_summary(cloud)
        if info:
            ex = info["extent_m"]
            st.caption(f"점 {info['points']:,}개 · 범위 {ex[0]:.0f} × {ex[1]:.0f} m, 높이 {ex[2]:.0f} m · "
                       f"색 {'있음' if info['rgb'] else '없음'} · 좌표계 {info['crs'] or '헤더에 없음'}")
            for w in info["warnings"]:
                st.warning(w)
    field = _pick("현장 절리 실측 (검증용)", found["field"], folder,
                  "dip, dip_direction 열이 있는 CSV", key="pick_field") if found["field"] else None
    if found["lidar"] or found["thermal"]:
        st.caption((f"라이다: {found['lidar'][0].name} — 망·식생 뒤 암반 보강에 사용 (고급 설정). " if found["lidar"] else "")
                   + ("열화상: '열화상' 화면에서 사용." if found["thermal"] else ""))
    if found["wall_line"]:
        st.caption(f"벽면 기준선: {found['wall_line'].name} 사용")
    else:
        st.caption("벽면 기준선 파일(wall_line.csv)이 없어 벽면을 직선 하나로 맞춥니다. "
                   "굽은 벽은 위의 '벽면 기준선 그리기'로 만드는 것이 좋습니다.")
    with st.expander("고급 설정", icon=":material/tune:"):
        auto = st.checkbox("분석 크기 자동 설정 (점 간격 기준)", **_w("adv_auto", True),
                           help="복셀 크기·법선 반경·패치 연결 거리·입면 영상 해상도를 점군 밀도에서 정합니다. "
                                "끄면 configs/example_site.yaml 값을 씁니다.")
        veg = st.checkbox("식생 자동 제외 (색 기준)", **_w("adv_veg", True),
                          help="초록색 점(나무·풀·이끼)을 절리 분석과 트레이스 검출에서 뺍니다. "
                               "입면 영상에는 그대로 보입니다. 마른 갈색 식생은 평면성 필터가 거릅니다.")
        clean_ground = st.checkbox("벽 아래 바닥·수면 면 제외", **_w("adv_clean_ground", True),
                                   help="벽 발치에서 위쪽 암면보다 앞으로 튀어나온 거의 수평인 면(바닥, 물 표면, 쌓인 돌)은 "
                                        "절리가 아니므로 절리군에서 뺍니다. 반구대에서는 하천 수면이 가장 큰 Set 0 면(159 m²)으로 "
                                        "잡혔습니다.")
        clean_net = st.checkbox("낙석방지망 덮인 곳 제외", **_w("adv_clean_net", True),
                                help="망이 드리운 곳은 모델 표면이 암반이 아니라 망입니다 (벽면과 나란한 매끈한 면, 지지 케이블은 "
                                     "긴 직선). 입면에서 1 m 주변 대부분이 망 특유의 청회색(색상 160~215°)인 곳을 절리 분석과 "
                                     "트레이스에서 뺍니다.")
        clean_flat = st.checkbox("평탄 지면·소단 제외", **_w("adv_clean_flat", True),
                                 help="거의 수평인 지면·소단·사면 위 땅(입면에서 경사 35° 미만인 띠)과, 사면 안쪽으로 1.5 m 이상 "
                                      "이어진 경사 30° 이하의 면을 절리군에서 빼고, 그 위의 트레이스(덮개 가장자리 등)도 뺍니다. "
                                      "암벽에 드러난 좁은 수평 절리 턱은 남깁니다.")
        use_lidar = st.checkbox("라이다로 망·식생 뒤 암반 보강", disabled=not found["lidar"], **_w("adv_lidar", True),
                                help="다중 반사 라이다의 마지막 반사는 낙석방지망·나무 뒤의 암반까지 닿습니다. 망·식생 구간에서는 "
                                     "라이다 점 중 가장 뒤쪽 층(암반)만 골라 절리 분석에 더합니다. 높이 기준을 맞추고(지오이드) 맨 "
                                     "암반에서 사진측량에 자동 정합합니다. 망이 암반에 바로 붙은 곳은 둘을 구분할 수 없습니다.")
        clean_veg_faces = st.checkbox("식생 위의 면 제외", disabled=not veg, **_w("adv_clean_veg", True),
                                      help="입면 사진에서 주변(반경 0.5 m)이 대부분 초록이고 맨 앞 표면에 있는 면은 마른 잎·가지 "
                                           "같은 식생 자체일 가능성이 커서 뺍니다. 잎 뒤로 보이는 암반 면은 그대로 둡니다.")
        exg = st.slider("식생 판정 기준 (초록 정도)", 0.02, 0.20, step=0.01, disabled=not veg, **_w("adv_exg", 0.06),
                        help="낮출수록 더 많이 제외 (연한 초록·이끼까지), 높일수록 진한 초록만 제외. "
                             "암반이 초록빛을 띠면 높이세요.")
        min_area = st.slider("최소 절리면 면적 (m²)", 0.0, 1.0, step=0.05, **_w("adv_min_area", 0.2),
                             help="이보다 작은 면은 절리면으로 세지 않습니다. 거친 암반면은 작은 조각으로 많이 "
                                  "쪼개지므로 0.2 m²(약 45 cm 사각형) 정도가 드론 해상도에 맞습니다. "
                                  "0이면 모두 포함.")
        fixed_txt = st.text_input("직접 지정할 절리군 (경사/경사방향)", **_w("adv_fixed_sets", ""),
                                  placeholder="예: 48/047, 30/120",
                                  help="자동 판정과 관계없이 항상 절리군으로 둘 방향. 그 방향 ±25° 안의 극점이 이 군에 들어가고 "
                                       "(더 가까운 다른 군이 있으면 그쪽), 평균 방향은 그 면들로 다시 계산합니다. 10° 안에 자동 "
                                       "절리군이 있으면 같은 군으로 봅니다. 스테레오넷 화면의 '후보 절리군'에서 바로 넣을 수도 "
                                       "있습니다. 다음 분석부터 적용됩니다.")
        fixed_sets = _parse_sets(fixed_txt)
        if fixed_sets is None:
            st.error("절리군은 '경사/경사방향' 형식으로, 쉼표로 구분해 넣으세요 (예: 48/047, 30/120).")
            fixed_sets = []
        excl_wall = st.checkbox("절벽면(굴착면)과 평행한 면은 절리군 찾기에서 빼기", **_w("adv_excl_wall", False),
                                help="석산 벽은 발파·기계로 깎은 면이라, 벽과 나란한 면(벤치면·발파면 또는 그 면을 따라 깨진 절리)이 "
                                     "넓게 드러나 절리군 판정을 좌우합니다. 켜면 경사 20° 초과이고 그 자리 벽면 방향과 20° 이내인 "
                                     "점은 절리군·절리면에서 빼고, 벽을 가로지르는 절리로만 절리군을 찾습니다. 끄면 (기본) 모두 "
                                     "쓰고, 결과에서 면마다 '벽면 평행 / 수평 / 교차'로 구분해 보여 줍니다 ('굴착면 구분' 화면).")
        trace_len = st.slider("최소 트레이스 길이 (m)", 0.2, 2.0, step=0.1, **_w("adv_trace_len", 0.5),
                              help="입면에서 이보다 짧은 선은 트레이스로 세지 않습니다.")
        friction = st.slider("마찰각 (°, 안정성 검토)", 20, 45, step=1, **_w("adv_friction", 30))
        st.markdown("**분석 범위** (측점·표고, 비워 두면 제한 없음)")
        c1, c2 = st.columns(2)
        u_lo, u_hi = c1.text_input("측점 최소 (m)", **_w("adv_u_lo", "")), c2.text_input("측점 최대 (m)", **_w("adv_u_hi", ""))
        v_lo, v_hi = c1.text_input("표고 최소 (m)", **_w("adv_v_lo", "")), c2.text_input("표고 최대 (m)", **_w("adv_v_hi", ""))
        from groundwater.facade.region import REGION_FILE
        has_poly = (folder / REGION_FILE).exists()
        if not has_poly:
            st.session_state["adv_use_poly"] = False
        use_poly = st.checkbox("입면 도구에서 그린 분석 범위 다각형 사용", disabled=not has_poly, **_w("adv_use_poly", has_poly))
        from groundwater.facade.region import BUFFER_FILE, SECTION_FILE
        has_buf = (folder / BUFFER_FILE).exists() and (folder / SECTION_FILE).exists()
        overlap = st.checkbox("구간 경계 겹침 처리 (버퍼)", disabled=not has_buf, **_w("adv_overlap", True),
                              help="U자 벽처럼 구간을 나눈 경우, 구간 경계 밖 약 20 m(section_buffer.csv)까지 함께 분석해서 "
                                   "경계에 걸친 절리면을 통째로 잡고, 결과는 중심이 이 구간(section_area.csv) 안에 있는 것만 "
                                   "남깁니다. 겹침 구간은 지도에 회색으로 표시됩니다.")
        st.markdown("**절벽면 평행 DEM·정사영상** (선택)")
        planar_run = _planar_input("폴더 (Metashape 평면 투영 등)", "adv_planar",
                                   help="있으면 입면 영상 색과 트레이스 검출에 이 고해상도 자료를 씁니다. "
                                        "처음 한 번 점군과 자동 정합합니다 (2~3분).")
    num = lambda x: float(x) if x.strip() else None  # noqa: E731
    try:
        region = {k: [num(a), num(b)] for k, (a, b) in {"chainage": (u_lo, u_hi), "elevation": (v_lo, v_hi)}.items()
                  if a.strip() or b.strip()}
    except ValueError:
        st.error("분석 범위에는 숫자만 넣으세요.")
        region = {}
    st.caption(f"결과 저장: {sdir / site_folder.OUTPUT_DIRNAME}")
    lidar = cloud if cloud_is_lidar else (found["lidar"][0] if found["lidar"] else None)
    if not cloud_is_lidar and sec_poly and found["lidar"]:      # the LiDAR that covers this section
        lidar = next((f for f in found["lidar"] if site_folder.las_coverage(f, sec_poly) >= 0.8), found["lidar"][0])
    return site_folder.build_config(folder, point_cloud=cloud, mesh=mesh, field=field, section=section,
                                    lidar=lidar,
                                    thermal=found["thermal"][0] if found["thermal"] else None,
                                    wall_line=found["wall_line"], auto_scale=auto, vegetation=veg,
                                    exg_threshold=exg, min_patch_area=min_area, region=region,
                                    use_region_polygon=use_poly, friction_deg=friction,
                                    trace_min_length=trace_len,
                                    planar_dir=planar_run.strip().strip('"') or None,
                                    clean_ground=clean_ground, clean_vegetation_faces=clean_veg_faces and veg,
                                    clean_netting=clean_net, clean_flat=clean_flat, use_lidar=use_lidar,
                                    section_overlap=overlap, fixed_sets=fixed_sets,
                                    exclude_wall_parallel=excl_wall)


def main():
    try:
        _main()
    finally:                                   # options changed anywhere this run are saved this run
        if st.session_state.get("_input_mode") == "현장 폴더":
            _store_settings()
        _note_program_modules()


STAGE_NAMES = {"load": "점군 읽기", "frame": "기준선·분석 범위", "auto_scale": "자동 설정 (점 간격)",
               "voxel": "식생 제외·복셀", "normals": "법선·곡률", "joints": "절리면·절리군",
               "facade": "입면 영상 (원본 전체 점)", "planar": "고해상도 평면 자료 (색·트레이스)",
               "kinematics": "안정성 검토", "save": "결과 저장"}


def _run_times(result):
    """Seconds per analysis stage of the run that made this result (results saved before this have none)."""
    t = getattr(result, "timings", None) or {}
    if not t:
        return
    with st.sidebar.expander(f"분석 소요 시간 {sum(t.values()) / 60:.1f}분", icon=":material/timer:"):
        st.dataframe(pd.DataFrame({"단계": [STAGE_NAMES.get(k, k) for k in t], "초": list(t.values())}),
                     hide_index=True, width="stretch")
        n = (result.config.get("load_thinning") or {}).get("points_total")
        if n:
            st.caption(f"원본 점 {n:,}개. 입면 영상은 원본의 모든 점으로, 절리 분석은 얇게 읽은 점으로 만듭니다.")


def _header(result):
    """Status badges and the key numbers of the result on screen."""
    cfg = result.config
    badges = [f":blue-badge[:material/location_on: {cfg.get('site_name') or '현장'}]"]
    if st.session_state.get("loaded_from"):
        badges.append(":gray-badge[:material/history: 저장된 결과]")
    else:
        badges.append(":green-badge[:material/check_circle: 방금 분석]")
    if cfg.get("crs"):
        badges.append(f":gray-badge[{cfg['crs']}]")
    if cfg.get("height_datum") == "orthometric":
        badges.append(":blue-badge[:material/height: 높이 = 해발고도 (KNGeoid18)]")
    else:
        badges.append(":orange-badge[:material/height: 높이 = 파일 그대로 (사진측량이면 타원체고, 해발 +수십 m)]")
    if cfg.get("lidar_rock"):
        n = cfg["lidar_rock"].get("points_added", 0)
        badges.append(f":violet-badge[:material/sensors: 라이다 망·식생 뒤 암반 {n / 1e6:.1f}백만 점]")
    if (cfg.get("planar") or {}).get("folder"):
        badges.append(":violet-badge[:material/layers: 절벽면 평행 고해상도 자료]")
    sec = (cfg.get("section") or {}).get("core_chainage")
    if sec:
        badges.append(f":gray-badge[:material/join: 담당 구간 측점 {sec[0]:.0f}–{sec[1]:.0f} m · 앞뒤 겹침 처리]")
    removed = cfg.get("vegetation_removed_fraction")
    if removed is not None:
        badges.append(f":green-badge[:material/park: 식생 제외 {removed * 100:.0f}%]")
    st.markdown(" ".join(badges))
    n_patch = sum(len(d.patches) for d in result.discontinuity_sets)
    k = result.kinematics
    modes = 0 if k is None or not len(k) else k["mode"].nunique()
    with st.container(horizontal=True):
        st.metric("절리군", f"{len(result.discontinuity_sets)}개", f"절리면 {n_patch:,}개", delta_color="off",
                  delta_arrow="off", border=True)
        st.metric("입면 트레이스", f"{len(result.traces):,}개", f"총 {sum(t.length_m for t in result.traces):,.0f} m",
                  delta_color="off", delta_arrow="off", border=True)
        st.metric("트레이스 방향군", f"{len(result.trace_families)}개", delta_color="off", border=True)
        st.metric("벽면 길이", f"{result.wall_frame.length:.0f} m", delta_color="off", border=True)
        st.metric("가능한 파괴 형태", f"{modes}종", "운동학적 검토" if modes else "없음", delta_color="off",
                  delta_arrow="off", border=True)


def _main():
    st.title("굴착사면 불연속면 · 지하수 유출통로 분석")
    st.caption("드론 사진측량 자료(1단계)로 절리를 찾고 물길과 그라우팅 계획을 검토합니다. 라이다·열화상 융합은 2단계에서 추가됩니다.")

    with st.sidebar:
        st.subheader(":material/folder_open: 1 · 현장 자료")
        mode = st.radio("입력 방식", ["현장 폴더", "설정 파일 (YAML)"], horizontal=True,
                        help="현장 폴더: 폴더 안의 점군·실측 자료를 자동으로 찾습니다. "
                             "설정 파일: 이전 분석의 분석결과/analysis_config.yaml 등을 불러와 세부 값을 조정할 때.")
        st.session_state["_input_mode"] = mode
        if mode == "현장 폴더":
            cfg = sidebar_folder()
        else:
            config_path = st.text_input("설정 파일 경로", value="configs/demo_site.yaml")
            cfg = load_config(config_path) if Path(config_path).is_file() else None
            if cfg is None:
                st.error("설정 파일을 찾을 수 없습니다.")
        st.subheader(":material/play_circle: 2 · 분석")
        run_clicked = st.button("분석 실행", type="primary", icon=":material/play_arrow:", disabled=cfg is None,
                                width="stretch")
        if mode == "현장 폴더" and cfg is not None:
            _store_settings()
        prev = st.session_state.get("prev_result_dir") if mode == "현장 폴더" else None
        load_clicked = False
        if prev:
            import datetime
            when = datetime.datetime.fromtimestamp((Path(prev) / RESULT_PICKLE).stat().st_mtime)
            load_clicked = st.button(f"이전 결과 보기 ({when:%Y-%m-%d %H:%M})", icon=":material/history:", width="stretch",
                                     help="이 폴더에 저장된 마지막 분석 결과를 다시 계산하지 않고 불러옵니다.")

    if run_clicked:
        import time

        for k in _RESULT_STATE:
            st.session_state.pop(k, None)
        with st.spinner("자료를 읽고 불연속면을 추출하는 중..."):
            result = run_pipeline(cfg)
            st.session_state["result"] = result
            st.session_state["output_dir"] = cfg["paths"].get("output_dir", "data/processed/output")
        with st.spinner("결과를 저장하는 중..."):
            t_save = time.perf_counter()
            save_results(result, st.session_state["output_dir"])
            result.timings["save"] = round(time.perf_counter() - t_save, 1)
        st.session_state["result_key"] = ("run", st.session_state["output_dir"], time.time())
        st.session_state["loaded_from"] = None
    elif load_clicked:
        for k in _RESULT_STATE:
            st.session_state.pop(k, None)
        with st.spinner("저장된 결과를 불러오는 중..."):
            mtime = (Path(prev) / RESULT_PICKLE).stat().st_mtime
            st.session_state["result"] = _load_results_cached(prev, mtime)
            st.session_state["result_key"] = ("saved", prev, mtime)
            st.session_state["output_dir"] = prev
            st.session_state["loaded_from"] = prev

    result = st.session_state.get("result")
    if result is None:
        st.info("왼쪽에서 현장 자료 폴더를 고르고 **분석 실행**을 누르세요. "
                "처음에는 합성 데모 자료(data/raw/demo_site)가 선택되어 있습니다.")
        return

    if st.session_state.get("loaded_from"):
        st.sidebar.info(f"저장된 결과를 보는 중: {st.session_state['loaded_from']}")
    else:
        st.sidebar.success(f"결과 저장됨: {st.session_state['output_dir']}")
    _run_times(result)
    _header(result)
    removed = result.config.get("faces_removed") or {}
    if any(removed.values()):
        st.caption(f"절리가 아닌 면 제외: 벽 아래 바닥·수면 {removed.get('ground_water', 0)}개, "
                   f"식생 위의 면 {removed.get('vegetation', 0)}개 (고급 설정에서 끄고 켬)")
    applied = result.config.get("auto_scale_applied")
    if applied:
        st.caption(f"자동 설정: 점 간격 {applied['point_spacing_m'] * 100:.1f} cm → 복셀 {applied['voxel_size'] * 100:.1f} cm, "
                   f"법선 반경 {applied['search_radius'] * 100:.0f} cm, 패치 연결 {applied['link_distance'] * 100:.0f} cm, "
                   f"입면 영상 {applied['facade_resolution'] * 100:.1f} cm/px")

    out_dir = st.session_state["output_dir"]
    if result.facade_rgb is not None and st.sidebar.button("입면 도구 열기 (수동 트레이스 · 분석 범위)",
                                                         width="stretch"):
        url = _open_facade_tool(out_dir, st.session_state.get("site_folder"))
        st.sidebar.markdown(f"새 탭이 안 열리면 [여기]({url}). 저장 후 **분석 실행**을 다시 누르세요.")

    ref_folder = _reference_folder(result)
    names = ["3D", "절리군 지도", "스테레오넷", "트레이스", "유출 가능성", "열화상", "그라우팅 입력", "안정성 검토", "라이다 단면", "지중 절리 (현장)", "굴착면 구분", "표",
             "현장 검증", "보고서"]
    if ref_folder:
        names.append("암각화 (참고 이미지)")
    # only the chosen view is computed and sent to the browser (st.tabs would run and send all of them)
    if st.session_state.get("view") not in names:
        st.session_state["view"] = names[0]
    view = st.segmented_control("화면", names, key="view", label_visibility="collapsed", required=True,
                                format_func=lambda n: f":material/{VIEW_ICONS.get(n, 'circle')}: {n}") or names[0]
    if view == "암각화 (참고 이미지)":
        render_reference_overlays(result, ref_folder)
    elif view == "3D":
        render_3d_view(result)
    elif view == "절리군 지도":
        render_setmap(result)
    elif view == "스테레오넷":
        render_stereonet(result)
    elif view == "트레이스":
        render_planar_redetect(result)
        render_trace_map(result)
        ft = families_table(result.trace_families, result.wall_frame, result.discontinuity_sets)
        if len(ft):
            st.subheader("트레이스 방향군")
            st.dataframe(ft, width="stretch")
            st.caption("linked_3d_set = 입면에 나타나는 각도가 맞는 3차원 절리군. '1 또는 2'처럼 둘 이상이면 그 절리군들이 이 "
                       "벽면에서 같은 각도의 선을 만들어 각도만으로는 가를 수 없다는 뜻입니다 (방향군 색은 첫 번째 절리군). "
                       "linked_3d_set이 비어 있는 방향군은 3차원 면으로 잡히지 않은 절리(주로 벽면을 가로지르는 절리)입니다. "
                       "est_dip/est_dip_direction은 트레이스들의 3차원 방향에서 추정한 값이며, 벽면이 평평하면 정할 수 없습니다.")
        angles = _figure(result, "trace_angles")
        if angles:
            st.image(angles)
    elif view == "유출 가능성":
        render_seepage(result)
    elif view == "열화상":
        render_thermal(result)
    elif view == "그라우팅 입력":
        render_grouting(result)
    elif view == "안정성 검토":
        render_kinematics(result)
    elif view == "라이다 단면":
        render_lidar_section(result)
    elif view == "지중 절리 (현장)":
        render_site_joints(result)
    elif view == "굴착면 구분":
        render_exposure(result)
    elif view == "표":
        render_tables(result)
    elif view == "현장 검증":
        render_validation(result)
    elif view == "보고서":
        rp = Path(out_dir) / report.REPORT_NAME
        if st.button("보고서 다시 만들기"):
            report.build_report(result, out_dir)
        if rp.exists():
            st.download_button("분석 보고서 내려받기 (HTML)", rp.read_bytes(), file_name=f"{result.config.get('site_name', '')}_{report.REPORT_NAME}")
            st.caption(f"저장 위치: {rp} — 브라우저로 열고 인쇄하면 PDF로 만들 수 있습니다.")
        else:
            st.info("아직 보고서가 없습니다. '보고서 다시 만들기'를 누르세요.")


@st.cache_resource(max_entries=2, show_spinner=False)
def _lidar_index(path: str, mtime: float):
    from groundwater import lidar_section as LS

    return LS.load_index(path)


@st.cache_resource(max_entries=1, show_spinner=False)
def _analysis_cloud(folder: str, mtime: float):
    """(points, set id per point, {set id: 'Set k  dip/dd'}) of a saved result's analysis cloud, or None."""
    import open3d as o3d

    d = Path(folder)
    ply, lab = d / "classified_point_cloud.ply", d / "point_labels.npy"
    if not ply.exists() or not lab.exists():
        return None
    pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    sets = np.load(lab)[:, 0]
    names = {}
    if (d / "discontinuity_sets.csv").exists():
        t = pd.read_csv(d / "discontinuity_sets.csv")
        names = {int(r.set_id): f"Set {int(r.set_id)} {r.mean_dip:.0f}°/{r.mean_dip_direction:03.0f}°" for r in t.itertuples()}
    return pts, sets, names


def _section_polygons(site: Path):
    from groundwater.facade.region import SECTION_FILE, load_section

    colours = {"E": "#1f77b4", "S": "#d62728", "W": "#2ca02c"}
    out = []
    for name in site_folder.sections(site):
        poly = load_section(site_folder.section_dir(site, name) / SECTION_FILE)
        if poly:
            out.append((f"구간 {name}", np.asarray(poly), colours.get(name[:1], "#7c3aed")))
    return out


def render_lidar_section(result):
    """Vertical section through the LiDAR scan along a line drawn on the plan (or across the wall at a chainage):
    the returns by kind and the rock layer the program keeps -- to check the scan and the classification."""
    from groundwater import lidar_section as LS
    from groundwater.viz import lidar_section_view as LV

    site = Path(str(st.session_state.get("site_folder", "")).strip().strip('"'))
    lidars = site_folder.scan(site)["lidar"] if site.is_dir() else []
    if not lidars:
        st.info("이 현장 폴더에 라이다(.las) 파일이 없습니다.")
        return
    st.caption("평면도에 선을 그리면 그 선을 따라 수직으로 자른 라이다 단면을 보여 줍니다. 얇은 단면 안의 점을 반사 종류(단일 · 첫 · "
               "중간 · 마지막)로 칠하고, 프로그램이 망·식생 뒤에서 암반으로 남기는 뒤쪽 층(검정, 벽면 안 완경사는 보라)과 벽면 범위 밖이라 빼는 층(주황: 원지반·나무 줄기·바닥), 실제 "
               "절리 분석에서 절리군에 들어간 점(절리군 색 ◆)을 겹칩니다. 라이다 품질과 암반 판정을 눈으로 확인하는 화면입니다.")
    names = [str(f.relative_to(site)) for f in lidars]
    from groundwater.facade.region import SECTION_FILE, load_section

    poly = load_section(site_folder.section_dir(site, st.session_state.get("_section")) / SECTION_FILE)
    default = next((n for n, f in zip(names, lidars) if site_folder.las_coverage(f, poly) >= 0.8), names[0])
    if st.session_state.get("ls_file") not in names:
        st.session_state.pop("ls_file", None)
    c = st.columns([2.6, 1.2, 1.3, 1.3], vertical_alignment="bottom")
    pick = c[0].selectbox("라이다 파일", names, **_w("ls_file", default),
                          help="처음에는 지금 구간을 덮는 파일이 골라집니다.")
    las = site / pick
    idx = _lidar_index(str(las), las.stat().st_mtime)
    if idx is None:
        st.warning(f"{las.name}은 아직 색인이 없습니다. 처음 한 번 파일 전체를 읽어 색인(작은 평면도와 위치 목록)을 만듭니다 "
                   f"({las.stat().st_size / 1e9:.0f} GB: 약 {max(1, round(las.stat().st_size / 6e9))}분). 그다음부터 단면은 1~2초입니다.")
        if st.button("색인 만들기", icon=":material/build:"):
            bar = st.progress(0.0, "라이다 색인 만드는 중...")
            LS.build_index(las, lambda f, msg: bar.progress(min(1.0, f), msg))
            _lidar_index.clear()
            st.rerun()
        return
    half = c[1].select_slider("단면 두께 (± cm)", [2, 5, 10, 20], **_w("ls_half", 5)) / 100
    show_cls = c[2].toggle("프로그램 판정 겹치기", **_w("ls_cls", True),
                           help="망·식생 뒤에서 암반으로 남기는 뒤쪽 층(검정·보라)과 벽면 범위 밖이라 빼는 뒤쪽 층(주황)")
    show_an = c[3].toggle("분석에 쓴 점 겹치기", **_w("ls_an", True), help="이 위치를 담당하는 구간의 분석 결과 점 (절리군 색)")

    modes = ["평면도에서 선 그리기", "구간·측점 (벽에 직각)", "좌표 입력"]
    mode = st.radio("단면 정하기", modes, horizontal=True, **_w("ls_mode", modes[0]))
    line = st.session_state.get("ls_line")
    secs = site_folder.sections(site)
    if mode == modes[1]:
        if not secs:
            st.info("구간 폴더(구간/<이름>/)가 없어 현장 폴더의 기준선을 씁니다.")
        cc = st.columns([1.4, 1.2, 1.0, 1.0, 1.0], vertical_alignment="bottom")
        sec = cc[0].selectbox("구간", secs or ["(현장 폴더)"], key="ls_sec")
        u = cc[1].number_input("측점 (m)", value=float(st.session_state.get("ls_u", 50.0)), step=1.0, key="ls_u")
        front = cc[2].number_input("굴착지 쪽 (m)", value=15.0, step=1.0, key="ls_front")
        back = cc[3].number_input("암반 쪽 (m)", value=40.0, step=1.0, key="ls_back")
        if cc[4].button("단면 보기", type="primary", width="stretch"):
            from groundwater.site_folder import WALL_LINE_NAME, read_wall_line
            from groundwater.facade.frame import build_wall_frame

            folder = site_folder.section_dir(site, sec if secs else None)
            poly, pit = read_wall_line(folder / WALL_LINE_NAME)
            P = np.asarray(poly, float)
            frame = build_wall_frame(np.column_stack([P, np.zeros(len(P))]), poly, pit)
            a, b = LS.perpendicular_line(frame, u, front, back)
            st.session_state["ls_line"] = line = (tuple(a), tuple(b))
    elif mode == modes[2]:
        a0, b0 = line if line else ((0.0, 0.0), (0.0, 0.0))
        cc = st.columns(5, vertical_alignment="bottom")
        ax_ = cc[0].number_input("A x", value=float(a0[0]), format="%.2f")
        ay_ = cc[1].number_input("A y", value=float(a0[1]), format="%.2f")
        bx_ = cc[2].number_input("B x", value=float(b0[0]), format="%.2f")
        by_ = cc[3].number_input("B y", value=float(b0[1]), format="%.2f")
        if cc[4].button("단면 보기", type="primary", width="stretch"):
            st.session_state["ls_line"] = line = ((ax_, ay_), (bx_, by_))

    left, right = st.columns([1, 1.35])
    with left:
        orthos = LV.find_orthos(site)
        bgs = [f.name for f in orthos] + ["라이다 윗면 색"]
        if st.session_state.get("ls_bg") not in bgs:
            st.session_state.pop("ls_bg", None)
        bg = st.selectbox("평면 바탕", bgs, **_w("ls_bg", bgs[0]),
                          help="현장 폴더의 정사영상(최근 날짜 먼저) 또는 라이다 점의 윗면 색. 정사영상은 구간 범위(+40 m)를 "
                               "고해상도로 보여 줍니다.")
        polys = _section_polygons(site)
        img, ext = _session_cache("_ls_plan", str(las), lambda: LV.plan_image(idx))
        if bg in [f.name for f in orthos]:
            if polys:                                   # the sections (+40 m), at a useful resolution
                allp = np.vstack([p for _, p, _ in polys])
                lo, hi = allp.min(axis=0) - 40, allp.max(axis=0) + 40
                ext = [lo[0], hi[0], lo[1], hi[1]]
            img, ext = _session_cache("_ls_ortho", (bg, tuple(np.round(ext, 1))),
                                      lambda: LV.ortho_image(site / bg, ext, 2000))
        fig = LV.plan_figure(img, ext, line, polys)
        ev = st.plotly_chart(fig, key="ls_plan", on_select="rerun", selection_mode="lasso",
                             config={"scrollZoom": False, "displaylogo": False})
        st.caption("올가미(lasso)로 선을 긋듯 드래그하세요: 시작점이 A, 끝점이 B. 휠 대신 위 도구 막대로 확대합니다. "
                   "노란 선 = 지금 단면.")
        lasso = (ev.selection.get("lasso") if ev and getattr(ev, "selection", None) else None) or []
        if mode == modes[0] and lasso:
            xs, ys = lasso[0].get("x") or [], lasso[0].get("y") or []
            if len(xs) >= 2 and np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) > 1.0:
                new = ((float(xs[0]), float(ys[0])), (float(xs[-1]), float(ys[-1])))
                if new != line:
                    st.session_state["ls_line"] = line = new
                    st.rerun()
    with right:
        if not line:
            st.info("왼쪽 평면도에 선을 그리거나, '구간·측점'으로 벽에 직각인 단면을 고르세요.")
            return
        a, b = np.asarray(line[0], float), np.asarray(line[1], float)
        if np.hypot(*(b - a)) < 1.0:
            st.info("선이 너무 짧습니다 (1 m 이상).")
            return
        key = (str(las), tuple(np.round(np.r_[a, b], 2)))
        with st.spinner("단면의 점을 읽는 중..."):
            slab = _session_cache("_ls_slab", key, lambda: LS.read_slab(las, idx, a, b))
        if not len(slab["xyz"]):
            st.warning("이 선 주변에 라이다 점이 없습니다.")
            return
        kinds = LS.return_kind(slab["ret"], slab["nret"])
        owner, ofolder = LS.owning_section(site, (a + b) / 2)
        res_dir = ofolder / site_folder.OUTPUT_DIRNAME
        ac = (_analysis_cloud(str(res_dir), (res_dir / "result.pkl").stat().st_mtime)
              if (show_cls or show_an) and (res_dir / "result.pkl").exists() else None)
        cls = None
        if show_cls:
            def classify():
                frame = LS.section_frame(ofolder, slab["xyz"][:: max(1, len(slab["xyz"]) // 200_000)])
                from groundwater.site_joints import find_ground

                # the section's analysis points: the whole wall column, as the analysis sees it (an oblique slab
                # holds only part of it)
                near = None
                if ac is not None:
                    lo, hi = slab["xyz"][:, :2].min(0) - 30.0, slab["xyz"][:, :2].max(0) + 30.0
                    near = ac[0][np.all((ac[0][:, :2] >= lo) & (ac[0][:, :2] <= hi), axis=1)]
                return LS.classify(slab["xyz"], frame, find_ground(site), near)
            cls = _session_cache("_ls_cls", (key, str(ofolder), "dtm", "outside", ac is not None), classify)
        analysis, set_names = None, None
        if show_an:
            if ac is not None:
                pts, sids, set_names = ac
                t = (b - a) / np.hypot(*(b - a))
                nrm = np.array([-t[1], t[0]])
                rel = pts[:, :2] - a
                ss, dd = rel @ t, rel @ nrm
                m = (np.abs(dd) <= max(half, 0.05)) & (ss >= -1) & (ss <= slab["length"] + 1)
                analysis = (ss[m], pts[m, 2], sids[m])
        title = (f"단면 A({a[0]:.1f}, {a[1]:.1f}) → B({b[0]:.1f}, {b[1]:.1f}) · 길이 {slab['length']:.1f} m · 두께 ±{half * 100:.0f} cm"
                 + (f" · 구간 {owner}" if owner else ""))
        sfig = LV.section_figure(slab, kinds, cls, half, analysis, set_names, title, show_cls)
        _figure_window(sfig, "lidar_section", 760, "새 창에서 크게 보기 — 라이다 단면 (휠 확대·축소, 점 정보)")
        stt = LV.slice_stats(slab, kinds, cls, half)
        multi = stt["첫 반사 (망·잎 등)"] + stt["중간 반사"] + stt["마지막 반사"]
        m_ = st.columns(4)
        m_[0].metric("단면 안 점", f"{stt['전체']:,}")
        m_[1].metric("다중 반사 비율", f"{100 * multi / max(1, stt['전체']):.0f}%",
                     help="망·잎을 지나 뒤에서도 돌아온 펄스의 점. 망·식생 구간에서 높습니다.")
        if cls is not None:
            m_[2].metric("암반으로 남긴 점", f"{stt['암반으로 남김']:,}",
                         help=f"그중 벽면 안 완경사(소단·사면) {stt.get('벽면 안 완경사 (암반)', 0):,}")
            m_[3].metric("벽면 밖·지면 위로 뺀 점", f"{stt['벽면 범위 밖 제외'] + stt['지면 위 제외']:,}",
                         help=f"벽면 범위 밖 {stt['벽면 범위 밖 제외']:,} · 지면 위(수목) {stt['지면 위 제외']:,}")
        st.caption(f"읽은 위치 블록 {slab['chunks_read']}개 · 판정은 이 위치를 담당하는 "
                   + (f"구간 {owner}의 기준선 기준" if owner else "현장 폴더 기준선 기준")
                   + ". 판정(뒤쪽 층)은 프로그램이 망·식생 셀에서 쓰는 규칙입니다: 5 cm 셀마다 가장 뒤쪽 점에서 6 cm 안, "
                     "그중 벽면 범위 안의 점(벽면 열마다 가장 아래부터 가장 위의 급경사 면(1 m 범위 경사 60° 이상)까지, 높이 4 m · 깊이 10 m 넘게 떨어지지 않고 이어진 범위; 그 사이 완경사 = 소단·사면도 암반, 그 위의 원지반·나무 줄기와 아래 바닥은 제외; 경사는 깊이 0.5 m 안의 점으로만 구함)이고, 주변 5 m 안 가장 높은 지면(DTM)보다 2 m 넘게 위가 아닌 점.")
        out = (res_dir if res_dir.exists() else site) / "단면확인"
        name = f"단면_{a[0]:.0f}_{a[1]:.0f}_{b[0]:.0f}_{b[1]:.0f}"
        bc = st.columns([1, 1, 2])
        if bc[0].button("그림 저장 (PNG)", icon=":material/save:", width="stretch"):
            from groundwater.viz import static_map

            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.png").write_bytes(static_map.preview_image(sfig, width_px=1400, max_height_px=1000))
            st.success(f"저장: {out / (name + '.png')}")
        csv = pd.DataFrame({"x": slab["xyz"][:, 0], "y": slab["xyz"][:, 1], "z": slab["xyz"][:, 2], "s_m": slab["s"],
                            "d_m": slab["d"], "return": slab["ret"], "returns": slab["nret"],
                            **({"rock_layer": cls["rock"], "ledge_rear": cls["ledge"], "outside_rear": cls["outside"],
                                  "above_ground_rear": cls["air"]}
                               if cls is not None else {})})
        csv = csv[np.abs(csv["d_m"]) <= half]
        bc[1].download_button("점 내려받기 (CSV)", csv.to_csv(index=False).encode("utf-8-sig"), f"{name}.csv", "text/csv",
                              icon=":material/download:", width="stretch")



def render_exposure(result):
    """Excavated surface vs joints: each face of the chosen sections classified by how it lies to the excavated
    wall it was found on (discontinuity/exposure.py) -- by section and set, on stereonets, elevation, plan."""
    from groundwater.discontinuity import exposure as X
    from groundwater.viz import exposure_view as EV
    from groundwater.viz import lidar_section_view as LV
    from groundwater.viz import site3d

    site = Path(str(st.session_state.get("site_folder", "")).strip().strip('"'))
    secs = site3d.section_results(site) if site.is_dir() else []
    if not secs:
        st.info("구간 결과가 없습니다 (현장 폴더/구간/<이름>/분석결과).")
        return
    st.caption("석산 벽은 절리를 따라 무너진 면이 아니라 발파·기계로 깎은 면입니다. 그래서 찾은 절리면에는 **굴착면 자체**(벤치면·"
               "발파면, 또는 그 면을 따라 깨진 절리 — 방향만으로는 구분 안 됨), **소단·바닥**(거의 수평), **벽을 가로지르는 "
               "절리**가 섞여 있습니다. 벽과 나란한 면은 넓게 드러나고 가로지르는 절리는 좁은 띠·트레이스로만 보여서, 면적으로 "
               "세면 벽면 평행이 과대평가됩니다. 벽면 방향은 그 자리의 벽(벽선 + 벽 앞면의 굽음, 5 m 규모) 기준입니다.")
    c = st.columns([1.2, 3, 1.2, 1.2], vertical_alignment="bottom")
    variant = _variant_picker(site, "ex_variant", c[0])
    names = [n for n, _ in site3d.section_results(site, variant)]
    c = c[1:]
    pick = c[0].multiselect("구간 결과", names, default=[n for n in names if n != "현장 폴더"] or names, key="ex_secs")
    par = c[1].select_slider("벽면 평행 기준 (°)", [10, 15, 20, 25, 30], **_w("ex_par", 20),
                             help="극점이 그 자리 벽면 법선과 이 각도 이내이고 경사가 수평 기준보다 크면 '벽면 평행'.")
    flat = c[2].select_slider("수평 기준: 경사 ≤ (°)", [10, 15, 20, 25, 30], **_w("ex_flat", 20))
    if not pick:
        return
    P = _session_cache("_ex_faces", (tuple(pick), par, flat, str(site), variant),
                       lambda: EV.load_sections(site, pick, float(par), float(flat), variant))
    if not len(P):
        st.info("절리면이 없습니다.")
        return
    tot = P["area_m2"].sum()
    m = st.columns(4)
    m[0].metric("절리면", f"{len(P):,}", help=f"면적 {tot:,.0f} m²")
    for i, cl in enumerate(EV.ORDER):
        R = P[P["exposure"] == cl]
        m[i + 1].metric(X.CLASSES[cl], f"{100 * R['area_m2'].sum() / tot:.0f}% (면적)", f"면 {len(R):,}개",
                        delta_color="off")
    tabs = st.tabs([":material/bar_chart: 구간·절리군별", ":material/explore: 스테레오넷", ":material/view_week: 입면",
                    ":material/map: 평면", ":material/table_chart: 면 목록"])
    with tabs[0]:
        st.plotly_chart(EV.bar_figure(P), width="stretch", key="ex_bar")
        st.dataframe(EV.table(P), hide_index=True, width="stretch")
        st.caption("거칠기 = 면 점들의 평면 오차(RMS). 이 자료에서는 벽면 평행 면과 교차 면의 거칠기가 거의 같아서 거칠기만으로 "
                   "발파면을 가려내기는 어렵습니다 (현장에서 발파공 흔적·풍화·충전물로 확인 필요).")
    with tabs[1]:
        st.image(_session_cache("_ex_st", (tuple(pick), par, flat, variant), lambda: EV.stereonets_png(P)),
                 width="stretch")
        st.markdown("**교차·수평 면만으로 다시 찾은 방향** (면 개수 기준)")
        R = _session_cache("_ex_refind", (tuple(pick), par, flat, variant), lambda: EV.refind(P))
        if len(R):
            st.dataframe(R, hide_index=True)
        else:
            st.caption("뚜렷한 방향이 없습니다.")
        st.caption("면은 분석에서 찾은 절리군의 방향으로만 만들어지므로, 여기서는 그 절리군들 중 벽을 가로지르는 쪽이 어디인지만 "
                   "보입니다. 벽면 평행 면에 가려진 다른 절리군을 찾으려면 분석 설정의 '절벽면(굴착면)과 평행한 면은 절리군 "
                   "찾기에서 빼기'를 켜고 다시 분석하세요.")
    with tabs[2]:
        for n in pick:
            Q = P[P["section"] == n]
            if len(Q):
                _figure_window(EV.elevation_figure(Q, Q["folder"].iloc[0]), f"ex_elev_{n.split('_')[0]}", 520,
                               f"새 창에서 크게 보기 — {n} 입면")
    with tabs[3]:
        orthos = LV.find_orthos(site)
        lo, hi = P[["centroid_x", "centroid_y"]].min().to_numpy() - 40, P[["centroid_x", "centroid_y"]].max().to_numpy() + 40
        bg = _session_cache("_ex_bg", (str(orthos[0]) if orthos else "", tuple(np.round(np.r_[lo, hi], 1))),
                            lambda: LV.ortho_image(orthos[0], [lo[0], hi[0], lo[1], hi[1]], 1600) if orthos else None)
        _figure_window(EV.plan_figure(P, bg), "ex_plan", 820, "새 창에서 크게 보기 — 평면")
    with tabs[4]:
        T = P[["section", "set_label", "patch_id", "dip", "dip_direction", "area_m2", "roughness_rms_m", "wall_az",
               "wall_angle_deg", "exposure", "chainage_m", "elevation_m", "centroid_x", "centroid_y", "centroid_z"]].copy()
        T["exposure"] = T["exposure"].map(X.CLASSES)
        T = T.rename(columns={"section": "구간", "set_label": "절리군", "patch_id": "면", "dip": "경사",
                              "dip_direction": "경사방향", "area_m2": "면적 (m²)", "roughness_rms_m": "거칠기 (m)",
                              "wall_az": "벽면 법선 방위", "wall_angle_deg": "벽면과의 각 (°)", "exposure": "구분",
                              "chainage_m": "측점", "elevation_m": "표고"}).round(2)
        st.dataframe(T, hide_index=True, width="stretch")
        st.download_button("CSV 내려받기", T.to_csv(index=False).encode("utf-8-sig"), "절리면_굴착면구분.csv", "text/csv",
                           icon=":material/download:")


def render_site_joints(result):
    """Site-wide joint model: every section's joint faces in one space, extended into the rock, cut by a plan
    section at an elevation and by a vertical section along a line -- how the discontinuities may run underground,
    apart from the cliffs that exposed them."""
    from groundwater import site_joints as SJ
    from groundwater.viz import lidar_section_view as LV
    from groundwater.viz import site3d
    from groundwater.viz import site_joints_view as V
    from groundwater.viz import tiles3d

    site = Path(str(st.session_state.get("site_folder", "")).strip().strip('"'))
    secs = site3d.section_results(site) if site.is_dir() else []
    if not secs:
        st.info("구간 결과가 없습니다 (현장 폴더/구간/<이름>/분석결과).")
        return
    st.caption("모든 구간의 절리면을 한 공간에 놓고, 같은 방향의 절리군은 구간이 달라도 같은 색(±20°)으로 그립니다. 관측된 면은 그 "
               "크기의 원판, **연장**은 그 원판을 몇 배로 키운 가정입니다 (굴착면에 드러난 절리는 잘려 있어 실제로는 더 멀리 "
               "이어집니다). **연속 가능면**은 같은 절리군이고 한 평면 위(1 m, 8° 이내, 120 m 범위)에 놓인 면들의 묶음으로, "
               "관측 자료가 직접 보여 주는 연속성입니다. 지표는 맨땅 DTM이고, 그 아래가 땅속(암반·토사)입니다.")
    c0 = st.columns([1.2, 4], vertical_alignment="bottom")
    variant = _variant_picker(site, "sj_variant", c0[0])
    names = [n for n, _ in site3d.section_results(site, variant)]
    c = st.columns([3, 1.2, 1.2, 1.6, 1.4], vertical_alignment="bottom")
    pick = c[0].multiselect("구간 결과", names, default=[n for n in names if n != "현장 폴더"] or names, key="sj_secs")
    min_area = c[1].select_slider("최소 면적 (m²)", [2, 5, 10, 20, 50], **_w("sj_area", 5))
    kinds_opt = {"모든 면": ("parallel", "flat", "cross"), "벽면 평행 제외": ("flat", "cross"), "교차만": ("cross",)}
    kinds = st.segmented_control("쓰는 면", list(kinds_opt), **_w("sj_kinds", "모든 면"),
                                 help="'굴착면 구분' 화면의 구분: 벽면 평행 면(굴착면 또는 그 면을 따른 절리)은 한 벽을 따라 "
                                      "이어져 보이기 쉬워서, 땅속 연속성을 볼 때는 빼 보는 것이 좋습니다.") or "모든 면"
    factor = c[2].select_slider("연장 배수", [1, 2, 3, 5, 10, 20], **_w("sj_factor", 3),
                                help="관측된 면 원판의 반지름을 몇 배로 키울지 (가정). 1 = 관측된 크기 그대로.")
    full = c[3].toggle("연속 가능면 끝까지 연장", **_w("sj_full", False),
                       help="연속 가능면을 평면 그대로 끝없이 이어 그립니다 (점선, 가정). 지중에서 어디를 지날 수 있는지 봅니다.")
    full_span = c[4].select_slider("끝까지 연장할 면: 길이 ≥ (m)", [0, 10, 20, 40, 60], **_w("sj_full_span", 20),
                                   disabled=not full,
                                   help="연속 가능면 중 관측된 면들이 이만큼 이상 퍼져 있는 것만 끝까지 이어 그립니다 (많으면 복잡해짐).")
    if not pick:
        return
    key = (tuple(pick), min_area, kinds, variant)

    def build():
        F = SJ.load_faces(site, pick, float(min_area), kinds_opt[kinds], variant)
        C = SJ.continuity(F)
        lo, hi = F[["x", "y"]].min().to_numpy() - 80, F[["x", "y"]].max().to_numpy() + 80
        gpath = SJ.find_ground(site)
        ground = SJ.ground_grid(gpath, (lo[0], lo[1], hi[0], hi[1]), 2.0) if gpath else None
        return F, C, ground, (lo, hi), gpath
    with st.spinner("구간 결과를 모으는 중..."):
        F, C, ground, (lo, hi), gpath = _session_cache("_sj_model", key, build)
    if not len(F):
        st.info("조건에 맞는 절리면이 없습니다.")
        return
    m = st.columns(4)
    m[0].metric("절리면", f"{len(F):,}")
    m[1].metric("현장 절리군", F["group"].nunique())
    m[2].metric("연속 가능면", f"{len(C):,}", help="같은 평면 위의 면 2개 이상")
    m[3].metric("여러 구간에 걸친 연속 가능면", int(C["sections"].str.contains("·").sum()) if len(C) else 0)
    tabs = st.tabs([":material/view_in_ar: 3D", ":material/satellite_alt: 정사영상 위 절리", ":material/layers: 수평 단면",
                    ":material/vertical_split: 수직 단면", ":material/table_chart: 연속 가능면 표"])
    orthos = LV.find_orthos(site)
    with tabs[0]:
        cc = st.columns([1.4, 3], vertical_alignment="center")
        under = cc[0].toggle("연장 부분은 땅속만", value=True, key="sj_under",
                             help="연장한 원판 중 지표 위로 나오는 부분은 그리지 않습니다 (공중으로 뻗은 부분).")
        with cc[1].popover("실사 타일 모델 위에 보기 (새 탭)", icon=":material/public:"):
            tp = str(st.session_state.get("site3d_tiles") or _saved("site3d_tiles", "")).strip()
            tiles = tiles3d.find_tileset(tp) if tp else tiles3d.find_tileset(site)
            st.caption(f"타일 모델: `{tiles.name if tiles else '없음'}` (현장 폴더에서 가장 큰 .3tz, '3D 보기 → 현장 전체 실사 "
                       "3D'에서 바꿀 수 있음). 관측 면, 땅속으로 연장한 면(반투명), 연속 가능면 범위, 지표 교선을 올립니다. "
                       "모델을 반투명으로 시작해서 땅속이 보입니다 (CesiumJS, 인터넷 연결 필요).")
            m_alpha = st.slider("처음 표면 불투명도", 0.1, 1.0, 0.5, 0.05, key="sj_model_alpha")
            if st.button("장면 열기", icon=":material/open_in_new:", disabled=tiles is None or ground is None):
                with st.spinner("장면을 만드는 중..."):
                    scene = V.tiles_scene(site, F, C, float(factor), ground, tiles, full, float(full_span),
                                          float(m_alpha), variant)
                    path = tiles3d.write_scene(site / "현장전체_지중절리_3D장면.json", scene)
                url = _open_tiles_viewer(path)
                st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path.name}`")
        fig = _session_cache("_sj_3d", (key, factor, under), lambda: V.figure_3d(F, C, float(factor), ground, True, True, under))
        _view3d(fig, "sj_3d", 760)
    with tabs[1]:
        if ground is None:
            st.info("지표 DTM이 없어 지표 교선은 그릴 수 없습니다 (주향·경사 기호만).")
        cc = st.columns([1.6, 1.3, 1.3, 1], vertical_alignment="bottom")
        sym = cc[0].segmented_control("주향·경사 기호", ["연속 가능면", "모든 관측 면", "없음"], **_w("sj_sym", "연속 가능면"),
                                      help="기호: 긴 선 = 주향, 짧은 선 = 경사 방향(80° 이상은 생략), 숫자 = 경사.")
        dl = cc[1].toggle(f"×{factor:g} 연장 면의 지표 교선", **_w("sj_disc_lines", True),
                          help="연장한 원판이 지표(DTM)와 만나는 선: 지표에 그 절리가 드러날 것으로 예상되는 자리.")
        sym_m = cc[3].number_input("기호 크기 (m)", 2.0, 30.0, **_w("sj_sym_m", 8.0), step=1.0)
        cc[2].caption("점선(연속 가능면 끝까지의 지표 교선)은 위의 '끝까지 연장' 스위치를 따릅니다.")
        bg = _session_cache("_sj_bg", (str(orthos[0]) if orthos else "", tuple(np.round(np.r_[lo, hi], 1))),
                            lambda: LV.ortho_image(orthos[0], [lo[0], hi[0], lo[1], hi[1]], 1600) if orthos else None)
        smode = {"연속 가능면": "planes", "모든 관측 면": "faces", "없음": "none"}.get(sym or "없음", "none")
        _figure_window(_session_cache("_sj_ortho", (key, factor, smode, dl, full, full_span, sym_m, str(orthos[:1])),
                                      lambda: V.ortho_map(F, C, float(factor), ground, bg, smode, dl, full,
                                                          float(full_span), float(sym_m))),
                       "sj_ortho", 860, "새 창에서 크게 보기 — 정사영상 위 절리 (휠 확대·축소)")
        st.caption("실선 = 관측 면을 연장한 면이 지표와 만나는 선(지표에 드러날 것으로 예상되는 자리), 점선 = 연속 가능면을 끝까지 "
                   "이었을 때의 지표 교선(가정). 지표는 맨땅 DTM이라 나무 아래 땅 기준입니다. 점에 마우스를 올리면 방향·크기.")
    with tabs[2]:
        zlo, zhi = float(np.floor(F["z"].min())) - 20, float(np.ceil(F["z"].max())) + 10
        z0 = st.slider("표고 (m, 해발)", zlo, zhi, value=float(round(F["z"].median())), step=1.0, key="sj_z")
        bg = _session_cache("_sj_bg", (str(orthos[0]) if orthos else "", tuple(np.round(np.r_[lo, hi], 1))),
                            lambda: LV.ortho_image(orthos[0], [lo[0], hi[0], lo[1], hi[1]], 1600) if orthos else None)
        _figure_window(V.plan_figure(F, C, float(factor), z0, ground, bg, planes_full=full,
                                     full_min_span=float(full_span)), "sj_plan", 860,
                       "새 창에서 크게 보기 — 수평 단면 (휠 확대·축소)")
        st.caption("노랑 바탕 = 그 표고에서 땅속인 곳 (DTM 위). 선 = 절리면(연장 포함)이 그 표고의 수평면을 지나는 자리, 굵은 선 = "
                   "연속 가능면, 점선 = 연속 가능면을 끝까지 이은 가정.")
    with tabs[3]:
        line = st.session_state.get("sj_line") or st.session_state.get("ls_line")
        if not line:                                           # across the middle of the site, west to east
            cy = float((lo[1] + hi[1]) / 2)
            line = ((float(lo[0] + 60), cy), (float(hi[0] - 60), cy))
        cc = st.columns(5, vertical_alignment="bottom")
        ax_ = cc[0].number_input("A x", value=float(line[0][0]), format="%.1f", key="sj_ax")
        ay_ = cc[1].number_input("A y", value=float(line[0][1]), format="%.1f", key="sj_ay")
        bx_ = cc[2].number_input("B x", value=float(line[1][0]), format="%.1f", key="sj_bx")
        by_ = cc[3].number_input("B y", value=float(line[1][1]), format="%.1f", key="sj_by")
        if cc[4].button("라이다 단면 선 쓰기", disabled=not st.session_state.get("ls_line"), width="stretch",
                        help="'라이다 단면' 화면에서 마지막으로 그린 선"):
            st.session_state["sj_line"] = st.session_state["ls_line"]
            for k_ in ("sj_ax", "sj_ay", "sj_bx", "sj_by"):
                st.session_state.pop(k_, None)
            st.rerun()
        a, b = (ax_, ay_), (bx_, by_)
        st.session_state["sj_line"] = (a, b)
        if np.hypot(bx_ - ax_, by_ - ay_) < 5:
            st.info("선이 너무 짧습니다.")
        else:
            _figure_window(V.section_figure(F, C, float(factor), a, b, ground, planes_full=full,
                                             full_min_span=float(full_span)), "sj_section", 640,
                           "새 창에서 크게 보기 — 수직 단면 (휠 확대·축소)")
            st.caption("회색 = 땅속 (DTM 아래). 선 = 절리면(연장 포함)이 이 수직면을 지나는 자리. 좌표는 EPSG:5186 (평면도·"
                       "라이다 단면 화면과 같음).")
    with tabs[4]:
        if not len(C):
            st.info("연속 가능면이 없습니다.")
        else:
            T = C[["id", "group_label", "sections", "n", "span_m", "area_m2", "dip", "dd", "rms_m", "x", "y", "z"]].rename(
                columns={"group_label": "현장 절리군", "sections": "구간", "n": "면 수", "span_m": "길이 (m)",
                         "area_m2": "관측 면적 (m²)", "dip": "경사", "dd": "경사방향", "rms_m": "평면 오차 (m)"}).round(1)
            st.dataframe(T, hide_index=True, width="stretch")
            st.download_button("CSV 내려받기", T.to_csv(index=False).encode("utf-8-sig"), "연속가능_불연속면.csv",
                               "text/csv", icon=":material/download:")
    st.caption(f"지표: {gpath.name if gpath else '없음'} · 절리면: " + ", ".join(pick))



FIGURES = {"stereonet": lambda r, **o: report.fig_stereonet(r, r.trace_families, **o),
           "trace_angles": lambda r: report.fig_trace_angles(r, r.trace_families),
           "kinematics": lambda r: report.fig_kinematics(r)}


def _figure(result, name: str, **opts):
    """A report figure for the current result, made the first time a view shows it (each takes up to seconds);
    with options (e.g. the stereonet's density map) one per option set."""
    key = _result_key()
    cache = st.session_state.get("_figs")
    if not cache or cache[0] != key:
        cache = (key, {})
        st.session_state["_figs"] = cache
    fkey = (name, repr(sorted(opts.items())))
    if fkey not in cache[1]:
        with st.spinner("그림을 만드는 중..."):
            cache[1][fkey] = FIGURES[name](result, **opts)
    return cache[1][fkey]


def _parse_sets(text: str):
    """[(dip, dip direction), ...] from "48/047, 30/120" ([] for an empty text, None if not readable)."""
    out = []
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            d, a = (float(x) for x in part.split("/"))
        except ValueError:
            return None
        if not (0 <= d <= 90 and 0 <= a <= 360):
            return None
        out.append((d, a % 360))
    return out


def _pin_set(dip: float, dd: float):
    """Callback: add a candidate to the sets pinned in the advanced settings (used from the next analysis)."""
    cur = (st.session_state.get("adv_fixed_sets") or "").strip()
    st.session_state["adv_fixed_sets"] = (cur + ", " if cur else "") + f"{dip:.0f}/{dd:03.0f}"


def _set_candidates(result):
    """Density peaks that did not become sets, with why, and the sets pinned by the user."""
    det = result.config.get("set_detection")
    if det is None:
        st.caption("후보 절리군: 이 결과는 후보 판정 기능 이전에 만들어져 정보가 없습니다 (다시 분석하면 나옵니다).")
        return
    pinned = det.get("fixed_set_ids") or []
    if det.get("fixed_sets"):
        st.markdown("**직접 지정한 절리군**: " + ", ".join(f"{d:.0f}/{a:03.0f}" for d, a in det["fixed_sets"])
                    + (" → " + ", ".join(f"Set {i}" for i in pinned) if pinned else " (지지 점이 부족해 절리군이 되지 못함)"))
    cands = det.get("candidates") or []
    st.markdown("##### :material/help: 후보 절리군")
    if not cands:
        st.caption("기준에 조금 못 미친 뚜렷한 밀도 봉우리가 없습니다.")
        return
    st.caption("극점 밀도가 주변보다 뚜렷하지만 자동 판정 기준(전체 중앙값의 3배 이상, 20° 안에서 가장 진함)을 넘지 못한 "
               "방향입니다. 스테레오넷에 ◇와 점선으로 표시됩니다. 현장 확인 후 절리군으로 넣으면 다음 분석부터 항상 "
               "절리군으로 다룹니다 (안정성 검토 포함).")
    pinned_txt = st.session_state.get("adv_fixed_sets") or ""
    for k, c in enumerate(cands):
        cols = st.columns([1.1, 1.1, 1.0, 3.2, 1.6], vertical_alignment="center")
        cols[0].markdown(f"**{c['dip']:.0f}/{c['dip_direction']:03.0f}**")
        cols[1].markdown(f"밀도 {c['contrast']:.1f}배")
        cols[2].markdown(f"극점 {c['share'] * 100:.1f}%", help="그 방향 15° 안에 드는 극점의 비율")
        cols[3].caption(c["reason"])
        tag = f"{c['dip']:.0f}/{c['dip_direction']:03.0f}"
        cols[4].button("절리군으로 추가", key=f"pin_{k}", icon=":material/push_pin:", disabled=tag in pinned_txt,
                       on_click=_pin_set, args=(c["dip"], c["dip_direction"]), width="stretch",
                       help="고급 설정의 '직접 지정할 절리군'에 넣습니다. 다음 분석부터 적용.")
    if pinned_txt:
        st.caption(f"다음 분석에 직접 지정될 절리군: {pinned_txt} (고급 설정에서 고치거나 지울 수 있음)")


def render_stereonet(result):
    """Poles, or a smoothly contoured pole-density map (with or without the poles), set planes and the face."""
    modes = ["극점", "밀도 지도", "밀도 지도 + 극점"]
    c = st.columns([1.7, 1.4, 1.1, 1.1, 0.9], vertical_alignment="bottom")
    mode = c[0].radio("표시", modes, horizontal=True, **_w("st_mode", modes[0]),
                      help="밀도 지도: 극점이 모인 정도를 부드러운 등치선으로 (Kamb 지수 평활). 절리군의 중심과 퍼짐, "
                           "군과 군 사이가 이어지는지를 한눈에 봅니다.")
    off = mode == modes[0]
    methods = {"Kamb (지수 평활)": "exponential_kamb", "Schmidt (1% 면적)": "schmidt"}
    method = methods[c[1].radio("밀도 계산", list(methods), horizontal=True, disabled=off,
                                **_w("st_method", "Kamb (지수 평활)"),
                                help="Kamb: 균등 분포 대비 몇 σ 많은지 (평활 정도 조절). Schmidt: 1 % 면적 원 안의 극점 비율.")]
    sigma = c[2].slider("평활 σ", 1.0, 5.0, step=0.5, disabled=off or method == "schmidt", **_w("st_sigma", 3.0),
                        help="클수록 더 부드럽게 (세는 원이 커짐). 3이 일반적입니다.")
    cmap = c[3].selectbox("색", list(report.DENSITY_CMAPS), disabled=off, **_w("st_cmap", "흰색→갈색"))
    weighted = c[4].checkbox("면적 가중", disabled=off, **_w("st_weighted", False),
                             help="큰 절리면일수록 더 크게 셉니다 (넓은 면 몇 개가 지도를 좌우할 수 있음).")
    density = None if off else {"method": method, "sigma": float(sigma), "weighted": bool(weighted),
                                "poles": mode == modes[2], "cmap": report.DENSITY_CMAPS[cmap]}
    st.image(_figure(result, "stereonet", density=density))
    st.caption(("색 = 극점 밀도 (" + ("균등 분포 대비 σ" if method != "schmidt" else "1 % 면적당 %") + "), " if density else
                "점 = 절리면 극점(크기 = 면적), ")
               + "실선 = 절리군 평균면, 점선 = 트레이스로만 잡힌 절리군(추정), ◇·회색 점선 = 후보 절리군, 별 = 구간별 벽면.")
    _set_candidates(result)


def _reference_folder(result):
    """Planar folder that holds reference_overlays.json (site-specific reference images), if any."""
    from groundwater.facade import reference_overlays as ro

    for d in ((result.config.get("planar") or {}).get("folder"), st.session_state.get("planar_dir")):
        if d and Path(str(d)).is_dir() and (Path(str(d)) / ro.OVERLAY_FILE).exists():
            return str(d)
    return None


def render_reference_overlays(result, folder):
    """Reference images (e.g. the petroglyph tracing) on the planar orthomosaic / DSM, in place."""
    import plotly.graph_objects as go

    from groundwater.facade import planar
    from groundwater.facade import reference_overlays as ro

    data = ro.load(folder)
    entries = data["overlays"]
    st.caption("현장 특별 자료: 참고 이미지(실측도·사진)를 절벽면 평행 정사영상·DSM 위 제자리에 겹쳐 봅니다. 사진은 정사영상과 "
               "특징점 정합(호모그래피), 실측도는 형상 대응점으로 맞춘 박판 스플라인입니다. 위치 정확도: 사진 수 cm, 실측도 "
               "수 cm~10 cm (암면 아래쪽 사진 밖 구간은 더 큼). 타일 모델 뷰어(그라우팅 입력 탭)에서도 같은 이미지가 표면에 붙습니다.")
    c = st.columns([2, 1, 1])
    labels = [e["name"] for e in entries]
    pick = c[0].multiselect("겹칠 이미지", labels, default=[labels[0]], key="ref_pick")
    bg = c[1].radio("바탕", ["정사영상", "DSM 음영기복", "DSM 미세기복"], key="ref_bg",
                    help="미세기복 = DSM에서 3 cm보다 넓은 굴곡을 뺀 것. 균열·턱은 보이지만, 몇 mm 깊이로 쪼아 새긴 암각화 선은 "
                         "드론 DSM(2.83 mm 격자)에서 거의 드러나지 않습니다 — 위치는 겹친 이미지로 확인하세요.")
    alpha = c[2].slider("이미지 불투명도", 0.0, 1.0, 0.75, 0.05, key="ref_alpha")
    res = c[2].select_slider("해상도 (mm)", [3, 5, 8], value=5, key="ref_res") / 1000
    chosen = [e for e in entries if e["name"] in pick] or entries[:1]
    bounds = ro.union_bounds(folder, chosen, 0.4)
    kind = {"정사영상": "ortho", "DSM 음영기복": "hillshade", "DSM 미세기복": "relief"}[bg]
    key = ("_ref", folder, tuple(e["name"] for e in chosen), kind, res)
    cache = st.session_state.setdefault("_ref_cache", {})
    if key not in cache:
        with st.spinner("이미지 맞추는 중..."):
            cache.clear()
            base = ro.background(folder, bounds, res, kind)
            cache[key] = (base, [ro.render(folder, e, bounds, res) for e in chosen])
    base, ovs = cache[key]
    img = ro.composite(base, ovs, alpha)
    # axes in chainage / elevation (the plane is parallel to the wall: corners mapped, linear in between)
    x0, y0, x1, y1 = bounds
    u_ax = v_ax = None
    reg = planar.load_registration(folder, _reg_crs(result))
    if reg is not None:
        T = np.array(reg["matrix"])
        # the plane's z is the DSM value (metres along its normal): the corners at the wall's own depth
        dz = ro._read_window(planar.find_planar_pair(folder)[0], bounds, 0.05, [1])[0].astype(float)
        dz = dz[np.isfinite(dz) & (dz > -1000) & (dz != 0)]
        z = float(np.median(dz)) if len(dz) else 0.0
        corners = np.array([[x0, y0, z], [x1, y0, z], [x0, y1, z]])
        uvw = result.wall_frame.to_wall(corners @ T[:3, :3].T + T[:3, 3])
        u_ax, v_ax = (uvw[0, 0], uvw[1, 0]), (uvw[0, 1], uvw[2, 1])
    # sent as a JPEG (a raw array is ~7 MB of chart data); row 0 is the top, so flipped and anchored at the bottom
    import base64

    import cv2

    jpg = cv2.imencode(".jpg", np.ascontiguousarray(img[::-1, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
    fig = go.Figure(go.Image(source="data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(),
                             x0=u_ax[0] if u_ax else x0,
                             dx=((u_ax[1] - u_ax[0]) if u_ax else (x1 - x0)) / img.shape[1],
                             y0=v_ax[0] if v_ax else y0, dy=((v_ax[1] - v_ax[0]) if v_ax else (y1 - y0)) / img.shape[0],
                             hoverinfo="skip"))
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="측점 (m)" if u_ax else "평면 x (m)"),
                      yaxis=dict(title="표고 (m)" if v_ax else "평면 y (m)", autorange=True, scaleanchor="x"))
    _map_chart(fig, "overlay")
    notes = [f"- **{e['name']}** ({e['file']}): " + ("특징점 정합 " + str(e.get("inliers")) + "점" if e["method"] == "homography"
                                                   else f"형상 대응점 {len(e['points'])}개") + (f" — {e['note']}" if e.get("note") else "")
             for e in chosen]
    st.markdown("\n".join(notes))
    if st.button("현재 그림을 결과 폴더에 저장 (PNG)", key="ref_save"):
        import cv2

        out = Path(st.session_state.get("output_dir") or folder) / f"암각화_{kind}.png"
        cv2.imencode(".png", img[:, :, ::-1])[1].tofile(str(out))
        st.success(f"저장했습니다: {out}")


def render_planar_redetect(result):
    from groundwater.facade import planar
    from groundwater.pipeline import apply_planar

    pl = result.config.get("planar")
    if pl:
        how = ("Metashape 평면 정의로 배치" if pl.get("registration") == "metashape"
               else f"점군 자동 정합, 잔차 {pl.get('registration_median_m', 0) * 100:.1f} cm")
        st.caption(f"트레이스 출처: 절벽면 평행 자료 {pl.get('ortho')} ({pl.get('resolution_m', 0.02) * 100:.0f} cm 격자, "
                   f"{how}, 입면 영상 {pl.get('facade_recoloured', 0) * 100:.0f}% 재채색)")
        return
    with st.expander("고해상도 절벽면 평행 자료로 트레이스 다시 검출"):
        st.caption("Metashape 평면 투영 정사영상·DEM이 있으면 그 자료(원본 사진 해상도)로 입면 영상을 다시 칠하고 트레이스를 "
                   "다시 찾습니다. 3차원 절리 분석은 그대로 두므로 전체 재분석 없이 1분 안팎이면 됩니다.")
        pdir = _planar_input("폴더", "planar_redetect_dir")
        if pdir.strip() and st.button("다시 검출", type="primary"):
            folder = pdir.strip().strip('"')
            dem, ortho = planar.find_planar_pair(folder)
            if dem is None or ortho is None:
                st.error("그 폴더에서 평면 투영 DEM·정사영상을 찾지 못했습니다.")
                return
            with st.spinner("정합·재검출 중 (처음이면 정합에 2~3분)..."):
                apply_planar(result, folder, 0.02)
                save_results(result, st.session_state["output_dir"])
            st.session_state["planar_dir"] = folder
            st.session_state["result_key"] = ("redetect", folder, __import__("time").time())
            st.rerun()


def _seep_defaults():
    for k, v in hydro.WEIGHTS.items():
        st.session_state[f"seep_w_{k}"] = v


def render_thermal(result):
    """RGB and thermal wall orthomosaics (Metashape planar exports) registered and overlaid with an opacity slider."""
    from groundwater.facade import planar
    from groundwater.facade import thermal as TH
    from groundwater.viz import thermal_view as TV

    st.caption("벽면 평행으로 내보낸 RGB·열화상 정사영상을 같은 벽면 좌표(측점 × 표고)에 겹쳐 봅니다. 열화상 폴더에 온도 "
               "정사영상(*temperature*.tif, °C — scripts/dji_rjpeg_to_temperature.py → metashape_swap_to_temperature.py)이 "
               "있으면 온도로, 없으면 팔레트 색(상대값)으로 봅니다.")
    site = st.session_state.get("site_folder") or str(Path(result.config["paths"].get("point_cloud") or ".").parent)
    auto = _session_cache("_th_folders", str(site), lambda: TH.find_folders(site))
    c = st.columns(2)
    for key, kind in (("th_rgb_dir", "rgb"), ("th_dir", "thermal")):
        if not st.session_state.get(key, _saved(key, "")) and auto.get(kind):
            st.session_state[key] = str(auto[kind])
    rgb_dir = c[0].text_input("RGB 평면 자료 폴더 (벽면 평행 DEM·정사영상 + 평면 투영 JSON)", **_w("th_rgb_dir", ""))
    th_dir = c[1].text_input("열화상 평면 자료 폴더 (DEM·정사영상)", **_w("th_dir", ""))
    rgb_dir, th_dir = rgb_dir.strip().strip('"'), th_dir.strip().strip('"')
    pr_r = planar.find_planar_pair(rgb_dir) if rgb_dir else (None, None)
    pr_t = planar.find_planar_pair(th_dir) if th_dir else (None, None)
    if None in pr_r or None in pr_t:
        st.info("두 폴더 모두에 평면(벽면 평행) DEM과 정사영상 GeoTIFF가 있어야 합니다. 현장 폴더 아래에서 자동으로 찾지 못하면 "
                "직접 입력하세요.")
        return
    crs = _reg_crs(result) or "EPSG:5186"
    reg_r = planar.load_registration(rgb_dir, crs)
    if reg_r is None:
        st.warning("RGB 평면 자료의 위치를 알 수 없습니다. Metashape에서 scripts/metashape_export_planar_projection.py로 만든 "
                   "*_planar_projection.json을 그 폴더에 넣으세요.")
        return
    reg_t = TH.load(th_dir, crs)
    cc = st.columns([3, 1], vertical_alignment="bottom")
    if reg_t is None:
        cc[0].warning("열화상 자료가 아직 RGB에 정합되지 않았습니다 (열화상 모델은 따로 정렬되어 10~30 cm 어긋남).")
    else:
        lc = reg_t.get("local_correction") or {}
        shift = np.abs(np.array(lc.get("t") or [[0, 0, 0]])).max() * 100
        cc[0].caption(f"정합: RGB DEM 기준 강체 ICP + 벽을 따라 국부 보정 ({len(lc.get('knots_xy') or [])}구간, 최대 {shift:.0f} cm). "
                      f"열화상 표면과 RGB 표면의 거리 중앙값 {reg_t.get('fit_median_m', float('nan')) * 100:.0f} cm "
                      f"(75 % {reg_t.get('fit_p75_m', float('nan')) * 100:.0f} cm, 나무·망 포함).")
    if cc[1].button("열화상 정합 다시 하기" if reg_t else "열화상 정합 실행 (2~3분)", icon=":material/join_inner:",
                    width="stretch", key="th_register"):
        with st.status("열화상 DEM을 RGB DEM에 정합하는 중...", expanded=True) as box:
            reg_t = TH.register(rgb_dir, th_dir, result.wall_frame, crs, progress=box.write)
            box.update(label="정합 완료", state="complete")
        st.session_state.pop("_th_imgs", None)
    if reg_t is None:
        return

    u_lo, u_hi = (float(np.floor(x)) for x in _chainage_range(result))
    key = _range_state("th_range", u_lo, u_hi + 1, (u_lo, u_hi + 1))
    c = st.columns([3, 1], vertical_alignment="bottom")
    rng = c[0].slider("측점 구간 (m) — 좁힐수록 해상도가 높아짐", u_lo, u_hi + 1, step=1.0, key=key)
    op = c[1].slider("열화상 불투명도", 0.0, 1.0, step=0.05, **_w("th_op", 0.5))
    stamp = (Path(th_dir) / planar.REG_FILE).stat().st_mtime
    with st.spinner("두 정사영상을 같은 벽면 좌표로 옮기는 중 (구간마다 처음 한 번, 벽 전체 1분 안팎)..."):
        pair = _session_cache("_th_imgs", (_result_key(), rgb_dir, th_dir, tuple(rng), stamp),
                              lambda: TV.pair_images(result, rgb_dir, th_dir, crs, tuple(rng), max_px=8000))
    title = f"RGB · 열화상 — 측점 {rng[0]:g}–{rng[1]:g} m"
    url = _session_cache("_pop_thermal", (_result_key(), rgb_dir, th_dir, tuple(rng), stamp, op),
                         lambda: _view_server().publish("thermal_overlay", TV.overlay_page(pair, title, op)))
    _popup_button(url, "thermal_overlay", "새 창에서 크게 보기 — 불투명도 조절·좌우 비교, 휠 확대·축소, 드래그 이동")
    img = TV.blend(pair["rgb"], pair["thermal"], op)
    k = max(1, int(np.ceil(max(img.shape[:2]) / 2400)))
    st.image(img[::k, ::k], width="stretch")
    tr = pair.get("t_range")
    st.caption(f"화소 {pair['res'] * 100:.1f} cm (열화상 원래 해상도 약 3.3 cm). "
               + (f"열화상 = 온도 {tr[0]:.1f}~{tr[1]:.1f} °C (inferno: 어두움 = 차가움, 밝음 = 따뜻함). " if tr else
                  "열화상 = 카메라 팔레트 색 (상대값). ")
               + "새 창에서는 불투명도 슬라이더로 바로 바꾸고, '좌우 비교'로 경계선 왼쪽은 RGB, 오른쪽은 열화상으로 볼 수 있습니다.")
    if st.toggle(":material/view_in_ar: 타일 모델에 열화상 표시", **_w("th_tiles", False)):
        from groundwater.viz import tiles3d

        st.caption("위 측점 구간의 열화상(온도 정사영상이 있으면 °C)을 실사 타일 모델 표면에 붙여 새 탭에서 엽니다. 열화상 TIFF로 "
                   "만든 타일 모델(IcSan_thermal_tiled_relative_inferno.3tz)을 고르면 그 자체 색은 상대 온도, 붙인 이미지는 °C "
                   "입니다 (그 화면의 '표면에 붙인 이미지'에서 켜고 끔. 인터넷 연결 필요: CesiumJS).")
        tiles = _tiles_path_input(st.session_state.get("mesh_path", ""), st.session_state.get("planar_dir", ""), "_th")
        out = st.session_state.get("output_dir")
        if st.button("타일 모델 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not out,
                     key="th_tiles_open"):
            with st.spinner("장면을 만드는 중..."):
                scene = tiles3d.scene_json(result, [], _flow_sets(result), tiles, None, None,
                                           title="열화상 — 타일 모델", joint_shade="none",
                                           overlay_folder=_reference_folder(result))
                ov = TV.tiles_overlay(result, pair)
                scene["overlays"] = ([ov] if ov else []) + scene["overlays"]
                path = tiles3d.write_scene(Path(out) / "thermal_scene_tiles.json", scene)
            url = _open_tiles_viewer(path)
            st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}`")
    _thermal_anomalies(result, th_dir, crs, stamp)


def _thermal_anomalies(result, th_dir, crs, stamp):
    """Thermal anomalies (relative to their surroundings, on bare rock) and how they line up with the joints."""
    from groundwater import hydro
    from groundwater import thermal_anomaly as TA
    from groundwater.viz import thermal_view as TV

    st.markdown("##### :material/thermostat: 열화상 이상 부분 × 절리")
    st.caption("온도(°C, 없으면 팔레트 상대 지수)를 맨 암반(식생·낙석방지망·평탄 지면 제외)에서 주변 몇 m의 평균·편차와 비교해 "
               "뚜렷이 차갑거나 따뜻한 부분을 뽑습니다. 지하수(연중 약 13~16 °C)는 저녁·밤·겨울에는 암반보다 따뜻하게, 햇빛 받는 "
               "여름 낮에는 차갑게 보입니다 (익산 10/21 열화상은 16:55~17:39 저녁 촬영 → 따뜻한 이상이 유출 후보). 그늘·벌어진 "
               "틈도 차갑게 보이므로 사진에서 훨씬 어두운 것은 '그늘 의심'으로 표시합니다.")
    c = st.columns(4)
    win = c[0].select_slider("비교 범위 (m)", [3.0, 5.0, 8.0], **_w("ta_win", 5.0),
                             help="이 크기 안의 평균·편차와 비교 (사진마다 조금씩 바뀐 색 범위를 상쇄)")
    zmin = c[1].slider("기준 (표준편차 배)", 1.5, 3.0, step=0.25, **_w("ta_z", 2.0))
    amin = c[2].select_slider("최소 면적 (m²)", [0.1, 0.25, 0.5, 1.0], **_w("ta_area", 0.25))
    near = c[3].select_slider("절리 근접 거리 (m)", [0.2, 0.3, 0.5], **_w("ta_near", 0.3))
    seep = st.session_state.get("_seep")                    # the seepage map of the 유출 가능성 view, if computed
    smap = seep[0] if seep else None
    if smap is None:
        smap = _session_cache("_ta_smap", _result_key(), lambda: hydro.seepage_potential(result))
    with st.spinner("열화상 이상 부분을 찾는 중 (30초 안팎)..."):
        an = _session_cache("_ta", (_result_key(), th_dir, stamp, win, zmin, amin, near),
                            lambda: TA.detect(result, th_dir, crs, win, zmin, amin, near, smap=smap))
    sm = an.summary
    cold, warm = sm.get("차가움") or {}, sm.get("따뜻함") or {}
    m = st.columns(4)
    m[0].metric("차가운 이상", f"{cold.get('n', 0)}곳", f"{cold.get('area_m2', 0)} m²", delta_color="off")
    m[1].metric("따뜻한 이상", f"{warm.get('n', 0)}곳", f"{warm.get('area_m2', 0)} m²", delta_color="off")
    for col, kind, d in ((m[2], "차가운", cold), (m[3], "따뜻한", warm)):
        col.metric(f"{kind} 이상 중 절리 {near:g} m 이내", f"{d.get('near_share', 0) * 100:.0f}%",
                   f"암면 전체 {sm['near_share_rock'] * 100:.0f}% 대비 {d.get('enrichment', 0):.1f}배"
                   + (f" (그늘 의심 제외 {d.get('enrichment_not_shadow', 0):.1f}배)" if kind == "차가운" else ""),
                   delta_color="off", help="1배보다 크면 이상 부분이 우연보다 절리 가까이에 몰려 있다는 뜻")
    bg, ext = _session_cache("_seep_facade", _result_key(), lambda: report.facade_preview(result, 6000))
    _map_chart(TV.anomaly_figure(result, an, bg, ext), "thermal_anom")
    st.caption(f"분석한 맨 암반 {sm['rock_m2']:,.0f} m². 파랑 = 차가운 이상, 빨강 = 따뜻한 이상, 노란 선 = 절리 트레이스. "
               "새 창에서 이상 부분에 마우스를 올리면 표의 값이 나옵니다.")
    if len(an.table):
        tbl = an.table.drop(columns=["u", "v"]).sort_values(["종류", "면적 (m²)"], ascending=[True, False])
        st.dataframe(tbl, hide_index=True, width="stretch")
        out = st.session_state.get("output_dir")
        b = st.columns([1, 1, 3])
        b[0].download_button("CSV 내려받기", tbl.to_csv(index=False).encode("utf-8-sig"), "thermal_anomalies.csv",
                             "text/csv", icon=":material/download:", width="stretch")
        if out and b[1].button("결과 폴더에 저장", icon=":material/save:", width="stretch", key="ta_save"):
            path = Path(out) / "thermal_anomalies.csv"
            tbl.to_csv(path, index=False, encoding="utf-8-sig")
            b[2].success(f"저장: {path}")


def render_seepage(result):
    st.caption("표면 절리망으로 본 지하수 유출 **가능성** 순위입니다. 격자(기본 1 m)마다 네 지표를 0~1로 바꿔 가중 합산하고, "
               "암면 격자 중 상위 15 %를 '상', 다음 35 %를 '중'으로 표시합니다. 실제 유출은 열화상·관측으로 확인해야 합니다.")
    with st.expander("지표와 가중치"):
        st.markdown("- **절리 밀도 P21**: 격자 면적당 트레이스 길이 (m/m²)\n"
                    "- **열린 정도**: 입면 깊이에서 홈의 평균 깊이 (표면 개구의 상한)\n"
                    "- **연결성**: 서로 닿는 트레이스로 이어진 절리망의 길이 — 길수록 안쪽까지 연결될 가능성\n"
                    "- **절리 옆 식생**: 트레이스 양옆 10 cm 띠 안의 초록 화소 비율 — 절리에 뿌리내린 식물은 물이 나오는 자리")
        if sum(float(st.session_state.get(f"seep_w_{k}", _saved(f"seep_w_{k}", v))) for k, v in hydro.WEIGHTS.items()) <= 0:
            _seep_defaults()                   # an all-zero set (saved earlier) would lock the tab
        c = st.columns(4)
        labels = {"p21": "절리 밀도", "openness": "열린 정도", "connectivity": "연결성", "wetness": "절리 옆 식생"}
        w = {k: c[i].slider(lab, 0.0, 1.0, step=0.05, **_w(f"seep_w_{k}", hydro.WEIGHTS[k]))
             for i, (k, lab) in enumerate(labels.items())}
        cell = st.select_slider("격자 크기 (m)", [0.5, 1.0, 2.0], **_w("seep_cell", 1.0))
        st.button("기본 가중치로", on_click=_seep_defaults, key="seep_reset")
    if sum(w.values()) <= 0:
        st.warning("가중치를 하나 이상 0보다 크게 두세요.")
        return
    key = ("_seep", _result_key(), tuple(w.values()), cell)
    if st.session_state.get("_seep_key") != key:
        with st.spinner("유출 가능성 계산 중..."):
            smap = hydro.seepage_potential(result, cell_m=cell, weights=w)
            st.session_state["_seep"] = (smap, None)
            st.session_state["_seep_key"] = key
    smap, _ = st.session_state["_seep"]
    if smap is None:
        st.info("색 정보가 없어 계산할 수 없습니다.")
        return
    _seepage_map_view(result, smap)
    st.subheader(f"유출 가능성 '상' 구역 ({len(smap.zones)}곳, 점수 순)")
    st.dataframe(smap.zones, width="stretch")
    st.caption("openness_cm = 홈 평균 깊이, network_m = 연결된 절리망 길이, plants_along_joints = 절리 옆 식생 비율, "
               "x·y·z = 구역 중심의 지도 좌표 (현장 좌표계).")
    st.download_button("구역 목록 CSV", smap.zones.to_csv(index=False).encode("utf-8-sig"), file_name="seepage_zones.csv")


def _seepage_map_view(result, smap):
    """The seepage map, zoomable (popup): background = the point-cloud facade image or the wall-parallel
    orthomosaic itself; layer = the grade, the score or one indicator; and the same on the tiled model."""
    from groundwater.facade import planar
    from groundwater.viz import seepage_view as SV

    folder = (result.config.get("planar") or {}).get("folder") or st.session_state.get("planar_dir") or ""
    folder = str(folder).strip().strip('"')
    dem, ortho = planar.find_planar_pair(folder) if folder and Path(folder).is_dir() else (None, None)
    reg = planar.load_registration(folder, _reg_crs(result)) if ortho is not None else None
    bgs = ["입면 영상 (점군)"] + (["고해상도 정사영상 (벽면 평행)"] if reg is not None else [])
    if st.session_state.get("seep_bg", _saved("seep_bg", bgs[0])) not in bgs:
        st.session_state["seep_bg"] = bgs[0]
    c = st.columns([1.5, 3.2, 1.1], vertical_alignment="bottom")
    bg = c[0].radio("바탕 영상", bgs, **_w("seep_bg", bgs[0]),
                    help="고해상도 정사영상: 올린 벽면 평행 정사영상(2~3 mm)을 벽면 좌표에 그대로 옮겨 그립니다. "
                         "벽 전체는 약 6.5 mm, 측점 구간을 좁히면 원래 해상도까지.")
    layer = c[1].radio("표시할 값", list(SV.LAYERS), format_func=lambda k: SV.LAYERS[k][0], horizontal=True,
                       **_w("seep_layer", "grade"),
                       help="등급 외에는 셀마다 그 값을 색으로 (암면 셀만, 2~98 백분위로 색 범위). 새 창에서 셀 위에 "
                            "마우스를 올리면 네 지표 값이 모두 나옵니다.")
    op = c[2].slider("덮개 불투명도", 0.2, 1.0, step=0.05, **_w("seep_op", 0.6))
    if bg.startswith("고해상도"):
        lo, hi = (float(np.floor(x)) for x in _chainage_range(result))
        key = _range_state("seep_range", lo, hi + 1, (lo, hi + 1))
        rng = st.slider("정사영상 측점 구간 (m) — 좁힐수록 정사영상 해상도가 높아짐", lo, hi + 1, step=0.5, key=key)
        with st.spinner("정사영상을 벽면 좌표로 옮기는 중 (구간마다 처음 한 번, 벽 전체 15초 안팎)..."):
            img, ext, res = _session_cache("_seep_ortho", (_result_key(), folder, tuple(rng)),
                                           lambda: planar.wall_image(result, ortho, reg["matrix"], rng, max_px=8000))
        note = f"바탕: 벽면 평행 정사영상, 화소 {res * 1000:.1f} mm"
    else:
        img, ext = _session_cache("_seep_facade", _result_key(), lambda: report.facade_preview(result, 6000))
        note = "바탕: 점군에서 만든 입면 영상"
    fig = SV.map_figure(smap, layer, img, ext, op)
    _map_chart(fig, "seep_map")
    st.caption(f"{note}. 흰 사각형 = 점수 상위 10개 '상' 구역 (번호 = 점수 순). 지표는 셀마다의 값입니다.")

    if st.toggle(":material/view_in_ar: 타일 모델에 표시", **_w("seep_tiles", False)):
        from groundwater.viz import tiles3d

        st.caption("등급과 네 지표, 종합 점수를 각각 이미지로 실사 타일 모델 표면에 붙여 새 탭에서 엽니다. 위에서 고른 값이 "
                   "먼저 켜지고, 그 화면의 '표면에 붙인 이미지'에서 바꿔 볼 수 있습니다 (인터넷 연결 필요: CesiumJS).")
        tiles = _tiles_path_input(st.session_state.get("mesh_path", ""), st.session_state.get("planar_dir", ""), "_seep")
        out = st.session_state.get("output_dir")
        if st.button("타일 모델 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not out,
                     key="seep_tiles_open"):
            with st.spinner("장면을 만드는 중..."):
                scene = tiles3d.scene_json(result, [], _flow_sets(result), tiles, None, None,
                                           title="유출 가능성 — 타일 모델", joint_shade="none", smap=smap,
                                           overlay_folder=_reference_folder(result))
                scene["overlays"] = SV.tiles_overlays(result, smap, layer) + scene["overlays"]
                path = tiles3d.write_scene(Path(out) / "seepage_scene_tiles.json", scene)
            url = _open_tiles_viewer(path)
            st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}`")


def render_grouting(result):
    st.caption("드론 절리망으로 만든 **그라우팅 설계 초기 입력값**입니다. 개구폭을 모르는 동안은 모든 절리군에 같은 개구폭을 "
               "가정하므로 투수 '방향'과 '비율'만 의미가 있습니다. 간격·연장성·개구폭은 아래 표에서 고칠 수 있고, "
               "시추·루전 시험 결과가 나오면 그 값으로 바꾸세요.")
    base = _flow_sets(result)
    if not len(base):
        st.info("절리군이 없습니다.")
        return
    base.insert(0, "use", True)
    edited = st.data_editor(
        base, width="stretch", hide_index=True, key=f"gr_sets_{abs(hash(str(_result_key())))}",
        disabled=["name", "dip", "dip_direction", "source", "note"],
        column_config={"use": st.column_config.CheckboxColumn("포함"),
                       "name": st.column_config.TextColumn("절리군"),
                       "source": st.column_config.TextColumn("출처"),
                       "note": st.column_config.TextColumn("비고"),
                       "spacing_m": st.column_config.NumberColumn("간격 (m)", min_value=0.05, step=0.1, format="%.2f"),
                       "persistence_m": st.column_config.NumberColumn("연장성 (m)", min_value=0.1, step=0.1, format="%.2f"),
                       "aperture_rel": st.column_config.NumberColumn("상대 개구폭", min_value=0.0, step=0.1, format="%.2f"),
                       "dip": st.column_config.NumberColumn("경사", format="%.0f"),
                       "dip_direction": st.column_config.NumberColumn("경사방향", format="%.0f")})
    sets = edited[edited["use"]].drop(columns=["use"])
    if not len(sets):
        st.warning("절리군을 하나 이상 포함하세요.")
        return
    seg = result.wall_frame.vertices
    d = seg[-1] - seg[0]
    wall_az = float(np.degrees(np.arctan2(d[0], d[1])) % 360)
    mode = st.radio("차수 커튼", ["없음", f"벽면 기준선과 평행 (가상, 방위 {wall_az:.0f}°)", "방위각 직접 입력"], horizontal=True)
    az = None
    if mode.startswith("벽면"):
        az = wall_az
    elif mode.startswith("방위각"):
        st.session_state.setdefault("curtain_az", float(wall_az))
        az = st.number_input("커튼 방위각 (°, 평면에서 커튼선 방향)", 0.0, 360.0, step=1.0, key="curtain_az")

    _, principal, share = hydro.snow_tensor(sets)
    sets = sets.assign(flow_share=np.round(share, 3))
    grid, best, in_plane, vertical = hydro.hole_orientations(sets, hydro.crossing_weights(sets), az)
    g = {"sets": sets, "principal": principal, "grid": grid, "best": best, "in_plane": in_plane, "vertical": vertical}

    c = st.columns(3)
    c[0].metric("최적 공 방향 (방위/경사)", f"{best['trend']:.0f}° / {best['plunge']:.0f}°", f"{best['joints_per_m'] * 10:.1f} 개/10 m")
    c[1].metric("연직공", "— / 90°", f"{vertical['joints_per_m'] * 10:.1f} 개/10 m", delta_color="off")
    if in_plane:
        c[2].metric("커튼 면 안 최적", f"{in_plane['trend']:.0f}° / {in_plane['plunge']:.0f}°",
                    f"{in_plane['joints_per_m'] * 10:.1f} 개/10 m")
    st.caption("'개/10 m' = 공 10 m당 만나는 절리 수(투수 기여도로 가중). 연직공보다 크게 높으면 경사공을 검토할 근거입니다.")
    left, right = st.columns([1, 1])
    with left:
        st.image(report.fig_hole_orientation(g))
    with right:
        st.markdown("**투수 주축** (1 = 물이 가장 잘 흐르는 방향)")
        st.dataframe(principal, hide_index=True, width="stretch")
        st.markdown("**절리군별 투수 기여도**")
        st.dataframe(sets[["name", "flow_share"]], hide_index=True, width="stretch")
        st.markdown("**최적 방향 공 / 연직공이 만나는 절리 (10 m당)**")
        a = pd.DataFrame(hydro.per_set_frequency(sets, best["trend"], best["plunge"]))
        b = pd.DataFrame(hydro.per_set_frequency(sets, 0, 90))
        st.dataframe(a.merge(b, on="set", suffixes=("_최적", "_연직")), hide_index=True, width="stretch")
    if az is not None:
        st.subheader("차수 커튼 검토")
        cc = hydro.curtain_check(sets, az)
        st.dataframe(cc, hide_index=True, width="stretch")
        conductive = sets[sets["flow_share"] >= 0.1]
        ext = float(np.percentile(conductive["persistence_m"], 90)) if len(conductive) else None
        st.caption("cut_at_Sm = 연직공을 간격 S로 한 줄 뚫었을 때 그 절리군의 절리 하나가 공에 걸릴 확률 "
                   "(커튼 면 위 트레이스 중 공을 가로지르는 길이 / S). "
                   + (f"커튼 끝 연장 권장: 전도성 절리군 연장성 p90 = {ext:.1f} m 이상 (드론 연장성은 하한값)." if ext else ""))
    render_grouting_3d(result, sets, best, in_plane, az)
    if any(sets["note"].str.contains("벽면과 평행")):
        st.warning("벽면과 평행한 절리군의 간격은 벽면이 계단식으로 물러나는 간격일 수 있어 투수 기여도를 과대평가할 수 있습니다. "
                   "현장 판단으로 간격을 고치거나 '포함'을 끄고 비교해 보세요.")


@_fragment
def render_grouting_3d(result, sets, best, in_plane, curtain_az):
    st.subheader("3차원 가상 시공")
    layouts = ["절리망 기반 표적 천공 (권장)", "얕은 균열 주입 (표면 가까운 균열 보수)", "균등 배치 (차수 커튼·격자)"]
    layout = st.radio("배치 방식", layouts, horizontal=True, **_w("g3d_layout2", layouts[0]),
                      help="절리망 기반: 표면 절리를 암반 속으로 연장한 절리면과 그 교차를 보고, 유출 가능성 높은 곳에서 "
                           "가장 많은 절리를 관통하도록 공 위치·방향·경사를 정합니다. "
                           "얕은 균열 주입: 트레이스 바로 뒤 1 m 안팎의 절리를 맞히는 짧은 공 (표면 균열 보수용). "
                           "균등: 위 통계로 정한 방향의 공을 일정 간격으로 뚫는 일반 차수 커튼.")
    if layout.startswith("절리망"):
        targeted = _network_controls(result)
    elif layout.startswith("얕은"):
        targeted = _targeted_controls(result)
    else:
        targeted = None
        st.caption("위 계산의 공 방향으로 그라우팅 공을 3차원 모델에 가상으로 배치합니다. 공을 따라 놓인 원판은 각 절리군의 "
                   "간격으로 계산한 **예상 관통 위치**(실측 절리가 아님)이며, 색은 절리군입니다.")
    if targeted is not None:
        trend = plunge = spacing = length = offset = 0.0
        rows, mode = None, "표적"
    else:
        trend, plunge, spacing, length, offset, rows, mode = _uniform_controls(st.columns(4), best, in_plane)
    _grouting_surface_and_draw(result, sets, targeted, trend, plunge, spacing, length, offset, rows, mode)


def _network_controls(result):
    st.caption("표면에서 본 절리를 **암반 속으로 연장한 절리면**(아래 '벽면 안쪽 절리면'의 연장 깊이 설정)을 기준으로 공을 설계합니다. "
               "유출 가능성 높은 칸마다 공 입구 후보를 두고, 방향(벽면 직각에서 ±60°)과 경사(10~60°)를 바꿔 가며 "
               "암반 안에 머무는 공 중 **아직 관통되지 않은 절리를 가장 많이 지나는 공**을 하나씩 고릅니다. "
               "여러 절리군을 함께 지나는 공(절리 교차 = 물길)에 가산점을 주고, 연장 범위 너머는 절리군 간격으로 기대값을 더합니다.")
    c = st.columns(4)
    cls = c[0].radio("공 입구 후보 칸", ["'상'만", "'상'+'중'"], **_w("nw_cls", "'상'만"))
    c[0].button("기본값으로", on_click=_reset_network, icon=":material/restart_alt:", key="nw_reset",
                help="공 길이 8 m, 최대 30개, 입구 간격 2 m, 통계 가중 0.2, 절리 연장 = 트레이스 길이 ×1 (최대 5 m)")
    length = c[1].slider("공 길이 (m)", 3.0, 20.0, step=1.0, **_w("nw_length", 8.0))
    max_holes = c[1].slider("최대 공 수", 5, 100, step=5, **_w("nw_max", 30))
    gap = c[2].slider("공 입구 최소 간격 (m)", 1.0, 5.0, step=0.5, **_w("nw_gap", 2.0))
    stat = c[2].slider("통계 절리 가중", 0.0, 1.0, step=0.1, **_w("nw_stat", 0.2),
                       help="관측 연장 범위 너머의 절리 (절리군 간격으로 계산한 기대값)에 주는 가중치")
    smap, how = _seepage_map(result)
    c[3].caption(how)
    if smap is None:
        st.info("유출 가능성 지도를 만들 수 없어 (색 정보 없음) 표적 배치를 할 수 없습니다.")
        return None
    return {"kind": "network", "smap": smap, "classes": (2,) if cls == "'상'만" else (2, 1), "length": length,
            "max_holes": max_holes, "min_gap": gap, "stat_w": stat}


def _targeted_controls(result):
    st.caption("유출 가능성 지도에서 점수가 높은 칸을 지나는 **매핑된 절리(트레이스)**에만 공을 둡니다. 공마다 트레이스 옆 벽면에서 "
               "비스듬히 뚫어 벽면 뒤 일정 깊이에서 그 절리면을 맞히고 (절리군 방향 사용), 공이 만나는 위치를 원판으로 그립니다. "
               "평탄한 절리는 위에서 아래로, 급경사 절리는 좌우 번갈아 뚫으며 모든 공은 아래로 기울입니다 (주입재가 흘러나오지 않게).")
    c = st.columns(4)
    cls = c[0].radio("대상 칸", ["'상'만", "'상'+'중'"], **_w("tg_cls", "'상'만"))
    min_len = c[0].slider("최소 트레이스 길이 (m)", 0.5, 5.0, step=0.5, **_w("tg_minlen", 1.0))
    spacing = c[1].slider("트레이스 따라 공 간격 (m)", 0.5, 3.0, step=0.25, **_w("tg_spacing", 1.0),
                          help="균열 주입의 관례: 절리를 맞히는 깊이와 비슷하게")
    depth = c[1].slider("절리를 맞히는 깊이 (m, 절리면 따라)", 0.3, 5.0, step=0.1, **_w("tg_depth", 1.0))
    angle = c[2].slider("벽면과 공의 각도 (°)", 20, 70, step=5, **_w("tg_angle", 45))
    extra = c[2].slider("절리 너머 연장 (m)", 0.0, 2.0, step=0.1, **_w("tg_extra", 0.5))
    smap, how = _seepage_map(result)
    c[3].caption(how)
    if smap is None:
        st.info("유출 가능성 지도를 만들 수 없어 (색 정보 없음) 표적 배치를 할 수 없습니다.")
        return None
    return {"kind": "crack", "smap": smap, "classes": (2,) if cls == "'상'만" else (2, 1), "min_trace_m": min_len,
            "spacing": spacing, "joint_depth": depth, "angle_deg": angle, "extra_m": extra}


def _seepage_map(result):
    """The seepage map of the seepage tab (its weights and cells), else one with the default weights."""
    sk = st.session_state.get("_seep_key")
    if "_seep" in st.session_state and sk and sk[1] == _result_key():
        return st.session_state["_seep"][0], "유출 가능성: '유출 가능성' 화면의 가중치·격자 사용"
    key = ("_seep_default", _result_key())
    if st.session_state.get("_seep_default_key") != key:
        st.session_state["_seep_default"] = hydro.seepage_potential(result)
        st.session_state["_seep_default_key"] = key
    return st.session_state["_seep_default"], "유출 가능성: 기본 가중치"


def _chainage_range(result):
    d = result.facade_depth
    u0 = float(d.transform.c)
    return u0, u0 + d.array.shape[2] * abs(d.transform.a)


def _joint_controls(result):
    """Options for the joint planes inside the rock and the cut-away."""
    with st.expander("벽면 안쪽 절리면 · 표면 잘라내기", expanded=True):
        st.caption("표면 분석 결과를 벽면 안쪽으로 이어 그립니다. **관측 절리 연장**은 실제로 매핑된 트레이스·절리면을 그 절리군 "
                   "방향으로 암반 속까지 연장한 것(위치는 실제, 크기만 가정), **통계 절리망**은 절리군의 간격·연장성으로 만든 "
                   "가상 절리(위치는 임의)입니다. 색은 절리군, 진하기는 유출 가능성(또는 크기)입니다. 공이 지나는 절리는 흰 점, "
                   "공이 닿지 않는 유출 '상' 절리는 빨간 테두리로 표시합니다.")
        c = st.columns(4)
        show = c[0].checkbox("관측 절리 연장", **_w("g3d_joints", True))
        dfn = c[0].checkbox("통계 절리망 (위치 임의)", **_w("g3d_dfn", False))
        pred = c[0].checkbox("공 따라 예상 관통 원판", **_w("g3d_pred", False),
                             help="균등 배치에서 절리군 간격으로 계산한 예상 관통 위치 (절리면을 그릴 때는 겹쳐 보여 기본 꺼짐)")
        jdepth = c[1].slider("연장 깊이 = 트레이스 길이 ×", 0.5, 3.0, step=0.25, **_w("g3d_jdepth", 1.0),
                             help="절리는 대략 길이만큼 깊다고 봅니다. 드론 트레이스 길이는 하한값이라 키워 볼 만합니다.")
        jmax = c[1].slider("최대 연장 (m)", 1.0, 15.0, step=0.5, **_w("g3d_jmax", 5.0))
        dfn_depth = c[2].slider("통계 절리망 두께 (벽면 뒤, m)", 1.0, 20.0, step=0.5, disabled=not dfn,
                                **_w("g3d_dfn_depth", 10.0), help="상부 커튼과 비교하려면 커튼 거리보다 두껍게")
        jop = c[2].slider("절리면 불투명도", 0.1, 1.0, step=0.05, **_w("g3d_jopacity", 0.5))
        shade = c[3].radio("진하기 기준", ["유출 가능성", "크기", "같게"], **_w("g3d_jshade", "유출 가능성"))
        lo, hi = _chainage_range(result)
        lo, hi = float(np.floor(lo)), float(np.ceil(hi))
        saved = _saved("g3d_cut_range", None)
        if not (isinstance(st.session_state.get("g3d_cut_range", saved), (list, tuple))
                and len(st.session_state.get("g3d_cut_range", saved)) == 2):
            st.session_state.pop("g3d_cut_range", None)
            saved = None
        mid, span = 0.5 * (lo + hi), hi - lo
        default = list(saved) if saved else [round(mid - 0.12 * span), round(mid + 0.12 * span)]
        cur = st.session_state.get("g3d_cut_range", default)                              # see _w
        st.session_state["g3d_cut_range"] = tuple(min(max(x, lo), hi) for x in cur)
        cut_on = st.checkbox("표면 잘라내기 (그 측점 구간의 표면을 빼고 안쪽 절리면과 공을 보기)", **_w("g3d_cut_on", True))
        rng = st.slider("잘라낼 측점 구간 (m)", lo, hi, step=0.5, key="g3d_cut_range", disabled=not cut_on,
                        help="반투명 표면은 웹 3D 특성상 얼룩져 보여, 표면을 잘라내 안쪽을 보는 방식을 씁니다. "
                             "바꾸면 장면이 바로 다시 그려집니다.")
    cut = (float(rng[0]), float(rng[1])) if cut_on else None
    return {"show": show, "dfn": dfn, "pred": pred, "depth_factor": jdepth, "max_depth": jmax, "dfn_depth": dfn_depth,
            "opacity": jop, "shade": {"유출 가능성": "seepage", "크기": "size", "같게": "none"}[shade], "cut": cut}


def _joint_planes(result, sets, opts):
    from groundwater.viz import joints3d

    if not (opts["show"] or opts["dfn"]):
        return None, None
    smap, _ = _seepage_map(result)
    key = ("_joints", id(result), id(smap), opts["show"], opts["dfn"], opts["depth_factor"], opts["max_depth"],
           opts["dfn_depth"], tuple(map(tuple, sets[["name", "spacing_m", "persistence_m"]].values.tolist())))
    cache = st.session_state.setdefault("_joints_cache", {})
    if key not in cache:
        rock = joints3d.RockTest(result)
        joints, shown = [], {}
        if opts["show"]:
            joints += joints3d.observed(result, smap, opts["depth_factor"], opts["max_depth"], rock=rock)
        if opts["dfn"]:
            dfn, shown = joints3d.statistical(result, sets, opts["dfn_depth"], rock=rock)
            joints += dfn
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[key] = (joints, shown)
    return cache[key]


def _section_view(result, sets):
    """Cross-section at one chainage: face profile, joint planes inside the rock, holes and their crossings."""
    from groundwater.viz import grouting3d, joints3d, section

    raw = st.session_state.get("_g3d_raw")
    if not raw:
        return
    holes, joints, title, shade = raw
    st.subheader("단면으로 보기 (벽면에 직각)")
    st.caption("3차원 장면을 한 측점에서 자른 단면입니다. 회색 점 = 암면 단면, 색 선 = 그 단면을 지나는 절리면 "
               "(실선: 관측 절리 연장, 점선: 통계 절리망), 굵은 갈색 선 = 그라우팅 공(가까운 공을 단면에 투영), "
               "흰 점 = 공이 절리를 지나는 곳, 붉은 띠 = 공이 닿지 않는 유출 '상' 절리. 오른쪽(+)이 벽면 앞, 왼쪽(−)이 암반 안쪽입니다.")
    lo, hi = _chainage_range(result)
    if holes:
        us = sorted(h["u"] for h in holes)
        default = float(us[len(us) // 2])
    else:
        default = 0.5 * (lo + hi)
    v = st.session_state.get("sec_u", _saved("sec_u", default))
    st.session_state["sec_u"] = float(min(max(v, lo), hi))
    c = st.columns([3, 1])
    u0 = c[0].slider("단면 측점 (m)", float(np.floor(lo)), float(np.ceil(hi)), step=0.5, key="sec_u")
    band = c[1].slider("공 투영 범위 (± m)", 0.25, 5.0, step=0.25, **_w("sec_band", 1.0))
    names = list(dict.fromkeys(r.name for r in sets.itertuples()))

    def colour_of(name):
        return "#9ca3af" if name not in names else grouting3d._set_color(name, names.index(name))

    def make():
        cov = joints3d.coverage(holes, joints) if joints and holes else None
        return section.section_figure(result, u0, holes, joints, cov, colour_of, band=band, shade=shade)

    fig = _session_cache("_sec_cache", (st.session_state.get("_g3d_key"), u0, band, shade), make)
    _map_chart(fig, "section")
    if not fig.layout.meta["holes_in_band"]:
        st.caption("이 단면 가까이에 공이 없습니다. 측점이나 투영 범위를 바꿔 보세요.")


NETWORK_DEFAULTS = {"nw_cls": "'상'만", "nw_length": 8.0, "nw_max": 30, "nw_gap": 2.0, "nw_stat": 0.2,
                    "g3d_jdepth": 1.0, "g3d_jmax": 5.0}


def _reset_network():
    for k, v in NETWORK_DEFAULTS.items():
        st.session_state[k] = v


def _guess_tileset(mesh_path, planar_dir) -> str:
    """A tiled model next to the mesh or the planar data, in the site folder, or in a folder beside it."""
    from groundwater.viz import tiles3d

    site = str(st.session_state.get("site_folder") or "").strip().strip('"')
    cands = [mesh_path, planar_dir, site]
    if site and Path(site).is_dir():
        cands += [str(d) for d in sorted(Path(site).parent.iterdir()) if d.is_dir()]
    for d in cands:
        d = str(d).strip().strip('"')
        if d and Path(d).exists():
            f = tiles3d.find_tileset(Path(d).parent if Path(d).is_file() else d)
            if f:
                return str(f)
    return ""


def _browse_tiles():
    """Native file dialog for the tiled model (.3tz, or a tileset.json whose folder is used)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        cur = str(st.session_state.get("tiles_path") or "").strip().strip('"')
        start = (Path(cur).parent if Path(cur).is_file() else Path(cur)) if cur else Path(
            str(st.session_state.get("site_folder") or "."))
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askopenfilename(
            title="타일 모델 선택 (.3tz 또는 tileset.json)", initialdir=str(start) if start.exists() else None,
            filetypes=[("3D Tiles 타일 모델", "*.3tz"), ("tileset.json", "tileset.json"), ("모든 파일", "*.*")])
        root.destroy()
    except Exception as e:  # noqa: BLE001
        st.session_state["browse_error"] = str(e)
        return
    if path:
        f = Path(path)
        st.session_state["tiles_path"] = str(f.parent if f.name.lower() == "tileset.json" else f)


def _tiles_path_input(mesh_path, planar_dir, key_suffix: str = ""):
    """Tiled-model path field (remembered per site, guessed when empty) with a file-dialog button; the tileset
    found there, or None."""
    from groundwater.viz import tiles3d

    st.session_state["tiles_path"] = st.session_state.get("tiles_path", _saved("tiles_path", ""))   # see _w
    if not str(st.session_state["tiles_path"]).strip():
        st.session_state["tiles_path"] = _guess_tileset(mesh_path, planar_dir)
    c = st.columns([5, 1], vertical_alignment="bottom")
    tp = c[0].text_input("타일 모델 (.3tz 파일 또는 tileset.json 폴더)", key="tiles_path",
                         help="비워 두면 메쉬·평면 자료 폴더, 현장 폴더와 그 옆 폴더에서 .3tz를 찾아 채웁니다.")
    c[1].button("찾아보기…", on_click=_browse_tiles, icon=":material/folder_open:", width="stretch",
                key=f"tiles_browse{key_suffix}")
    if st.session_state.pop("browse_error", None):
        st.warning("파일 선택 창을 열 수 없습니다. 경로를 직접 붙여 넣으세요.")
    tiles = tiles3d.find_tileset(tp) if tp.strip() else None
    if tp.strip() and tiles is None:
        st.warning("타일 모델을 찾을 수 없습니다.")
    return tiles


def _tiles_view(result, sets, mesh_path, planar_dir):
    """Same scene on the photo-textured tiled model (Cesium 3D Tiles .3tz) in a browser tab."""
    from groundwater.viz import joints3d, tiles3d

    with st.expander("실사 타일 모델로 보기 (Metashape 타일 모델, 사진 텍스처)", expanded=False):
        st.caption("타일 모델(.3tz 또는 tileset.json 폴더)은 사진 텍스처를 그대로 가진 3차원 모델을 화면에 보이는 만큼만 "
                   "불러와 그립니다. 위 장면의 공과 절리면을 그 위에 올리고, 측점 구간 잘라내기·투명도·선명도를 "
                   "브라우저에서 바로 조절합니다 (인터넷 연결 필요: 3D 엔진 CesiumJS).")
        tiles = _tiles_path_input(mesh_path, planar_dir)
        out = st.session_state.get("output_dir")
        raw = st.session_state.get("_g3d_raw")
        if st.button("타일 모델 장면 열기 (새 탭)", disabled=tiles is None or raw is None or not out):
            holes, joints, title, shade = raw
            cov = joints3d.coverage(holes, joints) if joints else None
            smap = _seepage_map(result)[0]
            scene = tiles3d.scene_json(result, holes, sets, tiles, joints, cov, title=title, joint_shade=shade,
                                       smap=smap, overlay_folder=_reference_folder(result))
            if st.session_state.get("g3d_seep"):           # the seepage map on the model too, as in the 3D view
                from groundwater.viz import setmap3d

                classes = (2,) if st.session_state.get("g3d_seep_cls") == "상만" else (2, 1)
                ov = setmap3d.seepage_tiles_overlay(result, smap, classes)
                scene["overlays"] = ([ov] if ov else []) + scene["overlays"]
            path = tiles3d.write_scene(Path(out) / tiles3d.SCENE_FILE, scene)
            url = _open_tiles_viewer(path)
            st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}`")


def _seepage_overlay_controls() -> dict:
    """Whether to drape the seepage-potential map on the wall of the grouting scene (3D view and tiled model): the
    holes and the cells where water may come out in one picture."""
    c = st.columns([1.6, 1, 1], vertical_alignment="bottom")
    on = c[0].toggle(":material/water_drop: 유출 가능성 지도 겹치기", **_w("g3d_seep", False),
                     help="'유출 가능성' 화면의 지도(같은 가중치·격자)를 암벽 표면에 반투명하게 덮고, 점수가 높은 구역 "
                          "10곳의 번호(#)를 붙입니다. 공이 유출 가능성이 높은 곳을 겨누는지 한 그림에서 비교합니다. "
                          "실사 타일 모델 장면에도 같은 지도가 붙습니다 (그 화면의 '표면에 붙인 이미지'에서 켜고 끄기).")
    lvl = c[1].radio("표시 등급", ["상·중", "상만"], horizontal=True, disabled=not on, **_w("g3d_seep_cls", "상·중"))
    op = c[2].slider("지도 불투명도", 0.2, 0.9, step=0.05, disabled=not on, **_w("g3d_seep_op", 0.55))
    return {"on": on, "classes": (2,) if lvl == "상만" else (2, 1), "opacity": op}


def _uniform_controls(c, best, in_plane):
    from groundwater.viz import grouting3d

    choices = {f"최적 ({best['trend']:.0f}°/{best['plunge']:.0f}°)": (best["trend"], best["plunge"]), "연직 (0°/90°)": (0.0, 90.0)}
    if in_plane:
        choices = {f"커튼 면 안 최적 ({in_plane['trend']:.0f}°/{in_plane['plunge']:.0f}°)": (in_plane["trend"], in_plane["plunge"]),
                   **choices}
    choices["직접 입력"] = None
    pick = c[0].radio("공 방향", list(choices), key="g3d_dir")
    if choices[pick] is None:
        st.session_state.setdefault("g3d_trend", float(best["trend"]))
        st.session_state.setdefault("g3d_plunge", 60.0)
        trend = c[0].number_input("방위 (°)", 0.0, 360.0, step=5.0, key="g3d_trend")
        plunge = c[0].number_input("경사 (°, 아래로)", 0.0, 90.0, step=5.0, key="g3d_plunge")
    else:
        trend, plunge = choices[pick]
    flat = plunge < grouting3d.MIN_CREST_PLUNGE
    mode = c[1].radio("천공 위치", ["상부에서 (커튼선)", "벽면에서"], key="g3d_mode", index=1 if flat else 0,
                      disabled=flat)
    if flat:
        mode = "벽면에서"
        c[1].caption(f"경사 {plunge:.0f}° 공은 상부에서 뚫으면 절벽 위에만 머물러 벽면에서 뚫는 배치로 그립니다 "
                     f"(상부 커튼은 경사 {grouting3d.MIN_CREST_PLUNGE:.0f}° 이상).")
    spacing = c[2].slider("공 간격 (m)", 1.0, 8.0, step=0.5, **_w("g3d_spacing", 3.0))
    length = c[2].slider("공 길이 (m)", 5.0, 40.0, step=1.0, **_w("g3d_length", 15.0))
    if mode.startswith("상부"):
        offset = c[3].slider("벽면 뒤 커튼 거리 (m)", 1.0, 20.0, step=0.5, **_w("g3d_offset", 5.0))
        rows = None
    else:
        offset = 5.0
        rows_txt = c[3].text_input("천공 표고 (m, 쉼표로 여러 줄)", help="비워 두면 벽 높이 전체에 5 m 간격으로 여러 줄",
                                   **_w("g3d_rows", ""))
        try:
            rows = [float(x) for x in rows_txt.split(",") if x.strip()] or None
        except ValueError:
            st.error("천공 표고에는 숫자만 넣으세요.")
            rows = None
    return trend, plunge, spacing, length, offset, rows, mode


def _grouting_surface_and_draw(result, sets, targeted, trend, plunge, spacing, length, offset, rows, mode):
    from groundwater.facade import planar
    from groundwater.viz import grouting3d

    st.session_state.setdefault("mesh_path", _saved("mesh_path", ""))
    mesh_path = st.text_input("메쉬 모델 OBJ (선택, 지도 좌표계 — 있으면 완전한 3차원 표면·지면 위 공 입구·암반 이탈 검사)",
                              value=st.session_state.get("mesh_path", ""), key="mesh_path_input")
    st.session_state["mesh_path"] = mesh_path
    site = None
    mp = mesh_path.strip().strip('"')
    mesh_ok = bool(mp) and Path(mp).is_file()
    hide_veg = False
    if mp and not mesh_ok:
        st.warning("메쉬 파일을 찾을 수 없습니다.")
    elif mesh_ok:
        hide_veg = st.checkbox("메쉬에서 나무·풀 숨기기", **_w("g3d_hideveg2", False),
                               help="초록 삼각형을 표시에서 뺍니다. 공 입구·암반 이탈 계산에는 항상 식생을 뺀 메쉬를 씁니다.")
        st.caption("메쉬는 3차원 장면을 처음 그릴 때 한 번 읽습니다 (1분 안팎, 이후 다시 읽지 않음).")
    c = st.columns(3)
    wall_modes = ["모델 표면 (메쉬·평면 DEM)", "점군"]
    wall_mode = c[0].radio("암벽 표시", wall_modes, horizontal=True, **_w("g3d_wall_mode", wall_modes[0]),
                           help="모델 표면: 메쉬(또는 절벽면 평행 DEM)가 있으면 그것을, 없으면 점군을 그립니다. "
                                "점군: 분석 점군을 정사영상 색으로 (점 수를 늘릴수록 형상이 또렷하고 화면은 무거움)")
    opacity = c[1].slider("암벽 불투명도", 0.05, 1.0, step=0.05, **_w("g3d_opacity", 1.0),
                          help="낮출수록 암벽 뒤의 공과 절리면이 비쳐 보입니다. 특정 구간만 안쪽을 보려면 아래 "
                               "'표면 잘라내기'를 함께 쓰세요.")
    as_points = wall_mode == "점군"
    npts = c[2].select_slider("암벽 점 수", [100_000, 300_000, 500_000, 1_000_000, 2_000_000],
                              **_w("g3d_wall_pts", 500_000), format_func=lambda n: f"{n // 10000}만",
                              help="점군으로 그릴 때 (메쉬·표면이 없을 때 포함). 벽 전체에 고르게 뽑습니다.")
    seep = _seepage_overlay_controls()
    jopts = _joint_controls(result)
    folder = st.session_state.get("site_folder") or ""
    pdir = _planar_input("절벽면 평행 DEM·정사영상 폴더 (선택, 메쉬가 없을 때 표면으로 씀)", "planar_dir_input")
    surface = None
    if pdir.strip():
        dem, ortho = planar.find_planar_pair(pdir.strip().strip('"'))
        if dem is None:
            st.warning("그 폴더에서 평면 투영 DEM을 찾지 못했습니다 (지도 좌표계가 아닌 GeoTIFF).")
        else:
            reg = planar.load_registration(pdir.strip().strip('"'), _reg_crs(result))
            if reg is None:
                st.info("평면 좌표와 지도 좌표의 관계가 아직 없습니다. 점군과 자동 정합합니다 (2~3분, 한 번만).")
                if st.button("점군과 자동 정합"):
                    with st.spinner("정합 중..."):
                        reg = planar.register(dem, np.asarray(result.point_cloud.points), result.wall_frame)
                        planar.save_registration(pdir.strip().strip('"'), reg)
                    st.success(f"정합 완료: 잔차 중앙값 {reg['median_residual_m'] * 100:.1f} cm")
            elif reg.get("source") == "metashape":
                st.caption(f"Metashape 평면 정의로 정확히 배치됨 ({reg['projection_file']})")
            else:
                st.caption(f"점군과 자동 정합됨 ({reg['dem']}, 잔차 중앙값 {reg['median_residual_m'] * 100:.1f} cm)")
            if reg is not None and not mesh_ok:
                key = ("_surf", str(dem), reg["median_residual_m"])
                if st.session_state.get("_surf_key") != key:
                    with st.spinner("표면 만드는 중..."):
                        st.session_state["_surf"] = planar.textured_surface(dem, ortho, reg["matrix"], step_m=0.08)
                        st.session_state["_surf_key"] = key
                surface = st.session_state["_surf"]
    tkey = None if targeted is None else tuple((k, id(v) if k == "smap" else v) for k, v in sorted(targeted.items()))
    smap_over = _seepage_map(result)[0] if seep["on"] else None
    draw_key = (_result_key(), mp if mesh_ok else None, hide_veg, id(surface), opacity, as_points, npts,
                tuple(sorted(jopts.items())), tkey,
                trend, plunge, spacing, length, offset, tuple(rows or ()), mode,
                tuple(sorted(seep.items())), id(smap_over))
    clicked = st.button("3차원 장면 그리기", type="primary",
                        help="한 번 그린 뒤에는 옵션을 바꾸면 자동으로 다시 그립니다.")
    if clicked or (st.session_state.get("_g3d") and st.session_state.get("_g3d_key") != draw_key):
        st.session_state["_g3d_key"] = draw_key
        if mesh_ok:
            with st.spinner("메쉬 읽는 중 (처음 한 번 1분 안팎)..."):
                site = _site_mesh_cached(mp, Path(mp).stat().st_mtime, _result_key(), hide_veg, result)
        with st.spinner("그리는 중..."):
            if targeted is not None and targeted["kind"] == "network":
                design, _ = _joint_planes(result, sets, {**jopts, "show": True, "dfn": False})
                opts = {k: v for k, v in targeted.items() if k not in ("kind", "smap")}
                holes = grouting3d.network_layout(result, targeted["smap"], design, sets, site=site, **opts)
                title = (f"절리망 기반 표적 천공: 공 {len(holes)}개 · 길이 {targeted['length']:g} m · "
                         f"절리 연장 = 트레이스 길이 ×{jopts['depth_factor']:g}")
            elif targeted is not None:
                holes = grouting3d.targeted_layout(result, site=site, **{k: v for k, v in targeted.items() if k != "kind"})
                title = (f"표적 주입: 유출 가능성 {'상' if targeted['classes'] == (2,) else '상·중'} 칸의 트레이스 · "
                         f"간격 {targeted['spacing']:g} m · 절리 깊이 {targeted['joint_depth']:g} m · 벽면과 {targeted['angle_deg']}°")
            else:
                holes = grouting3d.hole_layout(result, "crest" if mode.startswith("상부") else "face", trend, plunge,
                                               spacing, length, rows=rows, offset=offset, site=site)
                title = f"공 {trend:.0f}°/{plunge:.0f}° · 간격 {spacing:g} m · 길이 {length:g} m · {mode}"
            surf = None if as_points else (site.display if site is not None else surface)
            joints, shown = _joint_planes(result, sets, jopts)
            st.session_state["_g3d_raw"] = (holes, joints, title, jopts["shade"])
            fig, table = grouting3d.scene(result, holes, sets, surface=surf, title=title, surface_opacity=opacity,
                                          surface_name="암벽 (메쉬)" if site is not None else "암벽 표면",
                                          surface_points=npts,
                                          cut_chainage=jopts["cut"], joints=joints, joint_opacity=jopts["opacity"],
                                          joint_shade=jopts["shade"], show_predicted=jopts["pred"] or not joints,
                                          seepage=smap_over, seepage_classes=seep["classes"],
                                          seepage_opacity=seep["opacity"])
            tab = pd.DataFrame(table)
            st.session_state["_g3d"] = (fig, tab if targeted is not None else
                                        tab.fillna({c: 0 for c in tab.columns if c != "breaks_out_at_m"}), shown)
    if st.session_state.get("_g3d"):
        fig, table, shown = st.session_state["_g3d"]
        _view3d(fig, "g3d", 780)
        meta = fig.layout.meta
        if meta:
            share = meta["high_joints_hit"] / max(meta["high_joints"], 1)
            st.info(f"유출 가능성 '상' 칸의 관측 절리 **{meta['high_joints']}개 중 {meta['high_joints_hit']}개({share:.0%})**를 "
                    f"공이 지납니다 (관측 절리 전체 {meta['joints']}개 중 {meta['joints_hit']}개"
                    + (f", 통계 절리 {meta['dfn']}개 중 {meta['dfn_hit']}개" if meta.get("dfn") else "") + "). "
                    "빨간 테두리가 남은 절리입니다 — 배치 방식·간격·공 길이를 바꿔 비교해 보세요.")
        if shown and any(v < 1 for v in shown.values()):
            st.caption("통계 절리망이 너무 많아 일부만 만들었습니다 (공과의 교차도 그만큼만 셉니다): "
                       + ", ".join(f"{k} {v:.0%}" for k, v in shown.items() if v < 1))
        elif shown:
            st.caption("통계 절리망: 공과의 교차는 전부로 세고, 화면에는 절리군마다 최대 1,500개만 그립니다 (범례에 표시 비율).")
        nw_max = st.session_state.get("nw_max")
        if len(table) and "new_high_joints" in table and nw_max and len(table) >= nw_max:
            st.info(f"공 수가 '최대 공 수'({nw_max}개)에 닿아 멈췄습니다. 더 배치하려면 최대 공 수를 늘리세요.",
                    icon=":material/info:")
        if len(table) and "new_high_joints" in table:
            st.caption(f"공 {len(table)}개, 총 천공 {table['length_m'].sum():.0f} m. 방향·경사는 공마다 다릅니다 "
                       "(표의 trend/plunge). new_high_joints = 그 공이 새로 관통하는 유출 '상' 절리 수, "
                       "sets_crossed = 지나는 절리군. 3차원 장면에서 표면 잘라내기, 아래 단면 보기로 암반 속을 확인하세요.")
        elif len(table) and "target_joint" in table:
            st.caption(f"공 {len(table)}개, 총 천공 {table['length_m'].sum():.0f} m (공 길이 평균 {table['length_m'].mean():.1f} m). "
                       "대상 절리별: " + ", ".join(f"{k} {v}개" for k, v in table["target_joint"].value_counts().items()))
        elif len(table):
            st.caption(f"공 {len(table)}개, 총 천공 {len(table) * length:.0f} m, 공당 예상 관통 절리 평균 {table['n_joints'].mean():.1f}개 "
                       f"(최소 {table['n_joints'].min():.0f}, 최대 {table['n_joints'].max():.0f})")
        if len(table):
            if "breaks_out_at_m" in table and table["breaks_out_at_m"].notna().any():
                n = int(table["breaks_out_at_m"].notna().sum())
                st.warning(f"{n}개 공이 계획 길이 전에 암반 밖으로 나갑니다 (빨간 ✕, 표의 breaks_out_at_m). "
                           "방향·위치·길이를 조정하세요.")
            st.dataframe(table, hide_index=True, width="stretch")
        _section_view(result, sets)
        _tiles_view(result, sets, mp, pdir)
        out = st.session_state.get("output_dir")
        if out and st.button("3차원 장면을 결과 폴더에 저장 (grouting_3d.html)"):
            html = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            (Path(out) / "grouting_3d.html").write_bytes(html)
            st.success(f"저장했습니다: {Path(out) / 'grouting_3d.html'}")


def _kin_joints(result) -> dict:
    """The joints of the kinematic tests (as the pipeline): the 3D sets and the trace-only families with an
    estimated orientation; name -> (dip, dip direction)."""
    joints = {f"Set {d.set_id}": (d.mean_dip, d.mean_dip_direction) for d in result.discontinuity_sets}
    for f in result.trace_families:
        if f.linked_set is None and f.dip is not None:
            joints[f"트레이스 F{f.family_id}"] = (f.dip, f.dip_direction)
    return joints


def render_kinematics(result):
    from groundwater.discontinuity import kinematics as K
    from groundwater.viz import kinematic_view as KV

    fr0 = float(result.config.get("kinematics", {}).get("friction_deg", 30))
    st.caption("운동학적으로 '가능한지'만 보는 선별 검토(Markland 시험)입니다. 점착력·수압·절리 연장성·블록 크기는 "
               "고려하지 않습니다. 절리군은 측정된 방향(평균 경사·경사방향) 그대로 판정합니다 (일반적인 Markland 방식).")
    joints = _kin_joints(result)
    c = st.columns([1, 1, 2.2], vertical_alignment="bottom")
    phi = c[0].slider("마찰각 φ (°)", 20.0, 45.0, step=1.0, **_w("kin_phi", fr0))
    lat = c[1].slider("측방 한계 (± °)", 10.0, 30.0, step=5.0, **_w("kin_lat", 20.0),
                      help="평면·전도파괴: 절리 경사방향이 벽면 경사방향(전도는 그 반대)에서 이 각도 안일 때")
    ws = result.face_windows
    if not ws:
        st.info("벽면 방향을 구하지 못했습니다.")
        return
    scope = c[2].radio("평사투영 범위", ["벽 전체", "측점 구간"], horizontal=True, **_w("kin_scope", "벽 전체"),
                       help="벽 전체: 대표 벽면 하나와 모든 절리면. 측점 구간: 그 구간의 벽면(5 m 구간 벽면의 중앙값)과 "
                            "그 구간 안의 절리면만 — 벽이 꺾이거나 오버행인 곳을 따로 보는 구간(도메인) 해석.")
    bounds = sorted({round(w["u0"], 1) for w in ws} | {round(w["u1"], 1) for w in ws})
    u_range = None
    if scope == "측점 구간":
        lo, hi = bounds[0], bounds[-1]
        step = round(bounds[1] - bounds[0], 1) if len(bounds) > 1 else 5.0     # the windows' width (5 m)
        key = _range_state("kin_range", lo, hi, (lo, min(lo + 2 * step, hi)))
        lo_s, hi_s = st.session_state[key]                        # on the window boundaries, at least one window
        lo_s = lo + round((lo_s - lo) / step) * step
        hi_s = max(lo_s + step, lo + round((hi_s - lo) / step) * step)
        st.session_state[key] = (float(lo_s), float(min(hi_s, hi)))
        u_range = st.slider("측점 구간 (m) — 5 m 구간 경계 단위", lo, hi, step=step, key=key)
        face = KV.domain_face(ws, u_range)
        label = f"측점 {u_range[0]:g}–{u_range[1]:g} m 벽면 {face[0]:.0f}°/{face[1]:03.0f}°"
    else:
        face = K.representative_face(ws)
        label = f"벽 전체 대표 벽면 {face[0]:.0f}°/{face[1]:03.0f}°"

    st.markdown(f"##### :material/explore: 평사투영 운동학 검토 — {label}")
    png, counts = _session_cache("_kin_stereo", (_result_key(), tuple(u_range or ()), phi, lat),
                                 lambda: KV.stereonet_panels(result, face, phi, lat, u_range=u_range))
    st.image(png, width="stretch")
    n_faces = counts["평면파괴"]["total"]
    st.caption("하반구 등면적 투영. 평면·전도파괴는 절리면 극점, 쐐기파괴는 서로 다른 절리군 면 쌍의 교선(무작위 "
               "최대 4,000쌍)으로 봅니다. 색 영역에 든 점(테두리)이 그 파괴가 가능한 방향입니다."
               + (f" 이 구간 절리면 {n_faces}개." if u_range else "")
               + (" 절리면이 30개 미만이면 비율은 참고만 하세요." if u_range and n_faces < 30 else ""))
    with st.expander("5 m 구간별 평사투영 비교 (구간마다 그 구간의 벽면 × 그 구간의 절리면)"):
        rows = _session_cache("_kin_domains", (_result_key(), phi, lat), lambda: KV.domain_table(result, phi, lat))
        pct = {k: st.column_config.ProgressColumn(k, min_value=0, max_value=100, format="%d%%")
               for k in ("평면 %", "쐐기 %", "전도 %")}
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=pct)
        st.caption("벽 전체 대표 벽면 하나로 본 비율과 달리, 벽이 꺾이거나 오버행인 구간마다 비율이 달라지는 것을 봅니다. "
                   "절리면이 적은 구간(30개 미만)의 비율은 흔들림이 큽니다.")

    k = K.kinematic_check(result.face_windows, joints, phi, lat) if result.face_windows else None
    with st.expander("5 m 구간별 검토 (구간 평균 벽면)"):
        if k is None or not len(k):
            st.success("검토한 구간에서 운동학적으로 가능한 파괴 형태가 없습니다.")
        else:
            st.dataframe(k.groupby(["mode", "joints"]).size().reset_index(name="구간 수"), width="stretch")
            st.dataframe(k, width="stretch")
        if result.face_windows:
            st.dataframe(pd.DataFrame(result.face_windows).round(1), width="stretch")

    st.markdown("##### :material/map: 파괴 가능 구역 지도")
    st.caption("벽면을 셀로 나누어 셀마다 그 자리의 벽면 방향(입면 깊이 지도를 블록 크기로 평활해 구함)과 근처에서 "
               "관측된 절리군으로 같은 시험을 합니다. 위의 구간 평균과 달리 돌출부·오목부·오버행이 각자 판정됩니다.")
    c = st.columns([1, 1, 1.3, 1.2, 2.2], vertical_alignment="bottom")
    cell = c[0].select_slider("셀 크기 (m)", [0.5, 1.0, 2.0], **_w("kin_cell", 1.0))
    smooth = c[1].select_slider("벽면 평활 (m)", [1.0, 2.0, 3.0, 5.0], **_w("kin_smooth", 3.0),
                                help="벽면 방향을 구할 때의 평활 크기. 작을수록 작은 턱·홈까지, 클수록 큰 블록 규모의 면")
    local = c[2].checkbox("근처(2 m)에서 관측된 절리군만", **_w("kin_local", True),
                          help="끄면 모든 절리군이 벽 전체에 있다고 보고 벽면 방향만으로 판정합니다.")
    noveg = c[3].checkbox("식생 덮인 셀 제외", **_w("kin_noveg", True),
                          help="입면 사진에서 60 % 이상이 초록인 셀: 표면이 잎이라 벽면 방향을 믿을 수 없음")
    shown = c[4].multiselect("표시할 파괴 형태", list(K.MODES), **_w("kin_modes", list(K.MODES)),
                             help="하나만 고르면 그 형태의 셀을 그 색으로, 여럿이면 셀마다 가능한 형태 수로 칠합니다.")
    shown = [m for m in K.MODES if m in shown]
    veg_thr = (result.config.get("vegetation") or {}).get("exg_threshold", 0.06)
    kkey = (_result_key(), phi, lat, cell, smooth, local, noveg)
    kmap = _session_cache("_kin_map", kkey, lambda: K.kinematic_map(
        result, joints, phi, lat, cell, smooth, 2.0 if local else None, noveg, veg_thr))
    if not shown:
        st.info("표시할 파괴 형태를 하나 이상 고르세요.")
        return
    _map_chart(KV.map_figure(result, kmap, shown), "kin_map")
    st.dataframe(pd.DataFrame(KV.summary(kmap)), hide_index=True, width="stretch")
    st.caption(f"암면 셀 {int(kmap.valid.sum())}개 ({kmap.valid.sum() * kmap.cell_m ** 2:.0f} m²), 그중 오버행 "
               f"{int((kmap.overhang & kmap.valid).sum())}개 (벽면 경사 90°로 봄). 원인 절리군의 면적은 그 절리군 때문에 "
               "가능한 셀의 면적입니다.")

    row = st.container(horizontal=True)
    show3d = row.toggle(":material/scatter_plot: 3차원 점군에 표시", **_w("kin_3d", False))
    showtiles = row.toggle(":material/view_in_ar: 타일 모델에 표시", **_w("kin_tiles", False))
    if show3d:
        _kin_3d(result, kmap, shown, kkey)
    if showtiles:
        _kin_tiles(result, kmap, shown)


@_fragment
def _kin_3d(result, kmap, shown, kkey):
    from groundwater.viz import kinematic_view as KV

    c = st.columns(3)
    npts = c[0].select_slider("점 수", [200_000, 400_000, 800_000, 1_500_000], **_w("kin_3d_pts", 400_000),
                              format_func=lambda n: f"{n // 10000}만")
    op = c[1].slider("구역 불투명도", 0.2, 0.9, step=0.05, **_w("kin_3d_op", 0.6))
    cop = c[2].slider("점군 불투명도", 0.1, 1.0, step=0.05, **_w("kin_3d_cop", 1.0))
    fig = _session_cache("_kin_3dfig", (kkey, tuple(shown), npts, op, cop),
                         lambda: KV.scene_figure(result, kmap, shown, npts, op, cop))
    _view3d(fig, "kin_3d", 760)
    st.caption("점군 = 정사영상 색, 그 앞면에 파괴 형태별 셀을 덮었습니다 (왼쪽 범례에서 형태별로 켜고 끔).")


def _kin_tiles(result, kmap, shown):
    from groundwater.viz import kinematic_view as KV
    from groundwater.viz import tiles3d

    st.caption("파괴 형태별 셀을 실사 타일 모델 표면에 이미지로 붙여 새 탭에서 엽니다 (그 화면의 '표면에 붙인 이미지'에서 "
               "형태별로 켜고 끔. 인터넷 연결 필요: CesiumJS).")
    tiles = _tiles_path_input(st.session_state.get("mesh_path", ""), st.session_state.get("planar_dir", ""), "_kin")
    out = st.session_state.get("output_dir")
    if st.button("타일 모델 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not out,
                 key="kin_tiles_open"):
        with st.spinner("장면을 만드는 중..."):
            scene = tiles3d.scene_json(result, [], _flow_sets(result), tiles, None, None,
                                       title="파괴 가능 구역 — 타일 모델", joint_shade="none",
                                       overlay_folder=_reference_folder(result))
            scene["overlays"] = KV.tiles_overlays(result, kmap, shown) + scene["overlays"]
            path = tiles3d.write_scene(Path(out) / "kinematic_scene_tiles.json", scene)
        url = _open_tiles_viewer(path)
        st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}`")


if __name__ == "__main__":
    main()
