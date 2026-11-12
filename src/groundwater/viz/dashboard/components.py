"""Building blocks shared by the views: tables with labelled columns, the choice control, the explanation behind
a button, figures as a still image with a popup window, the 3D view, the saved figure files, the planar-data and
3D-tiles path fields, and the buttons that open the external tools."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from groundwater import report
from groundwater.facade.region import core_chainage
from groundwater.viz.dashboard.state import (_PLANAR_WIDGETS, _SRC_ROOT, _clean_path, _flow_sets, _reference_folder,
    _remember, _result_key, _saved, _session_cache)


MAP_HEIGHT = 880                       # px; the map views get as much of the screen as they can use
POPUP_TITLES = {"trace_map": "입면 트레이스 지도", "seep_map": "유출 가능성 지도", "thermal_anom": "열화상 이상 부분 × 절리", "kin_map": "파괴 가능 구역 지도", "kin_3d": "파괴 가능 구역 (3D)", "set_map": "절리군 위치 지도", "overlay": "참고 이미지 겹쳐 보기",
                "section": "그라우팅 단면", "sm_plan": "수평 단면 (x-y)", "sm_cross": "횡단면 (y-z)",
                "sm_par": "벽면 평행 단면 (x-z)", "v3d_main": "점군 3D", "sm_cloud": "점군 3D — 절리군 색",
                "sm_interior": "암반 내부 절리면 (3D)", "lidar_section": "라이다 단면", "sj_plan": "지중 절리 수평 단면", "sj_ortho": "정사영상 위 절리", "ex_plan": "굴착면 구분 — 평면", "sj_section": "지중 절리 수직 단면", "g3d": "그라우팅 3차원 가상 시공"}


# The tables' columns: name in the data -> label with its unit, or (label, number format), or (label, number format,
# help). One list for every view, so a quantity reads the same wherever it appears. Only the screen uses it: the
# files a view writes (CSV) keep the names of the data.
_XYZ = "지도 좌표 (현장 좌표계)"
COLUMNS = {
    # joint sets and faces
    "set_id": "절리군", "set": "절리군", "name": "절리군", "patch_id": "절리면",
    "mean_dip": ("평균 경사 (°)", "%.1f"), "mean_dip_direction": ("평균 경사방향 (°)", "%.1f"),
    "fisher_k": ("Fisher K", "%.1f", "방향 집중도: 클수록 그 절리군의 면 방향이 한데 모여 있습니다"),
    "n_points": "점 수", "n_patches": "절리면 수",
    "spacing_mean_m": ("평균 간격 (m)", "%.2f"), "spacing_m": ("간격 (m)", "%.2f"),
    "dip": ("경사 (°)", "%.1f"), "dip_direction": ("경사방향 (°)", "%.1f"),
    "chainage_m": ("측점 (m)", "%.1f"), "elevation_m": ("표고 (m)", "%.1f"),
    "trace_confirmed_2d": ("트레이스 확인", None, "True = 입면 트레이스로 확인된 면, False = 맞는 트레이스 없음, "
                                           "빈칸 = 벽면과 평행해 확인할 수 없음"),
    "roughness_rms_m": ("거칠기 (m)", "%.4f", "면 점들의 평면 오차(RMS)"),
    "persistence_length_m": ("연장성 (m)", "%.2f"), "persistence_m": ("연장성 (m)", "%.2f"),
    "area_m2": ("면적 (m²)", "%.2f"),
    "x": ("X (m)", "%.2f", _XYZ), "y": ("Y (m)", "%.2f", _XYZ), "z": ("Z (m)", "%.2f", _XYZ),
    # the same quantities under the Korean names some views give them
    "경사": ("경사 (°)", "%.1f"), "경사방향": ("경사방향 (°)", "%.1f"), "측점": ("측점 (m)", "%.1f"),
    "표고": ("표고 (m)", "%.1f"), "벽면 법선 방위": ("벽면 법선 방위 (°)", "%.0f"),
    # trace families
    "family": "방향군", "apparent_angle_deg": ("입면 각도 (°)", "%.0f", "입면 영상에 나타나는 선의 각도"),
    "n_traces": "트레이스 수", "total_length_m": ("총 길이 (m)", "%.1f"), "mean_length_m": ("평균 길이 (m)", "%.2f"),
    "linked_3d_set": ("맞는 3차원 절리군", None,
                      "입면에 나타나는 각도가 맞는 3차원 절리군. '1 또는 2'처럼 둘 이상이면 그 절리군들이 이 벽면에서 같은 "
                      "각도의 선을 만들어 각도만으로는 가를 수 없다는 뜻입니다 (방향군 색은 첫 번째 절리군). 비어 있으면 "
                      "3차원 면으로 잡히지 않은 절리(주로 벽면을 가로지르는 절리)입니다."),
    "trace_only": ("트레이스로만 잡힘", None, "맞는 3차원 절리군이 없는 방향군"),
    "est_dip": ("추정 경사 (°)", None, "트레이스들의 3차원 방향에서 추정한 값. 벽면이 평평하면 정할 수 없습니다."),
    "est_dip_direction": ("추정 경사방향 (°)", None, "트레이스들의 3차원 방향에서 추정한 값. 벽면이 평평하면 정할 수 없습니다."),
    "orientation_quality": "방향 추정 품질", "apparent_spacing_m": ("입면 간격 (m)", "%.2f"),
    # seepage zones
    "zone": "구역", "score": ("점수", "%.2f"), "p21": ("절리 밀도 P21", "%.2f"),
    "openness_cm": ("열린 정도 (cm)", "%.1f", "입면 깊이에서 홈의 평균 깊이"),
    "network_m": ("절리망 길이 (m)", "%.0f", "서로 닿는 트레이스로 이어진 절리망의 길이"),
    "plants_along_joints": ("절리 옆 식생", "%.2f", "트레이스 양옆 10 cm 띠 안의 초록 화소 비율 (0~1)"),
    "trace_families": "트레이스 방향군", "sets_on_face": "암면의 절리군",
    # kinematics
    "u0": ("측점 시작 (m)", "%.1f"), "u1": ("측점 끝 (m)", "%.1f"), "face": "벽면 (경사/경사방향)", "mode": "파괴 형태",
    "joints": "절리군", "detail": ("절리 방향", None, "평면·전도파괴: 그 절리군의 경사/경사방향. 쐐기파괴: 두 절리군 교선의 "
                                                 "경사→방향"),
    "face_dip": ("벽면 경사 (°)", "%.1f"), "face_dip_direction": ("벽면 경사방향 (°)", "%.1f"),
    # field validation
    "field_id": "실측 번호", "field_dip": ("실측 경사 (°)", "%.0f"), "field_dip_direction": ("실측 경사방향 (°)", "%.0f"),
    "matched_set_id": "맞는 절리군", "angular_error_deg": ("각도 오차 (°)", "%.1f"),
    # grouting
    "axis": ("주축", None, "1 = 물이 가장 잘 흐르는 방향"), "relative_k": ("상대 투수계수", "%.2f"),
    "trend": ("방위 (°)", "%.0f"), "plunge": ("경사 (°, 아래로)", "%.0f"),
    "flow_share": st.column_config.ProgressColumn("투수 기여도", min_value=0, max_value=1, format="percent"),
    "angle_to_curtain_deg": ("커튼과의 각 (°)", "%.0f"),
    "apparent_dip_on_curtain_deg": ("커튼 면 위 경사 (°)", "%.0f"), "remark": "비고",
    "hole": "공", "collar_z": ("공 입구 표고 (m)", "%.1f"), "length_m": ("공 길이 (m)", "%.2f"),
    "n_joints": ("예상 관통 절리 수", None, "절리군 간격으로 계산한 값 (실측 절리가 아님)"),
    "target_joint": "대상 절리", "seepage_score": ("유출 가능성 점수", "%.2f"),
    "joint_at_m": ("절리까지 (m)", "%.2f", "공 입구에서 대상 절리를 만나는 곳까지의 길이"),
    "new_high_joints": ("새로 관통하는 '상' 절리", None, "그 공이 처음으로 관통하는, 유출 가능성 '상' 칸의 절리 수"),
    "sets_crossed": "지나는 절리군",
    "breaks_out_at_m": ("암반 이탈 (m)", "%.1f", "공이 계획 길이 전에 암반 밖으로 나가는 곳 (공 입구에서의 길이)"),
    "crosses_mapped": "지나는 관측 절리", "crosses_statistical": "지나는 통계 절리",
    "crosses_high_seepage": "지나는 유출 '상' 절리",
}


def _table(data, columns: dict | None = None, **kw):
    """A table with the dashboard's column labels and units (COLUMNS; `columns` adds to or replaces them for this
    table, an entry may also be a ready st.column_config column), without the row numbers, across the page."""
    df = pd.DataFrame(data)
    config = {}
    for name in df.columns:
        spec = {**COLUMNS, **(columns or {})}.get(name)
        if spec is None:
            continue
        if not isinstance(spec, (str, tuple)):
            config[name] = spec
            continue
        label, fmt, tip = (spec, None, None) if isinstance(spec, str) else (tuple(spec) + (None, None))[:3]
        kind = df[name].dtype
        if fmt and pd.api.types.is_numeric_dtype(kind) and not pd.api.types.is_bool_dtype(kind):
            config[name] = st.column_config.NumberColumn(label, format=fmt, help=tip)
        else:
            config[name] = st.column_config.Column(label, help=tip)
    kw.setdefault("hide_index", True)
    kw.setdefault("width", "stretch")
    return st.dataframe(df, column_config=config, **kw)


def _choice(label, options, col=None, **kw):
    """One of a few options as a segmented control: the dashboard's control for a short choice (a radio only where
    the options are a long list). One option is always selected, and a remembered value that is no longer among
    the options falls back to `default` (the first option)."""
    options = list(options)
    default = kw.pop("default", options[0])
    key = kw.get("key")
    if key is None:
        kw["default"] = default
    elif st.session_state.get(key) not in options:
        st.session_state[key] = default
    return (col or st).segmented_control(label, options, required=True, **kw) or default


# width of a small popover button with a three-letter label: in a horizontal row Streamlit sizes the button to less
# than its label and cuts the label short ("자…"), so the width is given
POPOVER_PX = 112


def _note(summary: str, detail: str):
    """A view's explanation: one line on the page, the rest behind a small '자세히' button, so a view opens on its
    controls and figures and not on a block of text."""
    row = st.container(horizontal=True, vertical_alignment="center")
    row.caption(summary, width="content")
    with row.popover("자세히", icon=":material/info:", type="tertiary", width=POPOVER_PX):
        st.markdown(detail)


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


def _variant_picker(site: Path, key: str, col=None) -> str:
    """Which result folder of the sections to use (viz.site3d.result_variants): 분석결과 = '기본', else the folder
    suffix (e.g. '벽면평행제외'); shown only when there is more than one."""
    from groundwater.viz import site3d

    vs = site3d.result_variants(site) if site.is_dir() else []
    if len(vs) <= 1:
        return vs[0] if vs else ""
    labels = {v: (v.lstrip("_") or "기본") for v in vs}
    _remember(key)
    st.session_state[key] = st.session_state.get(key, _saved(key, vs[0]))
    if st.session_state[key] not in vs:
        st.session_state[key] = vs[0]
    return (col or st).selectbox("결과 종류", vs, format_func=labels.get, key=key,
                                 help="구간마다의 결과 폴더: '기본' = 분석결과, 그 밖 = 분석결과_<이름> (다른 설정으로 돌린 결과, "
                                      "예: 벽면평행제외 = 절벽면과 평행한 면을 절리군 찾기에서 뺀 분석).")


MAX_FACADE_DISPLAY_PX = 3000      # longer side of the facade image sent to the browser


def _planar_changed(src):
    v = _clean_path(st.session_state.get(src, ""))
    st.session_state["planar_dir"] = v
    for k in _PLANAR_WIDGETS:
        if k != src:
            st.session_state[k] = v


def _planar_input(label, key, **kw) -> str:
    """The planar-data folder field; the sidebar, trace and grouting fields all edit the same value."""
    _remember("planar_dir", site=True)
    st.session_state.setdefault("planar_dir", _saved("planar_dir", ""))
    st.session_state[key] = st.session_state.get(key, st.session_state["planar_dir"])     # see _w
    return _clean_path(st.text_input(label, key=key, on_change=_planar_changed, args=(key,), **kw))


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


def _guess_tileset(mesh_path, planar_dir) -> str:
    """A tiled model next to the mesh or the planar data, in the site folder, or in a folder beside it."""
    from groundwater.viz import tiles3d

    site = _clean_path(st.session_state.get("site_folder"))
    cands = [mesh_path, planar_dir, site]
    if site and Path(site).is_dir():
        cands += [str(d) for d in sorted(Path(site).parent.iterdir()) if d.is_dir()]
    for d in cands:
        d = _clean_path(d)
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

        cur = _clean_path(st.session_state.get("tiles_path"))
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

    _remember("tiles_path", site=True)
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


def _session_tiles_input(key: str):
    """The tiled-model path field of the view with key prefix `key`, guessed from this session's mesh and
    planar-data folders (_tiles_path_input)."""
    return _tiles_path_input(st.session_state.get("mesh_path", ""), st.session_state.get("planar_dir", ""), f"_{key}")


def _tiles_scene_button(result, tiles, key: str, file_name: str, title: str, overlays, sets=None, joints=None,
                        smap=None):
    """The button that puts a view on the photo-textured tiled model: it writes the scene file (the joint sets,
    with the images of `overlays()` draped on the surface in front of the reference images) into the result
    folder and opens it in a new tab. `overlays` and `joints` are functions: worked out only on the click."""
    from groundwater.viz import tiles3d

    out = st.session_state.get("output_dir")
    if st.button("타일 모델 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not out,
                 key=f"{key}_tiles_open"):
        with st.spinner("장면을 만드는 중..."):
            scene = tiles3d.scene_json(result, [], _flow_sets(result) if sets is None else sets, tiles,
                                       joints() if joints else None, None, title=title, joint_shade="none", smap=smap,
                                       overlay_folder=_reference_folder(result))
            scene["overlays"] = [o for o in overlays() if o] + scene["overlays"]
            path = tiles3d.write_scene(Path(out) / file_name, scene)
        url = _open_tiles_viewer(path)
        st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}`")
