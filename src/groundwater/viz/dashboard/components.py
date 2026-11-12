"""Building blocks shared by the views: figures as a still image with a popup window, the 3D view, the saved
figure files, the planar-data and 3D-tiles path fields, and the buttons that open the external tools."""
import sys
from pathlib import Path

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
