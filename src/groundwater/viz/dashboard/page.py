"""The page: title, sidebar, run / load, the result's header and the navigation between the views."""
from pathlib import Path

import streamlit as st

from groundwater.pipeline import RESULT_PICKLE, load_config
from groundwater.viz.dashboard.components import POPOVER_PX, _choice, _open_facade_tool
from groundwater.viz.dashboard.run import _last_timings, _run_dialog, _run_notes, _run_times, _take_run
from groundwater.viz.dashboard.sidebar import sidebar_folder
from groundwater.viz.dashboard.state import (_RESULT_STATE, _load_results_cached, _reference_folder, _site_dir,
    _store_settings)
from groundwater.viz.dashboard.views.cloud3d import render_3d_view
from groundwater.viz.dashboard.views.exposure import render_exposure
from groundwater.viz.dashboard.views.grouting import render_grouting
from groundwater.viz.dashboard.views.kinematics import render_kinematics
from groundwater.viz.dashboard.views.lidar_section import render_lidar_section
from groundwater.viz.dashboard.views.outputs import (render_reference_overlays, render_report, render_tables,
    render_validation)
from groundwater.viz.dashboard.views.seepage import render_seepage
from groundwater.viz.dashboard.views.setmap import render_setmap
from groundwater.viz.dashboard.views.site_joints import render_site_joints
from groundwater.viz.dashboard.views.site_tiles import render_site_tiles
from groundwater.viz.dashboard.views.stereonet import render_stereonet
from groundwater.viz.dashboard.views.thermal import render_thermal
from groundwater.viz.dashboard.views.traces import render_traces


VIEW_ICONS = {"3D": "view_in_ar", "절리군 지도": "map", "스테레오넷": "explore", "트레이스": "timeline",
              "유출 가능성": "water_drop", "열화상": "thermostat", "그라우팅 입력": "construction", "안정성 검토": "warning", "라이다 단면": "straighten", "지중 절리 (현장)": "landscape", "굴착면 구분": "foundation", "현장 전체 3D": "public",
              "표": "table_chart", "현장 검증": "fact_check", "보고서": "description", "암각화 (참고 이미지)": "image"}


def _header(result):
    """The result on screen in a line or two, so the views start near the top of the page: where it is from,
    what to be careful about, and its key numbers. The rest -- coordinate system, the data used, what was left
    out, the automatic settings -- opens from '분석 정보'."""
    cfg = result.config
    shown = [f":blue-badge[:material/location_on: {cfg.get('site_name') or '현장'}]",
             ":gray-badge[:material/history: 저장된 결과]" if st.session_state.get("loaded_from")
             else ":green-badge[:material/check_circle: 방금 분석]"]
    info = []
    if cfg.get("crs"):
        info.append(f":gray-badge[좌표계 {cfg['crs']}]")
    if cfg.get("height_datum") == "orthometric":
        info.append(":blue-badge[:material/height: 높이 = 해발고도 (KNGeoid18)]")
    else:                                      # a caution, not a fact to look up: it stays in sight
        shown.append(":orange-badge[:material/height: 높이 = 파일 그대로 (사진측량이면 타원체고, 해발 +수십 m)]")
    if cfg.get("lidar_rock"):
        n = cfg["lidar_rock"].get("points_added", 0)
        info.append(f":violet-badge[:material/sensors: 라이다 망·식생 뒤 암반 {n / 1e6:.1f}백만 점]")
    if (cfg.get("planar") or {}).get("folder"):
        info.append(":violet-badge[:material/layers: 절벽면 평행 고해상도 자료]")
    sec = (cfg.get("section") or {}).get("core_chainage")
    if sec:
        info.append(f":gray-badge[:material/join: 담당 구간 측점 {sec[0]:.0f}–{sec[1]:.0f} m · 앞뒤 겹침 처리]")
    veg = cfg.get("vegetation_removed_fraction")
    if veg is not None:
        info.append(f":green-badge[:material/park: 식생 제외 {veg * 100:.0f}%]")
    n_patch = sum(len(d.patches) for d in result.discontinuity_sets)
    k = result.kinematics
    modes = 0 if k is None or not len(k) else k["mode"].nunique()
    numbers = [f"절리군 **{len(result.discontinuity_sets)}개** (절리면 {n_patch:,}개)",
               f"입면 트레이스 **{len(result.traces):,}개** (총 {sum(t.length_m for t in result.traces):,.0f} m)",
               f"트레이스 방향군 **{len(result.trace_families)}개**",
               f"벽면 길이 **{result.wall_frame.length:.0f} m**",
               f"가능한 파괴 형태 **{f'{modes}종' if modes else '없음'}**"]
    row = st.container(horizontal=True, vertical_alignment="center", key="result_header")
    row.markdown(" ".join(shown), width="content")
    details = row.popover("분석 정보", icon=":material/info:", type="tertiary", width=POPOVER_PX + 22)
    row.markdown(" · ".join(numbers), width="content")      # last: on a narrow page the numbers take the second line
    with details:
        if info:
            st.markdown("  \n".join(info))
        removed = cfg.get("faces_removed") or {}
        if any(removed.values()):
            st.caption(f"절리가 아닌 면 제외: 벽 아래 바닥·수면 {removed.get('ground_water', 0)}개, "
                       f"식생 위의 면 {removed.get('vegetation', 0)}개 (고급 설정에서 끄고 켬)")
        applied = cfg.get("auto_scale_applied")
        if applied:
            st.caption(f"자동 설정: 점 간격 {applied['point_spacing_m'] * 100:.1f} cm → 복셀 "
                       f"{applied['voxel_size'] * 100:.1f} cm, 법선 반경 {applied['search_radius'] * 100:.0f} cm, "
                       f"패치 연결 {applied['link_distance'] * 100:.0f} cm, 입면 영상 "
                       f"{applied['facade_resolution'] * 100:.1f} cm/px")


def _main():
    top = st.container()                       # the title: on the start page only, a result has its own header line

    with st.sidebar:
        st.subheader(":material/folder_open: 1 · 현장 자료")
        mode = _choice("입력 방식", ["현장 폴더", "설정 파일 (YAML)"], format_func=lambda m: m.split(" (")[0],
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

    running = _take_run()                      # a finished run's result comes on screen here
    if run_clicked and running:
        st.toast("분석이 이미 진행 중입니다.", icon=":material/hourglass_top:")
    elif run_clicked:                          # asks first when a result would be replaced
        out_new = cfg["paths"].get("output_dir", "data/processed/output")
        _run_dialog(cfg, _run_notes(prev, out_new), _last_timings(out_new))
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
    site_views = mode == "현장 폴더" and _site_has_views()
    if result is None:
        top.title("굴착사면 불연속면 · 지하수 유출통로 분석")
        top.caption("드론 사진측량·라이다 점군에서 절리를 찾고, 유출 가능성·열화상·안정성과 그라우팅 계획을 검토합니다.")
        st.info("왼쪽에서 현장 자료 폴더를 고르고 **분석 실행**을 누르세요. "
                "처음에는 합성 데모 자료(data/raw/demo_site)가 선택되어 있습니다."
                + (f" 아래 **{SITE_GROUP}** 화면(라이다 단면, 모든 구간의 결과 모음)은 분석 결과를 열지 않아도 볼 수 있습니다."
                   if site_views else ""))
        if site_views:                         # these views read the site folder, not the result on screen
            _show_views(None, {SITE_GROUP: VIEW_GROUPS[SITE_GROUP]})
        return

    if st.session_state.get("loaded_from"):
        st.sidebar.info(f"저장된 결과를 보는 중: {st.session_state['loaded_from']}")
    else:
        st.sidebar.success(f"결과 저장됨: {st.session_state['output_dir']}")
    _run_times(result)
    _header(result)

    out_dir = st.session_state["output_dir"]
    if result.facade_rgb is not None and st.sidebar.button("입면 도구 열기 (수동 트레이스 · 분석 범위)",
                                                         width="stretch"):
        url = _open_facade_tool(out_dir, st.session_state.get("site_folder"))
        st.sidebar.markdown(f"새 탭이 안 열리면 [여기]({url}). 저장 후 **분석 실행**을 다시 누르세요.")

    has_ref = _reference_folder(result) is not None
    _show_views(result, {g: [v for v in views if v != REFERENCE_VIEW or has_ref] for g, views in VIEW_GROUPS.items()})


# The views, grouped by what they are for. The views of SITE_GROUP read the site folder (every section's saved
# result, the LiDAR scan), not the result on screen, so they are offered without one too.
REFERENCE_VIEW = "암각화 (참고 이미지)"
SITE_GROUP = "현장 전체"
VIEW_GROUPS = {"절리": ["3D", "절리군 지도", "스테레오넷", "트레이스"],
               "유출·그라우팅": ["유출 가능성", "열화상", "그라우팅 입력"],
               "안정성": ["안정성 검토"],
               SITE_GROUP: ["라이다 단면", "지중 절리 (현장)", "굴착면 구분", "현장 전체 3D"],
               "결과물": ["표", "현장 검증", "보고서", REFERENCE_VIEW]}
GROUP_ICONS = {"절리": "polyline", "유출·그라우팅": "water_drop", "안정성": "warning", SITE_GROUP: "public",
               "결과물": "summarize"}


def _site_has_views() -> bool:
    """Whether the site folder holds anything for the site-wide views: a LiDAR scan or a saved result."""
    from groundwater.viz import site3d

    site = _site_dir()
    return site.is_dir() and (bool(st.session_state.get("_site_lidar")) or bool(site3d.section_results(site)))


def _show_views(result, groups: dict):
    """The navigation -- the group, then the views of that group -- and the chosen view. Only that view is computed
    and sent to the browser (st.tabs would run and send all of them). Each group reopens on its last view."""
    renderers = {"3D": render_3d_view, "절리군 지도": render_setmap, "스테레오넷": render_stereonet,
                 "트레이스": render_traces, "유출 가능성": render_seepage, "열화상": render_thermal,
                 "그라우팅 입력": render_grouting, "안정성 검토": render_kinematics, "라이다 단면": render_lidar_section,
                 "지중 절리 (현장)": render_site_joints, "굴착면 구분": render_exposure, "현장 전체 3D": render_site_tiles,
                 "표": render_tables, "현장 검증": render_validation, "보고서": render_report,
                 REFERENCE_VIEW: lambda r: render_reference_overlays(r, _reference_folder(r))}
    names = list(groups)
    group = names[0]
    if len(names) > 1:                         # (the site group alone, without a result, is not a choice to keep)
        cur = st.session_state.get("nav_group")
        st.session_state["nav_group"] = cur if cur in names else names[0]       # assigned in every run, see _w
        group = st.segmented_control("화면 묶음", names, key="nav_group", label_visibility="collapsed", required=True,
                                     format_func=lambda g: f":material/{GROUP_ICONS[g]}: {g}") or names[0]
    views = groups[group]
    last = st.session_state.setdefault("_nav_last", {})
    key = f"nav_view_{group}"
    # the state of a view row that was not drawn (another group was open) is gone: start it from the group's last view
    cur = st.session_state.get(key, last.get(group))
    st.session_state[key] = cur if cur in views else views[0]
    view = st.pills(group, views, key=key, required=True, label_visibility="visible" if len(names) == 1 else "collapsed",
                    format_func=lambda n: f":material/{VIEW_ICONS.get(n, 'circle')}: {n}") or views[0]
    last[group] = view
    renderers[view](result)


def main():
    try:
        _main()
    finally:                                   # options changed anywhere this run are saved this run
        if st.session_state.get("_input_mode") == "현장 폴더":
            _store_settings()
