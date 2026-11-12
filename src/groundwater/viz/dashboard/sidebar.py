"""The sidebar of the site-folder input: folder and section, the files found, the advanced settings. Its result
is the analysis configuration."""
from pathlib import Path

import streamlit as st

from groundwater import site_folder
from groundwater.pipeline import RESULT_PICKLE
from groundwater.viz.dashboard.components import _open_digitizer, _planar_input
from groundwater.viz.dashboard.state import (DEMO_FOLDER, NONE, _apply_folder_settings, _clean_path, _folder_id,
    _remember, _restore_session, _saved, _settings_file, _site_dir, _w)


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
        _remember(key, site=True)
        if st.session_state.get(key, _saved(key, options[0])) not in options:
            st.session_state.pop(key, None)              # a file chosen last time is gone
            st.session_state[key] = options[0]
        st.session_state[key] = st.session_state.get(key, _saved(key, options[0]))     # see _w
    choice = st.selectbox(label, options, help=help_text, key=key)
    return None if choice == NONE else Path(folder) / choice


def wall_line_tool(folder: Path, found: dict):
    """Button that opens the click-to-digitize page for wall_line.csv on the drone orthomosaic."""
    with st.expander("벽면 기준선 그리기 (정사영상에서 클릭)", expanded=found["wall_line"] is None):
        st.caption("드론 정사영상 위에서 암사면 하단을 따라 클릭해 wall_line.csv를 만듭니다. "
                   "저장 위치는 위의 현장 자료 폴더입니다.")
        ortho = st.text_input("정사영상 (GeoTIFF 파일 또는 들어 있는 폴더)", value=str(folder),
                              help="폴더를 넣으면 그 안에서 가장 큰 컬러 GeoTIFF를 찾습니다 (DTM·DSM 제외).")
        if st.button("정사영상 열기", width="stretch"):
            url = _open_digitizer(_clean_path(ortho), folder)
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
    folder = _site_dir()
    if not folder.is_dir():
        st.error("폴더를 찾을 수 없습니다.")
        return None
    section = _section_picker(folder)
    _apply_folder_settings(folder, section)
    found = site_folder.scan(folder)
    st.session_state["_site_lidar"] = bool(found["lidar"])       # for _site_has_views
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
        u_lo, u_hi = c1.text_input("측점 최소 (m)", **_w("adv_u_lo", "", site=True)), c2.text_input("측점 최대 (m)", **_w("adv_u_hi", "", site=True))
        v_lo, v_hi = c1.text_input("표고 최소 (m)", **_w("adv_v_lo", "", site=True)), c2.text_input("표고 최대 (m)", **_w("adv_v_hi", "", site=True))
        from groundwater.facade.region import REGION_FILE
        has_poly = (folder / REGION_FILE).exists()
        if not has_poly:
            st.session_state["adv_use_poly"] = False
        use_poly = st.checkbox("입면 도구에서 그린 분석 범위 다각형 사용", disabled=not has_poly, **_w("adv_use_poly", has_poly, site=True))
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
                                    planar_dir=planar_run or None,
                                    clean_ground=clean_ground, clean_vegetation_faces=clean_veg_faces and veg,
                                    clean_netting=clean_net, clean_flat=clean_flat, use_lidar=use_lidar,
                                    section_overlap=overlap, fixed_sets=fixed_sets,
                                    exclude_wall_parallel=excl_wall)


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
