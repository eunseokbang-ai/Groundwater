"""View '열화상': thermal images on the wall and their anomalies against the joints."""
from pathlib import Path

import numpy as np
import streamlit as st

from groundwater import report
from groundwater.viz.dashboard.components import (_map_chart, _note, _popup_button, _session_tiles_input, _table,
    _tiles_scene_button, _view_server)
from groundwater.viz.dashboard.state import (_chainage_range, _clean_path, _range_state, _reg_crs, _result_key,
    _saved, _session_cache, _w)


def render_thermal(result):
    """RGB and thermal wall orthomosaics (Metashape planar exports) registered and overlaid with an opacity slider."""
    from groundwater.facade import planar
    from groundwater.facade import thermal as TH
    from groundwater.viz import thermal_view as TV

    _note("벽면 평행으로 내보낸 RGB·열화상 정사영상을 같은 벽면 좌표(측점 × 표고)에 겹쳐 봅니다.",
          "벽면 평행으로 내보낸 RGB·열화상 정사영상을 같은 벽면 좌표(측점 × 표고)에 겹쳐 봅니다. 열화상 폴더에 온도 "
          "정사영상(`*temperature*.tif`, °C — `scripts/dji_rjpeg_to_temperature.py` → `metashape_swap_to_temperature.py`)이 "
          "있으면 온도로, 없으면 팔레트 색(상대값)으로 봅니다.")
    site = st.session_state.get("site_folder") or str(Path(result.config["paths"].get("point_cloud") or ".").parent)
    auto = _session_cache("_th_folders", str(site), lambda: TH.find_folders(site))
    c = st.columns(2)
    for key, kind in (("th_rgb_dir", "rgb"), ("th_dir", "thermal")):
        if not st.session_state.get(key, _saved(key, "")) and auto.get(kind):
            st.session_state[key] = str(auto[kind])
    rgb_dir = c[0].text_input("RGB 평면 자료 폴더 (벽면 평행 DEM·정사영상 + 평면 투영 JSON)", **_w("th_rgb_dir", "", site=True))
    th_dir = c[1].text_input("열화상 평면 자료 폴더 (DEM·정사영상)", **_w("th_dir", "", site=True))
    rgb_dir, th_dir = _clean_path(rgb_dir), _clean_path(th_dir)
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
        st.caption("위 측점 구간의 열화상(온도 정사영상이 있으면 °C)을 실사 타일 모델 표면에 붙여 새 탭에서 엽니다. 열화상 TIFF로 "
                   "만든 타일 모델(IcSan_thermal_tiled_relative_inferno.3tz)을 고르면 그 자체 색은 상대 온도, 붙인 이미지는 °C "
                   "입니다 (그 화면의 '표면에 붙인 이미지'에서 켜고 끔. 인터넷 연결 필요: CesiumJS).")
        _tiles_scene_button(result, _session_tiles_input("th"), "th", "thermal_scene_tiles.json", "열화상 — 타일 모델",
                            lambda: [TV.tiles_overlay(result, pair)])
    _thermal_anomalies(result, th_dir, crs, stamp)


def _thermal_anomalies(result, th_dir, crs, stamp):
    """Thermal anomalies (relative to their surroundings, on bare rock) and how they line up with the joints."""
    from groundwater import hydro
    from groundwater import thermal_anomaly as TA
    from groundwater.viz import thermal_view as TV

    st.markdown("##### :material/thermostat: 열화상 이상 부분 × 절리")
    _note("맨 암반의 온도를 주변 몇 m의 평균·편차와 비교해 뚜렷이 차갑거나 따뜻한 부분을 뽑습니다.",
          "온도(°C, 없으면 팔레트 상대 지수)를 맨 암반(식생·낙석방지망·평탄 지면 제외)에서 주변 몇 m의 평균·편차와 비교해 "
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
        _table(tbl, {"id": "번호"})
        out = st.session_state.get("output_dir")
        b = st.columns([1, 1, 3])
        b[0].download_button("CSV 내려받기", tbl.to_csv(index=False).encode("utf-8-sig"), "thermal_anomalies.csv",
                             "text/csv", icon=":material/download:", width="stretch")
        if out and b[1].button("결과 폴더에 저장", icon=":material/save:", width="stretch", key="ta_save"):
            path = Path(out) / "thermal_anomalies.csv"
            tbl.to_csv(path, index=False, encoding="utf-8-sig")
            b[2].success(f"저장: {path}")
