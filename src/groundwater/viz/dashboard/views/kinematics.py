"""View '안정성 검토': kinematic failure modes on the stereonet, the wall map and in 3D."""
import pandas as pd
import streamlit as st

from groundwater.viz.dashboard.components import _map_chart, _session_tiles_input, _tiles_scene_button, _view3d
from groundwater.viz.dashboard.state import _fragment, _range_state, _result_key, _session_cache, _w


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

    st.caption("파괴 형태별 셀을 실사 타일 모델 표면에 이미지로 붙여 새 탭에서 엽니다 (그 화면의 '표면에 붙인 이미지'에서 "
               "형태별로 켜고 끔. 인터넷 연결 필요: CesiumJS).")
    _tiles_scene_button(result, _session_tiles_input("kin"), "kin", "kinematic_scene_tiles.json",
                        "파괴 가능 구역 — 타일 모델", lambda: KV.tiles_overlays(result, kmap, shown))
