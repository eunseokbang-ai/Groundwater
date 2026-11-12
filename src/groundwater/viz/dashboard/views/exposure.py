"""View '굴착면 구분': faces parallel to the excavated wall, horizontal, or crossing it (all sections)."""
import numpy as np
import streamlit as st

from groundwater.viz.dashboard.components import _figure_window, _note, _table, _variant_picker
from groundwater.viz.dashboard.state import _session_cache, _site_dir, _w


def render_exposure(result):
    """Excavated surface vs joints: each face of the chosen sections classified by how it lies to the excavated
    wall it was found on (discontinuity/exposure.py) -- by section and set, on stereonets, elevation, plan."""
    from groundwater.discontinuity import exposure as X
    from groundwater.viz import exposure_view as EV
    from groundwater.viz import lidar_section_view as LV
    from groundwater.viz import site3d

    site = _site_dir()
    secs = site3d.section_results(site) if site.is_dir() else []
    if not secs:
        st.info("구간 결과가 없습니다 (현장 폴더/구간/<이름>/분석결과).")
        return
    _note("석산 벽은 발파·기계로 깎은 면이라, 찾은 절리면에는 굴착면 자체·소단·벽을 가로지르는 절리가 섞여 있습니다.",
          "석산 벽은 절리를 따라 무너진 면이 아니라 발파·기계로 깎은 면입니다. 그래서 찾은 절리면에는 **굴착면 자체**(벤치면·"
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
    # dynamic tabs: only the open one is built (plain st.tabs builds all five on every rerun)
    tabs = st.tabs([":material/bar_chart: 구간·절리군별", ":material/explore: 스테레오넷", ":material/view_week: 입면",
                    ":material/map: 평면", ":material/table_chart: 면 목록"], on_change="rerun", key="ex_tab")
    if tabs[0].open:
        with tabs[0]:
            st.plotly_chart(EV.bar_figure(P), width="stretch", key="ex_bar")
            _table(EV.table(P))
            st.caption("거칠기 = 면 점들의 평면 오차(RMS). 이 자료에서는 벽면 평행 면과 교차 면의 거칠기가 거의 같아서 거칠기만으로 "
                       "발파면을 가려내기는 어렵습니다 (현장에서 발파공 흔적·풍화·충전물로 확인 필요).")
    if tabs[1].open:
        with tabs[1]:
            st.image(_session_cache("_ex_st", (tuple(pick), par, flat, variant), lambda: EV.stereonets_png(P)),
                     width="stretch")
            st.markdown("**교차·수평 면만으로 다시 찾은 방향** (면 개수 기준)")
            R = _session_cache("_ex_refind", (tuple(pick), par, flat, variant), lambda: EV.refind(P))
            if len(R):
                _table(R, width="content")
            else:
                st.caption("뚜렷한 방향이 없습니다.")
            st.caption("면은 분석에서 찾은 절리군의 방향으로만 만들어지므로, 여기서는 그 절리군들 중 벽을 가로지르는 쪽이 어디인지만 "
                       "보입니다. 벽면 평행 면에 가려진 다른 절리군을 찾으려면 분석 설정의 '절벽면(굴착면)과 평행한 면은 절리군 "
                       "찾기에서 빼기'를 켜고 다시 분석하세요.")
    if tabs[2].open:
        with tabs[2]:
            for n in pick:
                Q = P[P["section"] == n]
                if len(Q):
                    _figure_window(EV.elevation_figure(Q, Q["folder"].iloc[0]), f"ex_elev_{n.split('_')[0]}", 520,
                                   f"새 창에서 크게 보기 — {n} 입면")
    if tabs[3].open:
        with tabs[3]:
            orthos = LV.find_orthos(site)
            lo, hi = P[["centroid_x", "centroid_y"]].min().to_numpy() - 40, P[["centroid_x", "centroid_y"]].max().to_numpy() + 40
            bg = _session_cache("_ex_bg", (str(orthos[0]) if orthos else "", tuple(np.round(np.r_[lo, hi], 1))),
                                lambda: LV.ortho_image(orthos[0], [lo[0], hi[0], lo[1], hi[1]], 1600) if orthos else None)
            _figure_window(EV.plan_figure(P, bg), "ex_plan", 820, "새 창에서 크게 보기 — 평면")
    if tabs[4].open:
        with tabs[4]:
            T = P[["section", "set_label", "patch_id", "dip", "dip_direction", "area_m2", "roughness_rms_m", "wall_az",
                   "wall_angle_deg", "exposure", "chainage_m", "elevation_m", "centroid_x", "centroid_y", "centroid_z"]].copy()
            T["exposure"] = T["exposure"].map(X.CLASSES)
            T = T.rename(columns={"section": "구간", "set_label": "절리군", "patch_id": "면", "dip": "경사",
                                  "dip_direction": "경사방향", "area_m2": "면적 (m²)", "roughness_rms_m": "거칠기 (m)",
                                  "wall_az": "벽면 법선 방위", "wall_angle_deg": "벽면과의 각 (°)", "exposure": "구분",
                                  "chainage_m": "측점", "elevation_m": "표고"}).round(2)
            _table(T)
            st.download_button("CSV 내려받기", T.to_csv(index=False).encode("utf-8-sig"), "절리면_굴착면구분.csv", "text/csv",
                               icon=":material/download:")
