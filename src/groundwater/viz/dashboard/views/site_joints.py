"""View '지중 절리 (현장)': the joints of every section continued into the ground."""
import numpy as np
import streamlit as st

from groundwater.viz.dashboard.components import _figure_window, _open_tiles_viewer, _variant_picker, _view3d
from groundwater.viz.dashboard.state import _saved, _session_cache, _site_dir, _w


def render_site_joints(result):
    """Site-wide joint model: every section's joint faces in one space, extended into the rock, cut by a plan
    section at an elevation and by a vertical section along a line -- how the discontinuities may run underground,
    apart from the cliffs that exposed them."""
    from groundwater import site_joints as SJ
    from groundwater.viz import lidar_section_view as LV
    from groundwater.viz import site3d
    from groundwater.viz import site_joints_view as V
    from groundwater.viz import tiles3d

    site = _site_dir()
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
    # dynamic tabs: only the open one is built
    tabs = st.tabs([":material/view_in_ar: 3D", ":material/satellite_alt: 정사영상 위 절리", ":material/layers: 수평 단면",
                    ":material/vertical_split: 수직 단면", ":material/table_chart: 연속 가능면 표"],
                   on_change="rerun", key="sj_tab")
    orthos = LV.find_orthos(site)
    if tabs[0].open:
        with tabs[0]:
            cc = st.columns([1.4, 3], vertical_alignment="center")
            under = cc[0].toggle("연장 부분은 땅속만", value=True, key="sj_under",
                                 help="연장한 원판 중 지표 위로 나오는 부분은 그리지 않습니다 (공중으로 뻗은 부분).")
            with cc[1].popover("실사 타일 모델 위에 보기 (새 탭)", icon=":material/public:"):
                tp = str(st.session_state.get("site3d_tiles") or _saved("site3d_tiles", "")).strip()
                tiles = tiles3d.find_tileset(tp) if tp else tiles3d.find_tileset(site)
                st.caption(f"타일 모델: `{tiles.name if tiles else '없음'}` (현장 폴더에서 가장 큰 .3tz, '현장 전체 → 현장 전체 "
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
    if tabs[1].open:
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
    if tabs[2].open:
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
    if tabs[3].open:
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
    if tabs[4].open:
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
