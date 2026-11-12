"""View '스테레오넷': pole density, the joint sets and the candidate sets."""
import streamlit as st

from groundwater import report
from groundwater.viz.dashboard.components import _figure
from groundwater.viz.dashboard.state import _w


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
