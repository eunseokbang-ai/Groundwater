"""View '그라우팅 입력': the joint network as grouting input and the 3D virtual grouting layout."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from groundwater import hydro, report
from groundwater.viz.dashboard.components import (_choice, _map_chart, _note, _open_tiles_viewer, _planar_input,
    _table, _tiles_path_input, _view3d)
from groundwater.viz.dashboard.state import (_chainage_range, _clean_path, _flow_sets, _fragment, _reference_folder,
    _reg_crs, _remember, _result_key, _saved, _session_cache, _w, _w_clamped)


@st.cache_resource(max_entries=1, show_spinner=False)
def _site_mesh_cached(path: str, mtime: float, result_key, hide_veg: bool, _result):
    from groundwater.viz import grouting3d

    return grouting3d.load_site_mesh(path, _result, hide_vegetation=hide_veg)


def render_grouting(result):
    _note("드론 절리망으로 만든 **그라우팅 설계 초기 입력값**입니다 — 투수 '방향'과 '비율'만 의미가 있습니다.",
          "드론 절리망으로 만든 **그라우팅 설계 초기 입력값**입니다. 개구폭을 모르는 동안은 모든 절리군에 같은 개구폭을 "
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
                       "dip": st.column_config.NumberColumn("경사 (°)", format="%.0f"),
                       "dip_direction": st.column_config.NumberColumn("경사방향 (°)", format="%.0f")})
    sets = edited[edited["use"]].drop(columns=["use"])
    if not len(sets):
        st.warning("절리군을 하나 이상 포함하세요.")
        return
    seg = result.wall_frame.vertices
    d = seg[-1] - seg[0]
    wall_az = float(np.degrees(np.arctan2(d[0], d[1])) % 360)
    mode = _choice("차수 커튼", ["없음", f"벽면 기준선과 평행 (가상, 방위 {wall_az:.0f}°)", "방위각 직접 입력"])
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
        _table(principal)
        st.markdown("**절리군별 투수 기여도**")
        _table(sets[["name", "flow_share"]])
        st.markdown("**최적 방향 공 / 연직공이 만나는 절리 (10 m당)**")
        a = pd.DataFrame(hydro.per_set_frequency(sets, best["trend"], best["plunge"]))
        b = pd.DataFrame(hydro.per_set_frequency(sets, 0, 90))
        _table(a.merge(b, on="set", suffixes=("_최적", "_연직")),
               {f"{c}_{w}": (f"{label} — {w}", fmt) for w in ("최적", "연직") for c, label, fmt in (
                   ("angle_to_hole_deg", "공과의 각 (°)", "%.0f"), ("joints_per_10m", "절리 수 (개/10 m)", "%.1f"))})
    if az is not None:
        st.subheader("차수 커튼 검토")
        cc = hydro.curtain_check(sets, az)
        cut_help = ("연직공을 그 간격으로 한 줄 뚫었을 때 그 절리군의 절리 하나가 공에 걸릴 확률 "
                    "(커튼 면 위 트레이스 중 공을 가로지르는 길이 / 공 간격)")
        _table(cc, {c: (f"걸릴 확률 — 공 간격 {c[7:-1]} m", "%.2f", cut_help) for c in cc.columns if c.startswith("cut_at_")})
        conductive = sets[sets["flow_share"] >= 0.1]
        ext = float(np.percentile(conductive["persistence_m"], 90)) if len(conductive) else None
        if ext:
            st.caption(f"커튼 끝 연장 권장: 전도성 절리군 연장성 p90 = {ext:.1f} m 이상 (드론 연장성은 하한값).")
    render_grouting_3d(result, sets, best, in_plane, az)
    if any(sets["note"].str.contains("벽면과 평행")):
        st.warning("벽면과 평행한 절리군의 간격은 벽면이 계단식으로 물러나는 간격일 수 있어 투수 기여도를 과대평가할 수 있습니다. "
                   "현장 판단으로 간격을 고치거나 '포함'을 끄고 비교해 보세요.")


@_fragment
def render_grouting_3d(result, sets, best, in_plane, curtain_az):
    st.subheader("3차원 가상 시공")
    layouts = ["절리망 기반 표적 천공 (권장)", "얕은 균열 주입 (표면 가까운 균열 보수)", "균등 배치 (차수 커튼·격자)"]
    layout = _choice("배치 방식", layouts, **_w("g3d_layout2", layouts[0]),
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
    _note("표면 절리를 **암반 속으로 연장한 절리면**을 보고, 아직 관통되지 않은 절리를 가장 많이 지나는 공을 하나씩 고릅니다.",
          "표면에서 본 절리를 **암반 속으로 연장한 절리면**(아래 '벽면 안쪽 절리면'의 연장 깊이 설정)을 기준으로 공을 설계합니다. "
          "유출 가능성 높은 칸마다 공 입구 후보를 두고, 방향(벽면 직각에서 ±60°)과 경사(10~60°)를 바꿔 가며 "
          "암반 안에 머무는 공 중 **아직 관통되지 않은 절리를 가장 많이 지나는 공**을 하나씩 고릅니다. "
          "여러 절리군을 함께 지나는 공(절리 교차 = 물길)에 가산점을 주고, 연장 범위 너머는 절리군 간격으로 기대값을 더합니다.")
    c = st.columns(4)
    cls = _choice("공 입구 후보 칸", ["'상'만", "'상'+'중'"], c[0], **_w("nw_cls", "'상'만"))
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
    _note("유출 가능성 점수가 높은 칸을 지나는 **매핑된 절리(트레이스)**에만 짧은 공을 둡니다.",
          "유출 가능성 지도에서 점수가 높은 칸을 지나는 **매핑된 절리(트레이스)**에만 공을 둡니다. 공마다 트레이스 옆 벽면에서 "
          "비스듬히 뚫어 벽면 뒤 일정 깊이에서 그 절리면을 맞히고 (절리군 방향 사용), 공이 만나는 위치를 원판으로 그립니다. "
          "평탄한 절리는 위에서 아래로, 급경사 절리는 좌우 번갈아 뚫으며 모든 공은 아래로 기울입니다 (주입재가 흘러나오지 않게).")
    c = st.columns(4)
    cls = _choice("대상 칸", ["'상'만", "'상'+'중'"], c[0], **_w("tg_cls", "'상'만"))
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


def _joint_controls(result):
    """Options for the joint planes inside the rock and the cut-away."""
    with st.expander("벽면 안쪽 절리면 · 표면 잘라내기", expanded=True):
        _note("표면 분석 결과를 벽면 안쪽으로 이어 그립니다: **관측 절리 연장**(위치는 실제)과 **통계 절리망**(위치는 임의).",
              "표면 분석 결과를 벽면 안쪽으로 이어 그립니다. **관측 절리 연장**은 실제로 매핑된 트레이스·절리면을 그 절리군 "
              "방향으로 암반 속까지 연장한 것(위치는 실제, 크기만 가정), **통계 절리망**은 절리군의 간격·연장성으로 만든 "
              "가상 절리(위치는 임의)입니다. 색은 절리군, 진하기는 유출 가능성(또는 크기)입니다. 공이 지나는 절리는 흰 점, "
              "공이 닿지 않는 유출 '상' 절리는 빨간 테두리로 표시합니다.")
        c = st.columns([1, 1, 1, 1.2])
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
        shade = _choice("진하기 기준", ["유출 가능성", "크기", "같게"], c[3], **_w("g3d_jshade", "유출 가능성"))
        lo, hi = _chainage_range(result)
        lo, hi = float(np.floor(lo)), float(np.ceil(hi))
        _remember("g3d_cut_range", site=True)
        saved = _saved("g3d_cut_range", None)
        if not (isinstance(st.session_state.get("g3d_cut_range", saved), (list, tuple))
                and len(st.session_state.get("g3d_cut_range", saved)) == 2):
            st.session_state.pop("g3d_cut_range", None)
            saved = None
        mid, span = 0.5 * (lo + hi), hi - lo
        default = list(saved) if saved else [round(mid - 0.12 * span), round(mid + 0.12 * span)]
        cur = st.session_state.get("g3d_cut_range", default)                              # see _w
        st.session_state["g3d_cut_range"] = tuple(min(max(x, lo), hi) for x in cur)
        cut_on = st.checkbox("표면 잘라내기 (그 측점 구간의 표면을 빼고 안쪽 절리면과 공을 보기)", **_w("g3d_cut_on", True, site=True))
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
    _w_clamped("sec_u", default, lo, hi, site=True)
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
    lvl = _choice("표시 등급", ["상·중", "상만"], c[1], disabled=not on, **_w("g3d_seep_cls", "상·중"))
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
    positions = ["상부에서 (커튼선)", "벽면에서"]
    mode = _choice("천공 위치", positions, c[1], key="g3d_mode", default=positions[1 if flat else 0], disabled=flat,
                   format_func=lambda p: p.split(" (")[0])
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
                                   **_w("g3d_rows", "", site=True))
        try:
            rows = [float(x) for x in rows_txt.split(",") if x.strip()] or None
        except ValueError:
            st.error("천공 표고에는 숫자만 넣으세요.")
            rows = None
    return trend, plunge, spacing, length, offset, rows, mode


def _grouting_surface_and_draw(result, sets, targeted, trend, plunge, spacing, length, offset, rows, mode):
    from groundwater.facade import planar
    from groundwater.viz import grouting3d

    _remember("mesh_path", site=True)
    st.session_state.setdefault("mesh_path", _saved("mesh_path", ""))
    mesh_path = st.text_input("메쉬 모델 OBJ (선택, 지도 좌표계 — 있으면 완전한 3차원 표면·지면 위 공 입구·암반 이탈 검사)",
                              value=st.session_state.get("mesh_path", ""), key="mesh_path_input")
    st.session_state["mesh_path"] = mesh_path
    site = None
    mp = _clean_path(mesh_path)
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
    wall_mode = _choice("암벽 표시", wall_modes, c[0], **_w("g3d_wall_mode", wall_modes[0]),
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
        dem, ortho = planar.find_planar_pair(pdir)
        if dem is None:
            st.warning("그 폴더에서 평면 투영 DEM을 찾지 못했습니다 (지도 좌표계가 아닌 GeoTIFF).")
        else:
            reg = planar.load_registration(pdir, _reg_crs(result))
            if reg is None:
                st.info("평면 좌표와 지도 좌표의 관계가 아직 없습니다. 점군과 자동 정합합니다 (2~3분, 한 번만).")
                if st.button("점군과 자동 정합"):
                    with st.spinner("정합 중..."):
                        reg = planar.register(dem, np.asarray(result.point_cloud.points), result.wall_frame)
                        planar.save_registration(pdir, reg)
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
                       "(표의 방위·경사). 3차원 장면에서 표면 잘라내기, 아래 단면 보기로 암반 속을 확인하세요.")
        elif len(table) and "target_joint" in table:
            st.caption(f"공 {len(table)}개, 총 천공 {table['length_m'].sum():.0f} m (공 길이 평균 {table['length_m'].mean():.1f} m). "
                       "대상 절리별: " + ", ".join(f"{k} {v}개" for k, v in table["target_joint"].value_counts().items()))
        elif len(table):
            st.caption(f"공 {len(table)}개, 총 천공 {len(table) * length:.0f} m, 공당 예상 관통 절리 평균 {table['n_joints'].mean():.1f}개 "
                       f"(최소 {table['n_joints'].min():.0f}, 최대 {table['n_joints'].max():.0f})")
        if len(table):
            if "breaks_out_at_m" in table and table["breaks_out_at_m"].notna().any():
                n = int(table["breaks_out_at_m"].notna().sum())
                st.warning(f"{n}개 공이 계획 길이 전에 암반 밖으로 나갑니다 (빨간 ✕, 표의 '암반 이탈'). "
                           "방향·위치·길이를 조정하세요.")
            _table(table)
        _section_view(result, sets)
        _tiles_view(result, sets, mp, pdir)
        out = st.session_state.get("output_dir")
        if out and st.button("3차원 장면을 결과 폴더에 저장 (grouting_3d.html)"):
            html = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            (Path(out) / "grouting_3d.html").write_bytes(html)
            st.success(f"저장했습니다: {Path(out) / 'grouting_3d.html'}")
