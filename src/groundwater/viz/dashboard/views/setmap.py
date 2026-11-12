"""View '절리군 지도': where each joint set crops out on the wall, with the 3D cloud, the tiled model and the
joint planes inside the rock."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from groundwater.viz.dashboard.components import (MAX_FACADE_DISPLAY_PX, _figure_window, _map_chart,
    _session_tiles_input, _tiles_scene_button, _view3d)
from groundwater.viz.dashboard.state import (_chainage_range, _flow_sets, _fragment, _range_state, _result_key,
    _saved, _session_cache, _w, _w_clamped)


SETMAP_INTERIOR_SAMPLE = 250_000     # cloud points behind the interior view (face context)
SECTION_PANEL_H = 500               # px, the interior view's plan-section card (full width)
SECTION_ROW_H = 600                 # px, its wall-parallel and cross-section cards (side by side)

# the two side-by-side section images at one height on any screen: the columns' width ratio shifts with the
# screen (fixed card padding), so the cross section's image box follows the wall-parallel one (fitted, not stretched)
SAME_HEIGHT_JS = """<script>
(() => {
  const sync = () => {
    const b = document.querySelector('.st-key-sm_card_b img'), c = document.querySelector('.st-key-sm_card_c img');
    if (!b || !c) return;
    const h = Math.round(b.getBoundingClientRect().height);
    if (h > 0 && c.style.height !== h + 'px') {
      Object.assign(c.style, { height: h + 'px', width: '100%', objectFit: 'contain' });
    }
  };
  if (!window.__gwSameHeight) {
    window.__gwSameHeight = true;
    window.addEventListener('resize', sync);
    setInterval(sync, 700);                    // images are replaced when a section changes
  }
  sync();
})();
</script>"""


def _jpeg_uri_bottom_first(img) -> str:
    """JPEG data URI of an image for a Plotly Image trace anchored at its bottom row (see _facade_display)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _setmap_display(result):
    from groundwater.viz import setmap3d

    img = setmap3d.setmap_blend(result, MAX_FACADE_DISPLAY_PX)
    t = result.facade_rgb.transform
    H, W = result.facade_rgb.array.shape[1:]
    return _jpeg_uri_bottom_first(img), abs(t.a) * W / img.shape[1], abs(t.e) * H / img.shape[0]


def _sm_disp(result, u_range):
    """The 3D views' coordinates (setmap3d.Disp) as chosen in the set map view."""
    from groundwater.viz import setmap3d

    if st.session_state.get("sm_coords", _saved("sm_coords", "실제 평면 (직선)")).startswith("실제"):
        return setmap3d.Disp(result.wall_frame, 0.5 * (u_range[0] + u_range[1]))
    return setmap3d.UNROLLED


def _sm_colour_of(sets):
    from groundwater.viz import grouting3d

    names = list(dict.fromkeys(sets["name"])) if len(sets) else []
    return lambda name: "#9ca3af" if name not in names else grouting3d._set_color(name, names.index(name))


def render_setmap(result):
    """Joint-set map: on the orthophoto (default), then -- on request, one below the other -- on the point cloud,
    on the tiled model and continued into the rock with sections."""
    from groundwater.viz import setmap3d

    if result.facade_sets is None or result.facade_rgb is None:
        st.info("색 정보가 없어 입면 지도를 만들 수 없습니다.")
        return
    src, dx, dy = _session_cache("_sm_disp", _result_key(), lambda: _setmap_display(result))
    t = result.facade_rgb.transform
    v_bottom = t.f + result.facade_rgb.array.shape[1] * t.e
    fig = go.Figure(go.Image(source=src, x0=t.c + dx / 2, dx=dx, y0=v_bottom + dy / 2, dy=dy, hoverinfo="skip"))
    for d in result.discontinuity_sets:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Set {d.set_id}  {d.mean_dip:.0f}°/{d.mean_dip_direction:03.0f}°",
                                 marker=dict(symbol="square", size=13, color=setmap3d.set_colour(d.set_id))))
    fig.update_xaxes(title="측점 (m)")
    fig.update_yaxes(title="표고 (m)", autorange=True)
    fig.update_layout(title="절리군 위치 지도 — 정사영상 (색 = 그 자리 암면이 속한 절리군)", legend=dict(orientation="h"))
    _map_chart(fig, "set_map")
    st.caption("입면 위에 각 자리의 암면이 어느 절리군에 속하는지 색으로 표시. 색이 없는 곳은 식생·거친 면·작은 조각. "
               "확대·이동은 '새 창에서 크게 보기'로 엽니다.")

    st.markdown("**다른 모델에서 보기** — 켜면 아래에 차례로 열립니다")
    row = st.container(horizontal=True)
    show_cloud = row.toggle(":material/scatter_plot: 점군 3D", **_w("sm_cloud", False),
                            help="분석 점군에서 절리군에 속한 점을 그 색으로, 나머지는 정사영상 색으로. 돌리기·확대 가능")
    show_tiles = row.toggle(":material/view_in_ar: 타일 모델", **_w("sm_tiles", False),
                            help="절리군 지도를 실사 타일 모델 표면에 붙여 새 탭에서 봅니다")
    show_int = row.toggle(":material/layers: 암반 내부 절리면 (3D · 단면)", **_w("sm_interior", False),
                          help="노출된 암면의 절리면을 그 방향 그대로 암반 속으로 연장해 3차원과 단면으로 봅니다")
    sets = _flow_sets(result)
    if show_cloud or show_int:
        coord_modes = ["실제 평면 (직선)", "측점 기준 (벽 따라 펼침)"]
        st.radio("3D·수평 단면 좌표", coord_modes, horizontal=True, **_w("sm_coords", coord_modes[0]),
                 help="실제 평면: 지도 좌표를 보는 구간 가운데의 벽 방향으로 돌려 그립니다 (평면인 절리는 평면, 직선). "
                      "측점 기준: 벽 기준선을 따라 펼친 좌표 — 기준선이 꺾이는 곳(원호)에서는 축이 함께 돌아 평면인 "
                      "절리도 휘어 보입니다. 측점·깊이를 정확히 읽을 때 씁니다.")
    if show_cloud:
        st.divider()
        _setmap_cloud(result)
    if show_tiles:
        st.divider()
        _setmap_tiles(result, sets)
    if show_int:
        st.divider()
        _setmap_interior(result, sets)


@_fragment
def _setmap_cloud(result):
    from groundwater.viz import setmap3d

    st.subheader(":material/scatter_plot: 점군 3D — 절리군 색")
    lo, hi = (float(np.floor(x)) for x in _chainage_range(result))
    c = st.columns([2, 3, 1.4], vertical_alignment="bottom")
    npts = c[0].select_slider("표시 점 수", [50_000, 100_000, 150_000, 250_000, 400_000], **_w("sm_npts", 150_000),
                              format_func=lambda n: f"{n // 1000}천", help="많을수록 촘촘하지만 화면이 느려집니다")
    key = _range_state("sm_cloud_range", lo, hi + 1, (lo, hi + 1))
    rng = c[1].slider("표시 측점 구간 (m)", lo, hi + 1, step=0.5, key=key)
    other = c[2].checkbox("절리군 밖 점도", **_w("sm_other", True), help="식생·거친 면 등 (정사영상 색)")
    sample = _session_cache("_sm_sample", (_result_key(), npts), lambda: setmap3d.cloud_sample(result, npts))
    disp = _sm_disp(result, rng)
    fig = _session_cache("_sm_cloudfig", (_result_key(), npts, rng, other, disp.plan),
                         lambda: setmap3d.cloud_figure(sample, result.discontinuity_sets, rng, other, disp=disp))
    _view3d(fig, "sm_cloud", 820)
    st.caption(("실제 평면 좌표: 보는 구간 가운데의 벽 방향이 x, 그 안쪽이 y (직선 거리), z = 표고. " if disp.plan else
                "벽면 좌표로 펼친 점군: x = 측점(벽 따라), y = 깊이(+ 암반 안쪽), z = 표고. ")
               + "마우스는 세슘(타일 모델 화면)과 같습니다. 왼쪽 범례에서 절리군을 켜고 끕니다.")


def _sm_joint_opts():
    """Options of the interior view (also used by the tiled-model scene when the interior view is closed)."""
    g = lambda k, d: st.session_state.get(k, _saved(k, d))  # noqa: E731
    return {"depth_factor": g("sm_jdepth", 2.0), "max_depth": g("sm_jmax", 5.0), "dfn": g("sm_dfn", False),
            "dfn_depth": g("sm_dfn_depth", 5.0), "traces": g("sm_traces", True)}


def _sm_joints(result, sets, opts):
    """Joint planes continued into the rock (and their wall-coordinate mesh), cached per option set."""
    from groundwater.viz import joints3d, setmap3d

    def make():
        rock = joints3d.RockTest(result)
        joints = joints3d.observed(result, None, opts["depth_factor"], opts["max_depth"], rock=rock)
        if not opts["traces"]:
            joints = [j for j in joints if j.source != "trace"]
        n_obs = len(joints)
        if opts["dfn"]:
            joints += joints3d.statistical(result, sets, opts["dfn_depth"], rock=rock)[0]
        return {"rock": rock, "joints": joints, "mesh": setmap3d.joint_mesh(result, joints), "n_obs": n_obs}

    key = (_result_key(), tuple(sorted(opts.items())),
           tuple(map(tuple, sets[["name", "spacing_m", "persistence_m"]].values.tolist())))
    return _session_cache("_sm_joints", key, make)


@_fragment
def _setmap_tiles(result, sets):
    from groundwater.viz import setmap3d

    st.subheader(":material/view_in_ar: 타일 모델 — 실사 모델에 절리군 지도 붙이기")
    st.caption("절리군 지도를 타일 모델 표면에 이미지로 붙여 새 탭에서 엽니다. 불투명도·표면 잘라내기·이미지 켜기/끄기는 "
               "그 화면에서 조절합니다 (인터넷 연결 필요: 3D 엔진 CesiumJS).")
    tiles = _session_tiles_input("sm")
    with_joints = st.checkbox("암반 내부 절리면도 함께 (아래 '암반 내부 절리면' 설정)", **_w("sm_tiles_joints", True))
    _tiles_scene_button(result, tiles, "sm", "setmap_scene_tiles.json", "절리군 위치 지도 — 타일 모델",
                        lambda: [setmap3d.tiles_overlay(result)], sets=sets,
                        joints=(lambda: _sm_joints(result, sets, _sm_joint_opts())["joints"]) if with_joints else None)


@_fragment
def _setmap_interior(result, sets):
    from groundwater.viz import section, setmap3d

    st.subheader(":material/layers: 암반 내부 절리면 — 노출된 암면에서 안쪽으로 연장")
    modes = ["주요 절리면 구조 (거시)", "개별 절리 연장 (상세)"]
    mode = st.radio("표시 방식", modes, horizontal=True, **_w("sm_mode", modes[0]),
                    help="거시: 절리군마다 가장 길게 이어진 절리면 몇 개만 골라 암반 블록을 관통하게 그려, 큰 절리면이 "
                         "어떻게 이어지고 서로 만나는지(교선) 봅니다. 상세: 관측된 절리 하나하나를 그 크기만큼 연장합니다.")
    major = mode == modes[0]
    if major:
        st.caption("절리군마다, 같은 평면 위에 놓인 노출면과 트레이스를 모아(절리군 평균 방향, 법선 방향 위치가 간격의 1/3 "
                   "안) 합친 길이가 가장 긴 평면을 **주요 절리면**으로 골라 암반 블록 끝까지 연장합니다. 검은 선은 주요 "
                   "절리면끼리 만나는 교선, 굵은 색 선은 그 절리면이 암면에 드러나는 선, 블록 옆·윗·뒷면의 색 선은 블록을 "
                   "자른 단면에 나타나는 절리 흔적입니다. 위치는 관측, 연장은 해석(가정)입니다.")
    else:
        st.caption("노출된 암면에서 찾은 절리면(입면 트레이스와 3차원 노출면)을 각자의 면 방향 그대로 암반 속으로 연장한 "
                   "추정입니다. 위치와 방향은 관측값, 안쪽으로 이어지는 크기만 가정(트레이스·노출면 크기 × 배율, 최대값 "
                   "제한)이며 암면 앞으로 나오는 부분은 잘라냅니다. 벽면과 거의 평행한 노출면은 연장해도 암면 자체이므로 빼고, "
                   "그런 절리군은 **통계 절리망**(간격·연장성으로 만든 가상 절리, 위치 임의)으로 안쪽 분포를 볼 수 있습니다.")
    c = st.columns(4)
    if major:
        per_set = c[0].slider("절리군마다 최대 평면 수", 1, 10, **_w("sm_major_max", 4),
                              help="한 절리군은 몇 m 간격으로 나란한 여러 평면입니다. 근거가 긴 순서로 이 수까지 그립니다 "
                                   "(1 = 가장 길게 이어진 면 하나, 가장 단순한 거시 구조).")
        min_len = c[1].slider("평면마다 근거 길이 최소 (m)", 0.0, 60.0, step=5.0, **_w("sm_major_minlen", 20.0),
                              help="그 평면 위에 놓인 노출면 연장성 + 트레이스 길이의 합. 짧은 근거의 평면은 그리지 않습니다.")
        min_sep = c[1].slider("나란한 평면 사이 최소 간격 (m)", 0.5, 5.0, step=0.5, **_w("sm_major_sep", 1.5),
                              help="이보다 가까운 평면은 같은 면의 굴곡으로 보고 하나만 그립니다.")
        extent_lbl = c[0].radio("연장 범위", ["블록 전체", "관측 범위 + 3 m"], horizontal=True,
                                **_w("sm_major_ext", "블록 전체"),
                                help="블록 전체: 주요 절리면을 블록 끝까지 (거시 구조). 관측 범위: 그 절리면의 노출면·"
                                     "트레이스가 있는 범위에서 3 m만 더")
        bdepth = c[1].slider("블록 깊이 (암면 뒤, m)", 2.0, 20.0, step=1.0, **_w("sm_major_depth", 8.0))
        jmax = bdepth
    else:
        jdepth = c[0].slider("연장 크기 = 트레이스·노출면 크기 ×", 0.5, 3.0, step=0.25, **_w("sm_jdepth", 2.0),
                             help="절리는 대략 길이만큼 깊다고 봅니다. 드론으로 본 길이는 하한값이라 키워 볼 만합니다.")
        jmax = c[0].slider("최대 연장 (m)", 1.0, 15.0, step=0.5, **_w("sm_jmax", 5.0))
        traces = c[1].checkbox("입면 트레이스도 연장", **_w("sm_traces", True),
                               help="끄면 3차원 노출면(절리군 지도의 색 면)만 연장합니다")
        dfn = c[1].checkbox("통계 절리망 함께 (위치 임의)", **_w("sm_dfn", False))
        dfn_depth = c[1].slider("통계 절리망 두께 (m)", 1.0, 15.0, step=0.5, disabled=not dfn, **_w("sm_dfn_depth", 5.0))
    jop = c[2].slider("절리면 불투명도", 0.1, 1.0, step=0.05, **_w("sm_jop", 0.75))
    face_modes = {"점군": "points", "표면 (반투명)": "surface", "끄기": "none"}
    face = face_modes[c[2].radio("암면 표시", list(face_modes), horizontal=True, **_w("sm_face_mode", "점군"),
                                 help="점군: 분석 점군을 정사영상 색으로 촘촘하게 (점 수를 늘릴수록 형상이 또렷하고 화면은 "
                                      "무거움). 표면: 정사영상 색을 입힌 암면 표면을 반투명하게 (가벼움)")]
    face_op = c[2].slider("암면 불투명도", 0.05, 1.0, step=0.05, **_w("sm_face_op", 0.35), disabled=face == "none",
                          help="낮출수록 암면 뒤의 절리면이 잘 보입니다")
    npts = c[2].select_slider("암면 점 수 (표시 구간 안)", [100_000, 250_000, 500_000, 1_000_000, 2_000_000],
                              **_w("sm_face_pts", 500_000), format_func=lambda n: f"{n // 10000}만",
                              disabled=face != "points")
    lo, hi = (float(np.floor(x)) for x in _chainage_range(result))
    mid, span = 0.5 * (lo + hi), hi - lo
    key = _range_state("sm_range", lo, hi + 1, (round(mid - 0.15 * span), round(mid + 0.15 * span)))
    rng = c[3].slider("표시 측점 구간 (m)", lo, hi + 1, step=0.5, key=key,
                      help="좁힐수록 안쪽이 잘 보입니다. 단면도 이 구간을 그립니다. 거시 보기는 넓게 잡아도 가볍습니다.")
    sample = _session_cache("_sm_isample", _result_key(), lambda: setmap3d.cloud_sample(result, SETMAP_INTERIOR_SAMPLE))
    colour_of = _sm_colour_of(sets)
    v_lo, v_hi = (round(float(x), 1) for x in np.percentile(sample["uvw"][:, 1], [1, 99]))
    if major:
        rock = _sm_joints(result, sets, _sm_joint_opts())["rock"]
        mkey = (_result_key(), tuple(rng), per_set, min_len, min_sep, extent_lbl, bdepth,
                tuple(map(tuple, sets[["name", "dip", "dip_direction", "spacing_m"]].values.tolist())))

        def make_major():
            pl = setmap3d.major_planes(result, sets, rng, per_set, min_length_m=min_len, min_sep_m=min_sep)
            return setmap3d.major_model(result, rock, pl, rng, (v_lo, v_hi), bdepth,
                                        "block" if extent_lbl == "블록 전체" else "observed")

        model = _session_cache("_sm_major", mkey, make_major)
        if model is None:
            st.info("이 구간에서 주요 절리면을 찾지 못했습니다. 측점 구간을 넓혀 보세요.")
            return
        J = {"rock": rock, "joints": model["joints"], "mesh": model["mesh"]}
        rows = [{"주요 절리면": f"{p['name']}-{setmap3d.plane_tag(p['rank'])}",
                 "경사/경사방향": f"{p['dip']:.0f}°/{p['dd']:03.0f}°",
                 "근거 길이 (m)": round(p["length_m"], 1), "노출면·트레이스 수": p["pieces"]} for p in model["planes"]]
        st.caption(f"주요 절리면 {len(model['planes'])}개 · 교선 {len(model['intersections'])}개 (근거 길이 = 그 평면 위 "
                   "노출면 연장성 + 트레이스 길이의 합)")
        with st.expander("주요 절리면 목록", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        opts = {"depth_factor": jdepth, "max_depth": jmax, "dfn": dfn, "dfn_depth": dfn_depth, "traces": traces}
        with st.spinner("절리면을 암반 속으로 연장하는 중..."):
            J = _sm_joints(result, sets, opts)
        n_tr = sum(j.source == "trace" for j in J["joints"])
        n_fc = sum(j.source == "facet" for j in J["joints"])
        n_dfn = len(J["joints"]) - J["n_obs"]
        st.caption(f"관측 연장 {J['n_obs']}개 (트레이스 {n_tr} · 노출면 {n_fc})" + (f" · 통계 절리망 {n_dfn}개" if dfn else ""))

    # the section cards (each with its position slider) sit below the 3D view but are made first: the 3D view
    # shows where the sections cut
    view_box = st.container()
    _w_clamped("sm_sec_v", 0.5 * (v_lo + v_hi), v_lo, v_hi, site=True)
    _w_clamped("sm_sec_u", 0.5 * (rng[0] + rng[1]), lo, hi + 1, site=True)
    st.markdown("##### :material/splitscreen: 단면으로 보기")
    st.caption("세 단면 모두 1:1 축척. 색 선 = 그 면을 지나는 절리면 (실선: 관측 연장·주요 절리면, 점선: 통계 절리망) · "
               "회색 점 = 암면 점군 단면 · 갈색 = 암반 · 검은 파선과 A·B·C 표시 = 다른 단면이 이 단면을 자르는 곳. "
               "확대·이동·점 정보는 각 그림의 '새 창에서 크게 보기'로.")
    # laid out by shape: the long, flat plan section across the page; the wall-parallel (~1.7:1) and the cross
    # section (~0.8:1) side by side at one height
    card_a = st.container(border=True)
    row = st.columns([2, 1], gap="medium")
    card_b, card_c = row[0].container(border=True, key="sm_card_b"), row[1].container(border=True, key="sm_card_c")
    with card_a:
        st.markdown("**:blue-badge[A] 수평 단면** :gray[x-y · 위에서 내려다본 면]")
        v0 = st.slider("표고 z (m)", v_lo, v_hi, step=0.25, key="sm_sec_v")
    with card_b:
        st.markdown("**:green-badge[B] 벽면 평행 단면** :gray[x-z · 암면을 그 깊이만큼 벗겨낸 모습]")
        dpar = st.slider("암면에서 깊이 (m)", 0.0, 10.0, step=0.25, **_w("sm_sec_d", 1.0))
    with card_c:
        st.markdown("**:orange-badge[C] 횡단면** :gray[y-z · 벽에 직각]")
        u0 = st.slider("측점 x (m)", lo, hi + 1, step=0.25, key="sm_sec_u")
    planes = {"v": v0, "u": u0, "depth_shown": jmax}

    surface = dense = None
    if face == "surface":
        surface = _session_cache("_sm_surface", (_result_key(), tuple(rng)), lambda: setmap3d.face_surface(result, rng))
    elif face == "points":                    # the ~3 M point section sample, in photo colours
        dense = _session_cache("_sm_dense", _result_key(), lambda: setmap3d.dense_face_sample(result))
    disp = _sm_disp(result, rng)
    # the same figure object while nothing it shows changes: the 3D page is then not rebuilt (~1 s) and re-sent
    fkey = (mkey if major else (_result_key(), tuple(sorted(opts.items()))),
            tuple(map(tuple, sets[["name", "dip", "dip_direction"]].values.tolist())), tuple(rng), jop, face, face_op,
            npts, v0, u0, jmax, disp.plan)
    if major:
        fig = _session_cache("_sm_ifig", fkey, lambda: setmap3d.major_figure(
            dense or sample, model, colour_of, jop, face, face_op, npts, surface=surface, planes=planes, disp=disp))
    else:
        fig = _session_cache("_sm_ifig", fkey, lambda: setmap3d.interior_figure(
            dense or sample, J["joints"], J["mesh"], colour_of, rng, jop, face, face_op, npts, surface=surface,
            planes=planes, disp=disp))
    with view_box:
        _view3d(fig, ("sm_interior_major" if major else "sm_interior") + ("_plan" if disp.plan else ""), 820)
        st.caption("x = 측점, y = 깊이(+ 암반 안쪽), z = 표고. 점선 사각형 = 아래 단면 A·C의 위치. 왼쪽 범례에서 항목을 "
                   "켜고 끌 수 있습니다. 마우스는 세슘과 같습니다 (가운데 드래그 또는 Ctrl + 왼쪽 드래그 = 회전, 더블클릭 = "
                   "회전 중심 지정, Space = 모델 중심으로).")

    # panel sizes = their width on a wide screen, so the fonts match; B and C have the same height
    prof = {"uvw": section._profile_points(result)}
    label = "새 창에서 크게 보기"
    with card_a:
        _figure_window(setmap3d.plan_section(prof, J["rock"], J["joints"], J["mesh"], colour_of, v0, rng, depth_shown=jmax,
                                             frame=result.wall_frame if disp.plan else None, refs={"u": u0, "d": dpar}),
                       "sm_plan", SECTION_PANEL_H, label, width_px=1470, height_px=SECTION_PANEL_H, show_title=False)
        st.caption(f"표고 {v0:.2f} m — 아래가 암면, 위로 갈수록 암반 안쪽. "
                   + ("실제 평면: 구간 가운데의 벽 방향을 가로로 놓은 직선 거리." if disp.plan else "측점 기준: 벽 기준선을 따라 펼친 좌표."))
    with card_b:
        _figure_window(setmap3d.parallel_section(result, J["rock"], J["joints"], J["mesh"], colour_of, dpar,
                                                 refs={"v": v0, "u": u0}),
                       "sm_par", SECTION_ROW_H, label, width_px=960, height_px=SECTION_ROW_H, show_title=False)
        st.caption(f"암면에서 {dpar:.2f} m 안쪽 (벽 전체). 회색 가는 선 = 암면(깊이 0)의 트레이스 — 경사진 절리가 깊이에 "
                   "따라 어디로 옮겨 가는지 비교.")
    with card_c:
        _figure_window(setmap3d.cross_section(prof, J["joints"], J["mesh"], colour_of, u0, depth_shown=jmax,
                                              refs={"v": v0, "d": dpar}, rock=J["rock"]),
                       "sm_cross", SECTION_ROW_H, label, width_px=465, height_px=SECTION_ROW_H, show_title=False)
        st.caption(f"측점 {u0:.2f} m — 왼쪽이 암면, 오른쪽으로 암반 안쪽.")
    st.html(SAME_HEIGHT_JS, unsafe_allow_javascript=True)
