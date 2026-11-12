"""View '유출 가능성': the seepage-potential map from the joint network."""
from pathlib import Path

import numpy as np
import streamlit as st

from groundwater import hydro, report
from groundwater.viz.dashboard.components import _map_chart, _session_tiles_input, _tiles_scene_button
from groundwater.viz.dashboard.state import (_chainage_range, _clean_path, _range_state, _reg_crs, _result_key,
    _saved, _session_cache, _w)


def _seep_defaults():
    for k, v in hydro.WEIGHTS.items():
        st.session_state[f"seep_w_{k}"] = v


def render_seepage(result):
    st.caption("표면 절리망으로 본 지하수 유출 **가능성** 순위입니다. 격자(기본 1 m)마다 네 지표를 0~1로 바꿔 가중 합산하고, "
               "암면 격자 중 상위 15 %를 '상', 다음 35 %를 '중'으로 표시합니다. 실제 유출은 열화상·관측으로 확인해야 합니다.")
    with st.expander("지표와 가중치"):
        st.markdown("- **절리 밀도 P21**: 격자 면적당 트레이스 길이 (m/m²)\n"
                    "- **열린 정도**: 입면 깊이에서 홈의 평균 깊이 (표면 개구의 상한)\n"
                    "- **연결성**: 서로 닿는 트레이스로 이어진 절리망의 길이 — 길수록 안쪽까지 연결될 가능성\n"
                    "- **절리 옆 식생**: 트레이스 양옆 10 cm 띠 안의 초록 화소 비율 — 절리에 뿌리내린 식물은 물이 나오는 자리")
        if sum(float(st.session_state.get(f"seep_w_{k}", _saved(f"seep_w_{k}", v))) for k, v in hydro.WEIGHTS.items()) <= 0:
            _seep_defaults()                   # an all-zero set (saved earlier) would lock the tab
        c = st.columns(4)
        labels = {"p21": "절리 밀도", "openness": "열린 정도", "connectivity": "연결성", "wetness": "절리 옆 식생"}
        w = {k: c[i].slider(lab, 0.0, 1.0, step=0.05, **_w(f"seep_w_{k}", hydro.WEIGHTS[k]))
             for i, (k, lab) in enumerate(labels.items())}
        cell = st.select_slider("격자 크기 (m)", [0.5, 1.0, 2.0], **_w("seep_cell", 1.0))
        st.button("기본 가중치로", on_click=_seep_defaults, key="seep_reset")
    if sum(w.values()) <= 0:
        st.warning("가중치를 하나 이상 0보다 크게 두세요.")
        return
    key = ("_seep", _result_key(), tuple(w.values()), cell)
    if st.session_state.get("_seep_key") != key:
        with st.spinner("유출 가능성 계산 중..."):
            smap = hydro.seepage_potential(result, cell_m=cell, weights=w)
            st.session_state["_seep"] = (smap, None)
            st.session_state["_seep_key"] = key
    smap, _ = st.session_state["_seep"]
    if smap is None:
        st.info("색 정보가 없어 계산할 수 없습니다.")
        return
    _seepage_map_view(result, smap)
    st.subheader(f"유출 가능성 '상' 구역 ({len(smap.zones)}곳, 점수 순)")
    st.dataframe(smap.zones, width="stretch")
    st.caption("openness_cm = 홈 평균 깊이, network_m = 연결된 절리망 길이, plants_along_joints = 절리 옆 식생 비율, "
               "x·y·z = 구역 중심의 지도 좌표 (현장 좌표계).")
    st.download_button("구역 목록 CSV", smap.zones.to_csv(index=False).encode("utf-8-sig"), file_name="seepage_zones.csv")


def _seepage_map_view(result, smap):
    """The seepage map, zoomable (popup): background = the point-cloud facade image or the wall-parallel
    orthomosaic itself; layer = the grade, the score or one indicator; and the same on the tiled model."""
    from groundwater.facade import planar
    from groundwater.viz import seepage_view as SV

    folder = (result.config.get("planar") or {}).get("folder") or st.session_state.get("planar_dir") or ""
    folder = _clean_path(folder)
    dem, ortho = planar.find_planar_pair(folder) if folder and Path(folder).is_dir() else (None, None)
    reg = planar.load_registration(folder, _reg_crs(result)) if ortho is not None else None
    bgs = ["입면 영상 (점군)"] + (["고해상도 정사영상 (벽면 평행)"] if reg is not None else [])
    if st.session_state.get("seep_bg", _saved("seep_bg", bgs[0])) not in bgs:
        st.session_state["seep_bg"] = bgs[0]
    c = st.columns([1.5, 3.2, 1.1], vertical_alignment="bottom")
    bg = c[0].radio("바탕 영상", bgs, **_w("seep_bg", bgs[0]),
                    help="고해상도 정사영상: 올린 벽면 평행 정사영상(2~3 mm)을 벽면 좌표에 그대로 옮겨 그립니다. "
                         "벽 전체는 약 6.5 mm, 측점 구간을 좁히면 원래 해상도까지.")
    layer = c[1].radio("표시할 값", list(SV.LAYERS), format_func=lambda k: SV.LAYERS[k][0], horizontal=True,
                       **_w("seep_layer", "grade"),
                       help="등급 외에는 셀마다 그 값을 색으로 (암면 셀만, 2~98 백분위로 색 범위). 새 창에서 셀 위에 "
                            "마우스를 올리면 네 지표 값이 모두 나옵니다.")
    op = c[2].slider("덮개 불투명도", 0.2, 1.0, step=0.05, **_w("seep_op", 0.6))
    if bg.startswith("고해상도"):
        lo, hi = (float(np.floor(x)) for x in _chainage_range(result))
        key = _range_state("seep_range", lo, hi + 1, (lo, hi + 1))
        rng = st.slider("정사영상 측점 구간 (m) — 좁힐수록 정사영상 해상도가 높아짐", lo, hi + 1, step=0.5, key=key)
        with st.spinner("정사영상을 벽면 좌표로 옮기는 중 (구간마다 처음 한 번, 벽 전체 15초 안팎)..."):
            img, ext, res = _session_cache("_seep_ortho", (_result_key(), folder, tuple(rng)),
                                           lambda: planar.wall_image(result, ortho, reg["matrix"], rng, max_px=8000))
        note = f"바탕: 벽면 평행 정사영상, 화소 {res * 1000:.1f} mm"
    else:
        img, ext = _session_cache("_seep_facade", _result_key(), lambda: report.facade_preview(result, 6000))
        note = "바탕: 점군에서 만든 입면 영상"
    fig = SV.map_figure(smap, layer, img, ext, op)
    _map_chart(fig, "seep_map")
    st.caption(f"{note}. 흰 사각형 = 점수 상위 10개 '상' 구역 (번호 = 점수 순). 지표는 셀마다의 값입니다.")

    if st.toggle(":material/view_in_ar: 타일 모델에 표시", **_w("seep_tiles", False)):
        st.caption("등급과 네 지표, 종합 점수를 각각 이미지로 실사 타일 모델 표면에 붙여 새 탭에서 엽니다. 위에서 고른 값이 "
                   "먼저 켜지고, 그 화면의 '표면에 붙인 이미지'에서 바꿔 볼 수 있습니다 (인터넷 연결 필요: CesiumJS).")
        _tiles_scene_button(result, _session_tiles_input("seep"), "seep", "seepage_scene_tiles.json",
                            "유출 가능성 — 타일 모델", lambda: SV.tiles_overlays(result, smap, layer), smap=smap)
