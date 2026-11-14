"""View '유출 가능성': the seepage-potential map from the joint network."""
from pathlib import Path

import numpy as np
import streamlit as st

from groundwater import hydro, report
from groundwater.viz.dashboard.components import (_choice, _map_chart, _session_tiles_input, _table,
    _tiles_scene_button)
from groundwater.viz.dashboard.state import (_chainage_range, _clean_path, _option, _range_state, _reg_crs,
    _result_key, _saved, _session_cache, _w)


def _seep_defaults():
    for k, v in hydro.WEIGHTS.items():
        st.session_state[f"seep_w_{k}"] = v


def render_seepage(result):
    st.caption("표면 절리망으로 본 지하수 유출 **가능성** 순위입니다. 격자(기본 1 m)마다 네 지표를 0~1로 바꿔 가중 합산하고, "
               "암면 격자 중 상위 15 %를 '상', 다음 35 %를 '중'으로 표시합니다. 실제 유출은 열화상·관측으로 확인해야 합니다.")
    with st.expander("지표와 가중치"):
        st.markdown("- **절리 밀도 P21**: 격자 면적당 트레이스 길이 (m/m²)\n"
                    "- **열린 정도**: 입면 깊이에서 폭 15 cm 이하 홈의 평균 깊이 — 암면이 벌어지거나 떨어져 나간 정도의 "
                    "간접 지표입니다 (절리의 개구폭을 잰 값이 아님)\n"
                    "- **연결성**: 서로 닿는 트레이스로 이어진 절리망의 길이 — 벽면에서 본 2차원 연결이며, 암반 안쪽으로의 "
                    "연결은 가정입니다\n"
                    "- **절리 옆 식생**: 트레이스 양옆 10 cm 띠 안의 초록 화소 비율 — 절리를 따라 자란 식물은 물기가 있다는 "
                    "간접 신호입니다 (빗물만으로도 자라고, 촬영 계절에 따라 달라짐)\n\n"
                    "지하수위·침출수위는 점수에 들어 있지 않습니다: 수위보다 높은 곳은 절리가 많아도 침출수가 나올 수 없으니, "
                    "구역의 표고를 수위와 함께 보세요.\n\n"
                    "**낙석방지망 아래**는 위 지표를 그대로 쓸 수 없습니다. 망에 가려진 트레이스는 분석에서 빼므로 절리 밀도·"
                    "연결성·절리 옆 식생이 없고, 입면 깊이도 암반이 아니라 망의 앞면입니다 (익산 벽에서 망은 암반보다 중앙값 "
                    "25~78 cm 앞). 그래서 망 아래 칸은 **따로 순위**를 매깁니다: 망 뒤 암반에서 돌아온 라이다 점으로 만든 "
                    "암반면의 홈 깊이(폭 30 cm 이하) 하나가 기준이고, 가중치와 무관합니다. 이 값은 둘 다 잴 수 있는 맨 "
                    "암반에서 기존 '열린 정도'와 순위가 잘 맞습니다 (구간별 순위상관 0.87~0.93). 라이다 절리면 경계 밀도를 "
                    "절리 밀도 대신 쓰는 방법도 시험했지만 맨 암반에서 사진 트레이스와 맞지 않아 (순위상관 0.0~0.2) 쓰지 "
                    "않았습니다.")
        if sum(float(st.session_state.get(f"seep_w_{k}", _saved(f"seep_w_{k}", v))) for k, v in hydro.WEIGHTS.items()) <= 0:
            _seep_defaults()                   # an all-zero set (saved earlier) would lock the tab
        c = st.columns(4)
        labels = {"p21": "절리 밀도", "openness": "열린 정도", "connectivity": "연결성", "wetness": "절리 옆 식생"}
        w = {k: c[i].slider(lab, 0.0, 1.0, step=0.05, **_w(f"seep_w_{k}", hydro.WEIGHTS[k]))
             for i, (k, lab) in enumerate(labels.items())}
        cell = st.select_slider("격자 크기 (m)", [0.5, 1.0, 2.0], **_w("seep_cell", 1.0))
        st.button("기본 가중치로", on_click=_seep_defaults, key="seep_reset")
    modes = {"lidar": "따로 순위 (라이다 암반 홈 깊이)", "skip": "판정 안 함", "front": "이전 방식 (맨 암반과 함께)"}
    under_net = _choice("낙석방지망 아래 칸", list(modes), format_func=modes.get, **_w("seep_under_net", "lidar"),
                        help="따로 순위: 망 아래 칸끼리, 망 뒤 암반(라이다)의 홈 깊이로 상위 15 %를 '상'으로 (보라색). "
                             "판정 안 함: 망 아래는 비워 두고 맨 암반만. 이전 방식: 망의 앞면에서 잰 값으로 맨 암반과 한 "
                             "순위에 넣던 방식 — 이전 발표자료의 그림과 맞춰 볼 때만 쓰세요.")
    if sum(w.values()) <= 0:
        st.warning("가중치를 하나 이상 0보다 크게 두세요.")
        return
    key = ("_seep", _result_key(), tuple(w.values()), cell, under_net)
    if st.session_state.get("_seep_key") != key:
        with st.spinner("유출 가능성 계산 중..."):
            smap = hydro.seepage_potential(result, cell_m=cell, weights=w, under_net=under_net)
            st.session_state["_seep"] = (smap, None)
            st.session_state["_seep_key"] = key
    smap, _ = st.session_state["_seep"]
    if smap is None:
        st.info("색 정보가 없어 계산할 수 없습니다.")
        return
    _seepage_map_view(result, smap)
    st.subheader(f"유출 가능성 '상' 구역 ({len(smap.zones)}곳, 점수 순)")
    if smap.net is not None:
        st.caption("'기준' 열: **맨 암반** 구역이 먼저 (네 지표의 점수 순), 그다음 **망 아래** 구역 (라이다 암반 홈 깊이 순). "
                   "두 묶음의 점수는 서로 비교할 수 없습니다.")
    _table(smap.zones)
    st.caption("열 제목에 마우스를 올리면 그 값의 뜻이 나옵니다. X·Y·Z = 구역 중심의 지도 좌표. 점수와 등급은 **이 벽 안에서의 "
               "상대 순위**라서 다른 구간의 점수와 바로 비교할 수 없고, 구역 번호는 가중치·격자를 바꾸면 달라집니다 "
               "(구역을 가리킬 때는 측점·표고를 함께 적으세요).")
    st.download_button("구역 목록 CSV", smap.zones.to_csv(index=False).encode("utf-8-sig"), file_name="seepage_zones.csv")


def _seepage_map_view(result, smap):
    """The seepage map, zoomable (popup): background = the point-cloud facade image or the wall-parallel
    orthomosaic itself; layer = the grade, the score or one indicator; and the same on the tiled model."""
    from groundwater.facade import planar
    from groundwater.viz import seepage_view as SV

    folder = (result.config.get("planar") or {}).get("folder") or _option("planar_dir") or ""
    folder = _clean_path(folder)
    dem, ortho = planar.find_planar_pair(folder) if folder and Path(folder).is_dir() else (None, None)
    reg = planar.load_registration(folder, _reg_crs(result)) if ortho is not None else None
    bgs = ["입면 영상 (점군)"] + (["고해상도 정사영상 (벽면 평행)"] if reg is not None else [])
    if st.session_state.get("seep_bg", _saved("seep_bg", bgs[0])) not in bgs:
        st.session_state["seep_bg"] = bgs[0]
    layer = _choice("표시할 값", list(SV.LAYERS), format_func=lambda k: SV.LAYERS[k][0], **_w("seep_layer", "grade"),
                    help="등급 외에는 셀마다 그 값을 색으로 (암면 셀만, 2~98 백분위로 색 범위). 새 창에서 셀 위에 "
                         "마우스를 올리면 네 지표 값이 모두 나옵니다.")
    c = st.columns([3, 1.2], vertical_alignment="bottom")
    bg = _choice("바탕 영상", bgs, c[0], **_w("seep_bg", bgs[0]),
                 help="고해상도 정사영상: 올린 벽면 평행 정사영상(2~3 mm)을 벽면 좌표에 그대로 옮겨 그립니다. "
                      "벽 전체는 약 6.5 mm, 측점 구간을 좁히면 원래 해상도까지.")
    op = c[1].slider("덮개 불투명도", 0.2, 1.0, step=0.05, **_w("seep_op", 0.6))
    low = st.toggle("'하' 칸도 칠하기", disabled=layer != "grade", **_w("seep_low", False),
                    help="등급 지도에서 색이 없는 곳은 '하'이거나 **판정하지 않은 곳**(식생, 자료 없음, 절리면이 거의 잡히지 않은 "
                         "칸)입니다. 켜면 '하' 칸을 옅은 회청색으로 칠해, 아무 색도 없는 곳이 판정하지 않은 곳임을 보여 줍니다.")
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
    fig = SV.map_figure(smap, layer, img, ext, op, show_low=low)
    _map_chart(fig, "seep_map")
    st.caption(f"{note}. 흰 사각형 = 점수 상위 10개 '상' 구역 (번호 = 점수 순)"
               + (", 보라 점선 사각형 = 망 아래 상위 10개 구역" if smap.net is not None else "") + ". 지표는 셀마다의 값입니다. "
               f"점수를 매긴 암면은 {smap.rock.sum() * smap.cell_m ** 2:,.0f} m²이고, 그 밖(식생·자료 없음·절리면이 거의 "
               "잡히지 않은 칸)은 판정하지 않았습니다.")
    if smap.net is not None:                   # the ranking under the net: what it is, and how much of the wall
        n_net, n_all = int(smap.net.sum()), int(smap.rock.sum())
        st.info(f"**점수를 매긴 칸의 {n_net / max(n_all, 1):.0%}가 낙석방지망 아래입니다 (보라색).** 그곳은 맨 암반과 따로, 망 "
                "아래 칸끼리 순위를 매겼습니다 — 망 뒤 암반(라이다)의 홈 깊이 하나가 기준이며, 그중 상위 15 %가 '망 아래 "
                "상'입니다. 근거가 지표 하나뿐이라 맨 암반의 등급보다 약하고, 두 순위를 같은 잣대로 비교할 수 없습니다. "
                "열화상 이상 부분도 망 아래에서는 뽑지 않으므로, 망 아래 '상'은 현장 확인이 먼저 필요한 후보입니다.",
                icon=":material/grid_on:")
    elif smap.under_net == "skip":
        st.info("낙석방지망 아래 칸은 판정하지 않았습니다 (위 '낙석방지망 아래 칸'에서 바꿀 수 있음).", icon=":material/grid_on:")
    else:                                      # the earlier way, or a result without a net mask
        net = _session_cache("_seep_net", st.session_state.get("_seep_key"), lambda: hydro.netting_cells(result, smap))
        if net is not None and net.any():
            high = int(((smap.classes == 2) & net).sum())
            st.warning(f"**이전 방식입니다: 점수를 매긴 칸의 {net.sum() / max(smap.rock.sum(), 1):.0%}가 낙석방지망 아래인데 맨 "
                       "암반과 한 순위에 들어 있습니다.** 그곳은 트레이스 지표가 거의 0이고 '열린 정도'도 암반이 아니라 망의 "
                       f"앞면에서 잰 값입니다. '상' 칸 {int((smap.classes == 2).sum()):,}개 중 {high:,}개가 망 아래입니다.",
                       icon=":material/warning:")

    if st.toggle(":material/view_in_ar: 타일 모델에 표시", **_w("seep_tiles", False)):
        st.caption("등급과 네 지표, 종합 점수를 각각 이미지로 실사 타일 모델 표면에 붙여 새 탭에서 엽니다. 위에서 고른 값이 "
                   "먼저 켜지고, 그 화면의 '표면에 붙인 이미지'에서 바꿔 볼 수 있습니다 (인터넷 연결 필요: CesiumJS).")
        _tiles_scene_button(result, _session_tiles_input("seep"), "seep", "seepage_scene_tiles.json",
                            "유출 가능성 — 타일 모델", lambda: SV.tiles_overlays(result, smap, layer), smap=smap)
