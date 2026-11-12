"""View '현장 전체 3D': the tiled model of the whole site with every section's result."""
import streamlit as st

from groundwater.viz.dashboard.components import _note, _open_tiles_viewer, _variant_picker
from groundwater.viz.dashboard.state import _saved, _site_dir, _w


def render_site_tiles(result=None):
    """Every section's results on the photo-textured tiled model of the whole site, in a browser tab."""
    from groundwater.viz import site3d, tiles3d

    site = _site_dir()
    secs = site3d.section_results(site) if site.is_dir() else []
    if not secs:
        st.info("구간 결과가 없습니다 (현장 폴더/구간/<이름>/분석결과).")
        return
    st.subheader(":material/public: 현장 전체 실사 3D — 타일 모델 위에 모든 구간의 결과")
    with st.container():
        _note("현장 전체 타일 모델(Metashape .3tz, 사진 텍스처)을 배경으로 구간마다의 절리면을 올려 새 탭에서 엽니다.",
              "현장 전체 타일 모델(Metashape .3tz, 사진 텍스처)을 배경으로 구간마다의 절리면을 올립니다. 같은 방향의 "
              "절리군은 구간이 달라도 같은 색(방향 ±20°로 묶음)이고, 이름에 그 절리군이 나오는 구간을 적습니다. 트레이스와 "
              "유출 가능성 칸(1 m)도 함께 올릴 수 있습니다. 타일 모델은 분석에는 쓰지 않고 보기용입니다 "
              "(인터넷 연결 필요: 3D 엔진 CesiumJS).")
        if not str(st.session_state.get("site3d_tiles") or _saved("site3d_tiles", "")).strip():
            f = tiles3d.find_tileset(site)                    # the largest .3tz of the site folder
            st.session_state["site3d_tiles"] = str(f) if f else ""
        tp = st.text_input("현장 전체 타일 모델 (.3tz 또는 tileset.json 폴더)", **_w("site3d_tiles", ""),
                           help="비워 두면 현장 폴더에서 가장 큰 .3tz를 고릅니다.")
        tiles = tiles3d.find_tileset(tp) if tp.strip() else None
        if tp.strip() and tiles is None:
            st.warning("타일 모델을 찾을 수 없습니다.")
        variant = _variant_picker(site, "site3d_variant")
        secs = site3d.section_results(site, variant)
        names = [n for n, _ in secs]
        default = [n for n in names if n != "현장 폴더"] or names
        c = st.columns([3, 1, 1, 1.3], vertical_alignment="bottom")
        pick = c[0].multiselect("구간 결과", names, default=default, key="site3d_secs",
                                help="'현장 폴더'는 현장 폴더 자체의 분석결과(서측 사진측량)입니다. 서측 라이다 결과와 같은 벽이라 "
                                     "둘 다 고르면 겹칩니다.")
        tr = c[1].toggle("트레이스", value=True, key="site3d_tr")
        sp = c[2].toggle("유출 가능성", value=True, key="site3d_sp")
        kin = c[3].toggle("파괴 가능 구역", value=False, key="site3d_kin",
                          help="'안정성 검토' 화면의 파괴 가능 구역 지도(기본 설정: 마찰각 = 분석 설정, 측방 ±20°, 1 m 셀, "
                               "벽면 평활 3 m, 근처 2 m 절리군, 식생 셀 제외)를 구간마다 계산해 벽 표면에 붙입니다. "
                               "구간마다 30초 안팎 더 걸립니다.")
        if st.button("현장 전체 장면 열기 (새 탭)", icon=":material/open_in_new:", disabled=tiles is None or not pick):
            with st.spinner("구간 결과를 모으는 중 (구간마다 10초 안팎)..."):
                scene = site3d.site_scene(site, tiles, pick, traces=tr, seepage=sp, variant=variant, kinematics=kin)
                path = tiles3d.write_scene(site / "현장전체_3D장면.json", scene)
            url = _open_tiles_viewer(path)
            st.markdown(f"새 탭이 열리지 않으면 [여기]({url})를 누르세요. 장면 파일: `{path}` "
                        f"(절리면 {sum(len(g['polys']) for g in scene['joint_groups'] if not g['label'].startswith('유출')):,}개)")
