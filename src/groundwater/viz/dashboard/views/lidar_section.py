"""View '라이다 단면': vertical sections through the LiDAR scan of the site."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from groundwater import site_folder
from groundwater.viz.dashboard.components import _choice, _figure_window, _note
from groundwater.viz.dashboard.state import _session_cache, _site_dir, _w


@st.cache_resource(max_entries=2, show_spinner=False)
def _lidar_index(path: str, mtime: float):
    from groundwater import lidar_section as LS

    return LS.load_index(path)


@st.cache_resource(max_entries=1, show_spinner=False)
def _analysis_cloud(folder: str, mtime: float):
    """(points, set id per point, {set id: 'Set k  dip/dd'}) of a saved result's analysis cloud, or None."""
    import open3d as o3d

    d = Path(folder)
    ply, lab = d / "classified_point_cloud.ply", d / "point_labels.npy"
    if not ply.exists() or not lab.exists():
        return None
    pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    sets = np.load(lab)[:, 0]
    names = {}
    if (d / "discontinuity_sets.csv").exists():
        t = pd.read_csv(d / "discontinuity_sets.csv")
        names = {int(r.set_id): f"Set {int(r.set_id)} {r.mean_dip:.0f}°/{r.mean_dip_direction:03.0f}°" for r in t.itertuples()}
    return pts, sets, names


def _section_polygons(site: Path):
    from groundwater.facade.region import SECTION_FILE, load_section

    colours = {"E": "#1f77b4", "S": "#d62728", "W": "#2ca02c"}
    out = []
    for name in site_folder.sections(site):
        poly = load_section(site_folder.section_dir(site, name) / SECTION_FILE)
        if poly:
            out.append((f"구간 {name}", np.asarray(poly), colours.get(name[:1], "#7c3aed")))
    return out


def render_lidar_section(result):
    """Vertical section through the LiDAR scan along a line drawn on the plan (or across the wall at a chainage):
    the returns by kind and the rock layer the program keeps -- to check the scan and the classification."""
    from groundwater import lidar_section as LS
    from groundwater.viz import lidar_section_view as LV

    site = _site_dir()
    lidars = site_folder.scan(site)["lidar"] if site.is_dir() else []
    if not lidars:
        st.info("이 현장 폴더에 라이다(.las) 파일이 없습니다.")
        return
    _note("평면도에 그은 선을 따라 수직으로 자른 라이다 단면입니다 — 라이다 품질과 암반 판정을 눈으로 확인하는 화면.",
          "평면도에 선을 그리면 그 선을 따라 수직으로 자른 라이다 단면을 보여 줍니다. 얇은 단면 안의 점을 반사 종류(단일 · 첫 · "
          "중간 · 마지막)로 칠하고, 프로그램이 망·식생 뒤에서 암반으로 남기는 뒤쪽 층(검정, 벽면 안 완경사는 보라)과 벽면 "
          "범위 밖이라 빼는 층(주황: 원지반·나무 줄기·바닥), 실제 절리 분석에서 절리군에 들어간 점(절리군 색 ◆)을 겹칩니다. "
          "라이다 품질과 암반 판정을 눈으로 확인하는 화면입니다.")
    names = [str(f.relative_to(site)) for f in lidars]
    from groundwater.facade.region import SECTION_FILE, load_section

    poly = load_section(site_folder.section_dir(site, st.session_state.get("_section")) / SECTION_FILE)
    default = next((n for n, f in zip(names, lidars) if site_folder.las_coverage(f, poly) >= 0.8), names[0])
    if st.session_state.get("ls_file") not in names:
        st.session_state.pop("ls_file", None)
    c = st.columns([2.6, 1.2, 1.3, 1.3], vertical_alignment="bottom")
    pick = c[0].selectbox("라이다 파일", names, **_w("ls_file", default),
                          help="처음에는 지금 구간을 덮는 파일이 골라집니다.")
    las = site / pick
    idx = _lidar_index(str(las), las.stat().st_mtime)
    if idx is None:
        st.warning(f"{las.name}은 아직 색인이 없습니다. 처음 한 번 파일 전체를 읽어 색인(작은 평면도와 위치 목록)을 만듭니다 "
                   f"({las.stat().st_size / 1e9:.0f} GB: 약 {max(1, round(las.stat().st_size / 6e9))}분). 그다음부터 단면은 1~2초입니다.")
        if st.button("색인 만들기", icon=":material/build:"):
            bar = st.progress(0.0, "라이다 색인 만드는 중...")
            LS.build_index(las, lambda f, msg: bar.progress(min(1.0, f), msg))
            _lidar_index.clear()
            st.rerun()
        return
    half = c[1].select_slider("단면 두께 (± cm)", [2, 5, 10, 20], **_w("ls_half", 5)) / 100
    show_cls = c[2].toggle("프로그램 판정 겹치기", **_w("ls_cls", True),
                           help="망·식생 뒤에서 암반으로 남기는 뒤쪽 층(검정·보라)과 벽면 범위 밖이라 빼는 뒤쪽 층(주황)")
    show_an = c[3].toggle("분석에 쓴 점 겹치기", **_w("ls_an", True), help="이 위치를 담당하는 구간의 분석 결과 점 (절리군 색)")

    modes = ["평면도에서 선 그리기", "구간·측점 (벽에 직각)", "좌표 입력"]
    mode = _choice("단면 정하기", modes, **_w("ls_mode", modes[0]))
    line = st.session_state.get("ls_line")
    secs = site_folder.sections(site)
    if mode == modes[1]:
        if not secs:
            st.info("구간 폴더(구간/<이름>/)가 없어 현장 폴더의 기준선을 씁니다.")
        cc = st.columns([1.4, 1.2, 1.0, 1.0, 1.0], vertical_alignment="bottom")
        sec = cc[0].selectbox("구간", secs or ["(현장 폴더)"], key="ls_sec")
        u = cc[1].number_input("측점 (m)", value=float(st.session_state.get("ls_u", 50.0)), step=1.0, key="ls_u")
        front = cc[2].number_input("굴착지 쪽 (m)", value=15.0, step=1.0, key="ls_front")
        back = cc[3].number_input("암반 쪽 (m)", value=40.0, step=1.0, key="ls_back")
        if cc[4].button("단면 보기", type="primary", width="stretch"):
            from groundwater.site_folder import WALL_LINE_NAME, read_wall_line
            from groundwater.facade.frame import build_wall_frame

            folder = site_folder.section_dir(site, sec if secs else None)
            poly, pit = read_wall_line(folder / WALL_LINE_NAME)
            P = np.asarray(poly, float)
            frame = build_wall_frame(np.column_stack([P, np.zeros(len(P))]), poly, pit)
            a, b = LS.perpendicular_line(frame, u, front, back)
            st.session_state["ls_line"] = line = (tuple(a), tuple(b))
    elif mode == modes[2]:
        a0, b0 = line if line else ((0.0, 0.0), (0.0, 0.0))
        cc = st.columns(5, vertical_alignment="bottom")
        ax_ = cc[0].number_input("A x", value=float(a0[0]), format="%.2f")
        ay_ = cc[1].number_input("A y", value=float(a0[1]), format="%.2f")
        bx_ = cc[2].number_input("B x", value=float(b0[0]), format="%.2f")
        by_ = cc[3].number_input("B y", value=float(b0[1]), format="%.2f")
        if cc[4].button("단면 보기", type="primary", width="stretch"):
            st.session_state["ls_line"] = line = ((ax_, ay_), (bx_, by_))

    left, right = st.columns([1, 1.35])
    with left:
        orthos = LV.find_orthos(site)
        bgs = [f.name for f in orthos] + ["라이다 윗면 색"]
        if st.session_state.get("ls_bg") not in bgs:
            st.session_state.pop("ls_bg", None)
        bg = st.selectbox("평면 바탕", bgs, **_w("ls_bg", bgs[0]),
                          help="현장 폴더의 정사영상(최근 날짜 먼저) 또는 라이다 점의 윗면 색. 정사영상은 구간 범위(+40 m)를 "
                               "고해상도로 보여 줍니다.")
        polys = _section_polygons(site)
        img, ext = _session_cache("_ls_plan", str(las), lambda: LV.plan_image(idx))
        if bg in [f.name for f in orthos]:
            if polys:                                   # the sections (+40 m), at a useful resolution
                allp = np.vstack([p for _, p, _ in polys])
                lo, hi = allp.min(axis=0) - 40, allp.max(axis=0) + 40
                ext = [lo[0], hi[0], lo[1], hi[1]]
            img, ext = _session_cache("_ls_ortho", (bg, tuple(np.round(ext, 1))),
                                      lambda: LV.ortho_image(site / bg, ext, 2000))
        fig = LV.plan_figure(img, ext, line, polys)
        ev = st.plotly_chart(fig, key="ls_plan", on_select="rerun", selection_mode="lasso",
                             config={"scrollZoom": False, "displaylogo": False})
        st.caption("올가미(lasso)로 선을 긋듯 드래그하세요: 시작점이 A, 끝점이 B. 휠 대신 위 도구 막대로 확대합니다. "
                   "노란 선 = 지금 단면.")
        lasso = (ev.selection.get("lasso") if ev and getattr(ev, "selection", None) else None) or []
        if mode == modes[0] and lasso:
            xs, ys = lasso[0].get("x") or [], lasso[0].get("y") or []
            if len(xs) >= 2 and np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) > 1.0:
                new = ((float(xs[0]), float(ys[0])), (float(xs[-1]), float(ys[-1])))
                if new != line:
                    st.session_state["ls_line"] = line = new
                    st.rerun()
    with right:
        if not line:
            st.info("왼쪽 평면도에 선을 그리거나, '구간·측점'으로 벽에 직각인 단면을 고르세요.")
            return
        a, b = np.asarray(line[0], float), np.asarray(line[1], float)
        if np.hypot(*(b - a)) < 1.0:
            st.info("선이 너무 짧습니다 (1 m 이상).")
            return
        key = (str(las), tuple(np.round(np.r_[a, b], 2)))
        with st.spinner("단면의 점을 읽는 중..."):
            slab = _session_cache("_ls_slab", key, lambda: LS.read_slab(las, idx, a, b))
        if not len(slab["xyz"]):
            st.warning("이 선 주변에 라이다 점이 없습니다.")
            return
        kinds = LS.return_kind(slab["ret"], slab["nret"])
        owner, ofolder = LS.owning_section(site, (a + b) / 2)
        res_dir = ofolder / site_folder.OUTPUT_DIRNAME
        ac = (_analysis_cloud(str(res_dir), (res_dir / "result.pkl").stat().st_mtime)
              if (show_cls or show_an) and (res_dir / "result.pkl").exists() else None)
        cls = None
        if show_cls:
            def classify():
                frame = LS.section_frame(ofolder, slab["xyz"][:: max(1, len(slab["xyz"]) // 200_000)])
                from groundwater.site_joints import find_ground

                # the section's analysis points: the whole wall column, as the analysis sees it (an oblique slab
                # holds only part of it)
                near = None
                if ac is not None:
                    lo, hi = slab["xyz"][:, :2].min(0) - 30.0, slab["xyz"][:, :2].max(0) + 30.0
                    near = ac[0][np.all((ac[0][:, :2] >= lo) & (ac[0][:, :2] <= hi), axis=1)]
                return LS.classify(slab["xyz"], frame, find_ground(site), near)
            cls = _session_cache("_ls_cls", (key, str(ofolder), "dtm", "outside", ac is not None), classify)
        analysis, set_names = None, None
        if show_an:
            if ac is not None:
                pts, sids, set_names = ac
                t = (b - a) / np.hypot(*(b - a))
                nrm = np.array([-t[1], t[0]])
                rel = pts[:, :2] - a
                ss, dd = rel @ t, rel @ nrm
                m = (np.abs(dd) <= max(half, 0.05)) & (ss >= -1) & (ss <= slab["length"] + 1)
                analysis = (ss[m], pts[m, 2], sids[m])
        title = (f"단면 A({a[0]:.1f}, {a[1]:.1f}) → B({b[0]:.1f}, {b[1]:.1f}) · 길이 {slab['length']:.1f} m · 두께 ±{half * 100:.0f} cm"
                 + (f" · 구간 {owner}" if owner else ""))
        sfig = LV.section_figure(slab, kinds, cls, half, analysis, set_names, title, show_cls)
        _figure_window(sfig, "lidar_section", 760, "새 창에서 크게 보기 — 라이다 단면 (휠 확대·축소, 점 정보)")
        stt = LV.slice_stats(slab, kinds, cls, half)
        multi = stt["첫 반사 (망·잎 등)"] + stt["중간 반사"] + stt["마지막 반사"]
        m_ = st.columns(4)
        m_[0].metric("단면 안 점", f"{stt['전체']:,}")
        m_[1].metric("다중 반사 비율", f"{100 * multi / max(1, stt['전체']):.0f}%",
                     help="망·잎을 지나 뒤에서도 돌아온 펄스의 점. 망·식생 구간에서 높습니다.")
        if cls is not None:
            m_[2].metric("암반으로 남긴 점", f"{stt['암반으로 남김']:,}",
                         help=f"그중 벽면 안 완경사(소단·사면) {stt.get('벽면 안 완경사 (암반)', 0):,}")
            m_[3].metric("벽면 밖·지면 위로 뺀 점", f"{stt['벽면 범위 밖 제외'] + stt['지면 위 제외']:,}",
                         help=f"벽면 범위 밖 {stt['벽면 범위 밖 제외']:,} · 지면 위(수목) {stt['지면 위 제외']:,}")
        _note(f"읽은 위치 블록 {slab['chunks_read']}개 · 판정은 이 위치를 담당하는 "
              + (f"구간 {owner}의 기준선 기준." if owner else "현장 폴더 기준선 기준."),
              "판정(뒤쪽 층)은 프로그램이 망·식생 셀에서 쓰는 규칙입니다: 5 cm 셀마다 가장 뒤쪽 점에서 6 cm 안, 그중 벽면 범위 "
              "안의 점(벽면 열마다 가장 아래부터 가장 위의 급경사 면(1 m 범위 경사 60° 이상)까지, 높이 4 m · 깊이 10 m 넘게 "
              "떨어지지 않고 이어진 범위; 그 사이 완경사 = 소단·사면도 암반, 그 위의 원지반·나무 줄기와 아래 바닥은 제외; "
              "경사는 깊이 0.5 m 안의 점으로만 구함)이고, 주변 5 m 안 가장 높은 지면(DTM)보다 2 m 넘게 위가 아닌 점.")
        out = (res_dir if res_dir.exists() else site) / "단면확인"
        name = f"단면_{a[0]:.0f}_{a[1]:.0f}_{b[0]:.0f}_{b[1]:.0f}"
        bc = st.columns([1, 1, 2])
        if bc[0].button("그림 저장 (PNG)", icon=":material/save:", width="stretch"):
            from groundwater.viz import static_map

            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.png").write_bytes(static_map.preview_image(sfig, width_px=1400, max_height_px=1000))
            st.success(f"저장: {out / (name + '.png')}")
        csv = pd.DataFrame({"x": slab["xyz"][:, 0], "y": slab["xyz"][:, 1], "z": slab["xyz"][:, 2], "s_m": slab["s"],
                            "d_m": slab["d"], "return": slab["ret"], "returns": slab["nret"],
                            **({"rock_layer": cls["rock"], "ledge_rear": cls["ledge"], "outside_rear": cls["outside"],
                                  "above_ground_rear": cls["air"]}
                               if cls is not None else {})})
        csv = csv[np.abs(csv["d_m"]) <= half]
        bc[1].download_button("점 내려받기 (CSV)", csv.to_csv(index=False).encode("utf-8-sig"), f"{name}.csv", "text/csv",
                              icon=":material/download:", width="stretch")
