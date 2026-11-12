"""Views of the '결과물' group: the tables, the field validation, the report and the reference-image overlay."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from groundwater import report
from groundwater.validation.field_compare import summarize_errors
from groundwater.viz.dashboard.components import _map_chart
from groundwater.viz.dashboard.state import _reg_crs


def render_tables(result):
    sets_rows = [
        {
            "set_id": d.set_id,
            "mean_dip": round(d.mean_dip, 1),
            "mean_dip_direction": round(d.mean_dip_direction, 1),
            "fisher_k": round(d.fisher_k, 1) if np.isfinite(d.fisher_k) else None,
            "n_points": d.n_points,
            "n_patches": len(d.patches),
            "spacing_mean_m": round(d.spacing_mean, 3) if d.spacing_mean else None,
        }
        for d in result.discontinuity_sets
    ]
    sets_df = pd.DataFrame(sets_rows)
    st.subheader("절리군")
    st.dataframe(sets_df, width="stretch")

    patches = [p for d in result.discontinuity_sets for p in d.patches]
    # all centroids to wall coordinates in one call (one call per face took ~1 s per rerun on a full wall)
    uv = result.wall_frame.to_wall(np.array([p.centroid for p in patches])) if patches else np.zeros((0, 3))
    patch_rows = [
        {
            "set_id": p.set_id,
            "patch_id": p.patch_id,
            "chainage_m": round(float(u), 2),
            "elevation_m": round(float(v), 2),
            "trace_confirmed_2d": result.trace_confirmed.get(p.patch_id),
            "n_points": p.n_points,
            "dip": round(p.dip, 1),
            "dip_direction": round(p.dip_direction, 1),
            "roughness_rms_m": round(p.roughness_rms, 4),
            "persistence_length_m": round(p.persistence_length, 2),
            "area_m2": round(p.area, 3),
        }
        for p, (u, v, _) in zip(patches, uv)
    ]
    patches_df = pd.DataFrame(patch_rows)
    st.subheader("개별 절리면")
    st.dataframe(patches_df, width="stretch")
    st.download_button(
        "절리면 목록 CSV 내려받기", patches_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="discontinuity_patches.csv",
    )


FIELD_TEMPLATE = ("id,dip,dip_direction,x,y,z,note\n"
                  "J01,78,292,,,,예) 경사 판상 절리 (Set 1 추정)\n"
                  "J02,8,297,,,,예) 층리\n")


def render_field_help():
    st.info("현장 실측 자료가 없어 방향 정확도를 검증하지 않았습니다.")
    st.markdown(
        "**권장 측정** — 클리노미터(또는 스마트폰 앱)로 절리면의 **경사/경사방향**을 잽니다.\n"
        "- 절리군마다 **10~20개**, 벽면 여러 구간에 고르게\n"
        "- **벽면을 가로지르는 수직 절리**(입면에 선으로만 보이는 것)를 꼭 포함 — 드론 3차원 분석이 가장 약한 방향입니다\n"
        "- 좌표(x, y, z)는 있으면 위치별 비교에 쓰고, 없어도 됩니다 (EPSG 좌표계는 점군과 같게)\n"
        "- 파일을 현장 폴더에 넣으면 다음 분석부터 자동으로 비교합니다")
    st.download_button("실측 입력 양식 내려받기 (CSV)", FIELD_TEMPLATE.encode("utf-8-sig"), file_name="field_joints.csv")
    folder = st.session_state.get("site_folder")
    up = st.file_uploader("작성한 실측 CSV를 현장 폴더에 넣기", type=["csv"])
    if up is not None and folder and Path(folder).is_dir():
        dest = Path(folder) / "field_joints.csv"
        if st.session_state.get("_field_upload") != up.file_id:      # once per uploaded file, not every rerun
            dest.write_bytes(up.getvalue())
            st.session_state["_field_upload"] = up.file_id
        st.success(f"저장했습니다: {dest} — 분석을 다시 실행하면 비교 결과가 나옵니다.")


def render_validation(result):
    if result.field_comparison is None:
        render_field_help()
        return
    summary = summarize_errors(result.field_comparison)
    cols = st.columns(4)
    cols[0].metric("현장 실측 수", summary["n"])
    cols[1].metric("평균 각도 오차", f"{summary['mean_error_deg']:.1f}°" if summary["mean_error_deg"] is not None else "-")
    cols[2].metric("RMSE", f"{summary['rmse_deg']:.1f}°" if summary["rmse_deg"] is not None else "-")
    cols[3].metric("최대 오차", f"{summary['max_error_deg']:.1f}°" if summary["max_error_deg"] is not None else "-")
    st.dataframe(result.field_comparison, width="stretch")


def render_report(result):
    out_dir = st.session_state["output_dir"]
    rp = Path(out_dir) / report.REPORT_NAME
    if st.button("보고서 다시 만들기"):
        report.build_report(result, out_dir)
    if rp.exists():
        st.download_button("분석 보고서 내려받기 (HTML)", rp.read_bytes(), file_name=f"{result.config.get('site_name', '')}_{report.REPORT_NAME}")
        st.caption(f"저장 위치: {rp} — 브라우저로 열고 인쇄하면 PDF로 만들 수 있습니다.")
    else:
        st.info("아직 보고서가 없습니다. '보고서 다시 만들기'를 누르세요.")


def render_reference_overlays(result, folder):
    """Reference images (e.g. the petroglyph tracing) on the planar orthomosaic / DSM, in place."""
    import plotly.graph_objects as go

    from groundwater.facade import planar
    from groundwater.facade import reference_overlays as ro

    data = ro.load(folder)
    entries = data["overlays"]
    st.caption("현장 특별 자료: 참고 이미지(실측도·사진)를 절벽면 평행 정사영상·DSM 위 제자리에 겹쳐 봅니다. 사진은 정사영상과 "
               "특징점 정합(호모그래피), 실측도는 형상 대응점으로 맞춘 박판 스플라인입니다. 위치 정확도: 사진 수 cm, 실측도 "
               "수 cm~10 cm (암면 아래쪽 사진 밖 구간은 더 큼). 타일 모델 뷰어(그라우팅 입력 탭)에서도 같은 이미지가 표면에 붙습니다.")
    c = st.columns([2, 1, 1])
    labels = [e["name"] for e in entries]
    pick = c[0].multiselect("겹칠 이미지", labels, default=[labels[0]], key="ref_pick")
    bg = c[1].radio("바탕", ["정사영상", "DSM 음영기복", "DSM 미세기복"], key="ref_bg",
                    help="미세기복 = DSM에서 3 cm보다 넓은 굴곡을 뺀 것. 균열·턱은 보이지만, 몇 mm 깊이로 쪼아 새긴 암각화 선은 "
                         "드론 DSM(2.83 mm 격자)에서 거의 드러나지 않습니다 — 위치는 겹친 이미지로 확인하세요.")
    alpha = c[2].slider("이미지 불투명도", 0.0, 1.0, 0.75, 0.05, key="ref_alpha")
    res = c[2].select_slider("해상도 (mm)", [3, 5, 8], value=5, key="ref_res") / 1000
    chosen = [e for e in entries if e["name"] in pick] or entries[:1]
    bounds = ro.union_bounds(folder, chosen, 0.4)
    kind = {"정사영상": "ortho", "DSM 음영기복": "hillshade", "DSM 미세기복": "relief"}[bg]
    key = ("_ref", folder, tuple(e["name"] for e in chosen), kind, res)
    cache = st.session_state.setdefault("_ref_cache", {})
    if key not in cache:
        with st.spinner("이미지 맞추는 중..."):
            cache.clear()
            base = ro.background(folder, bounds, res, kind)
            cache[key] = (base, [ro.render(folder, e, bounds, res) for e in chosen])
    base, ovs = cache[key]
    img = ro.composite(base, ovs, alpha)
    # axes in chainage / elevation (the plane is parallel to the wall: corners mapped, linear in between)
    x0, y0, x1, y1 = bounds
    u_ax = v_ax = None
    reg = planar.load_registration(folder, _reg_crs(result))
    if reg is not None:
        T = np.array(reg["matrix"])
        # the plane's z is the DSM value (metres along its normal): the corners at the wall's own depth
        dz = ro._read_window(planar.find_planar_pair(folder)[0], bounds, 0.05, [1])[0].astype(float)
        dz = dz[np.isfinite(dz) & (dz > -1000) & (dz != 0)]
        z = float(np.median(dz)) if len(dz) else 0.0
        corners = np.array([[x0, y0, z], [x1, y0, z], [x0, y1, z]])
        uvw = result.wall_frame.to_wall(corners @ T[:3, :3].T + T[:3, 3])
        u_ax, v_ax = (uvw[0, 0], uvw[1, 0]), (uvw[0, 1], uvw[2, 1])
    # sent as a JPEG (a raw array is ~7 MB of chart data); row 0 is the top, so flipped and anchored at the bottom
    import base64

    import cv2

    jpg = cv2.imencode(".jpg", np.ascontiguousarray(img[::-1, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])[1]
    fig = go.Figure(go.Image(source="data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(),
                             x0=u_ax[0] if u_ax else x0,
                             dx=((u_ax[1] - u_ax[0]) if u_ax else (x1 - x0)) / img.shape[1],
                             y0=v_ax[0] if v_ax else y0, dy=((v_ax[1] - v_ax[0]) if v_ax else (y1 - y0)) / img.shape[0],
                             hoverinfo="skip"))
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="측점 (m)" if u_ax else "평면 x (m)"),
                      yaxis=dict(title="표고 (m)" if v_ax else "평면 y (m)", autorange=True, scaleanchor="x"))
    _map_chart(fig, "overlay")
    notes = [f"- **{e['name']}** ({e['file']}): " + ("특징점 정합 " + str(e.get("inliers")) + "점" if e["method"] == "homography"
                                                   else f"형상 대응점 {len(e['points'])}개") + (f" — {e['note']}" if e.get("note") else "")
             for e in chosen]
    st.markdown("\n".join(notes))
    if st.button("현재 그림을 결과 폴더에 저장 (PNG)", key="ref_save"):
        import cv2

        out = Path(st.session_state.get("output_dir") or folder) / f"암각화_{kind}.png"
        cv2.imencode(".png", img[:, :, ::-1])[1].tofile(str(out))
        st.success(f"저장했습니다: {out}")
