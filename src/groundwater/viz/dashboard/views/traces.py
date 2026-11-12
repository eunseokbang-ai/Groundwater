"""View '트레이스': the traces on the facade image, their direction families, and the re-detection on
planar (wall-parallel) data."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from groundwater import report
from groundwater.discontinuity.trace_sets import families_table, family_tag
from groundwater.pipeline import save_results
from groundwater.viz.dashboard.components import MAX_FACADE_DISPLAY_PX, _figure, _map_chart, _planar_input
from groundwater.viz.dashboard.state import _result_key, _session_cache


TRACE_STATES = [
    (True, "lime", "입면 트레이스로 확인된 절리면"),
    (False, "orange", "맞는 트레이스 없음"),
    (None, "deepskyblue", "벽면과 평행 (확인 불가)"),
]
MAX_DISPLAY_TRACES = 5000         # longest traces drawn; all of them are still used in the analysis


def _facade_display(result):
    """Facade orthophoto as a JPEG data URI, downscaled to MAX_FACADE_DISPLAY_PX, and its pixel size (m)."""
    import base64
    import io

    import cv2
    from PIL import Image

    img = np.moveaxis(result.facade_rgb.array, 0, -1).copy()
    img[~np.isfinite(result.facade_depth.array[0])] = 255
    h, w = img.shape[:2]
    k = max(1.0, max(h, w) / MAX_FACADE_DISPLAY_PX)
    if k > 1:
        img = cv2.resize(img, (max(1, round(w / k)), max(1, round(h / k))), interpolation=cv2.INTER_AREA)
    # Plotly places the first row of a `source` image at y0 and goes up with a positive dy (a negative dy is
    # ignored), so the image is sent bottom row first and anchored at the lowest elevation
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(buf, "JPEG", quality=85)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, abs(result.facade_rgb.transform.a) * w / img.shape[1], abs(result.facade_rgb.transform.e) * h / img.shape[0]


def _family_sets(result) -> dict:
    """family id -> the sets it may belong to (trace_sets.family_sets), once per result."""
    from groundwater.discontinuity.trace_sets import family_sets

    return _session_cache("_fam_sets", _result_key(), lambda: {
        f.family_id: family_sets(f, result.wall_frame, result.discontinuity_sets) for f in result.trace_families})


def render_trace_map(result):
    if result.facade_rgb is None:
        st.info("점군에 색이 없어 입면 영상과 트레이스 교차검증을 건너뛰었습니다.")
        return

    # the full-resolution image and tens of thousands of separate line traces exceed Streamlit's message
    # limit on a real wall: send a downscaled JPEG and all traces as one NaN-separated line
    src, dx, dy = _session_cache("_facade_disp", _result_key(), lambda: _facade_display(result))
    t = result.facade_rgb.transform
    v_bottom = t.f + result.facade_rgb.array.shape[1] * t.e
    fig = go.Figure(go.Image(source=src, x0=t.c + dx / 2, dx=dx, y0=v_bottom + dy / 2, dy=dy, hoverinfo="skip"))
    traces = sorted(result.traces, key=lambda tr: tr.length_m, reverse=True)[:MAX_DISPLAY_TRACES]
    fam_of = {id(t): f for f in result.trace_families for t in f.members}
    groups = {}
    for tr in traces:
        f = fam_of.get(id(tr))
        key = ("manual", None) if tr.source == "manual" else ("family", f.family_id if f else None)
        groups.setdefault(key, []).append(tr)
    for (kind, fid), trs in sorted(groups.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        seg = np.full((len(trs), 3, 2), np.nan)
        seg[:, :2] = np.array([tr.coords for tr in trs])
        xy = seg.reshape(-1, 2)
        if kind == "manual":
            name, color, width = "수동 트레이스", "#facc15", 3
        elif fid is None:
            name, color, width = "방향군 밖", "#9ca3af", 1.2
        else:
            f = next(f for f in result.trace_families if f.family_id == fid)
            tag = family_tag(_family_sets(result)[fid])
            name, color, width = f"F{fid} {f.apparent_angle:.0f}° ({tag})", report.family_colour(f), 2
        fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color=color, width=width),
                                   name=name, hoverinfo="skip", connectgaps=False))

    patches = [p for d in result.discontinuity_sets for p in d.patches]
    if patches:
        uv = result.wall_frame.to_wall(np.array([p.centroid for p in patches]))
        for state, color, label in TRACE_STATES:
            idx = [i for i, p in enumerate(patches) if result.trace_confirmed.get(p.patch_id) is state]
            if not idx:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=uv[idx, 0], y=uv[idx, 1], mode="markers", name=label,
                    marker=dict(color=color, size=9, line=dict(color="black", width=1)),
                    text=[f"Set {patches[i].set_id} / patch {patches[i].patch_id}<br>"
                          f"{patches[i].dip:.0f}/{patches[i].dip_direction:03.0f}" for i in idx],
                    hovertemplate="%{text}<br>측점 %{x:.2f} m<br>표고 %{y:.2f} m<extra></extra>",
                )
            )

    fig.update_xaxes(title="측점 (m)")
    fig.update_yaxes(title="표고 (m)", autorange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"))
    _map_chart(fig, "trace_map")
    st.caption(
        f"벽면을 따라 펼친 입면 영상 (측점 {result.wall_frame.length:.1f} m). 트레이스 {len(result.traces)}개 — 색 = 방향군, "
        "회색 = 방향군 밖, 노랑 = 수동"
        + (f" (긴 것 {MAX_DISPLAY_TRACES}개만 표시)" if len(result.traces) > MAX_DISPLAY_TRACES else "")
        + ". 점 = 3차원 절리면 (트레이스 교차검증 결과별 색)."
    )


def render_traces(result):
    render_planar_redetect(result)
    render_trace_map(result)
    ft = families_table(result.trace_families, result.wall_frame, result.discontinuity_sets)
    if len(ft):
        st.subheader("트레이스 방향군")
        st.dataframe(ft, width="stretch")
        st.caption("linked_3d_set = 입면에 나타나는 각도가 맞는 3차원 절리군. '1 또는 2'처럼 둘 이상이면 그 절리군들이 이 "
                   "벽면에서 같은 각도의 선을 만들어 각도만으로는 가를 수 없다는 뜻입니다 (방향군 색은 첫 번째 절리군). "
                   "linked_3d_set이 비어 있는 방향군은 3차원 면으로 잡히지 않은 절리(주로 벽면을 가로지르는 절리)입니다. "
                   "est_dip/est_dip_direction은 트레이스들의 3차원 방향에서 추정한 값이며, 벽면이 평평하면 정할 수 없습니다.")
    angles = _figure(result, "trace_angles")
    if angles:
        st.image(angles)


def render_planar_redetect(result):
    from groundwater.facade import planar
    from groundwater.pipeline import apply_planar

    pl = result.config.get("planar")
    if pl:
        how = ("Metashape 평면 정의로 배치" if pl.get("registration") == "metashape"
               else f"점군 자동 정합, 잔차 {pl.get('registration_median_m', 0) * 100:.1f} cm")
        st.caption(f"트레이스 출처: 절벽면 평행 자료 {pl.get('ortho')} ({pl.get('resolution_m', 0.02) * 100:.0f} cm 격자, "
                   f"{how}, 입면 영상 {pl.get('facade_recoloured', 0) * 100:.0f}% 재채색)")
        return
    with st.expander("고해상도 절벽면 평행 자료로 트레이스 다시 검출"):
        st.caption("Metashape 평면 투영 정사영상·DEM이 있으면 그 자료(원본 사진 해상도)로 입면 영상을 다시 칠하고 트레이스를 "
                   "다시 찾습니다. 3차원 절리 분석은 그대로 두므로 전체 재분석 없이 1분 안팎이면 됩니다.")
        pdir = _planar_input("폴더", "planar_redetect_dir")
        if pdir.strip() and st.button("다시 검출", type="primary"):
            folder = pdir
            dem, ortho = planar.find_planar_pair(folder)
            if dem is None or ortho is None:
                st.error("그 폴더에서 평면 투영 DEM·정사영상을 찾지 못했습니다.")
                return
            with st.spinner("정합·재검출 중 (처음이면 정합에 2~3분)..."):
                apply_planar(result, folder, 0.02)
                save_results(result, st.session_state["output_dir"])
            st.session_state["planar_dir"] = folder
            st.session_state["result_key"] = ("redetect", folder, __import__("time").time())
            st.rerun()
