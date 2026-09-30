"""Streamlit dashboard for the Phase 1 discontinuity-extraction pipeline.

Run with:
    streamlit run src/groundwater/viz/dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running via `streamlit run` without an editable install.
_SRC_ROOT = Path(__file__).resolve().parents[2].parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from groundwater.pipeline import load_config, run_pipeline, save_results  # noqa: E402
from groundwater.validation.field_compare import summarize_errors  # noqa: E402
from groundwater.viz.stereonet import plot_stereonet  # noqa: E402

st.set_page_config(page_title="Rock Face Discontinuity Explorer", layout="wide")

MAX_3D_POINTS = 80_000


def _downsample_for_display(points: np.ndarray, colors: np.ndarray, max_points: int):
    if len(points) <= max_points:
        return points, colors
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx], colors[idx]


def render_3d_view(result):
    points = np.asarray(result.point_cloud.points)
    set_ids = result.point_labels[:, 0]
    display_points, display_set_ids = _downsample_for_display(points, set_ids, MAX_3D_POINTS)

    fig = go.Figure()
    unassigned = display_set_ids == -1
    if unassigned.any():
        p = display_points[unassigned]
        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=1.5, color="lightgray"), name="unassigned",
            )
        )
    for sid in sorted(s for s in set(display_set_ids.tolist()) if s != -1):
        p = display_points[display_set_ids == sid]
        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=2), name=f"Set {sid}",
            )
        )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trace_map(result):
    if result.orthomosaic is None:
        st.info("No orthomosaic configured for this site -- 2D trace mapping skipped.")
        return

    array = result.orthomosaic.array
    if array.shape[0] >= 3:
        rgb = np.moveaxis(array[:3], 0, -1)
        rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-6)
        img = (rgb * 255).astype(np.uint8)
    else:
        band = array[0]
        img = ((band - band.min()) / max(band.max() - band.min(), 1e-6) * 255).astype(np.uint8)

    fig = go.Figure(go.Image(z=img))
    for trace in result.traces:
        rows_cols = trace.pixel_coords
        fig.add_trace(
            go.Scatter(
                x=rows_cols[:, 1], y=rows_cols[:, 0], mode="lines",
                line=dict(color="red", width=1), showlegend=False,
            )
        )
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(result.traces)} candidate lineaments detected on the orthomosaic (red).")


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
    st.subheader("Discontinuity sets")
    st.dataframe(sets_df, use_container_width=True)

    patch_rows = []
    for d in result.discontinuity_sets:
        for p in d.patches:
            patch_rows.append(
                {
                    "set_id": p.set_id,
                    "patch_id": p.patch_id,
                    "n_points": p.n_points,
                    "dip": round(p.dip, 1),
                    "dip_direction": round(p.dip_direction, 1),
                    "roughness_rms_m": round(p.roughness_rms, 4),
                    "persistence_length_m": round(p.persistence_length, 2),
                    "area_m2": round(p.area, 3),
                }
            )
    patches_df = pd.DataFrame(patch_rows)
    st.subheader("Individual discontinuity patches")
    st.dataframe(patches_df, use_container_width=True)
    st.download_button(
        "Download patches CSV", patches_df.to_csv(index=False).encode("utf-8"),
        file_name="discontinuity_patches.csv",
    )


def render_validation(result):
    if result.field_comparison is None:
        st.info("No field_measurements path configured (or file not found) -- validation skipped.")
        return
    summary = summarize_errors(result.field_comparison)
    cols = st.columns(4)
    cols[0].metric("Field measurements", summary["n"])
    cols[1].metric("Mean angular error", f"{summary['mean_error_deg']:.1f} deg" if summary["mean_error_deg"] is not None else "-")
    cols[2].metric("RMSE", f"{summary['rmse_deg']:.1f} deg" if summary["rmse_deg"] is not None else "-")
    cols[3].metric("Max error", f"{summary['max_error_deg']:.1f} deg" if summary["max_error_deg"] is not None else "-")
    st.dataframe(result.field_comparison, use_container_width=True)


def main():
    st.title("굴착사면 불연속면 / 지하수 유출통로 탐지 대시보드")
    st.caption("Phase 1 -- RGB photogrammetry only. LiDAR/thermal fusion arrives in Phase 2 once acquired.")

    with st.sidebar:
        st.header("Site configuration")
        config_path = st.text_input("Config YAML path", value="configs/demo_site.yaml")
        run_clicked = st.button("Run pipeline", type="primary")
        st.markdown("---")
        st.markdown(
            "Loaded with **configs/demo_site.yaml** (synthetic data) so you can try the "
            "pipeline immediately. Switch to **configs/example_site.yaml** (copy it per "
            "site) and point it at your Pix4D/Metashape outputs once real data is ready."
        )

    if run_clicked:
        with st.spinner("Loading data and extracting discontinuities..."):
            cfg = load_config(config_path)
            result = run_pipeline(cfg)
            st.session_state["result"] = result
            st.session_state["output_dir"] = cfg["paths"].get("output_dir", "data/processed/output")

    result = st.session_state.get("result")
    if result is None:
        st.info("Configure a site and click **Run pipeline** to begin.")
        return

    if st.sidebar.button("Save results to disk"):
        save_results(result, st.session_state["output_dir"])
        st.sidebar.success(f"Saved to {st.session_state['output_dir']}")

    tab_3d, tab_stereo, tab_trace, tab_tables, tab_validate = st.tabs(
        ["3D View", "Stereonet", "Trace Map", "Data Tables", "Field Validation"]
    )
    with tab_3d:
        render_3d_view(result)
    with tab_stereo:
        field_dips = field_dip_dirs = None
        if result.field_comparison is not None:
            field_dips = result.field_comparison["field_dip"].to_numpy()
            field_dip_dirs = result.field_comparison["field_dip_direction"].to_numpy()
        fig = plot_stereonet(result.discontinuity_sets, field_dips, field_dip_dirs)
        st.pyplot(fig)
    with tab_trace:
        render_trace_map(result)
    with tab_tables:
        render_tables(result)
    with tab_validate:
        render_validation(result)


if __name__ == "__main__":
    main()
