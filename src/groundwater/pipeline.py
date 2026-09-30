"""Phase 1 end-to-end orchestration: RGB photogrammetry point cloud/mesh ->
discontinuity extraction -> (optional) 2D trace cross-check -> (optional)
field-measurement validation -> outputs on disk for the dashboard.

CLI usage:
    python -m groundwater.pipeline --config configs/example_site.yaml
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .discontinuity.geometry import DiscontinuitySet
from .discontinuity.normals import estimate_local_geometry
from .discontinuity.segmentation import extract_discontinuities
from .discontinuity.trace_mapping import Trace2D, extract_traces, match_traces_to_patches
from .io.mesh import load_mesh, mesh_to_point_cloud
from .io.point_cloud import crop_bbox, downsample, load_point_cloud
from .io.raster import Raster, load_raster
from .validation.field_compare import compare_to_field, load_field_measurements, summarize_errors


@dataclass
class PipelineResult:
    config: dict
    point_cloud: object  # open3d.geometry.PointCloud
    point_labels: np.ndarray  # (N, 2): [set_id, patch_id] per point, -1 = unassigned
    discontinuity_sets: list[DiscontinuitySet]
    traces: list[Trace2D]
    patch_trace_confirmed: np.ndarray | None
    field_comparison: pd.DataFrame | None
    orthomosaic: Raster | None


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_input_cloud(cfg: dict):
    paths = cfg["paths"]
    if paths.get("point_cloud"):
        pcd = load_point_cloud(paths["point_cloud"])
    elif paths.get("mesh"):
        mesh = load_mesh(paths["mesh"])
        pcd = mesh_to_point_cloud(mesh)
    else:
        raise ValueError("Config must provide paths.point_cloud or paths.mesh")
    return pcd


def run_pipeline(cfg: dict) -> PipelineResult:
    pcd = _load_input_cloud(cfg)

    prep = cfg.get("preprocessing", {})
    pcd = crop_bbox(pcd, prep.get("crop_bbox"))
    pcd = downsample(pcd, prep.get("voxel_size", 0.02))

    surface_variation = estimate_local_geometry(pcd, **cfg.get("normals", {}))
    discontinuity_sets, point_labels = extract_discontinuities(
        pcd,
        surface_variation=surface_variation,
        **cfg.get("orientation_sets", {}),
        **cfg.get("patches", {}),
    )

    traces: list[Trace2D] = []
    patch_trace_confirmed = None
    orthomosaic = None
    ortho_path = cfg["paths"].get("orthomosaic")
    if ortho_path and Path(ortho_path).exists():
        orthomosaic = load_raster(ortho_path)
        tm_cfg = cfg.get("trace_mapping", {})
        traces = extract_traces(
            orthomosaic,
            canny_low=tm_cfg.get("canny_low", 50),
            canny_high=tm_cfg.get("canny_high", 150),
            hough_min_line_length_px=tm_cfg.get("hough_min_line_length_px", 20),
            hough_max_line_gap_px=tm_cfg.get("hough_max_line_gap_px", 5),
        )
        all_patches = [p for dset in discontinuity_sets for p in dset.patches]
        if all_patches:
            patch_xy = np.array([p.centroid[:2] for p in all_patches])
            patch_trace_confirmed = match_traces_to_patches(traces, patch_xy, tolerance_m=prep.get("voxel_size", 0.02) * 5)

    field_comparison = None
    field_path = cfg["paths"].get("field_measurements")
    if field_path and Path(field_path).exists():
        field_df = load_field_measurements(field_path)
        field_comparison = compare_to_field(discontinuity_sets, field_df)

    return PipelineResult(
        config=cfg,
        point_cloud=pcd,
        point_labels=point_labels,
        discontinuity_sets=discontinuity_sets,
        traces=traces,
        patch_trace_confirmed=patch_trace_confirmed,
        field_comparison=field_comparison,
        orthomosaic=orthomosaic,
    )


def save_results(result: PipelineResult, output_dir: str | Path) -> None:
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sets_rows = []
    patches_rows = []
    trace_idx = 0
    for dset in result.discontinuity_sets:
        sets_rows.append(
            {
                "set_id": dset.set_id,
                "mean_dip": dset.mean_dip,
                "mean_dip_direction": dset.mean_dip_direction,
                "fisher_k": dset.fisher_k,
                "n_points": dset.n_points,
                "n_patches": len(dset.patches),
                "spacing_mean_m": dset.spacing_mean,
                "spacing_std_m": dset.spacing_std,
            }
        )
        for p in dset.patches:
            confirmed = None
            if result.patch_trace_confirmed is not None and trace_idx < len(result.patch_trace_confirmed):
                confirmed = bool(result.patch_trace_confirmed[trace_idx])
            patches_rows.append(
                {
                    "set_id": p.set_id,
                    "patch_id": p.patch_id,
                    "n_points": p.n_points,
                    "centroid_x": p.centroid[0],
                    "centroid_y": p.centroid[1],
                    "centroid_z": p.centroid[2],
                    "dip": p.dip,
                    "dip_direction": p.dip_direction,
                    "roughness_rms_m": p.roughness_rms,
                    "persistence_length_m": p.persistence_length,
                    "area_m2": p.area,
                    "trace_confirmed_2d": confirmed,
                }
            )
            trace_idx += 1

    pd.DataFrame(sets_rows).to_csv(output_dir / "discontinuity_sets.csv", index=False)
    pd.DataFrame(patches_rows).to_csv(output_dir / "discontinuity_patches.csv", index=False)

    if result.field_comparison is not None:
        result.field_comparison.to_csv(output_dir / "field_comparison.csv", index=False)
        with open(output_dir / "field_comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(summarize_errors(result.field_comparison), f, indent=2)

    colors = _colors_by_set(result.point_labels[:, 0])
    colored_pcd = o3d.geometry.PointCloud(result.point_cloud)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_dir / "classified_point_cloud.ply"), colored_pcd)

    np.save(output_dir / "point_labels.npy", result.point_labels)


def _colors_by_set(set_ids: np.ndarray) -> np.ndarray:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    colors = np.full((len(set_ids), 3), 0.6)  # gray for unassigned
    for sid in set(set_ids):
        if sid == -1:
            continue
        colors[set_ids == sid] = cmap(int(sid) % 10)[:3]
    return colors


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 1 discontinuity-extraction pipeline")
    parser.add_argument("--config", required=True, help="Path to a site YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    output_dir = cfg["paths"].get("output_dir", "data/processed/output")
    save_results(result, output_dir)

    print(f"Extracted {len(result.discontinuity_sets)} discontinuity sets")
    for dset in result.discontinuity_sets:
        print(
            f"  Set {dset.set_id}: dip={dset.mean_dip:.1f} dip_dir={dset.mean_dip_direction:.1f} "
            f"n_patches={len(dset.patches)} spacing={dset.spacing_mean}"
        )
    if result.field_comparison is not None:
        print("Field comparison summary:", summarize_errors(result.field_comparison))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
