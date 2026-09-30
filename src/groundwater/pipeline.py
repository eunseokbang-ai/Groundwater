"""Phase 1 end-to-end orchestration: RGB photogrammetry point cloud/mesh ->
discontinuity extraction -> wall frame + facade orthophoto -> 2D trace
cross-check -> (optional) field-measurement validation -> outputs on disk for
the dashboard.

CLI usage:
    python -m groundwater.pipeline --config configs/example_site.yaml
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .discontinuity.geometry import DiscontinuitySet
from .discontinuity.normals import estimate_local_geometry
from .discontinuity.segmentation import extract_discontinuities
from .discontinuity.trace_mapping import Trace2D, extract_traces, match_traces_to_patches
from .facade.frame import WallFrame, build_wall_frame
from .facade.orthophoto import rasterize_facade
from .io.mesh import load_mesh, mesh_to_point_cloud
from .io.point_cloud import crop_bbox, downsample, load_point_cloud
from .io.raster import Raster, write_raster
from .validation.field_compare import compare_to_field, load_field_measurements, summarize_errors

FOOTPRINT_SAMPLE = 500


@dataclass
class PipelineResult:
    config: dict
    point_cloud: object  # open3d.geometry.PointCloud (analysis resolution)
    point_labels: np.ndarray  # (N, 2): [set_id, patch_id] per point, -1 = unassigned
    discontinuity_sets: list[DiscontinuitySet]
    wall_frame: WallFrame
    facade_rgb: Raster | None
    facade_depth: Raster | None
    field_comparison: pd.DataFrame | None
    traces: list[Trace2D] = field(default_factory=list)
    trace_confirmed: dict = field(default_factory=dict)  # patch_id -> True / False / None (n/a)


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
    cloud = _load_input_cloud(cfg)
    prep = cfg.get("preprocessing", {})
    cloud = crop_bbox(cloud, prep.get("crop_bbox"))
    pcd = downsample(cloud, prep.get("voxel_size", 0.02))

    surface_variation = estimate_local_geometry(pcd, **cfg.get("normals", {}))
    discontinuity_sets, point_labels = extract_discontinuities(
        pcd,
        surface_variation=surface_variation,
        **cfg.get("orientation_sets", {}),
        **cfg.get("patches", {}),
    )

    frame_cfg = cfg.get("wall_frame", {})
    frame = build_wall_frame(np.asarray(pcd.points), frame_cfg.get("polyline"), frame_cfg.get("viewer_point"))

    field_comparison = None
    field_path = cfg["paths"].get("field_measurements")
    if field_path and Path(field_path).exists():
        field_comparison = compare_to_field(discontinuity_sets, load_field_measurements(field_path))

    result = PipelineResult(
        config=cfg,
        point_cloud=pcd,
        point_labels=point_labels,
        discontinuity_sets=discontinuity_sets,
        wall_frame=frame,
        facade_rgb=None,
        facade_depth=None,
        field_comparison=field_comparison,
    )
    if cloud.has_colors():
        # Full-density cloud, not the downsampled analysis cloud: a sharper image for trace detection.
        result.facade_rgb, result.facade_depth = rasterize_facade(
            frame.to_wall(np.asarray(cloud.points)), np.asarray(cloud.colors), **cfg.get("facade", {})
        )
        result.traces = extract_traces(
            result.facade_rgb, valid_mask=np.isfinite(result.facade_depth.array[0]), **cfg.get("trace_mapping", {})
        )
        result.trace_confirmed = _check_traces(result, **cfg.get("trace_matching", {}))
    return result


def _check_traces(result: PipelineResult, tolerance: float = 0.1, max_angle_deg: float = 20.0) -> dict:
    patches = [p for dset in result.discontinuity_sets for p in dset.patches]
    if not patches:
        return {}
    points = np.asarray(result.point_cloud.points)
    patch_ids = result.point_labels[:, 1]
    order = np.argsort(patch_ids, kind="stable")
    sorted_ids = patch_ids[order]
    rng = np.random.default_rng(0)

    footprints, expected = [], []
    for p in patches:
        lo, hi = np.searchsorted(sorted_ids, [p.patch_id, p.patch_id + 1])
        idx = order[lo:hi]
        if len(idx) > FOOTPRINT_SAMPLE:
            idx = rng.choice(idx, FOOTPRINT_SAMPLE, replace=False)
        footprints.append(result.wall_frame.to_wall(points[idx])[:, :2])
        u = result.wall_frame.to_wall(p.centroid)[0, 0]
        expected.append(result.wall_frame.apparent_trace_direction(p.normal, u))

    confirmed = match_traces_to_patches(result.traces, footprints, expected, tolerance, max_angle_deg)
    return {p.patch_id: c for p, c in zip(patches, confirmed)}


def save_results(result: PipelineResult, output_dir: str | Path) -> None:
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = result.wall_frame

    sets_rows = []
    patches_rows = []
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
            u, v, w = frame.to_wall(p.centroid)[0]
            patches_rows.append(
                {
                    "set_id": p.set_id,
                    "patch_id": p.patch_id,
                    "n_points": p.n_points,
                    "chainage_m": u,
                    "elevation_m": v,
                    "offset_m": w,
                    "centroid_x": p.centroid[0],
                    "centroid_y": p.centroid[1],
                    "centroid_z": p.centroid[2],
                    "dip": p.dip,
                    "dip_direction": p.dip_direction,
                    "roughness_rms_m": p.roughness_rms,
                    "persistence_length_m": p.persistence_length,
                    "area_m2": p.area,
                    "face_parallel": frame.apparent_trace_direction(p.normal, u) is None,
                    "trace_confirmed_2d": result.trace_confirmed.get(p.patch_id),
                }
            )

    pd.DataFrame(sets_rows).to_csv(output_dir / "discontinuity_sets.csv", index=False)
    pd.DataFrame(patches_rows).to_csv(output_dir / "discontinuity_patches.csv", index=False)

    with open(output_dir / "wall_frame.json", "w", encoding="utf-8") as f:
        json.dump({"polyline": frame.vertices.tolist(), "length_m": frame.length}, f, indent=2)
    if result.facade_rgb is not None:
        write_raster(output_dir / "facade_rgb.tif", result.facade_rgb)
        write_raster(output_dir / "facade_depth.tif", result.facade_depth)

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

    print(f"Wall frame: {len(result.wall_frame.vertices)}-vertex polyline, {result.wall_frame.length:.1f} m of chainage")
    print(f"Extracted {len(result.discontinuity_sets)} discontinuity sets")
    for dset in result.discontinuity_sets:
        print(
            f"  Set {dset.set_id}: dip={dset.mean_dip:.1f} dip_dir={dset.mean_dip_direction:.1f} "
            f"n_patches={len(dset.patches)} spacing={dset.spacing_mean}"
        )
    if result.facade_rgb is None:
        print("Point cloud has no colors: facade orthophoto and trace cross-check skipped")
    else:
        states = list(result.trace_confirmed.values())
        print(
            f"Facade traces: {len(result.traces)} detected; patches confirmed={states.count(True)} "
            f"unconfirmed={states.count(False)} face-parallel(n/a)={states.count(None)}"
        )
    if result.field_comparison is not None:
        print("Field comparison summary:", summarize_errors(result.field_comparison))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
