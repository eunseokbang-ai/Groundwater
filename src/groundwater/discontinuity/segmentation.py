"""End-to-end discontinuity extraction: coplanarity filter -> orientation sets
(pole-density peaks) -> spatial splitting into patches -> per-patch plane fit ->
per-set aggregate stats.
"""
import numpy as np
import open3d as o3d

from .clustering import find_orientation_sets, voxel_connected_components
from .geometry import (
    DiscontinuityPatch,
    DiscontinuitySet,
    axial_mean,
    compute_spacing,
    fit_plane,
    normal_to_dip_dipdir,
    persistence,
)


def extract_discontinuities(
    pcd: o3d.geometry.PointCloud,
    surface_variation: np.ndarray | None = None,
    max_surface_variation: float = 0.1,
    grid_spacing_deg: float = 1.5,
    bandwidth_deg: float = 6.0,
    min_separation_deg: float = 20.0,
    cone_angle_deg: float = 25.0,
    min_set_fraction: float = 0.003,
    min_peak_contrast: float = 3.0,
    max_sets: int | None = None,
    link_distance: float = 0.05,
    min_patch_points: int = 30,
) -> tuple[list[DiscontinuitySet], np.ndarray]:
    """Run discontinuity extraction on a point cloud whose normals are already set.

    `surface_variation` (from normals.estimate_local_geometry) excludes non-planar
    points -- vegetation, netting edges, plane intersections -- before orientation
    analysis. Returns (discontinuity_sets, point_labels) where point_labels holds
    (set_id, patch_id) per point, (-1, -1) for points not in any discontinuity.
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals -- run normals.estimate_local_geometry() first")

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    planar = np.ones(len(points), dtype=bool)
    if surface_variation is not None:
        planar = surface_variation <= max_surface_variation
    planar_idx = np.nonzero(planar)[0]

    set_labels, _ = find_orientation_sets(
        normals[planar_idx],
        grid_spacing_deg=grid_spacing_deg,
        bandwidth_deg=bandwidth_deg,
        min_separation_deg=min_separation_deg,
        cone_angle_deg=cone_angle_deg,
        min_set_fraction=min_set_fraction,
        min_set_points=min_patch_points,
        min_peak_contrast=min_peak_contrast,
        max_sets=max_sets,
    )

    point_set_id = np.full(len(points), -1, dtype=int)
    point_patch_id = np.full(len(points), -1, dtype=int)
    discontinuity_sets: list[DiscontinuitySet] = []
    next_patch_id = 0

    for set_id in range(set_labels.max() + 1 if len(set_labels) else 0):
        set_idx = planar_idx[set_labels == set_id]
        components = voxel_connected_components(points[set_idx], link_distance)

        order = np.argsort(components, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(components[order])) + 1)
        groups = sorted((g for g in groups if len(g) >= min_patch_points), key=len, reverse=True)

        patches: list[DiscontinuityPatch] = []
        for group in groups:
            global_idx = set_idx[group]
            patch_points = points[global_idx]
            centroid, normal, rms_residual = fit_plane(patch_points)
            dip, dip_direction = normal_to_dip_dipdir(normal)
            max_chord, area = persistence(patch_points, centroid, normal)
            patches.append(
                DiscontinuityPatch(
                    set_id=set_id,
                    patch_id=next_patch_id,
                    n_points=len(global_idx),
                    centroid=centroid,
                    normal=normal,
                    dip=dip,
                    dip_direction=dip_direction,
                    roughness_rms=rms_residual,
                    persistence_length=max_chord,
                    area=area,
                )
            )
            point_set_id[global_idx] = set_id
            point_patch_id[global_idx] = next_patch_id
            next_patch_id += 1

        if not patches:
            continue

        # Orientation from the patch plane fits: point normals near patch edges and
        # plane junctions are blended and bias the mean. Scatter still comes from points.
        axis, _ = axial_mean(np.array([p.normal for p in patches]), weights=np.array([p.n_points for p in patches], float))
        _, fisher_k = axial_mean(normals[point_set_id == set_id])
        mean_dip, mean_dip_dir = normal_to_dip_dipdir(axis)
        spacing_mean, spacing_std = compute_spacing(
            np.array([p.centroid for p in patches]), axis, coplanar_tolerance=link_distance
        )
        discontinuity_sets.append(
            DiscontinuitySet(
                set_id=set_id,
                mean_dip=mean_dip,
                mean_dip_direction=mean_dip_dir,
                fisher_k=fisher_k,
                n_points=int((point_set_id == set_id).sum()),
                patches=patches,
                spacing_mean=spacing_mean,
                spacing_std=spacing_std,
            )
        )

    return discontinuity_sets, np.column_stack([point_set_id, point_patch_id])
