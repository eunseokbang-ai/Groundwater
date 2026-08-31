"""End-to-end discontinuity extraction: orientation clustering -> spatial
splitting into patches -> per-patch plane fit -> per-set aggregate stats.
"""
import numpy as np
import open3d as o3d

from .clustering import cluster_orientations, cluster_spatial
from .geometry import (
    DiscontinuityPatch,
    DiscontinuitySet,
    compute_spacing,
    fit_plane,
    mean_orientation,
    normal_to_dip_dipdir,
    persistence,
)


def extract_discontinuities(
    pcd: o3d.geometry.PointCloud,
    orientation_eps: float = 0.26,
    orientation_min_samples: int = 30,
    spatial_eps: float = 0.15,
    spatial_min_samples: int = 20,
    min_patch_points: int = 30,
) -> tuple[list[DiscontinuitySet], np.ndarray]:
    """Run the full discontinuity-extraction pipeline on a point cloud with normals.

    Returns (discontinuity_sets, point_labels) where point_labels is a
    (set_id, patch_id) pair per point (-1, -1) for points not assigned to any
    discontinuity, useful for coloring the point cloud in the 3D viewer.
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals -- run normals.estimate_normals() first")

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    orientation_labels = cluster_orientations(normals, eps=orientation_eps, min_samples=orientation_min_samples)

    point_set_id = np.full(len(points), -1, dtype=int)
    point_patch_id = np.full(len(points), -1, dtype=int)

    discontinuity_sets: list[DiscontinuitySet] = []
    next_patch_id = 0

    for set_id in sorted(l for l in set(orientation_labels) if l != -1):
        set_mask = orientation_labels == set_id
        set_points = points[set_mask]
        set_normals = normals[set_mask]
        set_indices = np.nonzero(set_mask)[0]

        spatial_labels = cluster_spatial(set_points, eps=spatial_eps, min_samples=spatial_min_samples)

        patches: list[DiscontinuityPatch] = []
        for local_patch_label in sorted(l for l in set(spatial_labels) if l != -1):
            patch_mask = spatial_labels == local_patch_label
            if patch_mask.sum() < min_patch_points:
                continue

            patch_points = set_points[patch_mask]
            centroid, normal, rms_residual = fit_plane(patch_points)
            dip, dip_direction = normal_to_dip_dipdir(normal)
            max_chord, area = persistence(patch_points, centroid, normal)

            patch = DiscontinuityPatch(
                set_id=set_id,
                patch_id=next_patch_id,
                n_points=int(patch_mask.sum()),
                centroid=centroid,
                normal=normal,
                dip=dip,
                dip_direction=dip_direction,
                roughness_rms=rms_residual,
                persistence_length=max_chord,
                area=area,
            )
            patches.append(patch)

            global_idx = set_indices[patch_mask]
            point_set_id[global_idx] = set_id
            point_patch_id[global_idx] = next_patch_id
            next_patch_id += 1

        if not patches:
            continue

        mean_dip, mean_dip_dir, fisher_k = mean_orientation(set_normals)
        centroids = np.array([p.centroid for p in patches])
        mean_normal = np.array(
            [
                np.sin(np.radians(mean_dip)) * np.sin(np.radians(mean_dip_dir)),
                np.sin(np.radians(mean_dip)) * np.cos(np.radians(mean_dip_dir)),
                np.cos(np.radians(mean_dip)),
            ]
        )
        spacing_mean, spacing_std = compute_spacing(centroids, mean_normal)

        discontinuity_sets.append(
            DiscontinuitySet(
                set_id=set_id,
                mean_dip=mean_dip,
                mean_dip_direction=mean_dip_dir,
                fisher_k=fisher_k,
                patches=patches,
                spacing_mean=spacing_mean,
                spacing_std=spacing_std,
            )
        )

    point_labels = np.column_stack([point_set_id, point_patch_id])
    return discontinuity_sets, point_labels
