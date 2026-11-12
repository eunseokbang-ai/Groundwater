"""End-to-end discontinuity extraction: coplanarity filter -> orientation sets
(pole-density peaks) -> spatial splitting into patches -> per-patch plane fit ->
per-set aggregate stats.
"""
import numpy as np

SUBVERTICAL_DEG = 85.0   # at and above this dip the dip direction is ambiguous
# A face is one plane: a connected piece of a set's points whose plane fit is off by more than this (RMS) is a
# stepped or curved stretch of wall -- several faces chained at their edges -- and is split. Iksan west wall
# 2026-10: within 0.5-1 m the rock is 3-5 cm off a plane (L2 / photogrammetry noise 1-1.3 cm), but without a limit
# the dense LiDAR behind the netting chained a 983 m2 "face" 0.56 m RMS off its plane.
MAX_PLANE_RMS = 0.10     # m
MAX_SPLIT_DEPTH = 16
import open3d as o3d

from .clustering import find_orientation_sets, voxel_connected_components
from .geometry import (
    DiscontinuityPatch,
    DiscontinuitySet,
    axial_mean,
    fit_plane,
    normal_to_dip_dipdir,
    persistence,
    scanline_spacing,
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
    min_patch_area: float = 0.0,
    max_plane_rms: float | None = MAX_PLANE_RMS,
    fixed_sets=None,
    exclude: np.ndarray | None = None,
    info: dict | None = None,
) -> tuple[list[DiscontinuitySet], np.ndarray]:
    """Run discontinuity extraction on a point cloud whose normals are already set.

    `surface_variation` (from normals.estimate_local_geometry) excludes non-planar
    points -- vegetation, netting edges, plane intersections -- before orientation
    analysis. Patches smaller than `min_patch_area` (m2, convex hull in their own plane) are dropped:
    on a real rough face the set's points break into many small facets, and only surfaces of some extent
    are discontinuities worth reporting. Patches more than `max_plane_rms` off their plane are split
    (planar_parts; None = no limit). `fixed_sets` / `info`: sets the user pins and what the set detection
    reports (clustering.find_orientation_sets). `exclude` (bool per point): points left out of the sets and faces
    altogether (e.g. normals parallel to the excavated wall, discontinuity.exposure.point_mask). Returns (discontinuity_sets, point_labels) where point_labels
    holds (set_id, patch_id) per point, (-1, -1) for points not in any discontinuity.
    """
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals -- run normals.estimate_local_geometry() first")

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    planar = np.ones(len(points), dtype=bool)
    if surface_variation is not None:
        planar = surface_variation <= max_surface_variation
    if exclude is not None:
        planar &= ~np.asarray(exclude, bool)
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
        fixed_sets=fixed_sets,
        info=info,
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
        groups = [g for g in groups if len(g) >= min_patch_points]
        if max_plane_rms:
            groups = [g[part] for g in groups
                      for part in planar_parts(points[set_idx[g]], normals[set_idx[g]], link_distance, max_plane_rms,
                                                    min_patch_points)]
        groups = sorted(groups, key=len, reverse=True)

        patches: list[DiscontinuityPatch] = []
        for group in groups:
            global_idx = set_idx[group]
            patch_points = points[global_idx]
            centroid, normal, rms_residual = fit_plane(patch_points)
            dip, dip_direction = normal_to_dip_dipdir(normal)
            max_chord, area = persistence(patch_points, centroid, normal)
            if area < min_patch_area:
                continue
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
        discontinuity_sets.append(summarise_set(set_id, patches, normals, point_set_id, link_distance))

    return discontinuity_sets, np.column_stack([point_set_id, point_patch_id])


def _components(points: np.ndarray, link_distance: float, min_points: int) -> list:
    comp = voxel_connected_components(points, link_distance)
    order = np.argsort(comp, kind="stable")
    return [g for g in np.split(order, np.flatnonzero(np.diff(comp[order])) + 1) if len(g) >= min_points]


def planar_parts(points: np.ndarray, normals: np.ndarray, link_distance: float, max_rms: float, min_points: int,
                 depth: int = 0) -> list:
    """Index groups of `points` (one connected patch, point `normals`) each within `max_rms` RMS of its own plane.
    A patch off by more is split across its mean point normal -- the face orientation; a plane fitted to a step
    tilts across it -- at the two-means threshold of the points' offsets along it (the front and the back plane
    of a step), each side into its connected pieces, recursively. Pieces under `min_points` are dropped."""
    _, fit_n, rms = fit_plane(points)
    if rms <= max_rms or depth >= MAX_SPLIT_DEPTH or len(points) < 2 * min_points:
        return [np.arange(len(points))]
    n = (normals * np.where(normals @ fit_n < 0, -1.0, 1.0)[:, None]).mean(axis=0)
    d = points @ (n / np.linalg.norm(n))
    t = float(np.median(d))
    for _ in range(20):                                 # 1D two-means
        lo, hi = d[d <= t], d[d > t]
        if not len(lo) or not len(hi):
            break
        t_new = 0.5 * (lo.mean() + hi.mean())
        if abs(t_new - t) < 1e-4:
            break
        t = t_new
    out = []
    for side in (d > t, d <= t):
        idx = np.flatnonzero(side)
        for g in _components(points[idx], link_distance, min_points):
            sub = idx[g]
            out += [sub[k] for k in planar_parts(points[sub], normals[sub], link_distance, max_rms, min_points,
                                                 depth + 1)]
    return out


def summarise_set(set_id: int, patches: list, normals: np.ndarray, point_set_id: np.ndarray,
                  link_distance: float) -> DiscontinuitySet:
    """A set from its faces: mean orientation, scatter and spacing (also used after faces are left out,
    cleaning.drop_faces)."""
    # Orientation from the patch plane fits: point normals near patch edges and
    # plane junctions are blended and bias the mean. Scatter still comes from points.
    axis, _ = axial_mean(np.array([p.normal for p in patches]), weights=np.array([p.n_points for p in patches], float))
    _, fisher_k = axial_mean(normals[point_set_id == set_id])
    mean_dip, mean_dip_dir = normal_to_dip_dipdir(axis)
    if mean_dip >= SUBVERTICAL_DEG:
        # a near-vertical plane's dip direction flips 180 deg with a 1 deg wobble of its normal: report the
        # patches of a sub-vertical set in the set's own convention
        for p in patches:
            if abs((p.dip_direction - mean_dip_dir + 180.0) % 360.0 - 180.0) > 90.0:
                p.dip_direction = (p.dip_direction + 180.0) % 360.0
    spacing_mean, spacing_std = scanline_spacing(
        np.array([p.centroid for p in patches]), np.array([p.area for p in patches]), axis,
        coplanar_tolerance=link_distance,
    )
    return DiscontinuitySet(
        set_id=set_id,
        mean_dip=mean_dip,
        mean_dip_direction=mean_dip_dir,
        fisher_k=fisher_k,
        n_points=int((point_set_id == set_id).sum()),
        patches=patches,
        spacing_mean=spacing_mean,
        spacing_std=spacing_std,
    )
