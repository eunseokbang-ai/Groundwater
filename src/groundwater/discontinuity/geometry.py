"""Discontinuity geometry: dip/dip-direction convention, plane fitting, spacing,
persistence and a roughness proxy.

Convention (matches common rock-mass point-cloud literature, e.g. Riquelme et al.):
x = East, y = North, z = Up. Poles are only folded to the upper hemisphere at the
final dip / dip-direction conversion; all statistics before that are axial.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull


def normal_to_dip_dipdir(normal: np.ndarray) -> tuple[float, float]:
    """Convert a unit normal to (dip_deg, dip_direction_deg)."""
    n = normal / np.linalg.norm(normal)
    if n[2] < 0:
        n = -n
    dip = np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))
    dip_direction = np.degrees(np.arctan2(n[0], n[1])) % 360.0
    return float(dip), float(dip_direction)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares plane fit via PCA. Returns (centroid, unit_normal, rms_residual)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    residuals = centered @ normal
    rms_residual = float(np.sqrt(np.mean(residuals**2)))
    return centroid, normal, rms_residual


def _local_plane_coords(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project points onto the plane and express them in a local 2D (u, v) basis."""
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    centered = points - centroid
    return np.column_stack([centered @ u, centered @ v])


def scanline_spacing(
    centroids: np.ndarray, areas: np.ndarray, mean_normal: np.ndarray, coplanar_tolerance: float,
    n_lines: int = 400, seed: int = 0,
) -> tuple[float | None, float | None]:
    """Normal spacing of a set from virtual scanlines (ISRM-style) parallel to the set normal.

    Each patch is a disc of its own area around its centroid in the set's plane coordinates. A scanline
    runs along the set normal through a point on a randomly chosen patch (chosen in proportion to area);
    the patches whose discs it passes through are its intersections, and the gaps between consecutive
    intersections (fragments closer than `coplanar_tolerance` merged) are the spacings. Unlike sorting
    all centroids along the normal, patches that sit side by side on a long wall without overlapping do
    not produce spurious small gaps. Returns (mean, std), or (None, None) when no line hits two patches.
    """
    if len(centroids) < 2:
        return None, None
    n = mean_normal / np.linalg.norm(mean_normal)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    plane = np.column_stack([centroids @ u, centroids @ v])
    along = centroids @ n
    radius = np.sqrt(np.maximum(np.asarray(areas, float), 1e-9) / np.pi)

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(centroids), size=n_lines, p=radius**2 / np.sum(radius**2))
    ang = rng.uniform(0, 2 * np.pi, n_lines)
    rad = radius[pick] * np.sqrt(rng.uniform(0, 1, n_lines))
    lines = plane[pick] + np.column_stack([np.cos(ang), np.sin(ang)]) * rad[:, None]

    gaps = []
    for q in lines:
        hit = np.linalg.norm(plane - q, axis=1) < radius
        if hit.sum() < 2:
            continue
        g = np.diff(np.sort(along[hit]))
        gaps.append(g[g > coplanar_tolerance])
    gaps = np.concatenate(gaps) if gaps else np.empty(0)
    if len(gaps) == 0:
        return None, None
    return float(gaps.mean()), float(gaps.std())


def persistence(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Return (max_chord_length, convex_hull_area) of the patch projected onto its own plane."""
    uv = _local_plane_coords(points, centroid, normal)
    if len(uv) < 3:
        return 0.0, 0.0
    hull = ConvexHull(uv)
    hull_pts = uv[hull.vertices]
    dists = np.linalg.norm(hull_pts[:, None, :] - hull_pts[None, :, :], axis=-1)
    max_chord = float(dists.max())
    return max_chord, float(hull.volume)  # `volume` is the 2D area for a 2D hull


@dataclass
class DiscontinuityPatch:
    set_id: int
    patch_id: int
    n_points: int
    centroid: np.ndarray
    normal: np.ndarray
    dip: float
    dip_direction: float
    roughness_rms: float
    persistence_length: float
    area: float


@dataclass
class DiscontinuitySet:
    set_id: int
    mean_dip: float
    mean_dip_direction: float
    fisher_k: float
    n_points: int
    patches: list = field(default_factory=list)
    spacing_mean: float | None = None
    spacing_std: float | None = None


def axial_mean(normals: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Mean axis of a set of poles -> (unit_axis, fisher_k).

    Poles are axial (n and -n are the same plane), so the mean is the principal
    eigenvector of the (optionally weighted) orientation tensor rather than a
    vector sum -- folding to a hemisphere first cancels out sub-vertical sets
    whose poles straddle nz = 0. fisher_k (unweighted) is computed after
    flipping each pole onto the mean axis; higher means a tighter set.
    """
    weighted = normals if weights is None else normals * weights[:, None]
    _, vecs = np.linalg.eigh(weighted.T @ normals)
    axis = vecs[:, -1]
    signs = np.where(normals @ axis < 0, -1.0, 1.0)
    r = np.linalg.norm((normals * signs[:, None]).sum(axis=0))
    n = len(normals)
    fisher_k = (n - 1) / (n - r) if n > r else float("inf")
    return axis, float(fisher_k)


def compute_spacing(
    patch_centroids: np.ndarray, mean_normal: np.ndarray, coplanar_tolerance: float
) -> tuple[float | None, float | None]:
    """Perpendicular spacing between parallel patches in the same set.

    Patch centroids are projected onto the set's mean normal axis. Projections
    within `coplanar_tolerance` of each other are treated as fragments of one
    discontinuity (a joint cut into pieces by netting, vegetation or occlusion)
    so they do not contribute near-zero spacings.
    """
    if len(patch_centroids) < 2:
        return None, None
    n = mean_normal / np.linalg.norm(mean_normal)
    proj = np.sort(patch_centroids @ n)
    gaps = np.diff(proj)
    gaps = gaps[gaps > coplanar_tolerance]
    if len(gaps) == 0:
        return None, None
    return float(gaps.mean()), float(gaps.std())
