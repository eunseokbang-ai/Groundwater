"""Discontinuity geometry: dip/dip-direction convention, plane fitting, spacing,
persistence and a roughness proxy.

Convention (matches common rock-mass point-cloud literature, e.g. Riquelme et al.):
x = East, y = North, z = Up. The plane's pole (normal) is folded to the upper
hemisphere (nz >= 0) before conversion so every plane has one unambiguous
dip / dip-direction pair.
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
    patches: list = field(default_factory=list)
    spacing_mean: float | None = None
    spacing_std: float | None = None


def mean_orientation(normals: np.ndarray) -> tuple[float, float, float]:
    """Fisher mean of a set of unit normals -> (mean_dip, mean_dip_direction, fisher_k).

    fisher_k is a concentration parameter (higher = tighter cluster, i.e. a more
    well-defined joint set); it is the standard summary statistic reported
    alongside orientation clusters in rock-mass discontinuity studies.
    """
    folded = np.where(normals[:, 2:3] < 0, -normals, normals)
    resultant = folded.sum(axis=0)
    r = np.linalg.norm(resultant)
    n = len(folded)
    mean_vec = resultant / r if r > 0 else np.array([0.0, 0.0, 1.0])
    dip, dip_dir = normal_to_dip_dipdir(mean_vec)
    r_bar = r / n if n else 0.0
    fisher_k = (n - 1) / max(n - r, 1e-9) if n > r else float("inf")
    return dip, dip_dir, fisher_k


def compute_spacing(patch_centroids: np.ndarray, mean_normal: np.ndarray) -> tuple[float | None, float | None]:
    """Perpendicular spacing between parallel patches in the same set.

    Patch centroids are projected onto the set's mean normal axis; spacing is
    the gap between consecutive projections once sorted.
    """
    if len(patch_centroids) < 2:
        return None, None
    n = mean_normal / np.linalg.norm(mean_normal)
    proj = patch_centroids @ n
    proj.sort()
    gaps = np.diff(proj)
    gaps = gaps[gaps > 1e-6]
    if len(gaps) == 0:
        return None, None
    return float(gaps.mean()), float(gaps.std())
