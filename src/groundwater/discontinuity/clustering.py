"""Orientation clustering: group points into discontinuity sets by normal direction.

Approach (standard in rock-mass point-cloud literature, e.g. Riquelme's DSE and
"Lab-DBSCAN"-style methods): fold normals to one hemisphere, then run DBSCAN on
the unit-sphere coordinates directly. For the small angular separations typical
between distinct joint sets (>~10-15 deg), Euclidean chord distance on the unit
sphere is a good proxy for angular distance, so plain (fast) Euclidean DBSCAN
can be used instead of an O(n^2) angular-distance matrix.
"""
import numpy as np
from sklearn.cluster import DBSCAN


def fold_to_upper_hemisphere(normals: np.ndarray) -> np.ndarray:
    return np.where(normals[:, 2:3] < 0, -normals, normals)


def cluster_orientations(normals: np.ndarray, eps: float = 0.26, min_samples: int = 30) -> np.ndarray:
    """Cluster unit normals into discontinuity sets.

    Returns integer labels the same length as `normals`; -1 marks points not
    belonging to any well-defined set (noise / too-scattered orientation).
    """
    folded = fold_to_upper_hemisphere(normals)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(folded)
    return labels


def cluster_spatial(points: np.ndarray, eps: float = 0.15, min_samples: int = 20) -> np.ndarray:
    """Within one orientation cluster, split into individual physical patches by
    3D proximity (two parallel but spatially separate joints must not be merged).
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    return labels
