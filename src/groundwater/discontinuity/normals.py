"""Per-point local geometry (normal + planarity) for rock-face point clouds."""
import numpy as np
import open3d as o3d

_EIG_CHUNK = 1_000_000


def estimate_local_geometry(
    pcd: o3d.geometry.PointCloud, search_radius: float = 0.1, max_nn: int = 30
) -> np.ndarray:
    """Set pcd.normals in place and return each point's surface variation.

    Surface variation = smallest / sum of the local covariance eigenvalues:
    ~0 on a flat patch, up to 1/3 for an isotropic blob. Vegetation, netting
    edges and plane intersections score high, which is what the coplanarity
    filter in segmentation relies on. Normal signs are left arbitrary -- all
    downstream orientation statistics are axial.
    """
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
    covariances = np.asarray(pcd.covariances)

    normals = np.empty((len(covariances), 3))
    variation = np.empty(len(covariances))
    for start in range(0, len(covariances), _EIG_CHUNK):
        vals, vecs = np.linalg.eigh(covariances[start:start + _EIG_CHUNK])
        total = vals.sum(axis=1)
        normals[start:start + _EIG_CHUNK] = vecs[:, :, 0]
        variation[start:start + _EIG_CHUNK] = np.where(total > 0, vals[:, 0] / np.where(total > 0, total, 1.0), 1.0 / 3.0)

    pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals))
    return variation
