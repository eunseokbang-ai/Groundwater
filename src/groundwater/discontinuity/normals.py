"""Normal estimation for rock-face point clouds."""
import numpy as np
import open3d as o3d


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    search_radius: float = 0.1,
    max_nn: int = 30,
    camera_location: tuple[float, float, float] | None = None,
) -> o3d.geometry.PointCloud:
    """Estimate normals and orient them to point away from the rock face (outward).

    For a drone-scanned cliff, orienting normals consistently is required before
    orientation clustering -- otherwise a single planar joint set could be split
    into two clusters pointing in opposite directions. If no camera_location is
    given, the centroid-outward heuristic (tangent-plane propagation) is used.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn)
    )
    if camera_location is not None:
        pcd.orient_normals_towards_camera_location(camera_location)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    return pcd


def normals_array(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    if not pcd.has_normals():
        raise ValueError("Point cloud has no normals -- call estimate_normals() first")
    return np.asarray(pcd.normals)
