"""Point cloud loading for LAS/LAZ (laspy) and PLY/PCD (Open3D)."""
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a point cloud into an Open3D PointCloud, normalizing across formats.

    Supports .las/.laz (via laspy) and anything Open3D reads natively
    (.ply, .pcd, .xyz). Color is carried over when present in the source file.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".las", ".laz"):
        return _load_las(path)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No points loaded from {path}")
    return pcd


def _load_las(path: Path) -> o3d.geometry.PointCloud:
    import laspy

    with laspy.open(str(path)) as f:
        las = f.read()

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    dims = set(las.point_format.dimension_names)
    if {"red", "green", "blue"} <= dims:
        max_val = np.iinfo(las.red.dtype).max if np.issubdtype(las.red.dtype, np.integer) else 1.0
        rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64) / max_val
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))

    return pcd


def crop_bbox(pcd: o3d.geometry.PointCloud, bbox: list[float] | None) -> o3d.geometry.PointCloud:
    """Crop to an axis-aligned box [xmin, ymin, zmin, xmax, ymax, zmax]. No-op if bbox is None."""
    if bbox is None:
        return pcd
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, zmin), max_bound=(xmax, ymax, zmax)
    )
    return pcd.crop(aabb)


def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if not voxel_size:
        return pcd
    return pcd.voxel_down_sample(voxel_size)
