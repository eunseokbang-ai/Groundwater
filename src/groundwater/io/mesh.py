"""Textured mesh loading (OBJ + material/texture) via Open3D."""
from pathlib import Path

import open3d as o3d


def load_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh, pulling in vertex colors from any bound texture."""
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if not mesh.has_vertices():
        raise ValueError(f"No vertices loaded from {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, number_of_points: int = 500_000) -> o3d.geometry.PointCloud:
    """Sample the textured mesh surface into a colored point cloud for discontinuity analysis."""
    return mesh.sample_points_poisson_disk(number_of_points)
