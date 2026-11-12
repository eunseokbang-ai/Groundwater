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


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, number_of_points: int | None = None) -> o3d.geometry.PointCloud:
    """Point cloud for the discontinuity analysis from a mesh.

    Default: the mesh vertices with their colours and normals (a photogrammetry mesh is already a dense,
    even surface; a fixed number of Poisson samples would be far too sparse on a whole-site mesh). With
    `number_of_points` the surface is sampled uniformly instead (texture colours via enable_post_processing).
    Note: a mesh rounds off ledge edges, so gently dipping planes come out steeper than from the dense cloud
    (Ulsan bedding: 23 deg from the mesh, 8 deg from the LAS) -- prefer the point cloud when there is one.
    """
    if number_of_points:
        return mesh.sample_points_uniformly(number_of_points)
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    return pcd
