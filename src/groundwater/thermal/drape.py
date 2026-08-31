"""Phase 2: drape the (registered) thermal raster onto the 3D mesh/point cloud
so each discontinuity patch can be looked up for a co-located temperature
value.

Planned interface:

    def sample_thermal_at_points(points_xyz: np.ndarray, thermal_raster: Raster) -> np.ndarray
        For each 3D point, project to (x, y) and bilinearly sample the
        registered thermal raster -- requires registration.align to have
        already placed thermal_raster in the same CRS/frame as points_xyz.
"""
