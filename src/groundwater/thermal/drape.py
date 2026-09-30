"""Phase 2: drape registered thermal imagery onto the wall so each
discontinuity patch can be looked up for a co-located temperature.

Not yet implemented. Sampling a nadir thermal orthomosaic by (x, y) is not an
option: a vertical wall collapses to a thin strip in plan view. Planned
interface, built on facade.frame.WallFrame so thermal, RGB facade and depth
rasters share one (chainage, elevation) pixel grid:

    def thermal_facade(frame, points_xyz, thermal_images, camera_poses, resolution) -> Raster
        Back-project each radiometric thermal image through its camera pose
        (the H20T records thermal and wide RGB together, so the poses come from
        the RGB photogrammetry) onto the point cloud, keep per point the view
        most perpendicular to the wall, then rasterize with the same z-buffer
        rule as facade.orthophoto.

    def sample_facade(raster, uv) -> np.ndarray
        Temperature at wall coordinates, e.g. along each patch footprint.
"""
