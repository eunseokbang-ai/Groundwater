"""Phase 2 (post LiDAR/thermal acquisition): co-register the LiDAR point cloud
and the thermal orthomosaic onto the RGB photogrammetry coordinate frame.

Not yet implemented -- both datasets will exist in project-relative or GPS
coordinates that may not perfectly agree with the RGB model unless the same
GCPs / RTK base were used for all three sensors. Implement once real data is
available so the ICP threshold and image-registration approach can be tuned
against it, rather than guessed at now.

Planned interface:

    def register_point_clouds(source, target, init_transform=None) -> (aligned_source, transform, fitness)
        Point-to-plane ICP (open3d.pipelines.registration.registration_icp),
        source = LiDAR cloud, target = RGB photogrammetry cloud.

    def register_raster_to_raster(moving: Raster, reference: Raster) -> Affine
        Feature-based (e.g. ORB/AKAZE keypoints + RANSAC homography) alignment
        of the thermal orthomosaic to the RGB orthomosaic, falling back to
        matching shared GCP pixel coordinates if available.
"""
