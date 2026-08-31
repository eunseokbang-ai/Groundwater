"""Phase 2: seepage-candidate thermal anomaly detection.

Not yet implemented -- deferred until real thermal imagery shows the actual
noise characteristics and contrast magnitude on this rock type (sun-facing
vs shaded portions of the face will need different local baselines).

Planned interface:

    def detect_anomalies(temp_raster: Raster, method="local_zscore",
                          window_px=51, z_threshold=2.0) -> np.ndarray
        Boolean mask, True where a pixel's temperature deviates from its local
        neighborhood beyond z_threshold standard deviations -- flags candidate
        seepage zones without assuming a fixed global temperature threshold
        (sun/shade variation across a cliff face makes a global threshold
        unreliable).

    def anomaly_polygons(mask: np.ndarray, raster: Raster, min_area_m2=0.1) -> GeoDataFrame
        Vectorize the mask into polygons in world coordinates for use by
        fusion.correlate.
"""
