"""Phase 2: correlate discontinuity patches (this module's Phase-1 output,
see discontinuity.segmentation.extract_discontinuities) with thermal anomaly
zones (thermal.anomaly.anomaly_polygons) to score groundwater seepage-pathway
candidates.

Planned interface:

    def score_seepage_candidates(patches: list[DiscontinuityPatch],
                                  anomaly_polygons: GeoDataFrame,
                                  max_distance_m: float = 0.3) -> pd.DataFrame
        For each patch, find overlapping/nearby thermal anomalies and combine
        anomaly intensity with the patch's persistence_length (a longer,
        more continuous discontinuity connected to a thermal anomaly is a
        stronger seepage-pathway candidate than an isolated short joint) into
        a single ranked score. Output columns: patch_id, set_id, dip,
        dip_direction, persistence_length, thermal_anomaly_intensity,
        seepage_score.

Kept as a stub until thermal data exists so the scoring weights can be fit to
observed anomaly magnitudes rather than guessed.
"""
