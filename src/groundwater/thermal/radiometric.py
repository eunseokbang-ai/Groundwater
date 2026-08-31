"""Phase 2: radiometric calibration/normalization of the thermal orthomosaic.

Not yet implemented -- needs real thermal data plus its acquisition metadata
(emissivity setting, ambient air temperature/humidity, capture time) to pick
the right calibration path:

- If the thermal camera exports calibrated temperature (already in degC,
  e.g. a R-JPEG/TIFF with embedded calibration from DJI/FLIR), skip straight
  to normalization.
- If it exports raw radiometric counts, apply the sensor's radiometric
  equation (object temperature from atmospheric transmission, emissivity,
  reflected apparent temperature) before anything downstream is meaningful.

Planned interface:

    def to_temperature(raw_raster: Raster, emissivity: float, ambient_temp_c: float,
                        humidity_pct: float) -> Raster
        Convert raw sensor counts to calibrated surface temperature (degC).

    def normalize_for_scene(temp_raster: Raster) -> Raster
        Subtract the local background trend (e.g. large-scale sun-exposure
        gradient across the face) so anomaly.py compares seepage candidates
        against their immediate surroundings, not the whole scene.
"""
