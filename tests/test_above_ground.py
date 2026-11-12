"""Points above the ground are not rock (cleaning.above_ground): tree crowns over the wall top, and the LiDAR rear
layer kept there (lidar_section.classify)."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")


def _cliff_dtm(path, foot=40.0, crest=60.0):
    """0.5 m DTM of a vertical cliff at x = 50: floor at `foot`, plateau at `crest`, and a ramp from 1 m in front
    of the foot to 4 m behind the crest (as the Iksan 0.5 m DTM crosses the east wall)."""
    from rasterio.transform import from_origin

    x = np.arange(0, 100, 0.5) + 0.25
    z = np.interp(x, [0, 49, 54, 100], [foot, foot, crest, crest]).astype(np.float32)
    z = np.tile(z, (80, 1))
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                       crs="EPSG:5186", transform=from_origin(0, 40, 0.5, 0.5), nodata=-9999) as d:
        d.write(z, 1)
    return path


def test_above_ground_flags_crowns_not_the_cliff(tmp_path):
    from groundwater.discontinuity.cleaning import above_ground

    dtm = _cliff_dtm(tmp_path / "dtm.tif")
    pts = np.array([
        [20, 20, 40.1],       # floor
        [80, 20, 60.2],       # plateau
        [50.2, 20, 59.5],     # vertical cliff face at the crest (the DTM ramp there is far below it)
        [50.2, 20, 48.0],     # cliff face, mid height
        [70, 20, 72.0],       # tree crown 12 m above the plateau
        [62, 20, 63.5],       # lower crown just behind the crest
        [20, 20, 41.5],       # shrub on the floor, within the margin
    ])
    pts = np.vstack([pts, np.column_stack([np.full(50, 30.0), np.linspace(5, 35, 50), np.full(50, 40.0)])])  # ground
    air, info = above_ground(pts, dtm)
    assert air[:7].tolist() == [False, False, False, False, True, True, False]
    assert not air[7:].any()
    assert info["removed"] == 2 and "skipped" not in info


def test_above_ground_skips_a_dtm_in_another_datum(tmp_path):
    from groundwater.discontinuity.cleaning import above_ground

    dtm = _cliff_dtm(tmp_path / "dtm.tif", foot=67.0, crest=87.0)       # ellipsoidal: ~27 m higher
    pts = np.column_stack([np.linspace(5, 45, 200), np.full(200, 20.0), np.full(200, 40.0)])
    pts = np.vstack([pts, [[70, 20, 95.0]]])
    air, info = above_ground(pts, dtm)
    assert not air.any() and "skipped" in info
    air, _ = above_ground(pts, dtm, check_datum=False)
    assert air[-1]


def test_section_rear_layer_above_ground_is_not_rock(tmp_path):
    from groundwater import lidar_section as LS
    from groundwater.facade.frame import build_wall_frame

    dtm = _cliff_dtm(tmp_path / "dtm.tif")
    rng = np.random.default_rng(0)
    # a vertical wall at x = 50 facing -x, 40-60 m, and a crown 64-70 m behind the crest (x 60-64)
    yy, zz = np.meshgrid(np.arange(10, 30, 0.02), np.arange(40, 60, 0.02))
    wall = np.column_stack([np.full(yy.size, 50.0) + rng.normal(0, 0.003, yy.size), yy.ravel(), zz.ravel()])
    yy, zz = np.meshgrid(np.arange(10, 30, 0.02), np.arange(64, 70, 0.02))
    crown = np.column_stack([62 + 2 * np.sin(yy.ravel() * 3), yy.ravel(), zz.ravel()])
    xyz = np.vstack([wall, crown])
    frame = build_wall_frame(xyz[::50], [[50, 30], [50, 10]], [0, 20])
    cls = LS.classify(xyz, frame, dtm)
    n = len(wall)
    assert cls["rock"][:n].mean() > 0.9
    assert not cls["rock"][n:].any()
    assert cls["air"][n:][cls["rear"][n:]].all()
    cls0 = LS.classify(xyz, frame)
    assert not cls0["air"].any() and not cls0["rock"][n:].any()    # without the DTM: outside the face span, too
    assert cls0["outside"][n:][cls0["rear"][n:]].all()
