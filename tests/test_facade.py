"""Wall frame, facade orthophoto, trace extraction/matching, and the pipeline
on a synthetic quarry wall."""
import numpy as np
import open3d as o3d
import pandas as pd
from rasterio.transform import Affine

from groundwater.discontinuity.trace_mapping import Trace2D, extract_traces, match_traces_to_patches
from groundwater.facade.frame import WallFrame, build_wall_frame
from groundwater.facade.orthophoto import rasterize_facade
from groundwater.io.raster import Raster
from groundwater.pipeline import run_pipeline, save_results
from groundwater.synthetic import SyntheticJoint, synthetic_wall
from groundwater.validation.field_compare import angular_difference, dip_dipdir_to_normal

BENT = [[0.0, 0.0], [12.0, 0.0], [20.0, -6.0]]  # second leg turns toward the viewer (south)


def test_bent_wall_chainage_offset_and_round_trip():
    frame = WallFrame(BENT)
    assert frame.length == 22.0

    on_first_leg = [6.0, -3.0, 101.0]  # 3 m in front (south) of the first leg
    on_second_leg = [12.0 + 5 * 0.8 - 2 * 0.6, 0.0 - 5 * 0.6 - 2 * 0.8, 104.0]  # 5 m along, 2 m in front
    past_the_end = [20.0 + 3 * 0.8, -6.0 - 3 * 0.6, 100.0]
    uvw = frame.to_wall(np.array([on_first_leg, on_second_leg, past_the_end]))

    np.testing.assert_allclose(uvw[0], [6.0, 101.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(uvw[1], [17.0, 104.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(uvw[2], [25.0, 100.0, 0.0], atol=1e-9)

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-2, 22, 500), rng.uniform(-8, 1, 500), rng.uniform(100, 110, 500)])
    # Behind a bend, points in the wedge project onto the vertex itself, so only front points invert exactly.
    front = frame.to_wall(pts)[:, 2] > 0.5
    np.testing.assert_allclose(frame.to_world(frame.to_wall(pts[front])), pts[front], atol=1e-9)


def _leaning_wall_points():
    """Straight wall along x whose crest leans back to the north: rock north, open pit south."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 20, 3000)
    z = rng.uniform(100, 110, 3000)
    return np.column_stack([x, 0.1 * (z - 100), z])


def test_fitted_frame_faces_the_viewer():
    points = _leaning_wall_points()
    for frame in (build_wall_frame(points, viewer_point=[5.0, -10.0]), build_wall_frame(points)):
        west, east, pit = frame.to_wall(np.array([[0.0, 0.0, 100.0], [20.0, 0.0, 100.0], [5.0, -10.0, 100.0]]))
        assert pit[2] > 0, "pit side must be in front (w > 0)"
        assert east[0] > west[0], "chainage must run left-to-right as seen from the pit"


def test_viewer_point_corrects_a_reversed_polyline():
    frame = build_wall_frame(_leaning_wall_points(), polyline=[[20.0, 0.0], [0.0, 0.0]], viewer_point=[5.0, -10.0])
    assert frame.to_wall(np.array([[5.0, -10.0, 100.0]]))[0, 2] > 0


def test_apparent_trace_directions_on_south_facing_wall():
    frame = WallFrame([[0.0, 0.0], [10.0, 0.0]])
    vertical = frame.apparent_trace_direction(dip_dipdir_to_normal(90.0, 90.0), 5.0)
    horizontal = frame.apparent_trace_direction(dip_dipdir_to_normal(0.0, 0.0), 5.0)
    np.testing.assert_allclose(np.abs(vertical), [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(np.abs(horizontal), [1.0, 0.0], atol=1e-9)
    assert frame.apparent_trace_direction(dip_dipdir_to_normal(88.0, 180.0), 5.0) is None


def test_facade_keeps_front_point_and_fills_only_small_holes():
    res = 0.1
    g = np.arange(0.0, 2.0, res) + res / 2  # pixel centres once the anchor fixes the origin at (0, 2)
    gu, gv = (a.ravel() for a in np.meshgrid(g, g))
    keep = ~((np.abs(gu - 0.55) < 0.01) & (np.abs(gv - 0.55) < 0.01))  # one-pixel hole
    keep &= ~((gu > 1.2) & (gv > 1.2))  # large empty corner
    uvw = np.column_stack([gu[keep], gv[keep], np.zeros(keep.sum())])
    colors = np.tile([0.5, 0.5, 0.5], (keep.sum(), 1))
    anchor = [0.0, 2.0, -1.0]  # sets the raster origin; behind the wall so it never wins a pixel
    in_front = [0.15, 1.95, 0.5]  # same pixel as a grid point, but nearer the viewer
    rgb, depth = rasterize_facade(
        np.vstack([uvw, [anchor, in_front]]), np.vstack([colors, [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]]), res
    )

    row, col = rgb.world_to_pixel(np.array([0.15]), np.array([1.95]))
    assert tuple(rgb.array[:, row[0], col[0]]) == (255, 0, 0)
    hole_row, hole_col = depth.world_to_pixel(np.array([0.55]), np.array([0.55]))
    assert np.isfinite(depth.array[0, hole_row[0], hole_col[0]])
    corner_row, corner_col = depth.world_to_pixel(np.array([1.75]), np.array([1.75]))
    assert np.isnan(depth.array[0, corner_row[0], corner_col[0]])


def test_traces_follow_dark_lines_not_nodata_edges():
    img = np.full((200, 300), 150, dtype=np.uint8)
    img[98:102, 20:180] = 20  # dark horizontal line, rows 98-101
    valid = np.ones_like(img, dtype=bool)
    valid[:, 200:] = False  # sharp no-data boundary at column 200
    img[~valid] = 0
    raster = Raster(array=np.stack([img] * 3), transform=Affine(0.01, 0, 0, 0, -0.01, 2.0), crs=None, nodata=None)

    traces = extract_traces(raster, valid_mask=valid)

    assert traces
    for trace in traces:
        cols = trace.pixel_coords[:, 1]
        assert cols.max() < 195, "no trace may run along the no-data boundary"
        assert np.all(np.abs(trace.pixel_coords[:, 0] - 99.5) < 4)


def test_trace_match_needs_proximity_and_orientation():
    trace = Trace2D(coords=np.array([[0.0, 5.0], [2.0, 5.0]]), pixel_coords=np.zeros((2, 2)), length_m=2.0)
    near = np.array([[1.0, 5.05], [1.2, 4.9]])
    far = near + [0.0, 1.0]
    horizontal, vertical = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    result = match_traces_to_patches(
        [trace], [near, near, far, near], [horizontal, vertical, horizontal, None], tolerance=0.1
    )
    assert result == [True, False, False, None]
    assert match_traces_to_patches([], [near, near], [horizontal, None], tolerance=0.1) == [False, None]


def test_pipeline_on_synthetic_wall(tmp_path):
    rng = np.random.default_rng(3)
    polyline = [[0.0, 0.0], [8.0, 0.0], [12.0, -3.0]]
    truth = {"A": (60.0, 150.0), "B": (35.0, 200.0), "C": (80.0, 90.0)}
    joints = [
        SyntheticJoint(*truth["A"], chainage=2.5, elevation=103.0),
        SyntheticJoint(*truth["B"], chainage=6.0, elevation=101.5),
        SyntheticJoint(*truth["C"], chainage=10.5, elevation=102.5),
    ]
    points, colors = synthetic_wall(rng, polyline, joints, height=5.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    cloud_path = tmp_path / "wall.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd)

    cfg = {
        "paths": {"point_cloud": str(cloud_path)},
        "preprocessing": {"voxel_size": 0.04},
        "wall_frame": {"polyline": polyline, "viewer_point": [4.0, -8.0]},
        "normals": {"search_radius": 0.15},
        "patches": {"link_distance": 0.08},
        "facade": {"resolution": 0.05},
        "trace_mapping": {"hough_min_line_length_px": 15},
        "trace_matching": {"tolerance": 0.12},
    }
    result = run_pipeline(cfg)

    natural = {}
    for name, (dip, dip_dir) in truth.items():
        errors = {s.set_id: angular_difference(dip, dip_dir, s.mean_dip, s.mean_dip_direction) for s in result.discontinuity_sets}
        best = min(errors, key=errors.get)
        assert errors[best] < 1.0, f"set {name} not recovered precisely"
        natural[name] = best
    natural_ids = set(natural.values())

    for dset in result.discontinuity_sets:
        for p in dset.patches:
            state = result.trace_confirmed[p.patch_id]
            if dset.set_id in natural_ids:
                assert state is True, f"natural joint patch {p.patch_id} (set {dset.set_id}) not confirmed"
            else:
                assert state is None, "excavated-face patches must be wall-parallel"

    save_results(result, tmp_path / "out")
    patches = pd.read_csv(tmp_path / "out" / "discontinuity_patches.csv")
    assert patches["chainage_m"].between(0.0, 13.0).all()
    assert patches["elevation_m"].between(100.0, 105.0).all()
    assert (tmp_path / "out" / "facade_rgb.tif").exists()
    assert (tmp_path / "out" / "wall_frame.json").exists()


def test_thermal_folders_correction_and_overlay(tmp_path):
    """Planar RGB / thermal pairs are told apart by name; the local correction interpolates world shifts by
    chainage; the blend shows the thermal image only where it has data; the popup page carries both images."""
    import rasterio
    from rasterio.transform import from_origin

    from groundwater.facade import planar
    from groundwater.facade import thermal as TH
    from groundwater.facade.frame import WallFrame
    from groundwater.viz import thermal_view as TV

    def write(path, bands):
        with rasterio.open(path, "w", driver="GTiff", width=8, height=6, count=bands, dtype="uint8",
                           crs="EPSG:4978", transform=from_origin(0, 6, 1, 1)) as ds:
            ds.write(np.zeros((bands, 6, 8), np.uint8))

    for sub, stem in (("planar_rgb", "west"), ("planar_thermal", "west")):
        (tmp_path / sub).mkdir()
        write(tmp_path / sub / f"{stem}_dem.tif", 1)
        write(tmp_path / sub / f"{stem}_ortho.tif", 4)
    f = TH.find_folders(tmp_path)
    assert f["rgb"].name == "planar_rgb" and f["thermal"].name == "planar_thermal"
    # a single-band temperature mosaic next to the thermal DEM is found as such, and not taken for the DEM
    write(tmp_path / "planar_thermal" / "a_temperature_ortho.tif", 1)
    assert TH.temperature_ortho(f["thermal"]).name == "a_temperature_ortho.tif"
    assert planar.find_planar_pair(f["thermal"])[0].name == "west_dem.tif"
    assert TH.temperature_ortho(f["rgb"]) is None

    frame = WallFrame([[0.0, 0.0], [20.0, 0.0]])
    corr = {"knots_xy": [[0.0, 0.0], [10.0, 0.0]], "t": [[0.0, 0.0, 0.0], [0.2, 0.0, -0.1]]}
    shift = planar._correction_by_chainage(frame, corr)
    assert np.allclose(shift([5.0, 10.0, 15.0]), [[0.1, 0, -0.05], [0.2, 0, -0.1], [0.2, 0, -0.1]])
    assert planar._correction_by_chainage(frame, None) is None

    rgb = np.full((4, 4, 3), 100, np.uint8)
    th = np.full((4, 4, 3), 255, np.uint8)
    th[:2] = (200, 0, 0)
    b = TV.blend(rgb, th, 0.5)
    assert (b[:2] == (150, 50, 50)).all() and (b[2:] == 100).all()          # no thermal data: RGB unchanged
    page = TV.overlay_page({"rgb": rgb, "thermal": th, "ext": [0, 4, 0, 4], "res": 1.0}, "t", 0.3)
    assert page.count("data:image/") == 2 and 'value="30"' in page


def test_thermal_palette_index_and_local_z():
    """An iron-palette ramp maps to a rising index; a cold spot stands out in the local z-score, its surroundings do not."""
    from groundwater import thermal_anomaly as TA

    ramp = np.array([[97, 3, 117], [181, 10, 144], [223, 42, 62], [247, 75, 7], [249, 138, 3]], np.uint8)
    img = np.repeat(ramp[None], 40, axis=0)                        # 40 x 5 image of the five palette steps
    valid = np.ones(img.shape[:2], bool)
    lut = TA.palette_lut(img, valid, steps=5)
    idx = TA.palette_index(img, valid, lut)
    assert np.all(np.diff(idx[0]) > 0) and idx[0, 0] == 0 and idx[0, -1] == 1
    field = np.full((80, 80), 0.6, np.float32) + np.random.default_rng(0).normal(0, 0.02, (80, 80)).astype(np.float32)
    field[38:42, 38:42] = 0.3                                      # a cold spot
    ok = np.ones_like(field, bool)
    m, sd, _ = TA._masked_mean_std(field, ok, 10.0)
    z = (field - m) / np.maximum(sd, 0.02)
    assert z[38:42, 38:42].max() < -2.0 and np.abs(z[:20, :20]).max() < 4.0


def test_section_polygon_in_plan_limits_the_region():
    """A section polygon in map coordinates keeps only that section's points, whatever their wall coordinates."""
    from groundwater.facade.region import is_active, region_mask

    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    uvw = np.array([[5.0, 1.0, 0.0], [5.0, 1.0, 0.0], [5.0, 1.0, 0.0]])        # same wall position for all
    region = {"map_polygon": [[-1, -1], [12, -1], [12, 1], [-1, 1]]}
    assert is_active(region)
    assert region_mask(uvw, region, xy).tolist() == [True, True, False]
    assert region_mask(uvw, region).all()                                     # without map coordinates: no effect
