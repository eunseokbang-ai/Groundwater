"""Trace families, kinematic screening, analysis region and the chunked facade accumulator."""
import numpy as np

from groundwater.discontinuity.kinematics import intersection, kinematic_check
from groundwater.discontinuity.trace_mapping import Trace2D, skeleton_segments, thin
from groundwater.discontinuity.trace_sets import apparent_spacing, trace_families
from groundwater.facade.orthophoto import FacadeAccumulator, rasterize_facade
from groundwater.facade.region import region_mask


def _trace(p, q, xyz=None):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return Trace2D(coords=np.array([p, q]), pixel_coords=np.zeros((2, 2)), length_m=float(np.linalg.norm(q - p)),
                   xyz=xyz)


def test_kinematics_textbook_cases():
    face = [{"u0": 0.0, "u1": 5.0, "face_dip": 80.0, "face_dip_direction": 180.0, "n_points": 1}]
    k = kinematic_check(face, {"A": (45, 175), "B": (80, 0), "C": (60, 120), "D": (60, 240), "E": (20, 180)})
    modes = {(r["mode"], r["joints"]) for _, r in k.iterrows()}
    assert ("평면파괴", "A") in modes                 # daylights, steeper than friction
    assert ("전도파괴", "B") in modes                 # steep, dipping into the face
    assert ("쐐기파괴", "C × D") in modes            # intersection plunges 41 deg toward the face
    assert not any(j == "E" for _, j in modes)       # 20 deg < friction angle
    plunge, trend = intersection((60, 120), (60, 240))
    assert abs(plunge - 40.9) < 0.5 and abs(trend - 180) < 0.5


def test_trace_families_link_and_trace_only_orientation():
    rng = np.random.default_rng(0)
    traces = []
    # family 1: horizontal traces on the facade (e.g. bedding)
    for i in range(20):
        traces.append(_trace([i, 100 + i % 5], [i + 1.0, 100 + i % 5]))
    # family 2: vertical traces of a vertical joint set striking N-S (pole along x) on a stepped face:
    # in 3D they run vertically, some also along y (on face segments of another orientation)
    for i in range(20):
        u = 30 + i
        d = np.array([0.0, rng.uniform(-0.6, 0.6), 1.0])
        d /= np.linalg.norm(d)
        xyz = np.array([[u, 0.0, 100.0], [u, 0.0, 100.0] + 2 * d])
        traces.append(_trace([u, 100.0], [u + 0.05, 102.0], xyz=xyz))
    fams = trace_families(traces)
    angles = sorted(round(f.apparent_angle) % 180 for f in fams)
    assert len(fams) == 2 and (angles[0] <= 3 or angles[0] >= 177) and 85 <= angles[1] <= 95
    vert = next(f for f in fams if 60 < f.apparent_angle < 120)
    assert vert.dip is not None and abs(vert.dip - 90) < 5 and min(abs(vert.dip_direction - 90), abs(vert.dip_direction - 270)) < 5
    flat = next(f for f in fams if f is not vert)
    assert flat.dip is None                          # no 3D directions: apparent angle only


def test_apparent_spacing_along_scanlines():
    traces = [_trace([0, z], [10, z]) for z in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert abs(apparent_spacing(traces, 0.0) - 0.5) < 1e-6


def test_region_mask_limits_and_polygon():
    uvw = np.array([[1, 1, 0], [5, 5, 0], [9, 1, 0], [5, 12, 0]], float)
    assert region_mask(uvw, {"chainage": [0, 6]}).tolist() == [True, True, False, True]
    assert region_mask(uvw, {"elevation": [None, 10]}).tolist() == [True, True, True, False]
    tri = [[0, 0], [10, 0], [5, 10]]
    assert region_mask(uvw, {"polygon": tri}).tolist() == [True, True, True, False]


def test_accumulator_in_chunks_matches_single_pass():
    rng = np.random.default_rng(1)
    uvw = np.column_stack([rng.uniform(0, 4, 20000), rng.uniform(0, 3, 20000), rng.normal(0, 0.05, 20000)])
    rgb = rng.uniform(0, 1, (20000, 3))
    one, d1 = rasterize_facade(uvw, rgb, 0.05)
    acc = FacadeAccumulator.covering(uvw, 0.05)
    for s in range(0, 20000, 3000):
        acc.add(uvw[s:s + 3000], rgb[s:s + 3000])
    two, d2 = acc.finish()
    assert np.array_equal(one.array, two.array)
    assert np.allclose(d1.array, d2.array, equal_nan=True)


def test_skeleton_segments_split_an_l_shape():
    m = np.zeros((60, 60), bool)
    m[10:12, 5:50] = True          # horizontal arm
    m[10:55, 48:50] = True         # vertical arm
    segs = skeleton_segments(thin(m), min_len_px=15, tol_px=1.0)
    dirs = sorted(abs(np.degrees(np.arctan2(*(b - a)))) % 180 for _, a, b in segs)
    assert len(segs) == 2 and dirs[0] < 10 and 80 < dirs[1] < 100


def test_traces_off_rock_are_dropped_manual_kept():
    from affine import Affine

    from groundwater.io.raster import Raster
    from groundwater.pipeline import drop_traces_off_rock

    labels = np.full((1, 100, 200), -1, np.int16)
    labels[0, :, :100] = 0                      # joint-set surfaces on the left half only (0.1 m cells)
    sets = Raster(array=labels, transform=Affine(0.1, 0, 0, 0, -0.1, 10.0), crs=None, nodata=-1)
    on_rock = _trace([2, 5], [4, 5])
    canopy = _trace([15, 5], [17, 5])
    hand = Trace2D(coords=np.array([[15.0, 6.0], [17.0, 6.0]]), pixel_coords=np.zeros((2, 2)), length_m=2.0, source="manual")
    kept = drop_traces_off_rock([on_rock, canopy, hand], sets)
    ids = {id(t) for t in kept}
    assert id(on_rock) in ids and id(hand) in ids and id(canopy) not in ids


def test_far_apart_collinear_pieces_are_not_one_trace():
    from groundwater.discontinuity.trace_mapping import _split_straight

    # two short collinear edges 2 m (100 px) apart -- a subset the splitter can be left with inside a branching
    # skeleton: they must stay two traces, not one long line across the sound rock between them
    rc = np.array([[20.0, c] for c in list(range(0, 30)) + list(range(130, 160))])
    out = []
    _split_straight(rc, tol=1.0, min_len=15, out=out)
    lengths = sorted(float(np.linalg.norm(b - a)) for a, b in out)
    assert len(lengths) == 2 and max(lengths) < 31


def test_wall_frame_runs_continuously_through_a_bend():
    from groundwater.facade.frame import WallFrame

    # the Ulsan wall line: a 16 deg bend 50.6 m along; the face is ~5 m behind it, on the outer side of the bend
    V = [[216133.812, 334135.821], [216096.445, 334101.757], [216071.313, 334089.251]]
    for fillet, smooth in ((False, False), (True, True)):
        f = WallFrame(V, fillet=fillet)
        # a line of rock points 5 m behind the wall line, 5 cm apart, along the whole wall
        u = np.arange(30, 70, 0.05)
        dense = WallFrame(V, fillet=True)
        P = dense.to_world(np.column_stack([u, np.zeros_like(u), np.full_like(u, -5.0)]))
        uu = f.to_wall(P)[:, 0]
        step = np.diff(uu)
        if smooth:
            assert step.min() > 0.03 and step.max() < 0.07           # no squeeze, no jump
            back = f.to_world(f.to_wall(P))
            assert np.abs(back - P).max() < 1e-6                      # exact round trip
        else:
            assert step.min() < 0.005                                 # the plain rule piles points on the vertex


def test_valley_detector_finds_a_wide_dark_fissure_not_noise():
    from affine import Affine

    from groundwater.discontinuity.trace_mapping import extract_traces, hysteresis, valley_strength
    from groundwater.io.raster import Raster

    rng = np.random.default_rng(3)
    H, W = 150, 150                                     # 3 x 3 m at 2 cm
    img = 150 + rng.normal(0, 6, (H, W))
    x = np.arange(W)
    img[:, :] -= 70 * np.exp(-((x - 75) / 10.0) ** 2)[None, :]  # a ~35 cm wide dark vertical fissure
    rgb = np.repeat(img[None].clip(0, 255).astype(np.uint8), 3, axis=0)
    r = Raster(array=rgb, transform=Affine(0.02, 0, 0, 0, -0.02, 3.0), crs=None, nodata=None)
    tr = extract_traces(r, resolution_m=0.02, use_depth=False)
    found = [t for t in tr if t.group // 1_000_000 == 3]
    assert any(t.length_m > 2.0 and abs(t.coords[0, 0] - 1.5) < 0.1 for t in found)
    assert not any(t.group // 1_000_000 == 0 for t in tr)          # the fixed 15 cm black-hat misses it

    noise = 150 + rng.normal(0, 6, (H, W))
    s = valley_strength(noise.astype(np.float32), 0.02)
    assert not hysteresis(s, 28.0, 14.0, np.ones_like(noise, bool)).any()


def test_elongated_keeps_fissures_drops_blotches():
    from groundwater.discontinuity.trace_mapping import elongated

    m = np.zeros((200, 200), bool)
    m[20:180, 95:110] = True              # 3.2 m x 0.3 m fissure at 2 cm
    m[150:190, 10:50] = True              # 0.8 m blotch
    out = elongated(m, 0.02)
    assert out[100, 100] and not out[170, 30]


def test_join_collinear_keeps_the_full_extent():
    from groundwater.discontinuity.trace_mapping import _join_collinear

    segs = [(1, np.array([0.0, 0.0]), np.array([0.0, 100.0])), (2, np.array([0.5, 90.0]), np.array([0.0, 95.0])),
            (3, np.array([0.0, 103.0]), np.array([0.2, 140.0])), (4, np.array([50.0, 0.0]), np.array([80.0, 0.0]))]
    out = _join_collinear(segs, gap_px=5, off_px=2.0)
    lengths = sorted(round(float(np.linalg.norm(b - a))) for _, a, b in out)
    assert 140 in lengths and 30 in lengths           # pieces along x=0 joined end to end, nothing lost


def test_to_wall_matches_nearest_point_on_a_densely_sampled_line():
    from groundwater.facade.frame import WallFrame

    rng = np.random.default_rng(3)
    verts = np.array([[0, 0], [10, 0], [14, 6], [8, 14]], float)
    for fillet in (False, True):
        f = WallFrame(verts, fillet=fillet)
        s = np.linspace(0, f.length, 40001)                       # the wall line every 0.5 mm
        line = f.to_world(np.column_stack([s, np.zeros_like(s), np.zeros_like(s)]))[:, :2]
        pts = np.column_stack([rng.uniform(-3, 17, 3000), rng.uniform(-4, 18, 3000), rng.uniform(0, 5, 3000)])
        uvw = f.to_wall(pts)
        d = np.linalg.norm(pts[:, None, :2] - line[None], axis=2)
        k = d.argmin(axis=1)
        inner = (k > 0) & (k < len(s) - 1)                        # beyond the ends the line is extended
        assert np.abs(np.abs(uvw[inner, 2]) - d[inner].min(axis=1)).max() < 1e-3
        # the chainage is the nearest point's, unless two parts of the line are equally near (inside a bend)
        far = np.abs(s[None, :] - s[k][:, None]) > 1.0                # another part of the line, not a neighbour
        tied = np.where(far, d, np.inf).min(axis=1) - d.min(axis=1) < 1e-3
        ok = inner & ~tied
        assert np.median(np.abs(uvw[ok, 0] - s[k[ok]])) < 1e-3
        assert np.allclose(uvw[:, 1], pts[:, 2])
        if fillet:                                                # and the way back (with arcs at the bends: a
            back = f.to_world(uvw)                                # sharp corner sends a fan of points to its vertex)
            assert np.abs(back[inner] - pts[inner]).max() < 1e-6


def test_stereonet_density_map_options_draw():
    """Poles only, Kamb density (area-weighted or not, with poles) and Schmidt counts all give a PNG."""
    from types import SimpleNamespace

    from groundwater import report

    rng = np.random.default_rng(0)
    def patch(dip, dd):
        return SimpleNamespace(dip=dip + rng.normal(0, 4), dip_direction=dd + rng.normal(0, 6), area=rng.uniform(0.1, 2))
    sets = [SimpleNamespace(set_id=k, patches=[patch(dip, dd) for _ in range(40)], mean_dip=dip, mean_dip_direction=dd)
            for k, (dip, dd) in enumerate([(10, 300), (80, 290), (88, 150)])]
    result = SimpleNamespace(discontinuity_sets=sets, face_windows=[])
    for density in (None, {"sigma": 3.0}, {"sigma": 2.0, "weighted": True, "poles": True, "cmap": "Blues"},
                    {"method": "schmidt", "cmap": "viridis"}):
        assert report.fig_stereonet(result, [], density)[:4] == b"\x89PNG"


def test_trace_family_lists_every_set_with_the_same_angle_on_the_wall():
    """Two sets cutting the wall at the same angle cannot be told apart: both are candidates, the linked one
    first; a set further off (beyond the tie margin or the link window) is not."""
    from types import SimpleNamespace

    from groundwater.discontinuity.trace_sets import _candidates, family_sets, family_tag

    expected = {0: 176.3, 1: 114.7, 2: 115.1, 3: 125.0}
    assert _candidates(102.0, expected, 1) == [1, 2]            # 12.7 vs 13.1 deg: a tie; set 3 (23 deg) is out
    assert _candidates(173.0, expected, 0) == [0]
    assert _candidates(102.0, expected, None) == []
    fam = SimpleNamespace(linked_set=1, apparent_angle=102.0, candidate_sets=[1, 2])
    assert family_sets(fam) == [1, 2] and family_tag([1, 2]) == "Set 1 또는 2" and family_tag([]) == "트레이스 전용"
    old = SimpleNamespace(linked_set=0, apparent_angle=173.0)    # saved before candidates were kept, no frame given
    assert family_sets(old) == [0]


def test_kinematic_rules_and_cell_map():
    """The three Markland rules, and the cell map on a flat 70 deg face dipping east with one set per mode."""
    from types import SimpleNamespace

    from rasterio.transform import Affine

    from groundwater.discontinuity import kinematics as K
    from groundwater.facade.frame import WallFrame
    from groundwater.io.raster import Raster

    # face 70/090, friction 30
    assert K.planar_possible(45, 95, 70, 90, 30)                       # daylights, steeper than friction
    assert not K.planar_possible(75, 90, 70, 90, 30)                   # steeper than the face: does not daylight
    assert not K.planar_possible(45, 140, 70, 90, 30)                  # outside +-20 deg
    assert K.toppling_possible(80, 270, 70, 90, 30)                    # (90 - 80) + 30 < 70
    assert not K.toppling_possible(50, 270, 70, 90, 30)
    assert K.wedge_possible(40, 100, 70, 90, 30) and not K.wedge_possible(40, 200, 70, 90, 30)
    # wall along x (u = x), viewer at -y: w toward the viewer, a face leaning back 20 deg (dip 70) toward -y (180)
    frame = WallFrame([[0.0, 0.0], [20.0, 0.0]])
    res = 0.05
    v = 10.0 - (np.arange(200) + 0.5) * res                              # rows: elevation 10 -> 0
    w = np.tile((-np.tan(np.radians(20.0)) * v)[:, None], (1, 400)).astype(np.float32)
    depth = Raster(array=w[None], transform=Affine(res, 0, 0.0, 0, -res, 10.0), crs=None, nodata=np.nan)
    result = SimpleNamespace(wall_frame=frame, facade_depth=depth, facade_rgb=None, facade_sets=None, trace_families=[],
                             discontinuity_sets=[])
    dip, dd, overhang, cover, _ = K.local_faces(result, cell_m=1.0, smooth_m=2.0)
    inner = dip[2:-2, 2:-2]
    assert np.allclose(inner, 70.0, atol=1.0) and np.allclose(dd[2:-2, 2:-2], 180.0, atol=1.0) and not overhang.any()
    joints = {"plane": (45.0, 185.0), "topple": (80.0, 0.0), "flat": (10.0, 180.0)}
    km = K.kinematic_map(result, joints, near_m=None, exclude_vegetation=False, cell_m=1.0, smooth_m=2.0)
    c = (slice(2, -2), slice(2, -2))
    assert km.modes["평면파괴"][c].all() and set(km.joints["평면파괴"][c].ravel()) == {"plane"}
    assert km.modes["전도파괴"][c].all() and set(km.joints["전도파괴"][c].ravel()) == {"topple"}


def test_near_vertical_set_tested_both_ways_only_on_request():
    """By default a set is tested in its measured direction; with vertical_deg a near-vertical one also the other
    way, so 89.8/349 and 89.8/169 then give the same result."""
    from groundwater.discontinuity import kinematics as K

    face = [{"u0": 0.0, "u1": 5.0, "face_dip": 80.0, "face_dip_direction": 350.0, "n_points": 1}]
    assert not len(kinematic_check(face, {"J": (89.8, 349.0)}))                      # dips out of the face
    assert list(kinematic_check(face, {"J": (89.8, 169.0)})["mode"]) == ["전도파괴"]  # dips into it
    a = kinematic_check(face, {"J": (89.8, 349.0)}, vertical_deg=85)
    b = kinematic_check(face, {"J": (89.8, 169.0)}, vertical_deg=85)
    assert list(a["mode"]) == list(b["mode"]) == ["전도파괴"]
    assert a["detail"].iloc[0] == "90/169 (반대 방향)" and b["detail"].iloc[0] == "90/169"
    # rule level, arrays of cell faces: both directions alike above the limit, only the measured one below it
    fdip, fdd = np.array([80.0, 80.0, 60.0]), np.array([350.0, 170.0, 350.0])
    for rule in (K.toppling_possible, K.planar_possible):
        assert (rule(88.0, 349.0, fdip, fdd, 30, vertical_deg=85) == rule(88.0, 169.0, fdip, fdd, 30, vertical_deg=85)).all()
    assert not K.toppling_possible(84.0, 349.0, 80, 350, 30, vertical_deg=85)
    assert K.planar_possible(86.0, 170.0, 88, 350, 30, vertical_deg=85) and not K.planar_possible(86.0, 170.0, 88, 350, 30)


def test_domain_stereonet_uses_the_range_face_and_joints():
    """A chainage range (domain): its face from the windows it overlaps (weighted by overlap), and only the joint
    faces whose centre lies in it are tested."""
    from types import SimpleNamespace

    from groundwater.discontinuity.geometry import DiscontinuityPatch, DiscontinuitySet
    from groundwater.facade.frame import WallFrame
    from groundwater.viz import kinematic_view as KV

    windows = [{"u0": 0.0, "u1": 5.0, "face_dip": 70.0, "face_dip_direction": 180.0, "height_m": 10.0},
               {"u0": 5.0, "u1": 10.0, "face_dip": 80.0, "face_dip_direction": 180.0, "height_m": 10.0},
               {"u0": 10.0, "u1": 15.0, "face_dip": 90.0, "face_dip_direction": 200.0, "height_m": 10.0}]
    assert KV.domain_face(windows, (0.0, 5.0)) == (70.0, 180.0)
    dip, dd = KV.domain_face(windows, (5.0, 15.0))
    assert dip == 85.0 and 185.0 < dd < 195.0
    frame = WallFrame([[0.0, 0.0], [20.0, 0.0]])

    def patch(u, d, a, i):
        c = frame.to_world(np.array([[u, 5.0, 0.0]]))[0]
        return DiscontinuityPatch(0, i, 50, c, np.zeros(3), d, a, 0.0, 1.0, 1.0)

    # planar-prone faces (45 deg toward the face) at u = 2, toppling-prone (80 deg into the slope) at u = 12
    sets = [DiscontinuitySet(0, 45.0, 180.0, 50.0, 100, [patch(2.0, 45.0, 180.0, i) for i in range(5)]),
            DiscontinuitySet(1, 80.0, 0.0, 50.0, 100, [patch(12.0, 80.0, 0.0, i) for i in range(5)])]
    result = SimpleNamespace(wall_frame=frame, discontinuity_sets=sets, face_windows=windows)
    near = KV.critical(result, (70.0, 180.0), 30.0, 20.0, (0.0, 5.0))["counts"]
    assert near["평면파괴"] == {"critical": 5, "total": 5, "share": 1.0} and near["전도파괴"]["critical"] == 0
    far = KV.critical(result, (85.0, 180.0), 30.0, 20.0, (10.0, 15.0))["counts"]
    assert far["평면파괴"]["total"] == 5 and far["전도파괴"]["critical"] == 5
    rows = KV.domain_table(result, 30.0, 20.0, max_pairs=50)
    assert [r["절리면 수"] for r in rows] == [5, 0, 5]


def test_site_folder_keeps_thermal_apart_and_reads_las_epsg(tmp_path):
    """Thermal models (named so) are listed apart, never taken as the photogrammetry cloud or mesh; a decimated
    mesh comes before a too-heavy one; the EPSG code is read from the LAS GeoTIFF keys without pyproj."""
    from types import SimpleNamespace

    from groundwater import site_folder as S

    for name, size in (("site_RGB.las", 300), ("site_thermal.las", 900), ("site_thermal.obj", 50), ("big.obj", 40),
                       ("small.obj", 20)):
        (tmp_path / name).write_bytes(b"0" * size)
    old = S.HEAVY_MESH
    S.HEAVY_MESH = 30                      # 'big.obj' counts as too heavy here
    try:
        f = S.scan(tmp_path)
    finally:
        S.HEAVY_MESH = old
    assert [p.name for p in f["point_cloud"]] == ["site_RGB.las"]
    assert {p.name for p in f["thermal"]} == {"site_thermal.las", "site_thermal.obj"}
    assert [p.name for p in f["mesh"]] == ["small.obj", "big.obj"]
    keys = [SimpleNamespace(id=1024, value_offset=1), SimpleNamespace(id=3072, value_offset=5186)]
    header = SimpleNamespace(vlrs=[SimpleNamespace(record_id=34735, geo_keys=keys)], parse_crs=lambda: None)
    assert S.las_epsg(header) == "EPSG:5186"


def test_folder_crs_skips_geocentric_planar_exports(tmp_path):
    """A wall-parallel planar export (geocentric EPSG:4978) must not become the site's map CRS."""
    import rasterio
    from rasterio.transform import from_origin

    from groundwater import site_folder as S

    for name, crs in (("a_planar_dem.tif", "EPSG:4978"), ("b_ortho_map.tif", "EPSG:5186")):
        with rasterio.open(tmp_path / name, "w", driver="GTiff", width=4, height=4, count=1, dtype="uint8", crs=crs,
                           transform=from_origin(0, 4, 1, 1)) as ds:
            ds.write(np.zeros((1, 4, 4), np.uint8))
    assert S.folder_crs(tmp_path) == "EPSG:5186"


def test_lidar_rear_layer_keeps_the_rock_behind_the_net():
    """Per wall cell only the rearmost layer (lowest w) is kept: the rock behind a net / canopy 0.5 m in front."""
    from groundwater import lidar

    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 2, 4000), rng.uniform(0, 2, 4000)
    rock = np.column_stack([u, v, rng.normal(-1.0, 0.01, 4000)])          # rock at w = -1
    net = np.column_stack([u[:2000], v[:2000], rng.normal(-0.5, 0.01, 2000)])
    uvw = np.vstack([rock, net])
    sel = lidar.rear_layer(uvw, cell_m=0.2, layer_m=0.06)
    assert sel[:4000].mean() > 0.95 and sel[4000:].mean() < 0.01


def test_height_datum_helpers():
    """reg_crs carries the geoid suffix only for orthometric results; a DJI Terra WKT is read as orthometric."""
    from types import SimpleNamespace

    from groundwater import heights as H

    assert H.reg_crs({"crs": "EPSG:5186", "height_datum": "orthometric"}) == "EPSG:5186+KNGeoid18"
    assert H.reg_crs({"crs": "EPSG:5186"}) == "EPSG:5186"
    assert H.base_crs("EPSG:5186+KNGeoid18") == "EPSG:5186"
    wkt = 'COMPD_CS["x",PROJCS["Korea 2000 / Central Belt 2010",AUTHORITY["EPSG","5186"]],VERT_CS["KNGeoid18"]]'
    hdr = SimpleNamespace(vlrs=[SimpleNamespace(record_id=2112, string=wkt)])
    assert H.las_datum(hdr) == "orthometric"
    from groundwater import site_folder as S
    assert S.las_epsg(SimpleNamespace(vlrs=hdr.vlrs, parse_crs=lambda: None)) == "EPSG:5186"
    if H.available():                       # the geoid model is in the repository's field-data folder
        from groundwater.viz.tiles3d import _to_ecef

        n = H.geoid_n_map([203127.0], [384865.0], "EPSG:5186")[0]
        assert 24.0 < n < 25.0                                            # Iksan
        p = np.array([[203127.0, 384865.0, 100.0]])
        e_orth = _to_ecef(p, "EPSG:5186+KNGeoid18")
        e_ell = _to_ecef(p + [0, 0, n], "EPSG:5186")
        assert np.allclose(e_orth, e_ell, atol=1e-3)


def test_lidar_steep_surface_keeps_the_wall_not_the_hillside():
    """A vertical wall (w constant) is kept, a 40-degree slope behind the trees is not."""
    from groundwater import lidar

    rng = np.random.default_rng(1)
    u, v = rng.uniform(0, 10, 20000), rng.uniform(0, 10, 20000)
    wall = np.column_stack([u, v, rng.normal(0, 0.01, 20000)])
    slope = np.column_stack([u, v + 20, -(v) / np.tan(np.radians(40))])      # rises 1 m per 1.19 m back
    keep = lidar.steep_surface(np.vstack([wall, slope]))
    assert keep[:20000].mean() > 0.95 and keep[20000:].mean() < 0.05


def test_lidar_rock_surface_keeps_ledges_not_the_ground_above_or_below():
    """A quarry wall in wall coordinates (v up, w toward the pit): floor in front, a vertical face with a 45-degree
    ledge in the middle, the vertical face again, then the gentle original ground (35 deg) above the crest. The
    face and its ledge are rock, the ground above and the floor are not, and the face just below the crest stays
    steep although the ground behind the crest falls in the same window."""
    from groundwater import lidar

    rng = np.random.default_rng(3)
    n = 4000

    def strip(v0, v1, w_of_v):
        u, v = rng.uniform(0, 10, n), rng.uniform(v0, v1, n)
        return np.column_stack([u, v, w_of_v(v)])

    floor = strip(-0.4, 0.0, lambda v: 5.0 - (v + 0.4) / 0.4 * 5.0)              # 5 m of floor, ~5 deg
    face1 = strip(0.0, 6.0, lambda v: np.zeros_like(v))
    ledge = strip(6.0, 8.0, lambda v: -(v - 6.0))                                  # 45 deg, 2 m high
    face2 = strip(8.0, 14.0, lambda v: np.full_like(v, -2.0))
    above = strip(14.0, 24.0, lambda v: -2.0 - (v - 14.0) / np.tan(np.radians(35)))
    # tree trunks / the scan's far edge behind the crest: seen from the pit, a vertical surface 20 m further back
    trunks = strip(24.0, 30.0, lambda v: np.full_like(v, -2.0 - 10.0 / np.tan(np.radians(35)) - 20.0))
    pts = np.vstack([floor, face1, ledge, face2, above, trunks])
    rock, led = lidar.rock_surface(pts)
    part = np.repeat(np.arange(6), n)
    u_in = (pts[:, 0] > 2) & (pts[:, 0] < 8)                                       # away from the strips' ends
    share = [rock[(part == i) & u_in].mean() for i in range(5)]
    assert share[1] > 0.95 and share[3] > 0.95                                     # the face
    assert share[2] > 0.9 and led[(part == 2) & u_in].mean() > 0.3                 # the ledge: gentle, kept
    #                                    (its edges read steep: the 1 m fit reaches the faces above and below)
    assert share[4] < 0.1                                                          # the original ground above
    assert rock[(part == 5) & u_in].mean() < 0.05                                  # steep, but outside the face
    assert rock[(part == 0) & u_in].mean() < 0.1                                   # the floor
    top = (part == 3) & u_in & (pts[:, 1] > 12.5)
    assert rock[top].mean() > 0.95 and not led[top].any()                          # steep up to the crest


def test_lidar_face_span_sees_the_face_top_in_front_of_the_far_edge():
    """A ridge-top wall: the scan's far edge 30 m behind lies at the same wall u, v as the top 6 m of the face, so
    the rear layer there follows it. The face span still runs up to the crest (the far points are left out and the
    span found again); the far points and the ground above the crest are outside, for the bare rock tested as it is."""
    from groundwater import lidar

    rng = np.random.default_rng(4)
    n = 6000

    def strip(v0, v1, w_of_v):
        u, v = rng.uniform(0, 10, n), rng.uniform(v0, v1, n)
        return np.column_stack([u, v, w_of_v(v)])

    face = strip(0.0, 20.0, lambda v: rng.normal(0, 0.02, len(v)))
    far = strip(14.0, 20.0, lambda v: np.full_like(v, -30.0))                    # behind the top 6 m of the face
    above = strip(20.0, 26.0, lambda v: -(v - 20.0) / np.tan(np.radians(30)))      # the ground past the crest
    base = np.vstack([face, far, above])
    test = lidar.face_span(np.zeros((0, 3)), base)
    on, _ = test(base)
    part = np.repeat(np.arange(3), n)
    u_in = (base[:, 0] > 2) & (base[:, 0] < 8)
    top = (part == 0) & u_in & (base[:, 1] > 14.0) & (base[:, 1] < 19.5)
    assert on[top].mean() > 0.95                                                   # the face up to the crest
    assert on[(part == 0) & u_in].mean() > 0.97
    assert on[(part == 1) & u_in].mean() < 0.02                                    # the far edge
    assert on[(part == 2) & u_in & (base[:, 1] > 21.0)].mean() < 0.05             # the ground above


def test_lidar_streaming_rear_prefilter_matches_the_rear_layer():
    """The running two rearmost depths per cell drop, chunk by chunk, only points the final rear layer drops too."""
    from groundwater import lidar

    rng = np.random.default_rng(2)
    idx = rng.integers(0, 50, 6000)
    w = np.where(rng.random(6000) < 0.5, rng.normal(-1.0, 0.01, 6000), rng.normal(-0.4, 0.01, 6000))
    r1, r2 = np.full(50, np.inf, np.float32), np.full(50, np.inf, np.float32)
    kept = []
    for part in np.array_split(np.arange(6000), 6):             # streamed in chunks
        lidar._update_rear(r1, r2, idx[part], w[part].astype(np.float32))
        kept.append(part[w[part] <= r2[idx[part]] + lidar.PREFILTER_M])
    kept = np.concatenate(kept)
    kept = kept[w[kept] <= r2[idx[kept]] + lidar.PREFILTER_M]
    for c in range(50):
        s = np.sort(w[idx == c])
        assert np.isclose(r1[c], s[0], atol=1e-6) and np.isclose(r2[c], s[1], atol=1e-6)
    rock = np.flatnonzero(w < -0.7)
    assert np.isin(rock, kept).all() and (w[kept] > -0.7).mean() < 0.05


def test_section_core_ownership_by_chainage():
    """With an overlap buffer only the core's chainage counts; without one everything does."""
    from groundwater.facade.region import core_chainage, in_core, in_polygon

    cfg = {"section": {"core_chainage": [10.0, 50.0]}}
    assert core_chainage(cfg) == (10.0, 50.0) and core_chainage({}) is None
    assert in_core([5, 10, 30, 51], cfg).tolist() == [False, True, True, False]
    assert in_core([5, 51], {}).all()
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert in_polygon(sq, np.array([[5.0, 5.0], [15.0, 5.0]])).tolist() == [True, False]


def test_scan_prefers_the_folders_own_wall_line(tmp_path):
    """A wall_line.csv in a sub-folder (drafts of other sections) does not replace the folder's own."""
    from groundwater import site_folder as S

    (tmp_path / "wall_line.csv").write_text("kind,x,y\nwall,0,0\nwall,10,0\n", encoding="utf-8")
    sub = tmp_path / "drafts" / "E"
    sub.mkdir(parents=True)
    (sub / "wall_line.csv").write_text("kind,x,y\nwall,5,5\nwall,9,9\n", encoding="utf-8")
    assert S.scan(tmp_path)["wall_line"] == tmp_path / "wall_line.csv"


def test_las_area_read_keeps_the_area_evenly_thinned(tmp_path):
    """Only the points in the box are read, thinned evenly (step doubling) to at most max_points."""
    import laspy

    from groundwater.io import point_cloud as PC

    rng = np.random.default_rng(4)
    n = 200_000
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = [0.001] * 3
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(0, 10, n)
    las.red = las.green = las.blue = np.full(n, 30000, np.uint16)
    las.write(tmp_path / "scan.las")
    pcd = PC.load_point_cloud(tmp_path / "scan.las", 5_000, bbox=((0, 0), (50, 50)))
    pts = np.asarray(pcd.points)
    assert len(pts) <= 5_000 and len(pts) > 2_000
    assert pts[:, :2].max() <= 50.001 and PC.load_stats["points_in_area"] > 40_000
    assert abs(np.mean(pts[:, 0] < 25) - 0.5) < 0.05                       # even over the area


def test_site_sections_have_their_own_files_and_results(tmp_path):
    """구간/<name>/ holds a section's line and polygons; its results go there; scan ignores it."""
    from groundwater import site_folder as S

    (tmp_path / "wall_line.csv").write_text("kind,x,y\nwall,0,0\nwall,10,0\n", encoding="utf-8")
    sec = tmp_path / "구간" / "E"
    sec.mkdir(parents=True)
    (sec / "wall_line.csv").write_text("kind,x,y\nwall,0,5\nwall,10,5\npit,5,10\n", encoding="utf-8")
    (sec / "section_area.csv").write_text("x,y\n-1,-1\n11,-1\n11,11\n-1,11\n", encoding="utf-8")
    assert S.sections(tmp_path) == ["E"]
    assert S.scan(tmp_path)["wall_line"] == tmp_path / "wall_line.csv"
    cfg = S.build_config(tmp_path, section="E")
    assert cfg["wall_frame"]["polyline"] == [[0.0, 5.0], [10.0, 5.0]]
    assert cfg["paths"]["output_dir"] == str(sec / "분석결과")
    assert cfg["region"]["map_polygon"][0] == [-1.0, -1.0]


def test_file_datum_reads_a_prj_beside_a_mesh(tmp_path):
    """An OBJ is ellipsoidal unless a .prj beside it (or its name) says KNGeoid18 / orthometric."""
    from groundwater import heights as H

    (tmp_path / "wall.obj").write_text("v 0 0 0\n", encoding="utf-8")
    assert H.file_datum(tmp_path / "wall.obj") == "ellipsoidal"
    (tmp_path / "wall.prj").write_text('COMPD_CS["x",PROJCS["y"],VERT_CS["KNGeoid18 height"]]', encoding="utf-8")
    assert H.file_datum(tmp_path / "wall.obj") == H.ORTHOMETRIC
    assert H.file_datum(tmp_path / "wall_해발.obj") == H.ORTHOMETRIC


def test_canopy_ground_faces_drops_flat_faces_under_the_trees():
    """Under a green front surface, a near-horizontal face 2 m behind it is ground; a steep one there, or a flat one
    at the front surface, stays."""
    from types import SimpleNamespace

    from rasterio.transform import from_origin

    from groundwater.discontinuity.cleaning import canopy_ground_faces
    from groundwater.facade.frame import WallFrame
    from groundwater.io.raster import Raster

    frame = WallFrame([[0.0, 0.0], [10.0, 0.0]])
    tf = from_origin(0.0, 10.0, 0.1, 0.1)
    rgb = np.zeros((3, 100, 100), np.uint8)
    rgb[1] = 160                                                       # green everywhere
    rgb[0] = rgb[2] = 60
    face_rgb = Raster(array=rgb, transform=tf, crs=None, nodata=None)
    depth = Raster(array=np.zeros((1, 100, 100), np.float32), transform=tf, crs=None, nodata=None)

    def patch(pid, dip, w):
        return SimpleNamespace(patch_id=pid, dip=dip, centroid=frame.to_world(np.array([[5.0, 5.0, w]]))[0])

    sets = [SimpleNamespace(patches=[patch(1, 10, -2.0), patch(2, 80, -2.0), patch(3, 10, -0.1)])]
    assert canopy_ground_faces(sets, frame, face_rgb, depth, 0.06) == {1}


def test_lidar_section_reads_only_the_slab_and_keeps_the_rock_behind_the_net(tmp_path, monkeypatch):
    """Index + section on a small synthetic scan: a vertical rock wall (y = 0) with a net 0.5 m in front of it; the
    slab holds only points near the line, the net returns are first returns, the rock is the layer kept."""
    import laspy

    from groundwater import lidar_section as LS
    from groundwater.facade.frame import build_wall_frame

    monkeypatch.setattr(LS, "CACHE_DIR", tmp_path / "idx")
    monkeypatch.setattr(LS, "CHUNK", 5_000)
    rng = np.random.default_rng(5)
    n = 200_000
    x, z = rng.uniform(0, 10, n), rng.uniform(0, 5, n)
    net = rng.random(n) < 0.5
    y = np.where(net, 0.5, 0.0) + rng.normal(0, 0.01, n)                 # the pit is at +y
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = [0.001] * 3
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = x, y, z
    las.return_number = np.where(net, 1, 2).astype(np.uint8)
    las.number_of_returns = np.full(n, 2, np.uint8)
    las.red = las.green = las.blue = np.full(n, 30000, np.uint16)
    las.write(tmp_path / "scan.las")
    idx = LS.build_index(tmp_path / "scan.las")
    assert len(idx["boxes"]) == 40 and LS.load_index(tmp_path / "scan.las") is not None
    slab = LS.read_slab(tmp_path / "scan.las", idx, (5.0, 5.0), (5.0, -5.0), half_width=0.5)
    assert len(slab["xyz"]) and np.all(np.abs(slab["xyz"][:, 0] - 5.0) <= 0.5 + 1e-6)
    kinds = LS.return_kind(slab["ret"], slab["nret"])
    assert set(np.unique(kinds)) == {1, 3}
    frame = build_wall_frame(slab["xyz"], [[0.0, 0.0], [10.0, 0.0]], [5.0, 10.0])
    cls = LS.classify(slab["xyz"], frame)
    rock = slab["xyz"][cls["rock"], 1]
    assert len(rock) > 100 and np.mean(np.abs(rock) < 0.1) > 0.95          # the wall, not the net


def test_scan_skips_renamed_result_folders(tmp_path):
    """A copy of a results folder (분석결과_이전_...) is not input data, even when its name says LiDAR."""
    from groundwater import site_folder as S

    for d in ("분석결과", "분석결과_이전_라이다"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "classified_point_cloud.ply").write_text("ply\n", encoding="utf-8")
    (tmp_path / "scan_lidar.las").write_bytes(b"")
    f = S.scan(tmp_path)
    assert f["point_cloud"] == [] and [p.name for p in f["lidar"]] == ["scan_lidar.las"]


def test_site3d_groups_sets_across_sections_and_finds_the_open_side():
    """Sets of different sections within 20 deg are one site-wide set, named by the sections it shows in; the
    camera side of a U is its open end."""
    import pandas as pd

    from groundwater.viz import site3d

    sets = {"E_east": pd.DataFrame({"set_id": [0, 1], "mean_dip": [81.0, 6.0], "mean_dip_direction": [265.0, 318.0],
                                    "area": [9000.0, 800.0]}),
            "W_west": pd.DataFrame({"set_id": [0, 2], "mean_dip": [87.0, 87.0], "mean_dip_direction": [35.0, 261.0],
                                    "area": [4000.0, 3500.0]})}
    g = site3d.site_sets(sets)
    assert g[("E_east", 0)][0] == g[("W_west", 2)][0] and g[("E_east", 0)][1].endswith("E·W")
    assert len({k for k, _ in g.values()}) == 3
    t = np.linspace(np.pi, 2 * np.pi, 50)                       # a U opening north: faces west, south and east
    A = np.column_stack([100 * np.cos(t), 100 * np.sin(t), np.zeros(50)])
    side = site3d._open_side(A, np.zeros(3))
    assert side[1] > 0.95
