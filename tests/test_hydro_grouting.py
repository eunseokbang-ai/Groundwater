"""Snow permeability tensor, hole-orientation search and curtain check (groundwater.hydro)."""
import numpy as np
import pandas as pd

from groundwater import hydro


def _sets(rows):
    df = pd.DataFrame(rows, columns=["name", "dip", "dip_direction", "spacing_m", "persistence_m"])
    df["aperture_rel"] = 1.0
    df["note"] = ""
    return df


def test_single_vertical_set_flows_in_its_plane():
    _, principal, share = hydro.snow_tensor(_sets([("A", 90, 90, 1.0, 2.0)]))   # plane striking N-S
    # zero permeability across the plane (E-W), equal along strike (N-S) and down dip (vertical)
    weakest = principal.iloc[-1]
    assert weakest["relative_k"] < 1e-6 and weakest["plunge"] < 1 and weakest["trend"] % 180 in (90.0,)
    assert share.tolist() == [1.0]


def test_best_hole_crosses_the_dominant_set_at_right_angle():
    sets = _sets([("A", 90, 90, 0.5, 2.0), ("B", 10, 0, 5.0, 2.0)])
    _, _, share = hydro.snow_tensor(sets)
    _, best, _, vertical = hydro.hole_orientations(sets, share)
    assert best["plunge"] == 0 and best["trend"] in (90, 270)       # horizontal, perpendicular to set A
    assert best["joints_per_m"] > 5 * vertical["joints_per_m"]      # a vertical hole misses set A


def test_curtain_parallel_set_is_flagged_and_never_cut():
    sets = _sets([("P", 88, 140, 1.0, 2.0), ("X", 85, 50, 1.0, 2.0), ("Bed", 5, 0, 3.0, 2.0)])
    cc = hydro.curtain_check(sets, curtain_azimuth=50.0).set_index("set")   # curtain strikes 050
    assert cc.loc["P", "angle_to_curtain_deg"] < 20 and cc.loc["P", "cut_at_1m"] == 0
    # X crosses the curtain in a near-vertical line: vertical holes run along it, inclined ones cut it
    assert cc.loc["X", "angle_to_curtain_deg"] > 80 and cc.loc["X", "cut_at_2m"] < 0.2
    inc = hydro.curtain_check(sets, curtain_azimuth=50.0, hole_plunge=45.0).set_index("set")
    assert inc.loc["X", "cut_at_2m"] > 0.8
    assert "하부" in cc.loc["Bed", "remark"]


def test_aimed_hole_meets_the_joint_behind_the_trace():
    from groundwater.viz.grouting3d import _aimed_hole, _trend_plunge_down

    outward = np.array([0.0, 1.0, 0.0])                # vertical face looking north
    p = np.zeros(3)                                    # point on a horizontal trace (bedding)
    h = _aimed_hole(p, -outward, np.array([0.0, 0.0, 1.0]), outward, depth=1.0, tan=1.0, min_plunge=10,
                    extra_m=0.5, site=None)
    assert np.allclose(h["collar"], [0, 0, 1]) and np.allclose(h["target"], [0, -1, 0])
    trend, plunge = _trend_plunge_down(h["dir"])
    assert abs(trend - 180) < 1e-6 and abs(plunge - 45) < 1e-6      # 45 deg down into the face
    assert abs(h["length"] - (np.sqrt(2) + 0.5)) < 1e-9
    # beside a vertical trace the hole is raised so it still dips >= 10 deg
    h = _aimed_hole(p, -outward, np.array([1.0, 0.0, 0.0]), outward, 1.0, 1.0, 10, 0.5, None)
    assert _trend_plunge_down(h["dir"])[1] >= 10 - 1e-6


def test_joint_planes_clip_to_rock_and_holes_cross_them():
    from groundwater.viz.joints3d import Joint, _clip_star, coverage

    class HalfSpace:                                    # rock where y <= 0 (face at y = 0, viewer at +y)
        def inside(self, xyz):
            return np.asarray(xyz)[:, 1] <= 0.0

    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    dirs = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(8)])   # a horizontal disc around the face point
    poly, r = _clip_star(np.array([0.0, -0.01, 0.0]), dirs, np.full(8, 2.0), HalfSpace())
    assert np.all(poly[1:, 1] <= 1e-9) and r[2] == 0.0 and abs(r[6] - 2.0) < 1e-9   # +y side cut, -y side kept
    sq = Joint("Set 0", np.array([[0, -1, 0], [1, -1, 0], [0, -2, 0], [-1, -1, 0], [0, 0, 0]], float),
               np.array([0, 0, 1.0]), "trace", 2, 0.9, 1.0)
    holes = [{"collar": np.array([0, -1, 1.0]), "dir": np.array([0, 0, -1.0]), "length": 2.0},   # through it
             {"collar": np.array([5, -1, 1.0]), "dir": np.array([0, 0, -1.0]), "length": 2.0},   # beside it
             {"collar": np.array([0, -1, 3.0]), "dir": np.array([0, 0, -1.0]), "length": 2.0}]   # too short
    cov = coverage(holes, [sq])
    assert [len(h) for h in cov["per_hole"]] == [1, 0, 0] and abs(cov["per_hole"][0][0][1] - 1.0) < 1e-5
    assert cov["n_high"] == 1 and cov["n_high_hit"] == 1


def test_tile_archive_reads_3tz_and_folder(tmp_path):
    import zipfile

    from groundwater.viz.tiles3d import TileArchive, find_tileset

    folder = tmp_path / "tiles"
    (folder / "Data").mkdir(parents=True)
    (folder / "tileset.json").write_text('{"asset": {"version": "1.1"}}')
    (folder / "Data" / "a.glb").write_bytes(b"glTF")
    tz = tmp_path / "model.3tz"
    with zipfile.ZipFile(tz, "w") as z:
        z.write(folder / "tileset.json", "tileset.json")
        z.write(folder / "Data" / "a.glb", "Data/a.glb")
    assert find_tileset(tmp_path) == tz and find_tileset(folder) == folder
    for src in (tz, folder):
        a = TileArchive(src)
        assert a.read("Data/a.glb") == b"glTF" and a.read("/tileset.json").startswith(b"{")
        assert a.read("missing.glb") is None
    assert TileArchive(folder).read("../model.3tz") is None          # no reading outside the tileset folder


def test_section_cuts_a_joint_polygon_into_a_segment():
    from groundwater.viz.section import _cut

    sq = np.array([[0, 0, 0], [2, 0, 0], [2, 1, -1], [0, 1, -1]], float)   # (u, v, w), crosses u = 1
    (v1, w1), (v2, w2) = _cut(sq, 1.0)
    assert {round(v1, 6), round(v2, 6)} == {0.0, 1.0} and {round(w1, 6), round(w2, 6)} == {0.0, -1.0}
    assert _cut(sq, 5.0) is None


def test_mesh_cut_by_planes_gives_joint_traces():
    from groundwater.viz.setmap3d import cut_segments

    # a square joint (wall coordinates u, v, w) dipping into the rock: from the face (w = 0) at v = 1 down to
    # w = -2 at v = 0, two triangles
    V = np.array([[0, 1, 0], [2, 1, 0], [2, 0, -2], [0, 0, -2]], float)
    mesh = {"V": V, "F": np.array([[0, 1, 2], [0, 2, 3]]), "owner": np.array([7, 7])}
    seg, own = cut_segments(mesh, V[:, 1] - 0.5)                 # horizontal section at v = 0.5
    assert len(seg) == 2 and set(own) == {7}
    assert np.allclose(seg[..., 1], 0.5) and np.allclose(seg[..., 2], -1.0)   # the joint is 1 m deep there
    assert abs(seg[..., 0].min()) < 1e-9 and abs(seg[..., 0].max() - 2) < 1e-9
    seg, _ = cut_segments(mesh, V[:, 0] - 1.0)                   # cross-section at u = 1: one line, face to depth
    ends = seg.reshape(-1, 3)
    assert np.isclose(ends[:, 2].max(), 0.0) and np.isclose(ends[:, 2].min(), -2.0)
    assert len(cut_segments(mesh, V[:, 1] - 5.0)[0]) == 0        # above the joint: nothing


def test_reference_overlay_homography_and_tps(tmp_path):
    import json

    import cv2

    from groundwater.facade import reference_overlays as ro

    img = np.full((100, 200, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 20), (90, 60), (30, 30, 200), -1)          # a red "figure" (BGR) on white paper
    cv2.imwrite(str(tmp_path / "t.png"), img)
    H = [[0.01, 0, 1.0], [0, -0.01, 2.0], [0, 0, 1]]                    # 1 cm / px, y up
    pts = [[x, y, 1 + 0.01 * x, 2 - 0.01 * y, ""] for x, y in ((0, 0), (200, 0), (0, 100), (200, 100), (100, 50))]
    (tmp_path / ro.OVERLAY_FILE).write_text(json.dumps({"overlays": [
        {"name": "h", "file": "t.png", "kind": "tracing", "method": "homography", "H": H},
        {"name": "s", "file": "t.png", "kind": "tracing", "method": "tps", "points": pts}]}), encoding="utf-8")
    data = ro.load(tmp_path)
    for e in data["overlays"]:
        b = ro.footprint(tmp_path, e)
        assert np.allclose(b, (1.0, 1.0, 3.0, 2.0), atol=1e-3)
        o = ro.render(tmp_path, e, b, 0.005)
        # the figure (image px 50-90, 20-60) -> plane x 1.5-1.9, y 1.4-1.8 -> render px cols 100-180, rows 40-120
        assert o[80, 140, 3] > 200 and o[80, 140, 0] > 150                # opaque red inside
        assert o[180, 20, 3] == 0                                          # white paper transparent


def test_view3d_reads_plotly_points_lines_and_meshes():
    import base64

    import plotly.graph_objects as go

    from groundwater.viz.view3d import page, scene_from_plotly

    fig = go.Figure([
        go.Scatter3d(x=[0, 1, 2], y=[0, 0, 0], z=[0, 0, 0], mode="markers", marker=dict(color=["#ff0000", "#00ff00", "#0000ff"], size=2), name="pts"),
        go.Scatter3d(x=[0, 1, None, 2, 3], y=[0] * 5, z=[0, 0, None, 1, 1], mode="lines", line=dict(color="black", width=5, dash="dash"), name="ln"),
        go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2], color="rgb(10, 20, 30)", opacity=0.5, name="m"),
    ])
    sc = scene_from_plotly(fig, "t")
    pts, ln, m = sc["items"]
    assert pts["kind"] == "points" and np.frombuffer(base64.b64decode(pts["col"]), np.uint8).tolist() == [255, 0, 0, 0, 255, 0, 0, 0, 255]
    seg = np.frombuffer(base64.b64decode(ln["pos"]), np.float32).reshape(-1, 2, 3)
    assert len(seg) == 2 and ln["dash"] and ln["color"] == "#000000"          # the None gap splits the polyline
    assert m["kind"] == "mesh" and m["color"] == "#0a141e" and m["opacity"] == 0.5
    assert np.allclose(sc["center"], [1.5, 0.5, 0.5])                          # positions sent relative to the centre
    assert "__SCENE_JSON__" not in page(sc)


def test_seepage_map_is_draped_on_the_face_per_class():
    """A flat wall (face at w = 0): the seepage cells lie on it, offset toward the viewer, one mesh per class with
    the cell area, and the best zones get their number."""
    from types import SimpleNamespace

    import pandas as pd
    from rasterio.transform import Affine

    from groundwater.facade.frame import WallFrame
    from groundwater.io.raster import Raster
    from groundwater.viz import setmap3d

    frame = WallFrame([[0.0, 0.0], [20.0, 0.0]])
    res = 0.05                                         # facade pixels: chainage -1-11 m, elevation -1-6 m, all at w = 0
    depth = Raster(array=np.zeros((1, 140, 240), np.float32), transform=Affine(res, 0, -1.0, 0, -res, 6.0), crs=None,
                   nodata=np.nan)
    result = SimpleNamespace(wall_frame=frame, facade_depth=depth)
    classes = np.zeros((5, 10), np.int8)
    classes[1:3, 2:4] = 2                              # 4 '상' cells
    classes[4, 6] = 1                                  # 1 '중' cell
    xyz = frame.to_world(np.array([[3.0, 3.0, 0.0]]))[0]
    zones = pd.DataFrame([{"zone": 7, "x": xyz[0], "y": xyz[1], "z": xyz[2]}])
    smap = SimpleNamespace(classes=classes, cell_m=1.0, u_min=0.0, v_max=5.0, zones=zones)
    tr = setmap3d.seepage_traces(result, smap, np.zeros(3), step_m=0.25, offset_m=0.1)
    meshes = {t["name"]: t for t in tr if t["type"] == "mesh3d"}
    assert set(meshes) == {setmap3d.SEEPAGE_NAMES[2], setmap3d.SEEPAGE_NAMES[1]}
    for n, (k, cells) in enumerate(((2, 4), (1, 1))):
        m = meshes[setmap3d.SEEPAGE_NAMES[k]]
        P = np.column_stack([m["x"], m["y"], m["z"]]).astype(float)
        uvw = frame.to_wall(P)
        # on the face, 0.1 m toward the viewer (each further layer 3 cm more, so overlapping layers stay visible)
        assert np.allclose(uvw[:, 2], 0.1 + 0.03 * n, atol=1e-5)
        assert len(m["i"]) * 0.25 ** 2 / 2 == cells * 1.0                   # the cells' area, no more
        assert m["color"] == setmap3d.SEEPAGE_COLOURS[k]
    hi = uvw = frame.to_wall(np.column_stack([meshes[setmap3d.SEEPAGE_NAMES[2]][c] for c in "xyz"]).astype(float))
    assert hi[:, 0].min() >= 2.0 - 1e-6 and hi[:, 0].max() <= 4.0 + 1e-6 and hi[:, 1].min() >= 2.0 - 1e-6
    labels = [t for t in tr if t["type"] == "scatter3d"][0]
    assert labels["text"] == ["#7"] and labels["mode"] == "text"


def test_seepage_zones_are_numbered_by_score():
    """Two zones of touching '상' cells; the one found second on the wall scores higher, so it is #1."""
    from types import SimpleNamespace

    from rasterio.transform import Affine

    from groundwater.facade.frame import WallFrame
    from groundwater.io.raster import Raster

    frame = WallFrame([[0.0, 0.0], [20.0, 0.0]])
    depth = Raster(array=np.zeros((1, 50, 100), np.float32), transform=Affine(0.1, 0, 0.0, 0, -0.1, 5.0), crs=None,
                   nodata=np.nan)
    result = SimpleNamespace(wall_frame=frame, facade_depth=depth, traces=[], trace_families=[], facade_sets=None)
    classes = np.zeros((5, 10), np.int8)
    classes[0:2, 0:2] = 2                                # found first (top left), lower score
    classes[3:5, 6:9] = 2                                # found second, higher score
    score = np.full((5, 10), 0.2)
    score[0:2, 0:2] = 0.5
    score[3:5, 6:9] = 0.9
    layers = {k: np.zeros((5, 10)) for k in ("p21", "openness", "connectivity", "wetness")}
    layers["score"] = score
    z = hydro._zones(result, classes, layers, 1.0, 0.0, 5.0)
    assert list(z["zone"]) == [1, 2] and list(z["score"]) == [0.9, 0.5] and z.loc[0, "chainage_m"] == "6–9"


def test_seepage_map_figure_layers_and_colour_fade():
    """The seepage map view: the grade as two class layers, an indicator in display units with a colour bar whose
    low end is nearly see-through, zone boxes, hover on every rock cell; the RGBA drape fades the same way."""
    from types import SimpleNamespace

    from groundwater.viz import seepage_view as SV

    classes = np.full((4, 6), -1, np.int8)
    classes[:, 1:5] = 0
    classes[1, 2] = 2
    classes[2, 3] = 1
    rock = classes >= 0
    p21 = np.where(rock, np.linspace(0, 2, 24).reshape(4, 6), 0.0)
    layers = {"p21": p21, "openness": np.full((4, 6), 0.02), "connectivity": np.log10(1 + np.full((4, 6), 9.0)),
              "wetness": np.full((4, 6), 0.5), "score": p21 / 2}
    zones = pd.DataFrame([{"zone": 1, "chainage_m": "2–3", "elevation_m": "2–3"}])
    smap = SimpleNamespace(classes=classes, rock=rock, layers=layers, cell_m=1.0, u_min=0.0, v_max=4.0, zones=zones)
    assert np.allclose(SV.layer_values(smap, "openness")[rock], 2.0)              # cm
    assert np.allclose(SV.layer_values(smap, "connectivity")[rock], 9.0)          # m of network
    assert np.isnan(SV.layer_values(smap, "p21")[~rock]).all()
    bg = np.full((40, 60, 3), 128, np.uint8)
    g = SV.map_figure(smap, "grade", bg, [0, 6, 0, 4])
    assert [t.name for t in g.data if t.type == "heatmap"] == ["중 (다음 35%)", "상 (상위 15%)"]
    f = SV.map_figure(smap, "p21", bg, [0, 6, 0, 4])
    hm = [t for t in f.data if t.type == "heatmap"]
    assert len(hm) == 1 and hm[0].showscale and f.layout.annotations[0].text == "<b>#1</b>"
    assert hm[0].colorscale[0][1].endswith(",0.12)") and hm[0].colorscale[-1][1].endswith(",1.00)")
    hover = [t for t in f.data if t.type == "scatter" and t.hoverinfo == "text"][0]
    assert len(hover.x) == rock.sum()
    a = SV._rgba(smap, "p21")[..., 3]
    assert (a[~rock] == 0).all() and a[rock].min() < 40 and a[rock].max() == 175
