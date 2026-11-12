import numpy as np
import pandas as pd

from groundwater import site_joints as SJ


def test_cut_vertical_disc_by_horizontal_plane():
    # a vertical disc (normal +x) of radius 5 at the origin, cut at z = 3: a chord along y of half-length 4
    seg = SJ.cut([[0, 0, 0]], [[1.0, 0, 0]], [5.0], np.array([0, 0, 3.0]), np.array([0, 0, 1.0]))[0]
    assert np.allclose(seg[:, 2], 3.0)
    assert np.allclose(seg[:, 0], 0.0)
    assert np.allclose(sorted(seg[:, 1]), [-4.0, 4.0])


def test_cut_misses_and_parallel():
    m = np.array([0, 0, 1.0])
    seg = SJ.cut([[0, 0, 0], [0, 0, 0]], [[1.0, 0, 0], [0, 0, 1.0]], [2.0, 5.0], np.array([0, 0, 3.0]), m)
    assert np.isnan(seg).all()                  # too small to reach z = 3; parallel to the cutting plane


def test_cut_dipping_disc():
    # dip 45 deg toward +x: the chord at z = 0 through the centre runs along y with the full diameter
    n = np.array([1.0, 0, 1.0]) / np.sqrt(2)
    seg = SJ.cut([[0, 0, 0]], [n], [3.0], np.zeros(3), np.array([0, 0, 1.0]))[0]
    assert np.isclose(np.linalg.norm(seg[1] - seg[0]), 6.0)
    # at z = 1 the centre of the chord is 1 m along the dip (x = -1 on the plane x + z = 0)
    seg = SJ.cut([[0, 0, 0]], [n], [3.0], np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))[0]
    mid = seg.mean(axis=0)
    assert np.allclose(mid, [-1, 0, 1])
    assert np.isclose(np.linalg.norm(seg[1] - seg[0]), 2 * np.sqrt(9 - 2))


def _faces(rows):
    F = pd.DataFrame(rows, columns=["section", "group", "x", "y", "z", "dip", "dd", "area"])
    n = SJ._pole(F["dip"], F["dd"])
    F["nx"], F["ny"], F["nz"] = n[:, 0], n[:, 1], n[:, 2]
    F["group_label"] = F["group"].astype(str)
    return F


def test_continuity_groups_coplanar_faces_only():
    # set 0: three faces on the vertical plane x = 0 (dd 90), two sections; one parallel face 5 m off it;
    # set 1: a face on the same plane but another set
    F = _faces([("E_a", 0, 0.0, 0, 0, 90, 90, 20), ("W_b", 0, 0.2, 30, 5, 88, 90, 10),
                ("W_b", 0, -0.3, 60, -4, 90, 92, 8), ("E_a", 0, 5.0, 10, 0, 90, 90, 15),
                ("E_a", 1, 0.0, 20, 0, 90, 90, 12)])
    C = SJ.continuity(F)
    assert len(C) == 1
    r = C.iloc[0]
    assert sorted(r.members) == [0, 1, 2]
    assert r.sections == "E·W"
    assert abs(r.dd - 90) < 3 and r.dip > 85
    assert np.isclose(r.span_m, np.linalg.norm([0.3 + 0.3, 60, 4]), atol=0.5)
    assert list(F["plane"]) == [0, 0, 0, -1, -1]


def test_continuity_no_chain_drift():
    # a stepped wall: each face 0.8 m behind the last -- neighbours are within 1 m, the ends are not
    F = _faces([("E", 0, 0.8 * i, 10.0 * i, 0, 90, 90, 10 + i) for i in range(8)])
    C = SJ.continuity(F)
    for r in C.itertuples():
        x = F["x"].to_numpy()[r.members]
        assert x.max() - x.min() <= 2.0


def test_ground_trace_vertical_disc_on_flat_ground():
    from affine import Affine

    from groundwater.viz import site_joints_view as V

    ground = (np.zeros((100, 100), np.float32), Affine(1.0, 0, -50, 0, -1.0, 50))      # z = 0 over -50..50
    # a vertical disc (strike N-S, normal +x) at x = 0.3, centre 5 m below ground, radius 13: meets the ground
    # along x = 0.3 for |y| <= 12
    lines = V.ground_trace([0.3, 0, -5], [1.0, 0, 0], 13.0, ground)
    P = np.vstack(lines)
    assert np.allclose(P[:, 0], 0.3, atol=0.05)
    assert np.allclose(P[:, 2], 0.0)
    assert 10.5 < np.ptp(P[:, 1]) / 2 <= 12.0
    assert V.ground_trace([0.3, 0, -20], [1.0, 0, 0], 13.0, ground) == []     # does not reach the ground
