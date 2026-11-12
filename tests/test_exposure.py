import numpy as np

from groundwater.discontinuity import exposure as X
from groundwater.facade.frame import WallFrame


def _dipdir(n):
    n = n if n[2] >= 0 else -n
    return np.degrees(np.arccos(n[2])), np.degrees(np.arctan2(n[0], n[1])) % 360


def test_classify_straight_wall():
    frame = WallFrame(np.array([[0.0, 0.0], [100.0, 0.0]]), fillet=True)
    along, out = frame.axes_at(50.0)
    wn = X.wall_normals(frame, [50.0, 50.0, 50.0, 50.0])
    assert np.allclose(np.abs(wn @ out), 1.0)
    steep_par = out * np.cos(np.radians(5)) + np.array([0, 0, np.sin(np.radians(5))])      # 85 deg, along the wall
    flat = np.array([0.05, 0.0, 1.0])
    cross = along.copy()                                                                  # strikes across the wall
    oblique = np.cos(np.radians(30)) * out + np.sin(np.radians(30)) * along                # 30 deg off the wall
    n = np.array([steep_par, flat, cross, oblique])
    dip = [_dipdir(v / np.linalg.norm(v))[0] for v in n]
    ang = X.wall_angle(n, wn)
    assert ang[0] < 6 and ang[2] > 85 and 29 < ang[3] < 31
    assert list(X.classify(dip, ang)) == ["parallel", "flat", "cross", "cross"]


def test_front_profile_follows_a_bend():
    # the wall front moves toward the pit by 0.3 m per m of chainage between u = 40 and 60: there a face
    # parallel to that front is 'parallel', though it is 17 deg off the wall line
    frame = WallFrame(np.array([[0.0, 0.0], [100.0, 0.0]]), fillet=True)
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 100, 20000)
    w = np.where(u < 40, 0, np.where(u < 60, 0.3 * (u - 40), 6.0)) - rng.uniform(0, 1, len(u))
    prof = X.front_profile(u, w, smooth_m=2.0)
    assert np.isclose(np.interp(50, *prof), 0.3, atol=0.03)
    assert abs(np.interp(10, *prof)) < 0.02
    along, out = frame.axes_at(50.0)
    face = out - 0.3 * along
    face /= np.linalg.norm(face)
    assert X.wall_angle([face], X.wall_normals(frame, [50.0], prof))[0] < 2
    assert X.wall_angle([face], X.wall_normals(frame, [50.0]))[0] > 15


def test_point_mask():
    frame = WallFrame(np.array([[0.0, 0.0], [100.0, 0.0]]), fillet=True)
    along, out = frame.axes_at(50.0)
    uvw = np.column_stack([np.linspace(1, 99, 200), np.zeros(200), np.zeros(200)])
    normals = np.vstack([np.tile(out, (100, 1)), np.tile(along, (100, 1))])
    m = X.point_mask(uvw, normals, frame)
    assert m[:100].all() and not m[100:].any()
