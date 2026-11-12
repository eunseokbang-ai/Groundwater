"""Discontinuity extraction against synthetic data with known ground truth.

The sub-vertical and rough-surface cases are regressions: the previous
hemisphere-folding + DBSCAN approach split vertical sets in two and merged
distinct sets into a non-existent orientation on rough rock.
"""
import numpy as np
import open3d as o3d
import pytest

from groundwater.discontinuity.clustering import find_orientation_sets, voxel_connected_components
from groundwater.discontinuity.geometry import axial_mean, compute_spacing, scanline_spacing
from groundwater.discontinuity.normals import estimate_local_geometry
from groundwater.discontinuity.segmentation import extract_discontinuities
from groundwater.validation.field_compare import angular_difference, dip_dipdir_to_normal


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _joint_set(rng, dip, dip_direction, origin, noise_std=0.0005, n_layers=3, layer_spacing=0.3):
    normal = dip_dipdir_to_normal(dip, dip_direction)
    u, v = _plane_basis(normal)
    grid = np.linspace(-0.75, 0.75, 50)
    gu, gv = (a.ravel() for a in np.meshgrid(grid, grid))
    layers = []
    for k in range(n_layers):
        pts = origin + k * layer_spacing * normal + gu[:, None] * u + gv[:, None] * v
        layers.append(pts + rng.normal(scale=noise_std, size=pts.shape))
    return np.vstack(layers)


def _poles_around(rng, axis, scatter_deg, n):
    """Poles scattered around `axis` with random sign (poles are axial)."""
    u, v = _plane_basis(axis)
    s = np.radians(scatter_deg)
    poles = axis + s * (rng.normal(size=(n, 1)) * u + rng.normal(size=(n, 1)) * v)
    poles /= np.linalg.norm(poles, axis=1, keepdims=True)
    return poles * rng.choice([-1.0, 1.0], size=(n, 1))


def _run(points, search_radius=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    variation = estimate_local_geometry(pcd, search_radius=search_radius, max_nn=30)
    return extract_discontinuities(pcd, surface_variation=variation)


def _assert_sets_match(found_sets, truth, tol_deg):
    assert len(found_sets) == len(truth), [(round(s.mean_dip, 1), round(s.mean_dip_direction, 1)) for s in found_sets]
    for dip, dip_dir in truth:
        errors = [angular_difference(dip, dip_dir, s.mean_dip, s.mean_dip_direction) for s in found_sets]
        assert min(errors) < tol_deg, f"no set within {tol_deg} deg of {dip}/{dip_dir}"


def test_recovers_orientations_and_spacing():
    rng = np.random.default_rng(42)
    points = np.vstack([
        _joint_set(rng, 15.0, 90.0, np.zeros(3)),
        _joint_set(rng, 75.0, 210.0, np.array([5.0, 5.0, 0.0])),
    ])
    sets, labels = _run(points)

    _assert_sets_match(sets, [(15.0, 90.0), (75.0, 210.0)], tol_deg=2.0)
    for dset in sets:
        assert len(dset.patches) == 3
        assert abs(dset.spacing_mean - 0.3) < 0.05
    assert (labels[:, 0] != -1).mean() > 0.9


@pytest.mark.parametrize("dip", [85.0, 88.0, 90.0])
def test_subvertical_set_is_not_split(dip):
    rng = np.random.default_rng(1)
    points = np.vstack([
        _joint_set(rng, dip, 150.0, np.zeros(3), noise_std=0.004),
        _joint_set(rng, 20.0, 30.0, np.array([6.0, 6.0, 0.0]), noise_std=0.004),
    ])
    sets, _ = _run(points)
    _assert_sets_match(sets, [(dip, 150.0), (20.0, 30.0)], tol_deg=2.0)


@pytest.mark.parametrize("noise_std", [0.015, 0.025])
def test_rough_surfaces_keep_sets_apart(noise_std):
    rng = np.random.default_rng(2)
    points = np.vstack([
        _joint_set(rng, 78.0, 150.0, np.zeros(3), noise_std=noise_std),
        _joint_set(rng, 20.0, 30.0, np.array([6.0, 6.0, 0.0]), noise_std=noise_std),
    ])
    sets, _ = _run(points)
    _assert_sets_match(sets, [(78.0, 150.0), (20.0, 30.0)], tol_deg=5.0)


def test_vertical_poles_with_mixed_signs_form_one_set():
    rng = np.random.default_rng(3)
    axis = dip_dipdir_to_normal(90.0, 60.0)
    labels, axes = find_orientation_sets(_poles_around(rng, axis, 5.0, 5000))
    assert len(axes) == 1
    assert np.degrees(np.arccos(abs(axes[0] @ axis))) < 1.0
    assert (labels == 0).mean() > 0.95


def test_scattered_sets_with_background_do_not_chain():
    rng = np.random.default_rng(4)
    a = dip_dipdir_to_normal(60.0, 100.0)
    b = dip_dipdir_to_normal(60.0, 145.0)
    background = rng.normal(size=(2000, 3))
    background /= np.linalg.norm(background, axis=1, keepdims=True)
    poles = np.vstack([_poles_around(rng, a, 8.0, 4000), _poles_around(rng, b, 8.0, 4000), background])

    _, axes = find_orientation_sets(poles)

    assert len(axes) == 2
    for truth in (a, b):
        assert min(np.degrees(np.arccos(np.abs(axes @ truth)))) < 3.0


def test_uniform_poles_yield_no_sets():
    rng = np.random.default_rng(5)
    poles = rng.normal(size=(20000, 3))
    poles /= np.linalg.norm(poles, axis=1, keepdims=True)
    _, axes = find_orientation_sets(poles)
    assert len(axes) == 0


def test_axial_mean_ignores_pole_signs():
    rng = np.random.default_rng(6)
    axis = dip_dipdir_to_normal(89.0, 10.0)
    mean_axis, fisher_k = axial_mean(_poles_around(rng, axis, 3.0, 1000))
    assert np.degrees(np.arccos(abs(mean_axis @ axis))) < 0.5
    assert fisher_k > 100


def test_voxel_components_split_parallel_planes_and_gaps():
    g = np.linspace(0.0, 1.0, 40)
    gx, gy = (a.ravel() for a in np.meshgrid(g, g))
    plane = np.column_stack([gx, gy, np.zeros_like(gx)])
    parallel = plane + [0.0, 0.0, 0.25]
    shifted = plane + [1.4, 0.0, 0.0]  # same plane, 0.4 m gap in x

    labels = voxel_connected_components(np.vstack([plane, parallel, shifted]), link_distance=0.05)

    n = len(plane)
    assert len(np.unique(labels[:n])) == 1
    assert len({labels[0], labels[n], labels[2 * n]}) == 3


def test_spacing_ignores_coplanar_fragments():
    normal = np.array([0.0, 0.0, 1.0])
    centroids = np.array([[0, 0, 0.0], [3, 0, 0.01], [0, 0, 0.5], [2, 1, 0.51], [0, 0, 1.0]])
    mean, _ = compute_spacing(centroids, normal, coplanar_tolerance=0.05)
    assert abs(mean - 0.5) < 0.02


def test_scanline_spacing_ignores_side_by_side_patches():
    # two stacks of joints 0.5 m apart, 10 m apart along the wall and offset by 0.2 m in depth: sorting
    # all centroids would report ~0.2-0.3 m gaps, scanlines only see each stack's own 0.5 m spacing
    normal = np.array([0.0, 0.0, 1.0])
    stack_a = [[0.0, 0.0, z] for z in (0.0, 0.5, 1.0, 1.5)]
    stack_b = [[10.0, 0.0, z + 0.2] for z in (0.0, 0.5, 1.0, 1.5)]
    centroids = np.array(stack_a + stack_b)
    mean, std = scanline_spacing(centroids, np.full(len(centroids), 1.0), normal, coplanar_tolerance=0.05)
    assert abs(mean - 0.5) < 1e-6 and std < 1e-6


def test_netting_and_flat_ground_masks():
    """Cyan-grey netting over a grey wall and a level bench band: the mask finds both; a flat face reaching 2 m
    back into the slope is ground, a 0.3 m ledge is a joint."""
    from types import SimpleNamespace

    from groundwater.discontinuity import cleaning as C

    res = 0.05
    H, W = 200, 300                                       # 10 m x 15 m facade
    rgb = np.full((H, W, 3), 128, np.uint8)               # grey rock
    rgb[:80, :150] = (94, 103, 106)                       # netting: hue ~190, saturation ~0.11
    valid = np.ones((H, W), bool)
    net = C.netting_area(rgb, valid, res)
    assert net[10:70, 10:140].mean() > 0.95 and net[100:, :].mean() == 0 and net[:80, 170:].mean() == 0
    depth = np.zeros((H, W), np.float32)                  # vertical wall at w = 0 ...
    v = (H - np.arange(H)) * res                          # elevation of each row (top = 10 m)
    band = (v > 4.0) & (v < 5.0)                          # ... and a ground band rising 1 m over 3 m back
    depth[band, :] = -(v[band, None] - 4.0) * 3.0
    depth[v >= 5.0, :] = -3.0                             # the wall above, set back by the bench
    flat = C.flat_area(depth, res)
    inner = (v > 4.25) & (v < 4.75)
    assert flat[inner][:, 20:-20].mean() > 0.95 and flat[v > 7.0].mean() == 0 and flat[v < 2.0].mean() == 0
    # flat faces in 3D: the depth extent decides
    uvw = np.r_[np.c_[np.zeros(50), np.zeros(50), np.linspace(0, -2.0, 50)],
                np.c_[np.zeros(50), np.zeros(50), np.linspace(0, -0.3, 50)]]
    patch = np.r_[np.full(50, 1), np.full(50, 2)]
    sets = [SimpleNamespace(patches=[SimpleNamespace(patch_id=1, dip=5.0), SimpleNamespace(patch_id=2, dip=5.0)])]
    assert C.flat_faces(uvw, patch, sets) == {1}


def test_planar_parts_splits_a_step_and_keeps_a_plane():
    """Two parallel faces 0.5 m apart joined at a step are two faces; one rough-free plane stays one."""
    from groundwater.discontinuity.segmentation import planar_parts

    rng = np.random.default_rng(0)
    xa, ya = rng.uniform(0, 5, 20000), rng.uniform(0, 5, 20000)
    front = np.column_stack([xa, ya, rng.normal(0, 0.01, 20000)])
    xb, yb = rng.uniform(5, 10, 20000), rng.uniform(0, 5, 20000)
    back = np.column_stack([xb, yb, rng.normal(-0.5, 0.01, 20000)])
    zs = rng.uniform(-0.5, 0, 2000)                                    # the riser joining them
    riser = np.column_stack([np.full(2000, 5.0) + rng.normal(0, 0.01, 2000), rng.uniform(0, 5, 2000), zs])
    pts = np.vstack([front, back, riser])
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    nrm[-2000:] = [1.0, 0.0, 0.0]
    parts = planar_parts(pts, nrm, 0.1, 0.10, 30)
    big = sorted(parts, key=len, reverse=True)[:2]
    assert len(big) == 2 and all(len(p) > 15000 for p in big)
    assert {int(np.round(pts[p, 2].mean() * 2)) for p in big} == {0, -1}   # one at z = 0, one at z = -0.5
    assert len(planar_parts(front, nrm[:20000], 0.1, 0.10, 30)) == 1


def test_weak_peak_is_a_candidate_and_a_pinned_set_is_kept():
    """A small pole cluster under the contrast bar is reported as a candidate; pinned, it becomes a set at that
    orientation."""
    from groundwater.discontinuity.clustering import find_orientation_sets, plane_normal
    from groundwater.discontinuity.geometry import normal_to_dip_dipdir

    rng = np.random.default_rng(3)

    def cluster(dip, dd, n, spread=4.0):
        v = plane_normal(dip, dd) + rng.normal(0, np.radians(spread), (n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    noise = rng.normal(0, 1, (60000, 3))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    normals = np.vstack([cluster(87, 35, 40000), cluster(88, 260, 40000), cluster(48, 47, 2000, 6.0), noise])
    info = {}
    _, axes = find_orientation_sets(normals, info=info)
    found = [normal_to_dip_dipdir(a) for a in axes]
    assert not any(abs(d - 48) < 8 and abs(a - 47) < 15 for d, a in found)
    assert any(abs(c["dip"] - 48) < 6 and abs((c["dip_direction"] - 47 + 180) % 360 - 180) < 10 for c in info["candidates"])
    info = {}
    _, axes = find_orientation_sets(normals, fixed_sets=[(48, 47)], info=info)
    pinned = [normal_to_dip_dipdir(axes[k]) for k in info["fixed"]]
    assert len(pinned) == 1 and abs(pinned[0][0] - 48) < 0.5 and abs(pinned[0][1] - 47) < 0.5
    assert len(axes) == 3
