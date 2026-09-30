"""Discontinuity extraction against synthetic data with known ground truth.

The sub-vertical and rough-surface cases are regressions: the previous
hemisphere-folding + DBSCAN approach split vertical sets in two and merged
distinct sets into a non-existent orientation on rough rock.
"""
import numpy as np
import open3d as o3d
import pytest

from groundwater.discontinuity.clustering import find_orientation_sets, voxel_connected_components
from groundwater.discontinuity.geometry import axial_mean, compute_spacing
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
