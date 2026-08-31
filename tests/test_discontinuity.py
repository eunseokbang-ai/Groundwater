"""Validate the discontinuity-extraction pipeline against synthetic point
clouds with known ground-truth plane orientations and spacing (no real drone
data is available yet -- LiDAR/thermal acquisition is scheduled for next week).
"""
import numpy as np
import open3d as o3d
import pytest

from groundwater.discontinuity.normals import estimate_normals
from groundwater.discontinuity.segmentation import extract_discontinuities
from groundwater.validation.field_compare import angular_difference, dip_dipdir_to_normal

RNG = np.random.default_rng(42)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _synthetic_joint_set(
    dip: float, dip_direction: float, origin: np.ndarray,
    n_side: int = 40, extent: float = 1.0, n_layers: int = 3, layer_spacing: float = 0.3,
    noise_std: float = 0.0005,
) -> np.ndarray:
    normal = dip_dipdir_to_normal(dip, dip_direction)
    u, v = _plane_basis(normal)
    grid = np.linspace(-extent / 2, extent / 2, n_side)
    gu, gv = np.meshgrid(grid, grid)
    gu, gv = gu.ravel(), gv.ravel()

    layers = []
    for k in range(n_layers):
        pts = origin + k * layer_spacing * normal + gu[:, None] * u + gv[:, None] * v
        pts += RNG.normal(scale=noise_std, size=pts.shape)
        layers.append(pts)
    return np.vstack(layers)


@pytest.fixture
def synthetic_two_sets():
    set_a = _synthetic_joint_set(dip=15.0, dip_direction=90.0, origin=np.array([0.0, 0.0, 0.0]))
    set_b = _synthetic_joint_set(dip=75.0, dip_direction=210.0, origin=np.array([5.0, 5.0, 0.0]))
    points = np.vstack([set_a, set_b])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, {"A": (15.0, 90.0), "B": (75.0, 210.0)}


def test_extract_discontinuities_recovers_known_orientations(synthetic_two_sets):
    pcd, ground_truth = synthetic_two_sets
    pcd = estimate_normals(pcd, search_radius=0.08, max_nn=30)

    discontinuity_sets, point_labels = extract_discontinuities(
        pcd,
        orientation_eps=0.26,
        orientation_min_samples=30,
        spatial_eps=0.15,
        spatial_min_samples=20,
        min_patch_points=30,
    )

    assert len(discontinuity_sets) == 2

    for dip, dip_dir in ground_truth.values():
        errors = [
            angular_difference(dip, dip_dir, d.mean_dip, d.mean_dip_direction)
            for d in discontinuity_sets
        ]
        assert min(errors) < 3.0, f"No extracted set matched ground truth dip={dip} dip_dir={dip_dir}"

    for dset in discontinuity_sets:
        assert len(dset.patches) == 3  # 3 layers per synthetic set
        assert dset.spacing_mean is not None
        assert abs(dset.spacing_mean - 0.3) < 0.05

    assert (point_labels[:, 0] != -1).sum() > 0.9 * len(point_labels)
