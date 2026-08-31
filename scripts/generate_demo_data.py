"""Regenerate the synthetic demo dataset used by configs/demo_site.yaml.

Produces a point cloud with 3 known joint sets (each 3 parallel layers,
0.25 m spacing), a matching noisy field-measurement CSV, and a synthetic
orthomosaic with 3 line-like features -- enough to exercise every stage of
the pipeline (extraction, spacing, 2D trace mapping, field validation)
without needing real drone data. Run from the repo root:

    python scripts/generate_demo_data.py
"""
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from groundwater.validation.field_compare import dip_dipdir_to_normal

RNG = np.random.default_rng(7)

DEMO_DIR = Path("data/raw/demo_site")
SETS_TRUTH = {"A": (15.0, 90.0), "B": (75.0, 210.0), "C": (50.0, 320.0)}


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _joint_set(dip, dip_direction, origin, n_side=50, extent=1.5, n_layers=3, spacing=0.25, noise=0.001):
    normal = dip_dipdir_to_normal(dip, dip_direction)
    u, v = _plane_basis(normal)
    grid = np.linspace(-extent / 2, extent / 2, n_side)
    gu, gv = np.meshgrid(grid, grid)
    gu, gv = gu.ravel(), gv.ravel()
    layers = []
    for k in range(n_layers):
        pts = origin + k * spacing * normal + gu[:, None] * u + gv[:, None] * v
        pts += RNG.normal(scale=noise, size=pts.shape)
        layers.append(pts)
    return np.vstack(layers)


def _draw_line(arr, r0, c0, r1, c1, width=2, value=40):
    n = max(abs(r1 - r0), abs(c1 - c0))
    for t in np.linspace(0, 1, n):
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        arr[max(0, r - width):r + width, max(0, c - width):c + width] = value


def generate_point_cloud() -> None:
    out_dir = DEMO_DIR / "rgb_photogrammetry"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = np.vstack(
        [
            _joint_set(15.0, 90.0, np.array([0.0, 0.0, 0.0])),
            _joint_set(75.0, 210.0, np.array([5.0, 5.0, 0.0])),
            _joint_set(50.0, 320.0, np.array([-4.0, 3.0, 1.0])),
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_path = out_dir / "pointcloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Wrote {len(points)} points to {ply_path}")


def generate_field_measurements() -> None:
    field_dir = DEMO_DIR / "field"
    field_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (dip, dip_dir) in SETS_TRUTH.items():
        for j in range(3):
            rows.append(
                {
                    "id": f"{name}-{j}",
                    "dip": dip + RNG.normal(scale=1.5),
                    "dip_direction": (dip_dir + RNG.normal(scale=2.0)) % 360,
                }
            )
    field_csv = field_dir / "scanline_joints.csv"
    pd.DataFrame(rows).to_csv(field_csv, index=False)
    print(f"Wrote {len(rows)} field measurements to {field_csv}")


def generate_orthomosaic() -> None:
    size = 400
    img = np.full((size, size), 180, dtype=np.uint8)
    img = np.clip(img.astype(int) + RNG.integers(-5, 5, size=(size, size)), 0, 255).astype(np.uint8)

    _draw_line(img, 20, 30, 380, 120)
    _draw_line(img, 350, 20, 40, 300)
    _draw_line(img, 10, 350, 390, 380)

    rgb = np.stack([img, img, img], axis=0)
    out_path = DEMO_DIR / "rgb_photogrammetry" / "orthomosaic.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0, 0, 0.01, 0.01)  # 1 cm/pixel, origin at (0, 0)

    with rasterio.open(
        out_path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype=rgb.dtype, crs="EPSG:5186", transform=transform,
    ) as dst:
        dst.write(rgb)
    print(f"Wrote synthetic orthomosaic to {out_path}")


if __name__ == "__main__":
    generate_point_cloud()
    generate_field_measurements()
    generate_orthomosaic()
