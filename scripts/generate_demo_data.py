"""Regenerate the synthetic demo dataset used by configs/demo_site.yaml.

Produces a colored point cloud of a bent quarry wall with three natural joint
sets exposed as faces (see groundwater.synthetic), plus a noisy field
measurement CSV of those sets for the validation tab. Run from the repo root:

    python scripts/generate_demo_data.py
"""
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from groundwater.synthetic import SyntheticJoint, synthetic_wall

DEMO_DIR = Path("data/raw/demo_site")
POLYLINE = [[0.0, 0.0], [12.0, 0.0], [20.0, -6.0]]  # must match configs/demo_site.yaml
SETS = {"A": (60.0, 150.0), "B": (35.0, 200.0), "C": (80.0, 90.0)}
PLACEMENTS = {  # (chainage, elevation) of each joint's trace centre
    "A": [(3.0, 105.0), (10.5, 102.2), (19.5, 105.5)],
    "B": [(6.0, 102.5), (14.5, 105.5), (20.0, 101.8)],
    "C": [(9.0, 105.5), (5.0, 106.0), (17.0, 102.5)],
}


def main() -> None:
    rng = np.random.default_rng(7)
    joints = [
        SyntheticJoint(dip=SETS[name][0], dip_direction=SETS[name][1], chainage=u, elevation=v)
        for name, spots in PLACEMENTS.items()
        for u, v in spots
    ]
    points, colors = synthetic_wall(rng, POLYLINE, joints)

    cloud_dir = DEMO_DIR / "rgb_photogrammetry"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(cloud_dir / "pointcloud.ply"), pcd)
    print(f"Wrote {len(points)} points to {cloud_dir / 'pointcloud.ply'}")

    field_dir = DEMO_DIR / "field"
    field_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"{name}-{j}",
            "dip": dip + rng.normal(scale=1.5),
            "dip_direction": (dip_dir + rng.normal(scale=2.0)) % 360,
        }
        for name, (dip, dip_dir) in SETS.items()
        for j in range(3)
    ]
    pd.DataFrame(rows).to_csv(field_dir / "scanline_joints.csv", index=False)
    print(f"Wrote {len(rows)} field measurements to {field_dir / 'scanline_joints.csv'}")

    # wall plan line for folder mode (configs/demo_site.yaml carries the same line and pit point)
    wall = pd.DataFrame([{"kind": "wall", "x": x, "y": y} for x, y in POLYLINE] + [{"kind": "pit", "x": 8.0, "y": -10.0}])
    wall.to_csv(DEMO_DIR / "wall_line.csv", index=False)
    print(f"Wrote {DEMO_DIR / 'wall_line.csv'}")


if __name__ == "__main__":
    main()
