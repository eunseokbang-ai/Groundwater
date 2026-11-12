"""The LiDAR section of the deck (slide 'LiDAR 단면 확인 도구'): the east-wall section A -> B as the dashboard's
section tool shows it -- returns by kind, the rear layer kept as rock, left out on gentle surfaces or above the
ground (tree crowns over the wall top), and the analysis points by joint set. Saved like the tool's PNG button.

    .venv\\Scripts\\python.exe docs\\figures\\make_lidar_section_figure.py

Writes docs/figures/round2/full/lidar_section.png and into 구간/E_동측벽/분석결과/단면확인/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from groundwater import lidar_section as LS  # noqa: E402
from groundwater import site_joints as SJ  # noqa: E402
from groundwater.viz import lidar_section_view as LV  # noqa: E402
from groundwater.viz import static_map  # noqa: E402

SITE = REPO / "Filed_Data" / "IcSan20261021"
LAS = SITE / "U자전체자료" / "IcSan20261024_LiDAR_L2_whole.las"
A, B = np.array([203172.6, 384992.1]), np.array([203214.5, 385027.8])
HALF = 0.05


def main():
    import open3d as o3d

    slab = LS.read_slab(LAS, LS.load_index(LAS), A, B)
    kinds = LS.return_kind(slab["ret"], slab["nret"])
    owner, folder = LS.owning_section(SITE, (A + B) / 2)
    frame = LS.section_frame(folder, slab["xyz"][:: max(1, len(slab["xyz"]) // 200_000)])
    res = folder / "분석결과"
    pts = np.asarray(o3d.io.read_point_cloud(str(res / "classified_point_cloud.ply")).points)
    cls = LS.classify(slab["xyz"], frame, SJ.find_ground(SITE), pts)       # the analysis points: the wall's columns
    sids = np.load(res / "point_labels.npy")[:, 0]
    t = pd.read_csv(res / "discontinuity_sets.csv")
    names = {int(r.set_id): f"Set {int(r.set_id)} {r.mean_dip:.0f}°/{r.mean_dip_direction:03.0f}°" for r in t.itertuples()}
    u = (B - A) / np.hypot(*(B - A))
    rel = pts[:, :2] - A
    ss, dd = rel @ u, rel @ np.array([-u[1], u[0]])
    m = (np.abs(dd) <= HALF) & (ss >= -1) & (ss <= slab["length"] + 1)
    title = (f"단면 A({A[0]:.1f}, {A[1]:.1f}) → B({B[0]:.1f}, {B[1]:.1f}) · 길이 {slab['length']:.1f} m · 두께 ±{HALF * 100:.0f} cm"
             + (f" · 구간 {owner}" if owner else ""))
    fig = LV.section_figure(slab, kinds, cls, HALF, (ss[m], pts[m, 2], sids[m]), names, title, True)
    png = static_map.preview_image(fig, width_px=1400, max_height_px=1000)
    (HERE / "round2" / "full" / "lidar_section.png").write_bytes(png)
    out = res / "단면확인"
    out.mkdir(exist_ok=True)
    (out / f"단면_{A[0]:.0f}_{A[1]:.0f}_{B[0]:.0f}_{B[1]:.0f}.png").write_bytes(png)
    print(LV.slice_stats(slab, kinds, cls, HALF))


if __name__ == "__main__":
    main()
