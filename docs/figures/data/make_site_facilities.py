"""Site facilities (EPSG:5186) for the figures and the viewer, tied to the 2026.09.07 drone orthomosaic.

* Upstream cut-off wall (L = 202 m): traced on the drone mosaic — it follows the concrete curb along the
  north and north-west edge of the HDPE cover (corner posts read at 2-5 cm/px).
* HDPE cover outline: segmented from the drone mosaic (sky-blue liner pixels) on gentle ground of the 0.5 m
  DTM — the turquoise rockfall netting on the cliffs is excluded by slope — and cut at the south tip along
  the sheet / rubble edge read on the mosaic (shaded rubble looks as blue as the sheet there).
* Downstream cut-off wall (L = 100 m): traced on the 0.5 m DTM (berm + trench, see WALL_DOWN).
* Wells, sumps and the storage tank on the facility plan figure are placed with an affine fitted on the
  three upstream-wall corners (plan pixel -> map), since they are not identifiable on the photo.

    .venv\\Scripts\\python.exe docs\\figures\\data\\make_site_facilities.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from shapely.geometry import LineString, Polygon, mapping, shape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "webapp" / "ortho_history"))
from large_ortho import read_tiles  # noqa: E402

DRONE = ROOT / "webapp" / "ortho_history" / "tiles" / "20260907"
DTM = ROOT / "DEM(5m)" / "dem(0.5m).tif"
LINER_MAX_SLOPE = 40.0    # deg; the cover lies on the gentle landfill surface, the netted cliffs are steeper
# south end of the cover: edge between the sheet and the bare rubble below it (drone mosaic, 8 cm/px)
LINER_SOUTH_EDGE = [(203174.0, 384803.0), (203181.0, 384797.5), (203188.0, 384800.0), (203195.0, 384801.0),
                    (203200.0, 384800.0), (203205.0, 384799.0), (203209.0, 384797.0), (203212.0, 384794.0),
                    (203215.0, 384791.0), (203218.0, 384789.0), (203223.0, 384789.0), (203227.0, 384788.0)]

# upstream wall corners on the drone mosaic (E, N): east end post, NW corner post (V vertex),
# SW corner post, south end of the last short segment
WALL_UP = [(203159.0, 385028.5), (203053.7, 385035.3), (203004.8, 384981.5), (203019.5, 384972.0)]
# the same corners on the facility plan figure (pixel x, y of images/6.webp, 831 x 1125)
PLAN_CP = [(243.3, 680.0), (446.7, 802.7), (596.7, 710.0)]
PLAN_POINTS = {  # name -> (kind, plan x, plan y)
    "PW-01": ("pumping_well", 545, 1001), "PW-02": ("pumping_well", 610, 966), "PW-03": ("pumping_well", 610, 906),
    "PW-04": ("pumping_well", 533, 910), "PW-05": ("pumping_well", 451, 877), "PW-06": ("pumping_well", 371, 734),
    "PW-07": ("pumping_well", 436, 741), "PW-08": ("pumping_well", 483, 716), "PW-09": ("pumping_well", 444, 828),
    "PW-10": ("pumping_well", 325, 769),
    "MW-01": ("monitoring_well", 269, 784), "MW-02": ("monitoring_well", 443, 914),
    "MW-03": ("monitoring_well", 659, 977), "MW-04": ("monitoring_well", 800, 882),
    "집수구-E": ("collection_sump", 236, 688), "집수구-C": ("collection_sump", 460, 806),
    "집수구-W": ("collection_sump", 607, 716), "저류조": ("storage_tank", 562, 698),
}
WALL_DOWN_PLAN = [(550.0, 1073.3), (581.7, 1120.0), (696.7, 981.7)]
# downstream wall traced on the 0.5 m DTM (local relief = DTM - 5 m Gaussian): the long arm follows a straight
# berm 0.25-0.45 m high (also the vegetation / bare-ground edge on the drone mosaic), the short arm a trench
# 0.5-0.8 m deep from the corner to ESE. 64 m + 39 m = 103 m, matching L = 100 m on the plan.
WALL_DOWN = [(202916.2, 385088.3), (202936.0, 385149.0), (202947.0, 385149.0), (202958.0, 385142.0),
             (202968.3, 385131.1)]
EVIDENCE = {
    "up": "드론 정사영상: HDPE 북·북서쪽 콘크리트 연석과 모서리 기둥 (판독 196 m ≈ 배치도 202 m)",
    "down": "0.5 m DTM: 긴 팔 = 직선 둔덕 높이 0.25~0.45 m(64 m, 드론 영상의 식생·나지 경계와 일치), "
            "짧은 팔 = 모서리에서 동남동 고랑 깊이 0.5~0.8 m(39 m). 합계 103 m ≈ 배치도 100 m. 위치 추정",
}
WALL_HALF_WIDTH = 1.0     # m, drawn as a thin polygon so the figures can fill it


def plan_affine():
    src = np.array(PLAN_CP, np.float32)
    dst = np.array(WALL_UP[:3], np.float64)
    o = dst.mean(axis=0)
    return cv2.getAffineTransform(src, (dst - o).astype(np.float32)), o


def to_map(A, o, pts):
    p = np.asarray(pts, float)
    return p @ A[:, :2].T + A[:, 2] + o


def hdpe_outline():
    """HDPE cover on the drone mosaic (0.25 m): mostly sky-blue liner pixels on gentle DTM ground, south tip cut."""
    e0, n0, e1, n1, res = 202950.0, 384700.0, 203300.0, 385060.0, 0.25
    w, h = int((e1 - e0) / res), int((n1 - n0) / res)
    rgb = read_tiles(DRONE, (e0, n0, e1, n1), w, h)[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # the black HDPE reflects the sky: blue hue in sun and shade alike; soil, gravel and trees are not blue
    # (the turquoise rockfall netting on the cliffs has green close to blue; the liner has blue well above green)
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    blue = (hsv[..., 0] >= 98) & (hsv[..., 0] <= 125) & (b - r >= 15) & (b - g >= 12) & (hsv[..., 2] < 200)
    # shaded netting still passes the colour test, so keep only ground gentler than the cliffs
    z = np.zeros((h, w), np.float32)
    with rasterio.open(DTM) as ds:
        reproject(ds.read(1), z, src_transform=ds.transform, src_crs=ds.crs, dst_transform=Affine(res, 0, e0, 0, -res, n1),
                  dst_crs=ds.crs, resampling=Resampling.bilinear)
    gy, gx = np.gradient(cv2.GaussianBlur(z, (0, 0), 2).astype(float), res)
    gentle = np.degrees(np.arctan(np.hypot(gx, gy))) < LINER_MAX_SLOPE
    frac = cv2.GaussianBlur(blue.astype(np.float32), (0, 0), 4)            # ~1 m: the sheet is blue throughout
    m = ((frac > 0.6) & gentle).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    big = 1 + np.argmax(st[1:, 4])
    cnts, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    poly = Polygon([(e0 + (x + 0.5) * res, n1 - (y + 0.5) * res) for x, y in c]).buffer(1.5).buffer(-1.5)
    edge = LINER_SOUTH_EDGE
    below = Polygon(edge + [(edge[-1][0], n0), (edge[0][0], n0)])
    poly = poly.difference(below)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly.simplify(0.5)


def main():
    A, o = plan_affine()
    feats = []
    up = LineString(WALL_UP)
    down = LineString(WALL_DOWN)
    plan_down = LineString(to_map(A, o, WALL_DOWN_PLAN))
    print(f"downstream wall: DTM trace vs plan placement, centroid offset {down.centroid.distance(plan_down.centroid):.0f} m")
    for name, line, source, ev in [("상류 연직차수벽 (L=202 m)", up, "drone", EVIDENCE["up"]),
                                   ("하류 연직차수벽 (L=100 m)", down, "dtm", EVIDENCE["down"])]:
        feats.append({"type": "Feature",
                      "properties": {"kind": "cutoff_wall", "name": name, "length_m": round(line.length, 1),
                                     "source": source, "evidence": ev,
                                     "centerline": [list(map(lambda v: round(v, 2), p)) for p in line.coords]},
                      "geometry": mapping(line.buffer(WALL_HALF_WIDTH, cap_style=2, join_style=2))})
    for name, (kind, x, y) in PLAN_POINTS.items():
        e, n = to_map(A, o, [(x, y)])[0]
        feats.append({"type": "Feature", "properties": {"kind": kind, "name": name, "source": "plan"},
                      "geometry": {"type": "Point", "coordinates": [round(e, 2), round(n, 2)]}})
    liner = hdpe_outline()
    feats.append({"type": "Feature", "properties": {"kind": "hdpe_liner", "area_ha": round(liner.area / 1e4, 2),
                                                    "source": "drone"},
                  "geometry": {"type": "Polygon", "coordinates": [[[round(a, 2), round(b, 2)] for a, b in liner.exterior.coords]]}})
    fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:5186"}},
          "note": "Upstream wall and HDPE outline traced on the 2026.09.07 drone orthomosaic; wells, sumps, storage "
                  "tank and the downstream wall placed from the facility plan with an affine fitted on the three "
                  "upstream-wall corners (positions of those approximate, a few m).",
          "features": feats}
    (HERE / "site_facilities.geojson").write_text(json.dumps(fc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"upstream wall {up.length:.0f} m, downstream wall {down.length:.0f} m, HDPE {liner.area / 1e4:.2f} ha")
    print("plan scale", np.round(np.sqrt(abs(np.linalg.det(A[:, :2]))), 3), "m/px")


if __name__ == "__main__":
    main()
