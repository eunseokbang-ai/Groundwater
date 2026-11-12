"""Layers of a new drone DSM for the orthophoto viewer (terrain/dsm_*):

    dsm_relief.webp      elevation-tinted shaded relief of the DSM (blue -> orange, as the DTM relief; 0.25 m)
    dsm_change.webp      surface change against the earlier 0.5 m DTM: red = higher now, blue = lower, on bare
                         ground only -- vegetation of the new orthophoto, rough surfaces (tree crowns the colour
                         misses, vehicles, panels) and cliffs left out: a DSM has the tree tops, the DTM the ground
                         under them -- transparent within the noise (+-0.15 m)
    dsm_meta.json        bounds, dates, elevation range, areas / volumes of change

A DSM is the top surface (trees, vehicles), so it is not used for the surface-water routing, which needs the
ground (DTM). On bare ground the two compare directly: Iksan 2026-10-23 DSM vs the 2026-09-07 0.5 m DTM agree to
0.10 m (median) on flat ground, both orthometric (KNGeoid18).

    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_dsm_layers.py [--dsm DSM.tif] [--ortho ORTHO.tif]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]
from groundwater.terrain import hydrology as hy  # noqa: E402
from build_terrain import CRS, OUT, ROOT, lonlat  # noqa: E402
from build_terrain_hr import HR_DEM, webp  # noqa: E402

SITE = ROOT / "Filed_Data" / "IcSan20261021"
RELIEF_RES = 0.25         # m
NOISE_M = 0.15            # |change| below this is not drawn
FULL_M = 1.5              # |change| drawn fully opaque from here
REPORT_M = 0.3            # change counted in the areas / volumes from here
EXG_THRESHOLD = 0.06      # as the joint analysis (discontinuity.vegetation)
ROUGH_M = 0.12            # DSM relief within ~1 m above this: trees (also dry / dark canopy the colour misses),
                          # vehicles -- not ground; bare earthwork is smooth at this scale
ROUGH2_M = 0.35           # and within ~2 m: rows of solar panels (smooth tops, ground between the rows)
STEEP_DEG = 35.0          # steeper ground (cliffs) left out: a few cm of horizontal offset is metres of height
SAME_LOOK_NCC = 0.6       # the earlier and the new orthophoto alike here (normalised cross-correlation over ~3.5 m):
SAME_LOOK_MAX_M = 0.5     # then a height difference above this is the DSM / DTM difference (solar panels, roofs the
                          # DTM left out), not a change of the ground; earthworks change the picture too


def _date(path: Path) -> str | None:
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", path.stem)
    return f"{m.group(1)}.{m.group(2)}.{m.group(3)}" if m else None


def newest(pattern_words, exclude=()) -> Path | None:
    """The newest (by the date in its name) GeoTIFF of the site folder whose name has one of the words."""
    c = [f for f in SITE.glob("*.tif") if any(w in f.name.lower() for w in pattern_words)
         and not any(w in f.name.lower() for w in exclude)]
    c = [f for f in c if _date(f)]
    return max(c, key=lambda f: _date(f)) if c else None


def read_to(path: Path, transform: Affine, shape, bands=1, resampling=Resampling.average) -> np.ndarray:
    out = np.full((bands,) + shape, np.nan, np.float32)
    with rasterio.open(path) as d:
        for b in range(bands):
            reproject(rasterio.band(d, b + 1), out[b], dst_transform=transform, dst_crs=CRS, resampling=resampling,
                      src_nodata=d.nodata, dst_nodata=np.nan)
    return out if bands > 1 else out[0]


def bounds_of(transform: Affine, shape):
    h, w = shape
    e1, n0 = transform * (w, h)
    lon, lat = lonlat([transform.c, e1], [n0, transform.f])
    return [[min(lat), min(lon)], [max(lat), max(lon)]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsm", help="the new DSM (default: the newest *DSM*.tif of the site folder)")
    ap.add_argument("--ortho", help="its orthophoto, for the vegetation mask (default: the newest *ortho*.tif)")
    ap.add_argument("--before-ortho", help="the orthophoto of the earlier DTM's date (default: the site folder's "
                                           "*20260907*.tif): where nothing looks different, no change is drawn")
    args = ap.parse_args()
    dsm = Path(args.dsm) if args.dsm else newest(("dsm",))
    ortho = Path(args.ortho) if args.ortho else newest(("ortho",), exclude=("planar",))
    if dsm is None:
        sys.exit(f"No DSM in {SITE}")
    with rasterio.open(dsm) as d:
        b = d.bounds
    print(f"DSM {dsm.name} ({_date(dsm)}), orthophoto {ortho.name if ortho else '-'}, earlier DTM {HR_DEM.name}")

    # relief of the DSM, 0.25 m
    t = Affine(RELIEF_RES, 0, b.left, 0, -RELIEF_RES, b.top)
    shape = (int((b.top - b.bottom) / RELIEF_RES), int((b.right - b.left) / RELIEF_RES))
    z = read_to(dsm, t, shape)
    valid = np.isfinite(z)
    lo, hi = (float(v) for v in np.percentile(z[valid], [1, 99.5]))
    webp(OUT / "dsm_relief.webp", hy.color_relief(z, valid, RELIEF_RES, lo, hi, ramp=hy.ELEV_RAMP))
    relief_bounds = bounds_of(t, shape)
    import cv2

    zf = np.where(valid, z, np.nanmedian(z)).astype(np.float32)
    rough = cv2.GaussianBlur(np.abs(zf - cv2.GaussianBlur(zf, (0, 0), 2)), (0, 0), 2)    # ~1 m, detrended
    rough2 = cv2.GaussianBlur(np.abs(zf - cv2.GaussianBlur(zf, (0, 0), 8)), (0, 0), 4)    # ~2 m
    rough = np.where(valid, np.maximum(rough, rough2 * ROUGH_M / ROUGH2_M), np.nan)       # one test: rough > ROUGH_M
    del z, zf

    # change against the earlier DTM, on its 0.5 m grid inside the DSM
    with rasterio.open(HR_DEM) as d:
        dt, cell = d.transform, d.res[0]
    c0, r0 = (int(np.floor(v)) for v in ~dt * (b.left, b.top))
    c1, r1 = (int(np.ceil(v)) for v in ~dt * (b.right, b.bottom))
    tc = dt * Affine.translation(c0, r0)
    cshape = (r1 - r0, c1 - c0)
    before = read_to(HR_DEM, tc, cshape, resampling=Resampling.nearest)
    after = read_to(dsm, tc, cshape)
    diff = after - before
    ok = np.isfinite(diff)
    veg = np.zeros(cshape, bool)
    if ortho is not None:
        from groundwater.discontinuity.vegetation import vegetation_mask

        rgb = read_to(ortho, tc, cshape, bands=3)
        veg = vegetation_mask(np.moveaxis(np.nan_to_num(rgb), 0, -1), EXG_THRESHOLD, scale=255.0)
        veg |= ~np.isfinite(rgb).all(axis=0)
    rough_c = np.full(cshape, np.nan, np.float32)
    reproject(rough, rough_c, src_transform=t, src_crs=CRS, dst_transform=tc, dst_crs=CRS, resampling=Resampling.max,
              src_nodata=np.nan, dst_nodata=np.nan)
    objects = np.nan_to_num(rough_c, nan=1.0) > ROUGH_M
    steep = np.hypot(*np.gradient(np.nan_to_num(before), cell)) > np.tan(np.radians(STEEP_DEG))
    bare = ok & ~veg & ~objects & ~steep
    flat0 = bare & (np.hypot(*np.gradient(np.nan_to_num(before), cell)) < np.tan(np.radians(4)))
    bias = float(np.median(diff[flat0])) if flat0.any() else 0.0
    same = np.zeros(cshape, bool)
    bo = Path(args.before_ortho) if args.before_ortho else next(iter(sorted(SITE.glob("*20260907*.tif"))), None)
    if ortho is not None and bo is not None:
        import cv2

        g_new = np.nan_to_num(read_to(ortho, tc, cshape, bands=3)).mean(axis=0).astype(np.float32)
        g_old = np.nan_to_num(read_to(bo, tc, cshape, bands=3)).mean(axis=0).astype(np.float32)
        k = (7, 7)
        m1, m2 = cv2.blur(g_new, k), cv2.blur(g_old, k)
        cov = cv2.blur(g_new * g_old, k) - m1 * m2
        v1 = np.maximum(cv2.blur(g_new * g_new, k) - m1 * m1, 1e-3)
        v2 = np.maximum(cv2.blur(g_old * g_old, k) - m2 * m2, 1e-3)
        ncc = cov / np.sqrt(v1 * v2)
        same = (ncc > SAME_LOOK_NCC) & (np.abs(np.nan_to_num(diff - bias)) > SAME_LOOK_MAX_M)
        bare &= ~same
        print(f"before orthophoto {bo.name}: {same.sum() * cell * cell / 1e4:.2f} ha alike in both photos left out")
    flat = bare & (np.hypot(*np.gradient(np.nan_to_num(before), cell)) < np.tan(np.radians(4)))
    bias = float(np.median(diff[flat])) if flat.any() else 0.0
    d = np.where(bare, diff - bias, np.nan)
    rgba = np.zeros(cshape + (4,), np.uint8)
    up = d > 0
    rgba[..., 0] = np.where(up, 215, 37)
    rgba[..., 1] = np.where(up, 48, 99)
    rgba[..., 2] = np.where(up, 39, 235)
    a = np.clip((np.abs(np.nan_to_num(d)) - NOISE_M) / (FULL_M - NOISE_M), 0, 1)
    rgba[..., 3] = (np.where(np.isfinite(d), a, 0) ** 0.6 * 230).astype(np.uint8)
    webp(OUT / "dsm_change.webp", rgba)
    area = cell * cell
    raised, lowered = np.nan_to_num(d) > REPORT_M, np.nan_to_num(d) < -REPORT_M
    stats = {"bare_ha": round(bare.sum() * area / 1e4, 2), "vegetation_ha": round((ok & veg).sum() * area / 1e4, 2),
             "rough_objects_ha": round((ok & ~veg & objects).sum() * area / 1e4, 2),
             "steep_ha": round((ok & ~veg & ~objects & steep).sum() * area / 1e4, 2),
             "unchanged_look_ha": round(same.sum() * area / 1e4, 2),
             "raised_m2": round(float(raised.sum() * area)), "lowered_m2": round(float(lowered.sum() * area)),
             "raised_m3": round(float(d[raised].sum() * area)), "lowered_m3": round(float(-d[lowered].sum() * area)),
             "flat_bias_m": round(bias, 3), "flat_spread_m": round(float(1.4826 * np.median(np.abs(diff[flat] - bias))), 3)
             if flat.any() else None}
    meta = {"dsm": dsm.name, "date": _date(dsm), "before": HR_DEM.name, "before_date": "2026.09.07",
            "relief_bounds": relief_bounds, "elev_range": [round(lo, 1), round(hi, 1)],
            "change_bounds": bounds_of(tc, cshape), "noise_m": NOISE_M, "full_m": FULL_M, "report_m": REPORT_M,
            "change": stats}
    (OUT / "dsm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta["change"], ensure_ascii=False))
    print(f"Wrote {OUT}/dsm_*")


if __name__ == "__main__":
    main()
