"""High-resolution surface-water analysis on the 0.5 m DTM (DEM(5m)/dem(0.5m).tif), only within its
own extent, for the orthophoto viewer (terrain/hr_*):

    hr_hillshade.webp, hr_mfd.webp   shaded relief and hillslope flow intensity (1 m display grid)
    hr_hillshade_elev.webp           elevation-tinted relief (blue -> orange, DTM min-max)
    hillshade_elev_site.webp         the 5 m DEM tinted on the same DTM range, so both match where they meet
    hr_basins.geojson                where rain ends up: onto the landfill / same outlet / other outlets
    hr_flows.geojson                 D8 channels >= HR_FLOW_HA, coloured by class
    hr_site.geojson                  landfill catchment + surface outflow path
    hr_meta.json                     bounds, class areas, comparison with the 5 m result

The 0.5 m DTM shows the present ground (HDPE cover, benches, roads, drains), so small channels and
divides follow built features that the 5 m DEM smooths away. Culverts under roads still look like
dams; the stream-burn lines in terrain_corrections.geojson are applied where they fall inside.

    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_terrain_hr.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]
from groundwater.terrain import hydrology as hy  # noqa: E402
from build_terrain import (COLOR_RANGE, CRS, FACILITIES, OUT, burn_corrections, cell_centres, find_dem,  # noqa: E402
                           lonlat, rgba_png, ring_lonlat, ROOT)

HR_DEM = ROOT / "DEM(5m)" / "dem(0.5m).tif"
HR_FLOW_HA = 0.2          # smallest channel drawn
DISPLAY_RES = 1.0         # m, raster overlays for the browser
LANDFILL_EDGE_M = 3.0     # the HDPE outline is registered to ~1 m and a 1 m-deep trough runs along the east cliff
                          # toe just outside it; without this margin the trough carries the cliff-face runoff
                          # north past the outline and the catchment edge follows the cover instead of the
                          # cliff top (2, 3 and 5 m all give the same catchment)
COLOURS = ["#ea580c", "#0d9488", "#2563eb", "#65a30d", "#9333ea", "#ca8a04", "#64748b"]


def direction_name(dx, dy):
    ang = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    return ["북쪽", "북동쪽", "동쪽", "남동쪽", "남쪽", "남서쪽", "서쪽", "북서쪽"][int(((ang + 22.5) % 360) // 45)]


def to_display(arr, t, resampling):
    """Resample a 0.5 m array to the DISPLAY_RES grid (same origin)."""
    k = int(round(DISPLAY_RES / t.a))
    h, w = arr.shape[0] // k, arr.shape[1] // k
    out = np.zeros((h, w), np.float32)
    reproject(arr.astype(np.float32), out, src_transform=t, dst_transform=t * Affine.scale(k), src_crs=CRS,
              dst_crs=CRS, resampling=resampling)
    return out, t * Affine.scale(k)


def webp(path, rgba):
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=82, method=4)


def main():
    with rasterio.open(HR_DEM) as ds:
        z = ds.read(1).astype(np.float64)
        t = ds.transform
        nodata = ds.nodata
    cell = t.a
    valid = (z != nodata) & (z > -1000)
    h, w = z.shape
    print(f"{HR_DEM.name}: {w}x{h} cells, {cell} m, valid {valid.sum() * cell * cell / 1e6:.2f} km2")

    zr = burn_corrections(np.where(valid, z, 0), valid, t)
    filled = hy.fill_depressions(zr, valid)
    code = hy.flow_directions(filled, valid, cell)
    acc = hy.flow_accumulation(filled, code, valid)
    print(f"routing done; max contributing area {acc.max() * cell * cell / 1e4:.0f} ha")

    fc = json.loads(FACILITIES.read_text(encoding="utf-8"))
    liner = next(f for f in fc["features"] if f["properties"]["kind"] == "hdpe_liner")["geometry"]
    from shapely.geometry import mapping, shape
    landfill = rasterize([(mapping(shape(liner).buffer(LANDFILL_EDGE_M)), 1)], out_shape=(h, w),
                         transform=t).astype(bool) & valid
    up = hy.upstream_mask(code, landfill)
    term = hy.terminal_outlet(code)
    lt = np.bincount(term[landfill]).argmax()

    # classes over the whole window
    cls = np.full((h, w), -1, np.int16)
    cls[valid & up] = 0
    cls[valid & ~up & (term == lt)] = 1
    rr, cc = np.nonzero(landfill)
    cx, cy = t * (cc.mean() + 0.5, rr.mean() + 0.5)
    classes = [{"key": "landfill", "name": "매립지로 유입", "color": COLOURS[0]},
               {"key": "same", "name": "매립지 유출수와 같은 출구 (매립지 미경유)", "color": COLOURS[1]}]
    ids, counts = np.unique(term[valid & (term != lt)], return_counts=True)
    order = np.argsort(-counts)
    for k, i in enumerate(order[:4]):
        r, c = divmod(int(ids[i]), w)
        ox, oy = t * (c + 0.5, r + 0.5)
        classes.append({"key": f"o{k}", "name": f"{direction_name(ox - cx, oy - cy)} 출구", "color": COLOURS[2 + k]})
        cls[valid & (term == ids[i])] = 2 + k
    rest = valid & (cls < 0)
    if rest.any():
        classes.append({"key": "rest", "name": "기타 작은 출구", "color": COLOURS[6]})
        cls[rest] = len(classes) - 1
    for k, c in enumerate(classes):
        c["area_ha"] = round(float((cls == k).sum()) * cell * cell / 1e4, 1)
    print("classes:", ", ".join(f"{c['name']} {c['area_ha']} ha" for c in classes))

    # polygons on the 1 m grid (0.5 m outlines are huge and add nothing at screen scale)
    cls_d, td = to_display(cls.astype(np.float32), t, Resampling.mode)
    feats = []
    for geom, v in shapes(cls_d.astype(np.int16), mask=cls_d >= 0, transform=td):
        g = shape(geom).simplify(1.0)
        c = classes[int(v)]
        for poly in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
            if poly.area < 25:
                continue
            feats.append({"type": "Feature", "properties": {"name": c["name"], "color": c["color"], "area_ha": c["area_ha"]},
                          "geometry": {"type": "Polygon", "coordinates": [ring_lonlat(poly.exterior.coords, 6)]
                                       + [ring_lonlat(i.coords, 6) for i in poly.interiors]}})
    (OUT / "hr_basins.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")

    # channels
    streams = acc * cell * cell >= HR_FLOW_HA * 1e4
    sorder = hy.strahler_order(code, streams)
    lines = []
    for cells, o, a in hy.stream_segments(code, streams, sorder, acc):
        if len(cells) < 4:
            continue
        cells = cells[::2] + ([cells[-1]] if len(cells) % 2 == 0 else [])
        k = int(cls[cells[0]])
        lon, lat = cell_centres(t, cells)
        lines.append({"type": "Feature", "properties": {"area_ha": round(a * cell * cell / 1e4, 2), "order": o,
                                                        "color": classes[k]["color"] if k >= 0 else "#334155"},
                      "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in zip(lon, lat)]}})
    (OUT / "hr_flows.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": lines}), encoding="utf-8")
    print(f"channels >= {HR_FLOW_HA} ha: {len(lines)} segments")

    # landfill catchment outline + outflow path
    site = []
    up_d, _ = to_display(up.astype(np.float32), t, Resampling.max)
    for geom, _ in shapes(up_d.astype(np.uint8), mask=up_d > 0, transform=td):
        site.append({"type": "Feature", "properties": {"kind": "landfill_catchment",
                                                       "area_ha": classes[0]["area_ha"]},
                     "geometry": {"type": "Polygon", "coordinates": [ring_lonlat(geom["coordinates"][0], 6)]}})
    k = np.argmax(acc[rr, cc])
    path = hy.trace_downstream(code, rr[k], cc[k])
    length = sum(hy.DIST[code[r, c]] for r, c in path[:-1]) * cell
    lon, lat = cell_centres(t, path[::4] + [path[-1]])
    site.append({"type": "Feature", "properties": {"kind": "outflow_path", "length_m": round(length)},
                 "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in zip(lon, lat)]}})
    (OUT / "hr_site.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": site}), encoding="utf-8")
    print(f"landfill catchment {classes[0]['area_ha']} ha, outflow path {length:.0f} m to the DTM edge")

    # rasters for display
    zf = np.where(valid, z, np.nan)
    zf = np.where(valid, z, np.nanmedian(zf))
    hs = hy.hillshade(zf, cell)
    hs_d, _ = to_display(hs, t, Resampling.average)
    val_d, _ = to_display(valid.astype(np.float32), t, Resampling.min)
    g = (hs_d * 255).astype(np.uint8)
    webp(OUT / "hr_hillshade.webp", np.dstack([g, g, g, np.where(val_d > 0, 255, 0).astype(np.uint8)]))
    cr = hy.color_relief(zf, valid, cell, *COLOR_RANGE)                     # at 0.5 m, then down to the 1 m display grid
    cr_d = np.stack([to_display(cr[..., i].astype(np.float32), t, Resampling.average)[0] for i in range(3)], -1)
    webp(OUT / "hr_hillshade_color.webp",
         np.dstack([cr_d.clip(0, 255).astype(np.uint8), np.where(val_d > 0, 255, 0).astype(np.uint8)]))
    elev_range = [round(float(z[valid].min()), 2), round(float(z[valid].max()), 2)]
    er = hy.color_relief(zf, valid, cell, *elev_range, ramp=hy.ELEV_RAMP)
    er_d = np.stack([to_display(er[..., i].astype(np.float32), t, Resampling.average)[0] for i in range(3)], -1)
    webp(OUT / "hr_hillshade_elev.webp",
         np.dstack([er_d.clip(0, 255).astype(np.uint8), np.where(val_d > 0, 255, 0).astype(np.uint8)]))
    with rasterio.open(find_dem()) as ds:                                    # 5 m DEM on the DTM's colour range
        z5 = ds.read(1).astype(np.float64)
        rgba_png(OUT / "hillshade_elev_site.png",
                 hy.color_relief(z5, z5 > 0, ds.transform.a, *elev_range, ramp=hy.ELEV_RAMP))
    mfd = hy.flow_accumulation_mfd(filled, valid, cell)
    mfd_d, _ = to_display(mfd * cell * cell, t, Resampling.max)
    la = np.log10(np.maximum(mfd_d, 1))
    v = np.clip((la - np.log10(50)) / (np.log10(2e4) - np.log10(50)), 0, 1)     # 0.005 .. 2 ha
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[..., 0] = (200 * (1 - v)).astype(np.uint8)
    rgba[..., 1] = (230 - 60 * v).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = (np.where(val_d > 0, v, 0) ** 1.2 * 235).astype(np.uint8)
    webp(OUT / "hr_mfd.webp", rgba)

    # grid for figures (landfill window, native 0.5 m)
    win = 360.0
    c0, r0 = map(int, ~t * (cx - win, cy + win))
    c1, r1 = map(int, ~t * (cx + win, cy - win))
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, w), min(r1, h)
    np.savez_compressed(OUT / "hr_grid.npz", cls=cls[r0:r1, c0:c1], z=z[r0:r1, c0:c1], valid=valid[r0:r1, c0:c1],
                        mfd=mfd[r0:r1, c0:c1] * cell * cell, transform=np.array((t * Affine.translation(c0, r0))[:6]))

    # micro-relief (DTM minus a 5 m Gaussian surface) at the native 0.5 m: berms red, trenches/ditches blue —
    # buried walls, curbs, drains and old benches show up as straight lines
    from scipy.ndimage import gaussian_filter
    lrm = zf - gaussian_filter(zf, 5.0 / cell)
    v = np.clip(lrm / 0.6, -1, 1)
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[..., 0] = np.where(v > 0, 220, 30)
    rgba[..., 1] = np.where(v > 0, 40, 90)
    rgba[..., 2] = np.where(v > 0, 40, 230)
    # transparent below 0.12 m so only distinct berms / trenches show over the orthophoto
    a = np.clip((np.abs(lrm) - 0.12) / 0.5, 0, 1)
    rgba[..., 3] = (np.where(valid, a, 0) ** 0.7 * 210).astype(np.uint8)
    webp(OUT / "hr_relief.webp", rgba)

    # downslope azimuth for the flow arrows, on the 1 m display grid (filled surface averaged to 1 m)
    filled_d, _ = to_display(np.where(valid, filled, np.nan), t, Resampling.average)
    vd = np.isfinite(filled_d) & (val_d > 0)
    (OUT / "hr_flowdir.bin").write_bytes(hy.downslope_azimuth(np.where(vd, filled_d, 0), vd, DISPLAY_RES, None,
                                                              sigma=1.5).tobytes())

    ne1, nn0 = t * (w, h)
    nlon, nlat = lonlat([t.c, ne1], [nn0, t.f])
    native_bounds = [[min(nlat), min(nlon)], [max(nlat), max(nlon)]]
    e1, n0 = td * (cls_d.shape[1], cls_d.shape[0])
    lon, lat = lonlat([td.c, e1], [n0, td.f])
    five = json.loads((OUT / "meta.json").read_text(encoding="utf-8")).get("site", {}) if (OUT / "meta.json").exists() else {}
    meta = {"dem": HR_DEM.name, "cell": cell, "bounds": [[min(lat), min(lon)], [max(lat), max(lon)]],
            "flowdir": {"x0": td.c, "y0": td.f, "cell": DISPLAY_RES, "width": cls_d.shape[1], "height": cls_d.shape[0]},
            "relief_bounds": native_bounds,
            "classes": classes, "flow_min_ha": HR_FLOW_HA, "outflow_path_m": round(length),
            "catchment_ha": classes[0]["area_ha"], "catchment_ha_5m": five.get("catchment_ha"),
            "elev_range": elev_range}
    (OUT / "hr_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}/hr_*")


if __name__ == "__main__":
    main()
