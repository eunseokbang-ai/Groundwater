"""Surface-water layers for the orthophoto viewer from the 5 m DEM (DEM(5m)/*.tif, EPSG:5186):

    terrain/hillshade.webp   shaded relief (image overlay)
    terrain/hillshade_elev.webp  elevation-tinted relief (blue -> orange, DEM min-max), range in meta.json
    terrain/flowacc.webp     flow accumulation, log-scaled blue with alpha
    terrain/twi.webp         topographic wetness index (likely wet / seepage-prone ground)
    terrain/streams.geojson  D8 stream network (>= STREAM_HA contributing area), Strahler order
    terrain/subbasins.geojson  catchment of every stream link >= SUBBASIN_KM2 (regional watersheds)
    terrain/detail_*         the landfill surroundings in detail (where rain ends up, small flow lines)

The largest DEM in DEM(5m)/ is used (a regional extract wins over the site one).
    terrain/site.geojson     catchment draining onto the landfill + downstream paths from it
    terrain/d8.bin, meta.json  flow directions for click-to-trace in the browser

The DEM is filled (priority-flood) before routing, so closed quarry pits spill over their lowest
rim; real pits hold water instead. Surface flow only: infiltration and joint flow are not modelled.

    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_terrain.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize, shapes
from rasterio.warp import transform as transform_coords

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from groundwater.terrain import hydrology as hy  # noqa: E402

OUT = Path(__file__).resolve().parent / "terrain"
FACILITIES = ROOT / "docs" / "figures" / "data" / "site_facilities.geojson"
CORRECTIONS = Path(__file__).resolve().parent / "terrain_corrections.geojson"
STREAM_HA = 3.0          # minimum contributing area (ha) for a mapped stream
COLOR_RANGE = (4.0, 250.0)  # m, shared hypsometric scale for the 5 m and 0.5 m colour relief
ELEV_PCT = (0.5, 99.9)      # elevation-tint stretch of the 5 m DEM (percentiles; isolated peaks saturate)
SUBBASIN_KM2 = 1.0       # sub-basins: catchment of every stream link draining >= this area
CRS = "EPSG:5186"


def lonlat(xs, ys):
    lon, lat = transform_coords(CRS, "EPSG:4326", list(xs), list(ys))
    return lon, lat


def cell_centres(t, cells):
    xs, ys = zip(*[t * (c + 0.5, r + 0.5) for r, c in cells])
    return lonlat(xs, ys)


def rgba_png(path, rgba, webp=True):
    """Overlay image; WebP keeps the regional rasters a few MB instead of tens."""
    path = Path(path)
    if not webp:
        Image.fromarray(rgba, "RGBA").save(path, optimize=True)
        return
    Image.fromarray(rgba, "RGBA").save(path.with_suffix(".webp"), "WEBP", quality=82, method=4)
    path.with_suffix(".png").unlink(missing_ok=True)


def burn_corrections(z, valid, t, log=print):
    """Stream burning: lower known channels (culverts under roads, drains the 5 m DEM smooths away)
    so routing follows them. Lines and burn depths come from terrain_corrections.geojson (WGS84)."""
    zr = z.copy()
    if not CORRECTIONS.exists():
        return zr
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["features"]
    h, w = z.shape
    inv = ~t
    for f in corr:
        lon, lat = zip(*f["geometry"]["coordinates"])
        xs, ys = transform_coords("EPSG:4326", CRS, list(lon), list(lat))
        # cells along the line in drawing order (lines are drawn downstream)
        cells = []
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / (t.a / 3)))
            for s in np.linspace(0, 1, n):
                c, r = inv * (x0 + (x1 - x0) * s, y0 + (y1 - y0) * s)
                rc = (int(r), int(c))
                if 0 <= rc[0] < h and 0 <= rc[1] < w and valid[rc] and (not cells or cells[-1] != rc):
                    cells.append(rc)
        # lower by burn_m and force a steady descent along the line: in flat floodplains a uniform
        # burn keeps the DEM's (wrong) direction, so each cell must sit below the one before it
        burn = f["properties"].get("burn_m", 2.0)
        prev = np.inf
        for rc in dict.fromkeys(cells):
            zr[rc] = min(zr[rc] - burn, prev - 0.002)
            prev = zr[rc]
    log(f"burned {len(corr)} correction line(s) from {CORRECTIONS.name} (descending in drawing order)")
    return zr


DETAIL_HALF =(500.0, 355.0)        # half width / height (m) of the detailed window around the landfill
DETAIL_FLOW_HA = 0.5                 # smallest D8 channel drawn as a line in the detailed window
# (smaller hillslope flow is shown as the MFD flow-intensity raster: D8 draws false parallel lines there)
DETAIL_CLASSES = [  # key, name, colour
    ("landfill", "매립지로 유입", "#ea580c"),
    ("west", "매립지 유출수와 같은 하천으로 (매립지 미경유)", "#0d9488"),
    ("other1", "", "#65a30d"),
    ("other2", "", "#2563eb"),
    ("other3", "", "#9333ea"),
]


def direction_name(dx, dy):
    ang = (np.degrees(np.arctan2(dx, dy)) + 360) % 360           # bearing from north
    names = ["북쪽", "북동쪽", "동쪽", "남동쪽", "남쪽", "남서쪽", "서쪽", "북서쪽"]
    return names[int(((ang + 22.5) % 360) // 45)]


def site_detail(z, valid, filled, code, acc, t, landfill):
    """Where rain falling around the landfill ends up: drainage classes and small flow lines."""
    h, w = z.shape
    cell = t.a
    term = hy.terminal_outlet(code).ravel()
    through = hy.upstream_mask(code, landfill).ravel()
    rr, cc = np.nonzero(landfill)
    cx, cy = t * (cc.mean() + 0.5, rr.mean() + 0.5)
    e0, e1 = cx - DETAIL_HALF[0], cx + DETAIL_HALF[0]
    n0, n1 = cy - DETAIL_HALF[1], cy + DETAIL_HALF[1]
    c0, c1 = int((e0 - t.c) / cell), int(np.ceil((e1 - t.c) / cell))
    r0, r1 = int((t.f - n1) / cell), int(np.ceil((t.f - n0) / cell))
    win = np.zeros((h, w), bool)
    win[r0:r1, c0:c1] = True
    win &= valid
    landfill_term = np.bincount(term[landfill.ravel()]).argmax()

    # classes: through the landfill / same outlet as the landfill / other outlets by size
    cls = np.full(h * w, -1, np.int16)
    wv = win.ravel()
    cls[wv & through] = 0
    cls[wv & ~through & (term == landfill_term)] = 1
    min_cells = 1e4 / (cell * cell)                   # ignore outlets draining < 1 ha of the window
    others = [tid for tid, n in sorted(zip(*np.unique(term[wv & (term != landfill_term)], return_counts=True)),
                                       key=lambda a: -a[1]) if n >= min_cells]
    classes = [dict(zip(("key", "name", "color"), DETAIL_CLASSES[0])),
               dict(zip(("key", "name", "color"), DETAIL_CLASSES[1]))]
    for k, tid in enumerate(others[:3]):
        r, c = divmod(tid, w)
        ox, oy = t * (c + 0.5, r + 0.5)
        key, _, colour = DETAIL_CLASSES[2 + k]
        classes.append({"key": key, "name": f"{direction_name(ox - cx, oy - cy)} 하천·계곡 방향", "color": colour})
        cls[wv & (term == tid)] = 2 + k
    cls = cls.reshape(h, w)
    for k, c in enumerate(classes):
        c["area_ha"] = round(float((cls == k).sum() * cell * cell / 1e4), 1)

    feats = []
    for geom, val in shapes(cls.astype(np.int16), mask=cls >= 0, transform=t):
        c = classes[int(val)]
        rings = [[[round(x, 6), round(y, 6)] for x, y in zip(*lonlat(*zip(*ring)))] for ring in geom["coordinates"]]
        feats.append({"type": "Feature", "properties": {"class": c["key"], "name": c["name"], "color": c["color"],
                                                        "area_ha": c["area_ha"]},
                      "geometry": {"type": "Polygon", "coordinates": rings}})
    (OUT / "detail_basins.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                                               encoding="utf-8")

    # small flow lines inside the window, coloured by class, width by contributing area
    small = (acc * cell * cell >= DETAIL_FLOW_HA * 1e4) & win
    order = hy.strahler_order(code, small, filled)
    lines = []
    for cells, o, a in hy.stream_segments(code, small, order, acc):
        if len(cells) < 2:
            continue
        cells = [rc for rc in cells if win[rc]] or cells
        k = int(cls[cells[0]])
        lon, lat = cell_centres(t, cells)
        lines.append({"type": "Feature", "properties": {"area_ha": round(a * cell * cell / 1e4, 2),
                                                        "color": classes[k]["color"] if k >= 0 else "#334155"},
                      "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)]
                                                                         for x, y in zip(lon, lat)]}})
    (OUT / "detail_flows.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": lines}),
                                              encoding="utf-8")
    # hillslope flow intensity (MFD), log-scaled, for the window only
    mfd = hy.flow_accumulation_mfd(filled, valid, cell)[r0:r1, c0:c1]
    la = np.log10(np.maximum(mfd * cell * cell, 1))
    v = np.clip((la - np.log10(500)) / (np.log10(2e4) - np.log10(500)), 0, 1)     # 0.05 ha .. 2 ha
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[..., 0] = (220 * (1 - v)).astype(np.uint8)
    rgba[..., 1] = (240 - 60 * v).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = (np.where(win[r0:r1, c0:c1], v, 0) ** 1.3 * 230).astype(np.uint8)
    rgba_png(OUT / "detail_mfd.png", rgba, webp=False)      # small; also embedded in the figure
    np.savez_compressed(OUT / "detail_grid.npz", cls=cls[r0:r1, c0:c1], acc=acc[r0:r1, c0:c1], mfd=mfd,
                        code=code[r0:r1, c0:c1], z=z[r0:r1, c0:c1],
                        transform=np.array((t * Affine_translation(c0, r0))[:6]))
    lon, lat = lonlat([e0, e1], [n0, n1])
    print("detail window:", ", ".join(f"{c['name']} {c['area_ha']} ha" for c in classes), f"; {len(lines)} flow lines")
    # exact window bounds (cell edges) for the MFD image overlay
    we0, wn1 = t * (c0, r0)
    we1, wn0 = t * (c1, r1)
    lon, lat = lonlat([we0, we1], [wn0, wn1])
    return {"bounds": [[min(lat), min(lon)], [max(lat), max(lon)]], "classes": classes, "flow_min_ha": DETAIL_FLOW_HA}


def Affine_translation(c, r):
    from affine import Affine
    return Affine.translation(c, r)


def find_dem() -> Path:
    """The DEM covering the largest area in DEM(5m)/ (a regional extract wins over the site one)."""
    def area(p):
        with rasterio.open(p) as ds:
            return ds.width * ds.height * abs(ds.res[0] * ds.res[1])
    return max((ROOT / "DEM(5m)").glob("*.tif"), key=area)


def ring_lonlat(ring, nd=5):
    lon, lat = lonlat(*zip(*ring))
    return [[round(x, nd), round(y, nd)] for x, y in zip(lon, lat)]


def write_subbasins(code, acc, valid, t, landfill, path_cells):
    """Sub-basins of every stream link draining >= SUBBASIN_KM2, flagged when the landfill or its
    surface outflow path lies in them."""
    from shapely.geometry import shape
    cell = t.a
    lab = hy.subbasins(code, acc, valid, SUBBASIN_KM2 * 1e6 / (cell * cell))
    ids, inv = np.unique(lab, return_inverse=True)
    lab32 = inv.reshape(lab.shape).astype(np.int32)
    keep = (ids >= 0)[lab32]
    area = dict(zip(ids.tolist(), (np.bincount(inv.ravel()) * cell * cell / 1e6).tolist()))
    on_path = {int(lab[r, c]) for r, c in path_cells} - {-1} if path_cells else set()
    in_landfill = set(np.unique(lab[landfill]).tolist()) - {-1} if landfill is not None else set()
    w = code.shape[1]
    feats = []
    for geom, v in shapes(lab32, mask=keep, transform=t):
        oid = int(ids[int(v)])
        g = shape(geom).simplify(cell * 1.2)
        if g.is_empty:
            continue
        r, c = divmod(oid, w)
        for poly in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
            feats.append({"type": "Feature",
                          "properties": {"id": oid, "area_km2": round(area[oid], 2),
                                         "outlet_km2": round(float(acc[r, c]) * cell * cell / 1e6, 2),
                                         "landfill": oid in in_landfill, "on_path": oid in on_path},
                          "geometry": {"type": "Polygon", "coordinates": [ring_lonlat(poly.exterior.coords)]
                                       + [ring_lonlat(i.coords) for i in poly.interiors]}})
    (OUT / "subbasins.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                                           encoding="utf-8")
    print(f"sub-basins (links >= {SUBBASIN_KM2} km2): {len(area) - (1 if -1 in area else 0)}, "
          f"on the landfill outflow path: {len(on_path)}")


def main():
    dem_path = find_dem()
    with rasterio.open(dem_path) as ds:
        z = ds.read(1).astype(np.float64)
        t = ds.transform
    cell = t.a
    valid = z > 0
    h, w = z.shape
    print(f"DEM {dem_path.name}: {w}x{h} cells, {cell} m, {valid.sum()} valid")

    zr = burn_corrections(z, valid, t)
    filled = hy.fill_depressions(zr, valid)
    code = hy.flow_directions(filled, valid, cell)
    acc = hy.flow_accumulation(filled, code, valid)
    zf = np.where(valid, z, np.nanmin(z[valid]))
    slope = hy.slope_radians(zf, cell)
    twi = hy.wetness_index(acc, slope, cell)
    depth = np.where(valid, filled - zr, 0)           # depression depth (closed pits)
    print(f"filled depressions: {(depth > 0.5).sum()} cells deeper than 0.5 m, max {depth.max():.1f} m")

    OUT.mkdir(exist_ok=True)
    # shaded relief
    hs = hy.hillshade(zf, cell)
    g = (hs * 255).astype(np.uint8)
    rgba_png(OUT / "hillshade.png", np.dstack([g, g, g, np.where(valid, 255, 0).astype(np.uint8)]))
    rgba_png(OUT / "hillshade_color.png", hy.color_relief(z, valid, cell, *COLOR_RANGE))
    elev_range = [round(float(v), 2) for v in np.percentile(z[valid], ELEV_PCT)]
    rgba_png(OUT / "hillshade_elev.png", hy.color_relief(z, valid, cell, *elev_range, ramp=hy.ELEV_RAMP))
    # flow accumulation: log area, transparent on slopes
    area = acc * cell * cell
    la = np.log10(np.maximum(area, 1))
    v = np.clip((la - 3.0) / (6.0 - 3.0), 0, 1)          # 0.1 ha .. 100 ha
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = (30 * (1 - v)).astype(np.uint8)
    rgba[..., 1] = (120 + 60 * v).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = (np.where(valid, v, 0) ** 0.8 * 235).astype(np.uint8)
    rgba_png(OUT / "flowacc.png", rgba)
    # wetness index: high values = converging, gentle ground
    tv = np.clip((twi - 6) / (14 - 6), 0, 1)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = (255 * (1 - tv)).astype(np.uint8)
    rgba[..., 1] = (200 * tv + 40).astype(np.uint8)
    rgba[..., 2] = (90 + 160 * tv).astype(np.uint8)
    rgba[..., 3] = (np.where(valid, tv, 0) * 200).astype(np.uint8)
    rgba_png(OUT / "twi.png", rgba)

    # streams
    streams = area >= STREAM_HA * 1e4
    order = hy.strahler_order(code, streams, filled)
    feats = []
    for cells, o, a in hy.stream_segments(code, streams, order, acc):
        if len(cells) < 2:
            continue
        lon, lat = cell_centres(t, cells)
        feats.append({"type": "Feature", "properties": {"order": o, "area_ha": round(a * cell * cell / 1e4, 1)},
                      "geometry": {"type": "LineString", "coordinates": [[round(x, 5), round(y, 5)] for x, y in zip(lon, lat)]}})
    (OUT / "streams.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    print(f"streams: {len(feats)} segments, max Strahler order {order.max()}")

    # landfill catchment and outflow paths
    site_feats, summary, mask, path = [], {}, None, None
    if FACILITIES.exists():
        fc = json.loads(FACILITIES.read_text(encoding="utf-8"))
        liner = next(f for f in fc["features"] if f["properties"]["kind"] == "hdpe_liner")
        mask = rasterize([(liner["geometry"], 1)], out_shape=(h, w), transform=t).astype(bool) & valid
        up = hy.upstream_mask(code, mask)
        catch_ha = up.sum() * cell * cell / 1e4
        own_ha = mask.sum() * cell * cell / 1e4
        for geom, val in shapes(up.astype(np.uint8), mask=up, transform=t):
            ring = geom["coordinates"][0]
            lon, lat = lonlat(*zip(*ring))
            site_feats.append({"type": "Feature", "properties": {"kind": "landfill_catchment",
                                                                 "area_ha": round(catch_ha, 1)},
                               "geometry": {"type": "Polygon", "coordinates": [[[round(x, 6), round(y, 6)]
                                                                                for x, y in zip(lon, lat)]]}})
        # downstream path from the landfill's outlet cell (largest accumulation inside the footprint)
        rr, cc = np.nonzero(mask)
        k = np.argmax(acc[rr, cc])
        path = hy.trace_downstream(code, rr[k], cc[k])
        lon, lat = cell_centres(t, path)
        length = sum(hy.DIST[code[r, c]] for r, c in path[:-1]) * cell
        site_feats.append({"type": "Feature", "properties": {"kind": "outflow_path", "length_m": round(length)},
                           "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)]
                                                                              for x, y in zip(lon, lat)]}})
        summary = {"landfill_ha": round(own_ha, 1), "catchment_ha": round(catch_ha, 1),
                   "outflow_path_m": round(length)}
        print(f"landfill {own_ha:.1f} ha, total catchment draining onto it {catch_ha:.1f} ha, "
              f"outflow path {length:.0f} m to the DEM edge")
    (OUT / "site.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": site_feats}),
                                      encoding="utf-8")

    write_subbasins(code, acc, valid, t, mask, path)
    detail = site_detail(z, valid, filled, code, acc, t, mask) if FACILITIES.exists() else None

    # flow directions for click-to-trace
    (OUT / "d8.bin").write_bytes(code.astype(np.uint8).tobytes())
    # smooth downslope azimuth for the flow-direction arrows
    (OUT / "flowdir.bin").write_bytes(hy.downslope_azimuth(filled, valid, cell, code, sigma=1.0).tobytes())
    xs = [t.c, t.c + w * cell, t.c + w * cell, t.c]
    ys = [t.f, t.f, t.f - h * cell, t.f - h * cell]
    lon, lat = lonlat(xs, ys)
    meta = {
        "dem": dem_path.name, "crs": CRS,
        "proj4": "+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 +ellps=GRS80 +units=m +no_defs",
        "width": w, "height": h, "cell": cell, "x0": t.c, "y0": t.f,
        "bounds": [[min(lat), min(lon)], [max(lat), max(lon)]],
        "stream_min_ha": STREAM_HA, "subbasin_km2": SUBBASIN_KM2, "site": summary, "detail": detail,
        "elev_range": elev_range,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    write_facilities_wgs84()
    print(f"Wrote {OUT}")


def write_facilities_wgs84():
    """Site facilities (walls, wells, sumps, HDPE outline) in WGS84 for the viewer's facility layer."""
    if not FACILITIES.exists():
        return
    fc = json.loads(FACILITIES.read_text(encoding="utf-8"))
    feats = []
    for f in fc["features"]:
        g, p = f["geometry"], dict(f["properties"])
        if f["properties"]["kind"] == "cutoff_wall" and "centerline" in p:
            g = {"type": "LineString", "coordinates": p.pop("centerline")}
        if g["type"] == "Point":
            lon, lat = lonlat([g["coordinates"][0]], [g["coordinates"][1]])
            g = {"type": "Point", "coordinates": [round(lon[0], 7), round(lat[0], 7)]}
        elif g["type"] == "LineString":
            g = {"type": "LineString", "coordinates": ring_lonlat(g["coordinates"], 7)}
        else:
            g = {"type": "Polygon", "coordinates": [ring_lonlat(r, 7) for r in g["coordinates"]]}
        feats.append({"type": "Feature", "properties": p, "geometry": g})
    (OUT / "facilities.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                       ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
