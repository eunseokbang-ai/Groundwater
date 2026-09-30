"""Build web-map tiles for the historical orthophoto viewer (webapp/ortho_history/index.html).

Scans a folder of NGII (국토지리정보원) 1:5,000 orthophoto sheets, e.g.
    (B060)정사영상_2025_36713072.tif  +  (B060)정사영상메타데이터_202512000336713072.xml
plus any *.kmz site boundaries, and writes

    webapp/ortho_history/tiles/<year>/{z}/{x}/{y}.webp   (XYZ / Web Mercator)
    webapp/ortho_history/layers.json                     (years, metadata, boundaries)

Most downloaded sheets carry no GeoTIFF georeferencing. NGII sheets cover the 1:5,000 map-sheet
extent (in EPSG:5186, Korea 2000 / Central Belt) plus a 50 m margin, so an image without a
transform is centred on that extent at its nominal GSD. A sheet that does carry a transform (the
2012 one here) is used as-is and serves as the reference for a phase-correlation alignment check.

Usage:
    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_tiles.py [--src 과거정사영상] [--force]
"""
import argparse
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.warp import Resampling, reproject, transform as transform_coords

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SHEET_CRS = "EPSG:5186"
SHEET_MARGIN_M = 50.0
MERC_HALF = 20037508.342789244
TILE = 256
MIN_ZOOM = 13
META_KEYS = ["촬영년도", "제작일자", "사업지구명", "입력회사", "지상표본거리", "사용수치표고", "해당도엽명", "행정구역명"]


# --- map sheet geometry ------------------------------------------------------------------------

def sheet_lonlat(sheet_no: str):
    """(west, south, east, north) of a Korean 1:5,000 map sheet, e.g. '36713072'.

    1:50,000 sheet = 2-digit latitude, last digit of longitude, 15' cell 01-16 (row-major from NW);
    1:5,000 sheet = 3-digit 1'30" cell 001-100 within it (row-major from NW).
    """
    lat0, lon0, cell50, cell5 = int(sheet_no[:2]), 120 + int(sheet_no[2]), int(sheet_no[3:5]), int(sheet_no[5:8])
    r50, c50 = divmod(cell50 - 1, 4)
    r5, c5 = divmod(cell5 - 1, 10)
    north = lat0 + 1 - r50 * 0.25 - r5 * 0.025
    west = lon0 + c50 * 0.25 + c5 * 0.025
    return west, north - 0.025, west + 0.025, north


def sheet_bounds_tm(sheet_no: str):
    w, s, e, n = sheet_lonlat(sheet_no)
    xs, ys = transform_coords("EPSG:4326", SHEET_CRS, [w, e, e, w], [n, n, s, s])
    return min(xs), min(ys), max(xs), max(ys)


def georeference(ds, sheet_no: str, nominal_gsd: float | None) -> Affine:
    t = ds.transform
    if not t.is_identity and abs(t.a) > 0.01 and abs(t.a) < 10:
        return t
    x0, y0, x1, y1 = sheet_bounds_tm(sheet_no)
    width_m = x1 - x0 + 2 * SHEET_MARGIN_M
    gsd = nominal_gsd or round(width_m / ds.width, 2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return Affine(gsd, 0, cx - ds.width * gsd / 2, 0, -gsd, cy + ds.height * gsd / 2)


# --- inputs -------------------------------------------------------------------------------------

def read_metadata(folder: Path) -> dict:
    """{촬영년도: {key: value}} from the NGII metadata XMLs."""
    out = {}
    for f in folder.glob("*.xml"):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        meta = {k: (root.findtext(f".//{k}") or "").strip() for k in META_KEYS}
        if meta["촬영년도"].isdigit():
            meta["파일"] = f.name
            out[int(meta["촬영년도"])] = meta
    return out


def read_boundaries(folder: Path) -> list:
    features = []
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    for f in sorted(folder.glob("*.kmz")) + sorted(folder.glob("*.kml")):
        kml = zipfile.ZipFile(f).read("doc.kml") if f.suffix == ".kmz" else f.read_bytes()
        for pm in ET.fromstring(kml).iter("{http://www.opengis.net/kml/2.2}Placemark"):
            for coords in pm.iterfind(".//k:coordinates", ns):
                ring = [[float(v) for v in p.split(",")[:2]] for p in coords.text.split()]
                features.append({
                    "type": "Feature",
                    "properties": {"name": pm.findtext("k:name", default=f.stem, namespaces=ns), "file": f.name},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
    return features


def find_sheets(folder: Path) -> list:
    sheets = []
    for f in sorted(folder.glob("*.tif")):
        m = re.search(r"_(\d{4})_(\d{8})", f.name)
        if m:
            sheets.append({"path": f, "year": int(m.group(1)), "sheet": m.group(2)})
    return sheets


# --- tiling -------------------------------------------------------------------------------------

def merc_x(lon):
    return lon * MERC_HALF / 180


def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * MERC_HALF / math.pi


def tile_range(bounds_merc, z):
    size = 2 * MERC_HALF / 2 ** z
    x0, y0, x1, y1 = bounds_merc
    return (int((x0 + MERC_HALF) // size), int((MERC_HALF - y1) // size),
            int((x1 + MERC_HALF) // size), int((MERC_HALF - y0) // size))


def load_rgba(path: Path):
    with rasterio.open(path) as ds:
        rgb = ds.read([1, 2, 3])
    alpha = np.full((1,) + rgb.shape[1:], 255, np.uint8)
    return np.concatenate([rgb, alpha])


def write_tiles(rgba, src_transform, out_dir: Path, zooms, quality=82) -> int:
    h, w = rgba.shape[1:]
    corners_x, corners_y = zip(*[src_transform * p for p in [(0, 0), (w, 0), (w, h), (0, h)]])
    lon, lat = transform_coords(SHEET_CRS, "EPSG:4326", corners_x, corners_y)
    bounds_merc = (merc_x(min(lon)), merc_y(min(lat)), merc_x(max(lon)), merc_y(max(lat)))
    count = 0
    for z in zooms:
        tx0, ty0, tx1, ty1 = tile_range(bounds_merc, z)
        size = 2 * MERC_HALF / 2 ** z
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        dst = np.zeros((4, ny * TILE, nx * TILE), np.uint8)
        dst_transform = Affine(size / TILE, 0, -MERC_HALF + tx0 * size, 0, -size / TILE, MERC_HALF - ty0 * size)
        src_px = abs(src_transform.a)
        resampling = Resampling.average if size / TILE > 1.5 * src_px else Resampling.bilinear
        reproject(rgba, dst, src_transform=src_transform, src_crs=SHEET_CRS, dst_transform=dst_transform,
                  dst_crs="EPSG:3857", resampling=resampling, num_threads=4)
        for j in range(ny):
            for i in range(nx):
                tile = dst[:, j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
                a = tile[3]
                if a.max() == 0:
                    continue
                path = out_dir / str(z) / str(tx0 + i) / f"{ty0 + j}.webp"
                path.parent.mkdir(parents=True, exist_ok=True)
                if a.min() == 255:
                    Image.fromarray(np.moveaxis(tile[:3], 0, -1)).save(path, "WEBP", quality=quality, method=4)
                else:
                    Image.fromarray(np.moveaxis(tile, 0, -1), "RGBA").save(path, "WEBP", quality=quality, method=4)
                count += 1
        print(f"    z{z}: {nx}x{ny} grid", flush=True)
    return count


# --- alignment check ----------------------------------------------------------------------------

def edge_image(rgba, src_transform, bounds, res=1.0):
    x0, y0, x1, y1 = bounds
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    dst = np.zeros((3, h, w), np.uint8)
    reproject(rgba[:3], dst, src_transform=src_transform, src_crs=SHEET_CRS,
              dst_transform=Affine(res, 0, x0, 0, -res, y1), dst_crs=SHEET_CRS, resampling=Resampling.average)
    gray = cv2.cvtColor(np.moveaxis(dst, 0, -1), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)


def shift_vs_reference(ref_edges, rgba, src_transform, bounds, res=1.0):
    """(dx, dy) in metres (east, north) of this image relative to the reference, and peak response."""
    edges = edge_image(rgba, src_transform, bounds, res)
    win = cv2.createHanningWindow(edges.shape[::-1], cv2.CV_32F)
    (sx, sy), response = cv2.phaseCorrelate(ref_edges, edges, win)
    return sx * res, -sy * res, response


# --- main ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "과거정사영상"), help="folder with orthophoto TIFs, XMLs, KMZs")
    ap.add_argument("--force", action="store_true", help="rebuild tiles even for sheets that are up to date")
    ap.add_argument("--no-check", action="store_true", help="skip the alignment check against the reference")
    args = ap.parse_args()

    src = Path(args.src)
    sheets = find_sheets(src)
    if not sheets:
        sys.exit(f"No '*_YYYY_<sheet>.tif' orthophotos in {src}")
    meta = read_metadata(src)
    boundaries = read_boundaries(src)

    for s in sheets:
        with rasterio.open(s["path"]) as ds:
            m = meta.get(s["year"])
            nominal = float(m["지상표본거리"]) if m and m["지상표본거리"] else None
            s["embedded"] = not ds.transform.is_identity
            s["transform"] = georeference(ds, s["sheet"], nominal)
            s["size"] = (ds.width, ds.height)

    # Tiles are rebuilt only for sheets that are new or whose source file changed since the last run.
    manifest_path = OUT_DIR / "layers.json"
    previous = {}
    if manifest_path.exists():
        previous = {d["year"]: d for d in json.loads(manifest_path.read_text(encoding="utf-8"))["layers"]}
    for s in sheets:
        st = s["path"].stat()
        s["stamp"] = f"{st.st_size}-{int(st.st_mtime)}"
        prev = previous.get(s["year"], {})
        s["todo"] = (args.force or not (OUT_DIR / "tiles" / str(s["year"])).exists()
                     or prev.get("file") != s["path"].name or prev.get("source_stamp", s["stamp"]) != s["stamp"])

    ref = next((s for s in sheets if s["embedded"]), None)
    check_bounds = None
    if ref and not args.no_check and any(s["todo"] and s is not ref for s in sheets):
        x0, y0, x1, y1 = sheet_bounds_tm(ref["sheet"])
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 900
        check_bounds = (cx - half, cy - half, cx + half, cy + half)
        ref_edges = edge_image(load_rgba(ref["path"]), ref["transform"], check_bounds)
        print(f"Alignment reference: {ref['path'].name} (embedded georeferencing)")

    layers = []
    for s in sheets:
        year, t = s["year"], s["transform"]
        gsd = abs(t.a)
        max_zoom = 19 if gsd < 0.35 else 18
        out = OUT_DIR / "tiles" / str(year)
        print(f"{year}: {s['path'].name}  {s['size'][0]}x{s['size'][1]} px  GSD {gsd:.3f} m  "
              f"{'embedded' if s['embedded'] else 'from sheet ' + s['sheet']}", flush=True)
        shift = previous.get(year, {}).get("shift_vs_reference")
        if not s["todo"]:
            print("    up to date, skipped (use --force to rebuild)")
        else:
            rgba = load_rgba(s["path"])
            shift = None
            if check_bounds and s is not ref:
                dx, dy, resp = shift_vs_reference(ref_edges, rgba, t, check_bounds)
                shift = {"dx_m": round(dx, 2), "dy_m": round(dy, 2), "response": round(resp, 3)}
                print(f"    shift vs reference: dx {dx:+.2f} m, dy {dy:+.2f} m (response {resp:.3f})", flush=True)
            if out.exists():
                shutil.rmtree(out)
            n = write_tiles(rgba, t, out, range(MIN_ZOOM, max_zoom + 1))
            print(f"    wrote {n} tiles", flush=True)
            del rgba

        w, h = s["size"]
        xs, ys = zip(*[t * p for p in [(0, 0), (w, 0), (w, h), (0, h)]])
        lon, lat = transform_coords(SHEET_CRS, "EPSG:4326", xs, ys)
        m = meta.get(year)
        layers.append({
            "year": year,
            "file": s["path"].name,
            "sheet": s["sheet"],
            "gsd_m": round(gsd, 3),
            "georef": "embedded" if s["embedded"] else "sheet",
            "tiles": f"tiles/{year}/{{z}}/{{x}}/{{y}}.webp",
            "min_zoom": MIN_ZOOM,
            "max_zoom": max_zoom,
            "bounds": [[min(lat), min(lon)], [max(lat), max(lon)]],
            "metadata": m,
            "shift_vs_reference": shift,
            "source_stamp": s["stamp"],
        })

    unmatched = sorted(set(meta) - {s["year"] for s in sheets})
    manifest = {
        "title": "익산 폐석산 정사영상 변천",
        "source": "국토지리정보원 정사영상 (1:5,000 도엽)",
        "reference_year": ref["year"] if ref else None,
        "layers": sorted(layers, key=lambda d: d["year"]),
        "unmatched_metadata_years": unmatched,
        "boundaries": {"type": "FeatureCollection", "features": boundaries},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    if unmatched:
        print(f"Metadata XML without a matching TIF year: {unmatched}")


if __name__ == "__main__":
    main()
