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

Screen captures of older aerial photos named by year (`1973.png`, more of one year as `1973_2.png`)
are added for the years that have no orthophoto; they are georeferenced against the orthophotos by
captures.py (LoFTR matching, needs torch + kornia) or by a world file next to them (`1973.pgw`,
EPSG:5186). Of several captures of one year, the best-registered, widest one is shown.

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


def find_captures(folder: Path) -> list:
    """`1984.png`, `1984_2.png`, `1984_w.png` ...: aerial-photo captures, several allowed per year."""
    return [{"path": f, "id": f.stem, "year": int(f.stem[:4])} for f in sorted(folder.iterdir())
            if f.suffix.lower() in (".png", ".jpg", ".jpeg") and re.fullmatch(r"\d{4}(_[0-9A-Za-z]+)?", f.stem)]


def parse_stamps(s) -> dict:
    """'1984=123-456;1984_2=789-012' -> {'1984': '123-456', '1984_2': '789-012'}."""
    return dict(p.split("=", 1) for p in (s or "").split(";") if "=" in p)


def stamp(*paths) -> str:
    return "|".join(f"{p.stat().st_size}-{int(p.stat().st_mtime)}" for p in paths if p.exists())


def max_zoom_for(gsd: float, lat: float = 36.0) -> int:
    """Web Mercator zoom whose ground pixel size is closest to the image GSD."""
    return int(np.clip(round(math.log2(156543.03 * math.cos(math.radians(lat)) / gsd)), MIN_ZOOM, 19))


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


# --- aerial-photo captures ----------------------------------------------------------------------

def gray_reference(sheet, bounds, res=1.0):
    """An orthophoto resampled to `res` over `bounds`, as a captures.Reference."""
    import captures

    x0, y0, x1, y1 = bounds
    grid = Affine(res, 0, x0, 0, -res, y1)
    dst = np.zeros((3, int((y1 - y0) / res), int((x1 - x0) / res)), np.uint8)
    reproject(load_rgba(sheet["path"])[:3], dst, src_transform=sheet["transform"], src_crs=SHEET_CRS,
              dst_transform=grid, dst_crs=SHEET_CRS, resampling=Resampling.average)
    return captures.Reference(sheet["year"], cv2.cvtColor(np.moveaxis(dst, 0, -1), cv2.COLOR_RGB2GRAY), grid)


def site_bounds_tm(boundaries, sheets, buffer_m=300.0):
    if not boundaries:
        return sheet_bounds_tm(sheets[0]["sheet"])
    lon, lat = zip(*[c for f in boundaries for c in f["geometry"]["coordinates"][0]])
    xs, ys = transform_coords("EPSG:4326", SHEET_CRS, lon, lat)
    return min(xs) - buffer_m, min(ys) - buffer_m, max(xs) + buffer_m, max(ys) + buffer_m


def find_drone_orthos(folder: Path) -> list:
    """`YYYYMMDD*.tif` with its own georeferencing: drone orthomosaics (any size)."""
    out = []
    for f in sorted(folder.glob("*.tif")):
        m = re.match(r"(\d{4})(\d{2})(\d{2})", f.stem)
        if not m:
            continue
        with rasterio.open(f) as ds:
            if ds.crs is None or ds.transform.is_identity:
                print(f"{f.name}: no georeferencing, skipped")
                continue
        out.append({"path": f, "id": f.stem, "year": int(m.group(1)), "date": f"{m.group(1)}.{m.group(2)}.{m.group(3)}"})
    return out


def drone_shift_check(tile_dir: Path, sheets, z=19, n=4, win_tiles=4):
    """Offset (m) of the drone mosaic against the newest NGII orthophoto. Phase correlation of edge
    images in an n x n grid of ~240 m windows on the z19 (~0.25 m) tiles; the median of the windows
    that correlate well is reported. (One whole-mosaic comparison is dominated by relief
    displacement on the steep forested slopes of the NGII orthophoto and overstates the offset.)"""
    from large_ortho import MERC, TILE, tile_size
    zdir = tile_dir / str(z)
    if not zdir.exists() or not sheets:
        return None
    present = {(int(p.name), int(f.stem)) for p in zdir.iterdir() for f in p.iterdir()}
    xs, ys = [x for x, _ in present], [y for _, y in present]
    size = tile_size(z)
    ref = max(sheets, key=lambda s: s["year"])
    ref_rgba = load_rgba(ref["path"])[:3]
    k = size / TILE * math.cos(math.radians(36.06))

    def edges(gray):
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        return cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.5)

    shifts = []
    for fx in np.linspace(0.15, 0.85, n):
        for fy in np.linspace(0.15, 0.85, n):
            x0 = int(min(xs) + fx * (max(xs) - min(xs))) - win_tiles // 2
            y0 = int(min(ys) + fy * (max(ys) - min(ys))) - win_tiles // 2
            tiles = [(x, y) for x in range(x0, x0 + win_tiles) for y in range(y0, y0 + win_tiles)]
            if not all(t in present for t in tiles):
                continue
            m = np.zeros((win_tiles * TILE, win_tiles * TILE, 4), np.uint8)
            for x, y in tiles:
                m[(y - y0) * TILE:(y - y0 + 1) * TILE, (x - x0) * TILE:(x - x0 + 1) * TILE] = \
                    np.asarray(Image.open(zdir / str(x) / f"{y}.webp").convert("RGBA"))
            if (m[..., 3] < 255).mean() > 0.05:
                continue
            grid = Affine(size / TILE, 0, -MERC + x0 * size, 0, -size / TILE, MERC - y0 * size)
            r = np.zeros((3,) + m.shape[:2], np.uint8)
            reproject(ref_rgba, r, src_transform=ref["transform"], src_crs=SHEET_CRS, dst_transform=grid,
                      dst_crs="EPSG:3857", resampling=Resampling.bilinear)
            a = edges(cv2.cvtColor(np.moveaxis(r, 0, -1), cv2.COLOR_RGB2GRAY).astype(np.float32))
            b = edges(cv2.cvtColor(m[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32))
            (sx, sy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
            if resp >= 0.1:
                shifts.append((sx * k, -sy * k, resp))
    if not shifts:
        return None
    dx, dy, resp = np.median(np.array(shifts), axis=0)
    return {"reference_year": ref["year"], "dx_m": round(float(dx), 2), "dy_m": round(float(dy), 2),
            "response": round(float(resp), 3), "windows": len(shifts)}


def build_drone_orthos(src: Path, sheets, old_layers, force):
    """Tile georeferenced drone mosaics (YYYYMMDD*.tif) with the out-of-core pyramid builder."""
    from large_ortho import build_pyramid

    previous = {d.get("id"): d for d in old_layers if d.get("kind") == "drone"}
    layers = []
    for o in find_drone_orthos(src):
        s = stamp(o["path"])
        out = OUT_DIR / "tiles" / o["id"]
        prev = previous.get(o["id"])
        print(f"{o['id']}: {o['path'].name}  drone orthomosaic ({o['path'].stat().st_size / 1e9:.1f} GB)", flush=True)
        if prev and prev.get("source_stamp") == s and out.exists() and not force:
            print("    up to date, skipped")
            layers.append(prev)
            continue
        if out.exists():
            shutil.rmtree(out)
        max_zoom, bounds, gsd = build_pyramid(o["path"], out, min_zoom=MIN_ZOOM)
        shift = drone_shift_check(out, sheets)
        if shift:
            print(f"    shift vs NGII {shift['reference_year']}: dx {shift['dx_m']:+.2f} m, dy {shift['dy_m']:+.2f} m "
                  f"(response {shift['response']})", flush=True)
        layers.append({
            "year": o["year"], "id": o["id"], "label": o["date"], "kind": "drone", "file": o["path"].name,
            "gsd_m": round(gsd, 3), "georef": "embedded", "tiles": f"tiles/{o['id']}/{{z}}/{{x}}/{{y}}.webp",
            "min_zoom": MIN_ZOOM, "max_zoom": max_zoom, "bounds": bounds, "metadata": None,
            "shift_vs_reference": shift, "source_stamp": s,
        })
    return layers


CONFIDENCE_RANK = {"수동": 0, "높음": 1, "보통": 2, "낮음": 3}


def build_captures(src: Path, sheets, boundaries, previous, previous_skipped, force):
    """Tiles + layer entries for aerial-photo captures in years without an orthophoto.

    Every capture of a year (1984.png, 1984_2.png, ...) is registered; the year shows the one with
    the best registration confidence and, among equals, the one covering the most ground.
    Returns (layers, skipped, duplicate_years).
    """
    import captures
    import edits

    edit_cfg = edits.load_edits()
    found = find_captures(src)
    if not found:
        return [], [], []
    sheet_years = {s["year"] for s in sheets}
    duplicates = sorted({c["year"] for c in found if c["year"] in sheet_years})
    caps = [c for c in found if c["year"] not in sheet_years]
    if duplicates:
        print(f"Captures skipped because an orthophoto exists for the year: {duplicates}")
    years = sorted({c["year"] for c in caps})
    by_year = {y: [c for c in caps if c["year"] == y] for y in years}
    for c in caps:
        world = [c["path"].with_suffix(e) for e in (".pgw", ".pngw", ".jgw", ".wld") if c["path"].with_suffix(e).exists()]
        c["world"] = captures.read_world_file(c["path"])
        c["stamp"] = stamp(c["path"], *world)

    # A year keeps its previous result while the capture shown is unchanged and no capture of that
    # year is new or modified (captures removed since then do not matter).
    year_stamp = {y: ";".join(f"{c['id']}={c['stamp']}" for c in by_year[y]) for y in years}
    settled = {}
    for y in years:
        prev, skip = previous.get(y, {}), previous_skipped.get(y, {})
        current = {c["id"]: c["stamp"] for c in by_year[y]}
        before = parse_stamps(prev.get("source_stamp"))
        shown = Path(prev.get("file", "")).stem
        if (not force and prev.get("kind") == "capture" and (OUT_DIR / "tiles" / str(y)).exists()
                and shown in current and all(before.get(k) == v for k, v in current.items())):
            # same photos: keep the registration; only re-tile when its edit recipe changed
            same_edit = prev.get("edit_stamp", "") == edits.edit_stamp(shown, edit_cfg)
            settled[y] = "layer" if same_edit else "retile"
        elif not force and parse_stamps(skip.get("source_stamp")) == current:
            settled[y] = "skipped"
        else:
            settled[y] = None

    results, rgbs, masks, failure = {}, {}, {}, "정사영상과의 대응점이 부족해 자동 정합 실패"
    if not all(v in ("layer", "skipped") for v in settled.values()):
        rgbs = {c["id"]: captures.load_rgb(c["path"]) for c in caps}
        masks = {cid: captures.icon_mask(rgb) for cid, rgb in rgbs.items()}
        # Only captures of changed years are registered; the captures shown for settled years keep
        # their earlier registration and serve as fixed chaining partners.
        fixed = {Path(previous[y]["file"]).stem: Affine(*previous[y]["transform"])
                 for y in years if settled[y] in ("layer", "retile")}
        auto = [c["id"] for c in caps if c["world"] is None and settled[c["year"]] is None]
        if auto:
            print(f"Registering aerial-photo captures {auto} against the orthophotos "
                  f"(kept as reference: {sorted(fixed)})...", flush=True)
            ordered = sorted(sheets, key=lambda s: s["year"])
            first = ordered[0]
            second = next((s for s in ordered if s["year"] >= first["year"] + 3), None)
            try:
                refs = [gray_reference(s, sheet_bounds_tm(s["sheet"])) for s in (first, second) if s]
                ids = auto + sorted(fixed)
                gray = {cid: cv2.cvtColor(captures.clean_capture(rgbs[cid], masks[cid]), cv2.COLOR_RGB2GRAY)
                        for cid in ids}
                results = captures.register(gray, {cid: masks[cid] for cid in ids}, refs,
                                            site_bounds_tm(boundaries, sheets), fixed=fixed, log=print)
            except ImportError as e:
                failure = f"자동 정합 모듈 없음 ({e.name}); pip install torch kornia"
                print(f"    {failure}")

    def display_image(cid):
        """The capture as shown on the map: icons removed, then any edits.json recipe applied."""
        rgb = captures.clean_capture(rgbs[cid], masks[cid])
        return edits.apply_edits(cid, rgb, masks[cid], edit_cfg, captures.clean_capture)

    def edit_fields(cid):
        return {"edit_stamp": edits.edit_stamp(cid, edit_cfg),
                "edits": [op.get("note", op["op"]) for op in edit_cfg.get(cid, [])]}

    layers, skipped = [], []
    for y in years:
        if settled[y] == "retile":
            prev = previous[y]
            cid = Path(prev["file"]).stem
            rgb = display_image(cid)
            h, w = rgb.shape[:2]
            out = OUT_DIR / "tiles" / str(y)
            shutil.rmtree(out)
            print(f"{y}: {prev['file']}  edit recipe changed, re-tiling with the kept registration", flush=True)
            n = write_tiles(np.concatenate([np.moveaxis(rgb, -1, 0), np.full((1, h, w), 255, np.uint8)]),
                            Affine(*prev["transform"]), out, range(MIN_ZOOM, prev["max_zoom"] + 1))
            print(f"    wrote {n} tiles", flush=True)
            present = {c["path"].name for c in by_year[y]}
            layers.append({**prev, **edit_fields(cid), "source_stamp": year_stamp[y],
                           "alternates": [a for a in prev.get("alternates", []) if a["file"] in present]})
            continue
        if settled[y] == "layer":
            print(f"{y}: {previous[y]['file']}  capture up to date, kept")
            present = {c["path"].name for c in by_year[y]}
            layers.append({**previous[y], "source_stamp": year_stamp[y],
                           "alternates": [a for a in previous[y].get("alternates", []) if a["file"] in present]})
            continue
        if settled[y] == "skipped":
            skipped.append(previous_skipped[y])
            continue
        options = []
        for c in by_year[y]:
            if c["world"] is not None:
                reg = {"transform": c["world"], "method": "월드파일 (수동 정합)", "confidence": "수동",
                       "model": "affine", "inliers": None, "rms_m": None}
            else:
                reg = results.get(c["id"])
            if reg is None:
                continue
            h, w = rgbs[c["id"]].shape[:2]
            t = reg["transform"]
            gsd = math.sqrt(abs(t.a * t.e - t.b * t.d))
            options.append({"capture": c, "reg": reg, "gsd": gsd, "area": w * h * gsd * gsd})
        placed = {o["capture"]["id"] for o in options}

        if not options:
            photos = []
            for c in by_year[y]:  # not on the map, but still viewable as plain photos
                photo = OUT_DIR / "photos" / f"{c['id']}.jpg"
                photo.parent.mkdir(exist_ok=True)
                Image.fromarray(captures.clean_capture(rgbs[c["id"]], masks[c["id"]])).save(photo, quality=90)
                photos.append(f"photos/{c['id']}.jpg")
            skipped.append({"year": y, "files": [c["path"].name for c in by_year[y]], "reason": failure,
                            "source_stamp": year_stamp[y], "photos": photos})
            print(f"{y}: not placed: {failure}")
            continue

        options.sort(key=lambda o: (CONFIDENCE_RANK[o["reg"]["confidence"]], -o["area"]))
        best = options[0]
        c, reg, gsd, t = best["capture"], best["reg"], best["gsd"], best["reg"]["transform"]
        alternates = [{"file": o["capture"]["path"].name, "confidence": o["reg"]["confidence"],
                       "area_km2": round(o["area"] / 1e6, 2)} for o in options[1:]]
        alternates += [{"file": x["path"].name, "confidence": None, "area_km2": None}
                       for x in by_year[y] if x["id"] not in placed]
        rgb = display_image(c["id"])
        h, w = rgb.shape[:2]
        rgba = np.concatenate([np.moveaxis(rgb, -1, 0), np.full((1, h, w), 255, np.uint8)])
        max_zoom = max_zoom_for(gsd)
        out = OUT_DIR / "tiles" / str(y)
        if out.exists():
            shutil.rmtree(out)
        print(f"{y}: {c['path'].name}  {w}x{h} px  ~{gsd:.2f} m/px  {best['area'] / 1e6:.2f} km2  {reg['method']}"
              + (f"  (chosen over {[a['file'] for a in alternates]})" if alternates else ""), flush=True)
        n = write_tiles(rgba, t, out, range(MIN_ZOOM, max_zoom + 1))
        print(f"    wrote {n} tiles", flush=True)
        xs, ys = zip(*[t * p for p in [(0, 0), (w, 0), (w, h), (0, h)]])
        lon, lat = transform_coords(SHEET_CRS, "EPSG:4326", xs, ys)
        layers.append({
            "year": y,
            "kind": "capture",
            "file": c["path"].name,
            "gsd_m": round(gsd, 2),
            "area_km2": round(best["area"] / 1e6, 2),
            "georef": "worldfile" if reg["confidence"] == "수동" else "registered",
            "tiles": f"tiles/{y}/{{z}}/{{x}}/{{y}}.webp",
            "min_zoom": MIN_ZOOM,
            "max_zoom": max_zoom,
            "bounds": [[min(lat), min(lon)], [max(lat), max(lon)]],
            "metadata": None,
            "registration": {
                "method": reg["method"],
                "confidence": reg["confidence"],
                "model": reg["model"],
                "inliers": reg["inliers"],
                "rms_m": None if reg["rms_m"] is None else round(reg["rms_m"], 1),
                "uncertainty_m": None if reg.get("uncertainty_m") is None else round(reg["uncertainty_m"]),
            },
            "alternates": alternates,
            "transform": [round(v, 6) for v in t[:6]],
            "source_stamp": year_stamp[y],
            **edit_fields(c["id"]),
        })
    return layers, skipped, duplicates


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
    previous, previous_skipped = {}, {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = {d["year"]: d for d in old["layers"]}
        previous_skipped = {d["year"]: d for d in old.get("skipped_captures", [])}
    for s in sheets:
        s["stamp"] = stamp(s["path"])
        prev = previous.get(s["year"], {})
        s["todo"] = (args.force or not (OUT_DIR / "tiles" / str(s["year"])).exists() or prev.get("kind") == "capture"
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
        max_zoom = max_zoom_for(gsd)
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
            "kind": "ortho",
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

    capture_layers, skipped, duplicates = build_captures(src, sheets, boundaries, previous, previous_skipped,
                                                         args.force)
    layers += capture_layers
    old_layers = json.loads(manifest_path.read_text(encoding="utf-8"))["layers"] if manifest_path.exists() else []
    layers += build_drone_orthos(src, sheets, old_layers, args.force)

    # Drop tiles and photos of images that are no longer in the source folder.
    keep_tiles = {d["tiles"].split("/")[1] for d in layers}
    for d in (OUT_DIR / "tiles").glob("*"):
        if d.is_dir() and d.name not in keep_tiles:
            shutil.rmtree(d)
            print(f"Removed tiles of {d.name} (no longer in {src.name})")
    keep_photos = {Path(p).name for s in skipped for p in s.get("photos", [])}
    for f in (OUT_DIR / "photos").glob("*"):
        if f.name not in keep_photos:
            f.unlink()

    unmatched = sorted(set(meta) - {s["year"] for s in sheets})
    manifest = {
        "title": "익산 폐석산 정사영상 변천",
        "source": "국토지리정보원 정사영상 (1:5,000 도엽)·항공사진",
        "reference_year": ref["year"] if ref else None,
        "layers": sorted(layers, key=lambda d: (d["year"], d.get("kind") == "drone", d.get("id", ""))),
        "skipped_captures": skipped,
        "duplicate_capture_years": duplicates,
        "unmatched_metadata_years": unmatched,
        "boundaries": {"type": "FeatureCollection", "features": boundaries},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    if unmatched:
        print(f"Metadata XML without a matching TIF year: {unmatched}")
    if skipped:
        print(f"Captures not placed: {[d['year'] for d in skipped]} (add a .pgw world file to place by hand)")


if __name__ == "__main__":
    main()
