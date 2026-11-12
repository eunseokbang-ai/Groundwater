"""Web tiles for very large georeferenced orthomosaics (e.g. a 13 GB, 2 cm drone mosaic) that do not
fit in memory: the deepest zoom is warped block by block in parallel worker processes straight from
the GeoTIFF (GDAL WarpedVRT, average resampling), and every shallower zoom is built by merging the
four child tiles, so the source is read only once.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

MERC = 20037508.342789244
TILE = 256
BLOCK = 8                    # tiles per block side at the deepest zoom (2048 px warps)


def tile_size(z):
    return 2 * MERC / 2 ** z


def deepest_zoom(gsd_m, lat_deg):
    """Shallowest Web Mercator zoom whose ground pixel is within 1.5x the GSD (capped at 22)."""
    ground = lambda z: tile_size(z) / TILE * math.cos(math.radians(lat_deg))
    return next((z for z in range(13, 23) if ground(z) <= gsd_m * 1.5), 22)


def _save(rgba, path, quality):
    a = rgba[..., 3]
    if a.max() == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if a.min() == 255:
        Image.fromarray(rgba[..., :3]).save(path, "WEBP", quality=quality, method=4)
    else:
        Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=quality, method=4)
    return True


def _warp_block(args):
    src_path, out_dir, z, bx, by, nx, ny, x0t, y0t, quality = args
    size = tile_size(z)
    res = size / TILE
    tx0, ty0 = x0t + bx * BLOCK, y0t + by * BLOCK
    w = min(BLOCK, nx - bx * BLOCK) * TILE
    h = min(BLOCK, ny - by * BLOCK) * TILE
    transform = Affine(res, 0, -MERC + tx0 * size, 0, -res, MERC - ty0 * size)
    with rasterio.open(src_path) as src, WarpedVRT(src, crs="EPSG:3857", transform=transform, width=w, height=h,
                                                     resampling=Resampling.average, warp_mem_limit=512) as vrt:
        data = vrt.read(indexes=[1, 2, 3, 4])
    rgba = np.moveaxis(data, 0, -1)
    n = 0
    for j in range(h // TILE):
        for i in range(w // TILE):
            t = rgba[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
            n += _save(np.ascontiguousarray(t), Path(out_dir) / str(z) / str(tx0 + i) / f"{ty0 + j}.webp", quality)
    return n


def _merge_parent(args):
    out_dir, z, x, y, quality = args
    canvas = np.zeros((2 * TILE, 2 * TILE, 4), np.uint8)
    found = False
    for dx in (0, 1):
        for dy in (0, 1):
            p = Path(out_dir) / str(z + 1) / str(2 * x + dx) / f"{2 * y + dy}.webp"
            if p.exists():
                canvas[dy * TILE:(dy + 1) * TILE, dx * TILE:(dx + 1) * TILE] = np.asarray(Image.open(p).convert("RGBA"))
                found = True
    if not found:
        return 0
    c = canvas.astype(np.float32).reshape(TILE, 2, TILE, 2, 4)
    a = c[..., 3]
    asum = a.sum(axis=(1, 3))
    rgb = (c[..., :3] * a[..., None]).sum(axis=(1, 3)) / np.maximum(asum, 1)[..., None]
    out = np.dstack([rgb, asum / 4]).clip(0, 255).astype(np.uint8)
    return int(_save(out, Path(out_dir) / str(z) / str(x) / f"{y}.webp", quality))


def read_tiles(tile_dir, bounds, width, height, crs="EPSG:5186", z=None):
    """RGBA image of `bounds` (x0, y0, x1, y1 in `crs`) at width x height px, assembled from a tile
    pyramid (the zoom closest to the requested resolution) — avoids re-reading a huge source."""
    from rasterio.warp import reproject, transform_bounds as tb
    tile_dir = Path(tile_dir)
    w_, s_, e_, n_ = tb(crs, "EPSG:4326", *bounds)
    res_m = (bounds[2] - bounds[0]) / width
    zooms = sorted(int(p.name) for p in tile_dir.iterdir() if p.name.isdigit())
    z = z or min(zooms, key=lambda k: abs(tile_size(k) / TILE * math.cos(math.radians((s_ + n_) / 2)) - res_m * 0.7))
    size = tile_size(z)
    mx = lambda lon: lon * MERC / 180
    my = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * MERC / math.pi
    x0, x1 = int((mx(w_) + MERC) // size), int((mx(e_) + MERC) // size)
    y0, y1 = int((MERC - my(n_)) // size), int((MERC - my(s_)) // size)
    mos = np.zeros(((y1 - y0 + 1) * TILE, (x1 - x0 + 1) * TILE, 4), np.uint8)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            p = tile_dir / str(z) / str(x) / f"{y}.webp"
            if p.exists():
                mos[(y - y0) * TILE:(y - y0 + 1) * TILE, (x - x0) * TILE:(x - x0 + 1) * TILE] = np.asarray(Image.open(p).convert("RGBA"))
    out = np.zeros((4, height, width), np.uint8)
    reproject(np.moveaxis(mos, -1, 0), out, src_transform=Affine(size / TILE, 0, -MERC + x0 * size, 0, -size / TILE, MERC - y0 * size),
              src_crs="EPSG:3857", dst_crs=crs, resampling=Resampling.bilinear,
              dst_transform=Affine((bounds[2] - bounds[0]) / width, 0, bounds[0], 0, -(bounds[3] - bounds[1]) / height, bounds[3]))
    return np.moveaxis(out, 0, -1)


def build_pyramid(src_path, out_dir, min_zoom=13, max_zoom=None, quality=80, workers=None, log=print):
    """Tiles for z in [min_zoom, max_zoom]; returns (max_zoom, bounds [[s, w], [n, e]], gsd)."""
    src_path, out_dir = Path(src_path), Path(out_dir)
    with rasterio.open(src_path) as src:
        gsd = abs(src.transform.a)
        w_, s_, e_, n_ = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    max_zoom = max_zoom or deepest_zoom(gsd, (s_ + n_) / 2)
    mx0, my0 = w_ * MERC / 180, math.log(math.tan(math.pi / 4 + math.radians(n_) / 2)) * MERC / math.pi
    mx1, my1 = e_ * MERC / 180, math.log(math.tan(math.pi / 4 + math.radians(s_) / 2)) * MERC / math.pi
    workers = workers or max(1, (os.cpu_count() or 4) - 2)

    size = tile_size(max_zoom)
    x0t, x1t = int((mx0 + MERC) // size), int((mx1 + MERC) // size)
    y0t, y1t = int((MERC - my0) // size), int((MERC - my1) // size)
    nx, ny = x1t - x0t + 1, y1t - y0t + 1
    jobs = [(str(src_path), str(out_dir), max_zoom, bx, by, nx, ny, x0t, y0t, quality)
            for by in range(math.ceil(ny / BLOCK)) for bx in range(math.ceil(nx / BLOCK))]
    log(f"    z{max_zoom}: {nx}x{ny} tiles in {len(jobs)} blocks, {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        total = 0
        for k, n in enumerate(pool.map(_warp_block, jobs, chunksize=1), 1):
            total += n
            if k % max(1, len(jobs) // 10) == 0:
                log(f"      {k}/{len(jobs)} blocks, {total} tiles", flush=True)
        for z in range(max_zoom - 1, min_zoom - 1, -1):
            xs = sorted({int(p.name) // 2 for p in (out_dir / str(z + 1)).iterdir()})
            ys = sorted({int(f.stem) // 2 for d in (out_dir / str(z + 1)).iterdir() for f in d.iterdir()})
            jobs2 = [(str(out_dir), z, x, y, quality) for x in xs for y in ys]
            n = sum(pool.map(_merge_parent, jobs2, chunksize=64))
            total += n
            log(f"    z{z}: {n} tiles", flush=True)
    log(f"    wrote {total} tiles", flush=True)
    return max_zoom, [[s_, w_], [n_, e_]], gsd
