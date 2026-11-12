"""DJI R-JPEG thermal photos (H20T / H20N / M30T / M2EA ..., *_T.JPG) -> 32-bit float TIFFs of temperature in
degC, one per photo with the same name (.tif), through the DJI Thermal SDK's dji_irp.exe. The photo's metadata
goes along: EXIF (make, model, focal length, time), GPS (position, altitude) and DJI's XMP (RTK, gimbal / flight
attitude, laser range) -- so the TIFFs can be aligned in Metashape from scratch (positions and attitudes read as
from the JPGs), or swapped into a project already aligned on the photos (metashape_swap_to_temperature.py).
The TIFFs are uncompressed (~1.3 MB each): Pillow writes the EXIF / GPS sub-directories only without libtiff.

    .venv\\Scripts\\python.exe scripts\\dji_rjpeg_to_temperature.py <photo folder> <output folder>
        [--sdk C:\\Groundwater\\dji_thermal_sdk_v1.9_20260526] [--emissivity 0.95] [--humidity 70]
        [--ambient 12] [--reflection 12] [--distance auto]

--distance auto takes each photo's laser-rangefinder distance (XMP LRFTargetDistance), clipped to the SDK's
1-25 m range. The measurement parameters set the absolute temperature; differences between nearby surfaces
hardly depend on them. A summary (time, distance, min / median / max) goes to <output>/temperature_summary.csv.
"""
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin, TiffTags

W, H = 640, 512
EXIF_IFD, GPS_IFD, MAKER_NOTE, INTEROP_IFD, XMP_TAG = 0x8769, 0x8825, 0x927C, 0xA005, 700


def photo_metadata(jpg: Path):
    """TIFF tags carrying the JPG's EXIF (camera, focal length, time), GPS and XMP (DJI attitude, RTK)."""
    src = Image.open(jpg)
    ex = src.getexif()
    info = TiffImagePlugin.ImageFileDirectory_v2()
    for k in (271, 272, 305, 306, 315):                  # Make, Model, Software, DateTime, Artist
        if k in ex:
            info[k] = ex[k]
    exif = {k: v for k, v in ex.get_ifd(EXIF_IFD).items() if k not in (MAKER_NOTE, INTEROP_IFD)}
    if exif:
        info[EXIF_IFD] = exif
    gps = dict(ex.get_ifd(GPS_IFD))
    if gps:
        info[GPS_IFD] = gps
    xmp = src.info.get("xmp")
    if xmp:
        info[XMP_TAG] = xmp if isinstance(xmp, bytes) else xmp.encode("utf-8")
        info.tagtype[XMP_TAG] = TiffTags.BYTE
    return info


def xmp_value(data: bytes, key: str):
    m = re.search(rb'drone-dji:' + key.encode() + rb'="([^"]*)"', data)
    return m.group(1).decode() if m else None


def convert(jpg: Path, out: Path, exe: Path, args) -> dict:
    data = jpg.read_bytes()[:200_000]
    dist = args.distance
    if dist == "auto":
        lrf = xmp_value(data, "LRFTargetDistance")
        try:
            dist = float(lrf)
        except (TypeError, ValueError):
            dist = 25.0
    dist = min(max(float(dist), 1.0), 25.0)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "t.raw"
        cmd = [str(exe), "-s", str(jpg), "-a", "measure", "-o", str(raw), "--measurefmt", "float32",
               "--distance", f"{dist:.2f}", "--humidity", str(args.humidity), "--emissivity", str(args.emissivity),
               "--ambient", str(args.ambient), "--reflection", str(args.reflection)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(exe.parent))
        if r.returncode != 0 or not raw.exists():
            return {"file": jpg.name, "error": (r.stdout + r.stderr).strip()[-200:]}
        t = np.fromfile(raw, np.float32)
    if t.size != W * H:
        return {"file": jpg.name, "error": f"unexpected size {t.size}"}
    t = t.reshape(H, W)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(t, mode="F").save(out, tiffinfo=photo_metadata(jpg))
    stamp = re.search(r"_(\d{14})_", jpg.name)
    return {"file": jpg.name, "time": stamp.group(1) if stamp else "", "distance_m": round(dist, 2),
            "min_c": round(float(t.min()), 2), "median_c": round(float(np.median(t)), 2), "max_c": round(float(t.max()), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--sdk", default=str(Path(__file__).resolve().parents[1] / "dji_thermal_sdk_v1.9_20260526"))
    ap.add_argument("--emissivity", type=float, default=0.95)
    ap.add_argument("--humidity", type=float, default=70.0)
    ap.add_argument("--ambient", type=float, default=12.0)
    ap.add_argument("--reflection", type=float, default=12.0)
    ap.add_argument("--distance", default="auto")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    exe = Path(args.sdk) / "utility" / "bin" / "windows" / "release_x64" / "dji_irp.exe"
    if not exe.exists():
        sys.exit(f"dji_irp.exe not found: {exe}")
    src, out = Path(args.src), Path(args.out)
    jpgs = sorted(p for p in src.rglob("*_T.JPG") if out not in p.parents)
    print(f"{len(jpgs)} thermal photos -> {out}", flush=True)
    jobs = [(j, out / j.relative_to(src).with_suffix(".tif")) for j in jpgs]
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, row in enumerate(pool.map(lambda a: convert(a[0], a[1], exe, args), jobs), 1):
            rows.append(row)
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    bad = [r for r in rows if "error" in r]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "temperature_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "time", "distance_m", "min_c", "median_c", "max_c", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows) - len(bad)} converted, {len(bad)} failed"
          + (f" (first: {bad[0]['file']}: {bad[0]['error']})" if bad else ""))
    print(f"parameters: emissivity {args.emissivity}, humidity {args.humidity} %, ambient {args.ambient} C, "
          f"reflection {args.reflection} C, distance {args.distance}")


if __name__ == "__main__":
    main()
