"""Run with Agisoft Metashape Professional on a project whose photos are temperature TIFFs (degC, from
dji_rjpeg_to_temperature.py) and which already has a tiled model: exports the tiled model with its textures
coloured through the chunk's raster transform, so the 8-bit textures keep the temperature contrast.

With the default raster transform (formula B1, range (1, 1)) every texel saturates to white. The textures of a
tiled model built from float photos do NOT hold degC: Iksan 2026-10 (photos 7.5-13.7 degC) gave texel values
~1.45-2.9 -- linear in the temperature, scale undocumented -- so use these exports for looking at relative
temperature, and the temperature orthomosaic (metashape_swap_to_temperature.py) for degC. Found by probing: the
palette stops set the colours, and the range must stay wide (1-5 worked, 1.4-3.2 saturated everything).
Set in memory only (the project is opened read-only and never saved):

    metashape.exe -r metashape_export_tiled_temperature.py <project.psx> <output.3tz> <low> <high> [inferno|gray]

low / high: texel values at the two ends of the colour ramp (Iksan 2026-10: 1.4 and 3.0).
"""
import json
import os
import sys

import Metashape

INFERNO = [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66), (245, 125, 21), (250, 193, 39),
           (252, 255, 164)]

path, out, low, high = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
mode = sys.argv[5] if len(sys.argv) > 5 else "inferno"
doc = Metashape.Document()
doc.open(path, read_only=True, ignore_lock=True)
chunk = doc.chunk or doc.chunks[0]
if chunk.tiled_model is None:
    raise SystemExit("no tiled model in the chunk")

stops = [(0, 0, 0), (255, 255, 255)] if mode == "gray" else INFERNO
rt = chunk.raster_transform
rt.formula = ["B1"]
rt.range = (min(1.0, low), max(5.0, high))         # wide: a tight range saturated everything
rt.palette = {low + (high - low) * k / (len(stops) - 1): stops[k] for k in range(len(stops))}
rt.interpolation = True
rt.enabled = True
chunk.raster_transform = rt

fmt = None
for name in ("TiledModelFormat3DTiles", "TiledModelFormatCesium"):
    fmt = getattr(Metashape.TiledModelFormat, name, None) or getattr(Metashape, name, None)
    if fmt is not None:
        break
kw = {"path": out, "raster_transform": Metashape.RasterTransformPalette}
if fmt is not None:
    kw["format"] = fmt
chunk.exportTiledModel(**kw)
with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
    json.dump({"source_project": path, "mapping": mode, "low_texel": low, "high_texel": high,
               "note": "relative temperature: texel values of the tiled model, linear in degC but not degC"},
              f, indent=1)
print("exported", out, mode, low, high)
