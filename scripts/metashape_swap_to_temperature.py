"""Run inside Agisoft Metashape Professional (Tools > Run Script...) in the thermal project, after
dji_rjpeg_to_temperature.py has turned the R-JPEG photos into temperature TIFFs (degC, 32-bit float, same names).

Copies the active chunk (the original stays as it is), points the copy's cameras at the temperature TIFFs -- the
alignment and the model are kept, the photos only change their values -- builds an orthomosaic on the SAME
projection as the chunk's existing orthomosaic (e.g. the wall plane from metashape_build_planar_like.py or a
Current view), and exports it as a single-band 32-bit float GeoTIFF of temperature next to the project:
<project>_temperature_ortho.tif. Put that file in the same folder as the thermal DEM / palette orthomosaic
(planar_thermal): the groundwater dashboard then reads degC instead of palette colours.

Without the GUI (the project is opened read-only and never saved -- the original stays as it is):

    metashape.exe -r metashape_swap_to_temperature.py <project.psx> <temperature TIFF folder> [<output .tif>]
"""
import os
import sys

import Metashape


def _const(*names):
    for n in names:
        for where in (Metashape, getattr(Metashape, "DataSource", None), getattr(Metashape, "ImageFormat", None),
                      getattr(Metashape, "RasterTransformType", None), getattr(Metashape, "BlendingMode", None)):
            if where is not None and hasattr(where, n):
                return getattr(where, n)
    raise AttributeError(names)


cli = len(sys.argv) >= 3
if cli:                                                   # metashape.exe -r script <project> <tif folder> [<out>]
    doc = Metashape.Document()
    doc.open(sys.argv[1], read_only=True, ignore_lock=True)
    folder = sys.argv[2]
else:
    doc = Metashape.app.document
    folder = None
src = doc.chunk or doc.chunks[0]
if src.orthomosaic is None:
    raise SystemExit("the chunk has no orthomosaic: build the palette one on the wall plane first (its projection is reused)")
proj = src.orthomosaic.projection
res = src.orthomosaic.resolution

if not cli:
    folder = Metashape.app.getExistingDirectory("온도 TIFF 폴더 (dji_rjpeg_to_temperature.py 출력)")
if not folder:
    raise SystemExit("cancelled")
tifs = {}
for root, _, files in os.walk(folder):
    for f in files:
        if f.lower().endswith(".tif"):
            tifs[os.path.splitext(f)[0].lower()] = os.path.join(root, f)

chunk = src.copy()
chunk.label = (src.label or "Chunk") + " (temperature)"
swapped = missing = 0
for cam in chunk.cameras:
    if cam.photo is None:
        continue
    stem = os.path.splitext(os.path.basename(cam.photo.path))[0].lower()
    if stem in tifs:
        cam.photo.path = tifs[stem]
        swapped += 1
    else:
        missing += 1
print(f"cameras pointed at temperature TIFFs: {swapped}, without one: {missing}")
if not swapped:
    raise SystemExit("no matching TIFF names")

surface = _const("ModelData") if chunk.model is not None else _const("ElevationData")
kw = {"surface_data": surface, "projection": proj}
try:
    kw["blending_mode"] = _const("MosaicBlending")
except AttributeError:
    pass
if res:
    kw["resolution"] = res
chunk.buildOrthomosaic(**kw)
print("temperature orthomosaic built on the existing projection")

if cli and len(sys.argv) >= 4:
    out = sys.argv[3]
else:
    out = (os.path.splitext(doc.path)[0] + "_temperature_ortho.tif" if doc.path
           else os.path.join(folder, "temperature_ortho.tif"))
ekw = {"path": out, "source_data": _const("OrthomosaicData"), "projection": proj}
try:
    ekw["raster_transform"] = _const("RasterTransformNone")
except AttributeError:
    pass
try:
    ekw["image_format"] = _const("ImageFormatTIFF")
except AttributeError:
    pass
chunk.exportRaster(**ekw)
if not cli:
    doc.save()
print("exported:", out)
