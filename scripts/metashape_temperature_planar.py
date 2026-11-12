"""Run with Agisoft Metashape Professional on a project aligned on temperature TIFFs (dji_rjpeg_to_temperature.py):
builds a DEM and a temperature orthomosaic (degC, 32-bit float) on the plane of another project's planar products
(its *_planar_projection.json, e.g. the RGB wall plane) and exports both as GeoTIFFs.

The project is first saved under a new name in <work folder> and processed there, so the original project and
its data stay untouched:

    metashape.exe -r metashape_temperature_planar.py <project.psx> <plane json> <work folder> <out folder> [<res m>]

Writes <out folder>/<name>_temperature_dem.tif and <name>_temperature_ortho.tif (band 1 degC, band 2 coverage).
"""
import json
import os
import sys

import Metashape

src, plane_json, work, out = sys.argv[1:5]
res = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
name = os.path.splitext(os.path.basename(src))[0]

doc = Metashape.Document()
doc.open(src, read_only=True, ignore_lock=True)
os.makedirs(work, exist_ok=True)
copy = os.path.join(work, name + "_work.psx")
doc.save(copy)                                    # a copy: all further processing writes here only
doc = Metashape.Document()
doc.open(copy)
chunk = doc.chunk or doc.chunks[0]

ref = json.load(open(plane_json, encoding="utf-8"))
prods = [p for p in ref["products"] if p.get("matrix") and "Planar" in str(p.get("projection_type"))
         and any(abs(p["matrix"][i][3]) > 1.0 for i in range(3))]
pick = next((p for p in prods if p["kind"] == "orthomosaic"), prods[0])
proj = Metashape.OrthoProjection()
proj.type = Metashape.OrthoProjection.Type.Planar
proj.matrix = Metashape.Matrix(pick["matrix"])
proj.crs = Metashape.CoordinateSystem("EPSG::4978")

rt = chunk.raster_transform                       # values as they are (degC), no palette
rt.enabled = False
chunk.raster_transform = rt

kw = {"projection": proj}
if res:
    kw["resolution"] = res
chunk.buildDem(source_data=Metashape.ModelData if chunk.model is not None else Metashape.PointCloudData, **kw)
print("DEM built")
okw = dict(kw, surface_data=Metashape.ModelData if chunk.model is not None else Metashape.ElevationData)
try:
    okw["blending_mode"] = Metashape.MosaicBlending
except AttributeError:
    pass
chunk.buildOrthomosaic(**okw)
print("orthomosaic built")
doc.save()

os.makedirs(out, exist_ok=True)
dem_path = os.path.join(out, name + "_temperature_dem.tif")
ortho_path = os.path.join(out, name + "_temperature_ortho.tif")
chunk.exportRaster(path=dem_path, source_data=Metashape.ElevationData, projection=proj)
chunk.exportRaster(path=ortho_path, source_data=Metashape.OrthomosaicData, projection=proj,
                   raster_transform=Metashape.RasterTransformNone, image_format=Metashape.ImageFormatTIFF)
print("exported", dem_path, ortho_path)
