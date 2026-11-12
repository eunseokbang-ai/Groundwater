"""Run inside Agisoft Metashape Professional (Tools > Run Script..., or paste into View > Console), in the project
to process -- e.g. the thermal model of a wall whose RGB model was already projected with "Current view".

Builds a DEM and an orthomosaic of the active chunk on exactly the same plane as another project's planar
products: the plane (4x4 matrix, geocentric -> plane) is read from that project's *_planar_projection.json
(written by metashape_export_planar_projection.py). Same axes and orientation, so the two orthomosaics line up
pixel axis to pixel axis. Afterwards export both (File > Export > Export DEM / Export Orthomosaic, GeoTIFF) and
run metashape_export_planar_projection.py in this project too, so the groundwater pipeline knows this one's
plane as well.

The chunk must be georeferenced (any map CRS): the matrix is in geocentric WGS 84 coordinates.
"""
import json

import Metashape

RESOLUTION = 0          # m per pixel for both products; 0 = Metashape's default (from the images)


def _const(*names):
    """A Metashape constant by its name in the 2.x API or the older one (e.g. PointCloudData / DenseCloudData)."""
    for n in names:
        for where in (Metashape, getattr(Metashape, "DataSource", None)):
            if where is not None and hasattr(where, n):
                return getattr(where, n)
    raise AttributeError(names)


path = Metashape.app.getOpenFileName("기준 평면 JSON (..._planar_projection.json)", filter="JSON (*.json)")
if not path:
    raise SystemExit("cancelled")
with open(path, encoding="utf-8") as f:
    ref = json.load(f)
planar = [p for p in ref["products"] if p.get("matrix") and "Planar" in str(p.get("projection_type"))
          and any(abs(p["matrix"][i][3]) > 1.0 for i in range(3))]      # not an identity (top-down) product
src = next((p for p in planar if p["kind"] == "orthomosaic"), planar[0] if planar else None)
if src is None:
    raise SystemExit("no planar product with a plane matrix in " + path)

proj = Metashape.OrthoProjection()
proj.type = Metashape.OrthoProjection.Type.Planar
proj.matrix = Metashape.Matrix(src["matrix"])
proj.crs = Metashape.CoordinateSystem("EPSG::4978")                       # the matrix is geocentric -> plane
print("plane from", path, "(", src["kind"], ", resolution there", src.get("resolution"), ")")

chunk = Metashape.app.document.chunk
if chunk.crs is None:
    raise SystemExit("the chunk is not georeferenced")
has_model = chunk.model is not None
dem_source = _const("ModelData") if has_model else _const("PointCloudData", "DenseCloudData")
kw = {"projection": proj}
if RESOLUTION:
    kw["resolution"] = RESOLUTION
chunk.buildDem(source_data=dem_source, **kw)
print("DEM built on the plane from", "the model" if has_model else "the point cloud")
chunk.buildOrthomosaic(surface_data=_const("ModelData") if has_model else _const("ElevationData"), **kw)
print("Orthomosaic built on the same plane. Next: File > Export > Export DEM / Export Orthomosaic (GeoTIFF),")
print("then Tools > Run Script > metashape_export_planar_projection.py in this project.")
Metashape.app.document.save()
