"""Run inside Agisoft Metashape Professional (Tools > Run Script..., or paste into View > Console).

Writes the projection of every orthomosaic and DEM of the active chunk -- type (planar / geographic /
cylindrical), CRS and the 4x4 projection matrix -- to <project name>_planar_projection.json next to the
.psx project, and prints it. A planar orthomosaic built from "Current view" keeps that plane here even after
the view has changed, so this also works for products made earlier.

The groundwater pipeline uses the matrix to place wall-parallel exports in map coordinates exactly
(instead of the ICP registration to the point cloud, facade/planar.py).
"""
import json

import Metashape


def matrix_rows(m):
    if m is None:
        return None
    return [[float(m[i, j]) for j in range(4)] for i in range(4)]


def describe(kind, obj):
    if obj is None:
        return None
    p = obj.projection
    crs = getattr(p, "crs", None)
    return {
        "kind": kind,
        "label": getattr(obj, "label", ""),
        "key": getattr(obj, "key", None),
        "projection_type": str(getattr(p, "type", "")),
        "crs_name": crs.name if crs is not None else None,
        "crs_wkt": crs.wkt if crs is not None else None,
        "matrix": matrix_rows(getattr(p, "matrix", None)),
        "resolution": getattr(obj, "resolution", None),
        "width": getattr(obj, "width", None),
        "height": getattr(obj, "height", None),
    }


doc = Metashape.app.document
chunk = doc.chunk
out = {
    "project": doc.path,
    "chunk": chunk.label,
    "metashape_version": Metashape.app.version,
    "chunk_crs": chunk.crs.wkt if chunk.crs else None,
    "chunk_transform": matrix_rows(chunk.transform.matrix) if chunk.transform and chunk.transform.matrix else None,
    "products": [],
}
for o in (getattr(chunk, "orthomosaics", None) or [chunk.orthomosaic]):
    d = describe("orthomosaic", o)
    if d:
        out["products"].append(d)
for e in (getattr(chunk, "elevations", None) or [chunk.elevation]):
    d = describe("dem", e)
    if d:
        out["products"].append(d)

path = (doc.path.rsplit(".", 1)[0] if doc.path else "metashape") + "_planar_projection.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("Saved:", path)
for p in out["products"]:
    print(p["kind"], p["label"], p["projection_type"], p["crs_name"])
    print("  matrix:", p["matrix"])
