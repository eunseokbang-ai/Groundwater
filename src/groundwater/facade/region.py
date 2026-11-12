"""Analysis region: chainage and elevation limits and/or a polygon drawn on the facade (wall coordinates), and
the section of the site it belongs to (a polygon in plan, map coordinates).

Used to leave out what is not rock face -- the tree line above the crest, the river or talus below, the
ends of a scan -- before any analysis. The facade polygon comes from the facade tool (analysis_region.csv in the
site folder, columns u,v in metres of chainage and elevation). A wall that turns (a quarry pit's U) is analysed in
sections, each facing one way; section_area.csv (columns x,y, map coordinates) keeps a section's points only, so
the next section's wall -- behind a corner, in a bay -- does not project onto this one's facade.

Overlap: with section_buffer.csv (x,y) as well, the section is processed over that larger area (its core plus
~20 m along the wall into the neighbouring sections) and owns only what lies in its core: faces whose centroid is
in section_area.csv, traces, face windows, seepage cells and thermal anomalies within the core's chainage range
(cfg["section"]["core_chainage"]). A joint face crossing the boundary is then segmented whole, and the
neighbourhood measures (5 m windows, 1 m kernels) see rock on both sides; each face is counted in one section only
(Iksan west wall 2026-10, hard cut: 21 faces, 4.4 % of the face area, cut at the boundary).
"""
import csv
from pathlib import Path

import numpy as np

REGION_FILE = "analysis_region.csv"
SECTION_FILE = "section_area.csv"
BUFFER_FILE = "section_buffer.csv"
BUFFER_M = 20.0          # ~ the largest neighbourhood used (10 m) + a large face (~10 m)


def load_polygon(path: str | Path, cols=("u", "v")) -> list | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        pts = [[float(r[cols[0]]), float(r[cols[1]])] for r in csv.DictReader(f)]
    return pts if len(pts) >= 3 else None


def load_section(path: str | Path) -> list | None:
    """The section polygon (map x, y) of a section_area.csv."""
    return load_polygon(path, ("x", "y"))


def save_polygon(path: str | Path, polygon) -> None:
    lines = ["u,v"] + [f"{u:.3f},{v:.3f}" for u, v in polygon]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_active(region: dict | None) -> bool:
    return bool(region) and any(region.get(k) for k in ("chainage", "elevation", "polygon", "map_polygon"))


def in_polygon(poly, xy: np.ndarray) -> np.ndarray:
    """True for points (x, y, ...) inside the polygon."""
    return _inside(poly, np.asarray(xy), np.ones(len(xy), bool))


def core_chainage(cfg: dict | None):
    """(lo, hi) chainage of the section core of a run with an overlap buffer, else None."""
    c = ((cfg or {}).get("section") or {}).get("core_chainage")
    return tuple(c) if c else None


def in_core(u, cfg: dict | None) -> np.ndarray:
    """True for chainages in the section core (everything without an overlap buffer)."""
    u = np.asarray(u, float)
    c = core_chainage(cfg)
    return np.ones(u.shape, bool) if c is None else (u >= c[0]) & (u <= c[1])


def _inside(poly, pts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    path = MplPath(np.asarray(poly, float))
    idx = np.flatnonzero(keep)
    for s in range(0, len(idx), 5_000_000):
        part = idx[s:s + 5_000_000]
        keep[part] = path.contains_points(pts[part, :2])
    return keep


def region_mask(uvw: np.ndarray, region: dict | None, xy: np.ndarray | None = None) -> np.ndarray:
    """True for points inside the region (everything when no limits are set). The section polygon needs the
    points' map coordinates `xy`."""
    keep = np.ones(len(uvw), bool)
    if not region:
        return keep
    for key, col in (("chainage", 0), ("elevation", 1)):
        lim = region.get(key)
        if lim:
            lo, hi = lim
            if lo is not None:
                keep &= uvw[:, col] >= lo
            if hi is not None:
                keep &= uvw[:, col] <= hi
    poly = region.get("polygon")
    if poly:
        keep = _inside(poly, uvw, keep)
    sec = region.get("map_polygon")
    if sec and xy is not None:
        keep = _inside(sec, np.asarray(xy), keep)
    return keep
