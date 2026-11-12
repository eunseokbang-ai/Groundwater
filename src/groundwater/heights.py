"""Height datum: analysis and reports in orthometric height (해발고도, EL.) over the Korean geoid KNGeoid18.

Inputs come in two datums. Photogrammetry exports (Metashape with a WGS 84-based chunk: LAS, OBJ, planar
GeoTIFFs and their *_planar_projection.json) carry ellipsoidal heights. A DJI Terra / L2 LiDAR LAS carries
orthometric heights, declared in its WKT ("... + Korean National Geoid Model (KNGeoid18)"). The difference is the
geoid height N (Iksan 24.52-24.53 m, Ulsan ~26 m); h_orthometric = h_ellipsoidal - N.

A config with "height_datum": "orthometric" (new site-folder configs) converts every input on load; older configs
and results without the key stay as they were (heights as in the files). The 3D tiled-model viewer works in
geocentric coordinates from ellipsoidal heights, so ECEF conversions add N back (tiles3d._to_ecef with a
'+KNGeoid18' crs, see reg_crs).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEOID_FILES = [ROOT / "Filed_Data" / "Geoid_Model" / "KNGeoid18.dat", ROOT / "data" / "geoid" / "KNGeoid18.dat"]
CACHE = Path.home() / ".groundwater" / "KNGeoid18.npz"
SUFFIX = "+KNGeoid18"
ORTHOMETRIC = "orthometric"
_grid = None


def geoid_file() -> Path | None:
    return next((f for f in GEOID_FILES if f.exists()), None)


def available() -> bool:
    return CACHE.exists() or geoid_file() is not None


def _load():
    """(lat0, lon0, dlat, dlon, N[rows lat, cols lon]) of KNGeoid18 (text: lat lon N per line on a regular grid),
    cached as npz."""
    global _grid
    if _grid is not None:
        return _grid
    if CACHE.exists():
        z = np.load(CACHE)
        _grid = (float(z["lat0"]), float(z["lon0"]), float(z["dlat"]), float(z["dlon"]), z["N"])
        return _grid
    f = geoid_file()
    if f is None:
        raise FileNotFoundError("KNGeoid18.dat not found (Filed_Data/Geoid_Model)")
    d = np.loadtxt(f)
    lats, lons = np.unique(d[:, 0]), np.unique(d[:, 1])
    N = np.full((len(lats), len(lons)), np.nan)
    N[np.searchsorted(lats, d[:, 0]), np.searchsorted(lons, d[:, 1])] = d[:, 2]
    dlat, dlon = float(np.median(np.diff(lats))), float(np.median(np.diff(lons)))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, lat0=lats[0], lon0=lons[0], dlat=dlat, dlon=dlon, N=N)
    _grid = (float(lats[0]), float(lons[0]), dlat, dlon, N)
    return _grid


def geoid_n(lon, lat) -> np.ndarray:
    """Geoid height N (m) by bilinear interpolation of KNGeoid18."""
    lat0, lon0, dlat, dlon, N = _load()
    fy = (np.asarray(lat, float) - lat0) / dlat
    fx = (np.asarray(lon, float) - lon0) / dlon
    i = np.clip(np.floor(fy).astype(int), 0, N.shape[0] - 2)
    j = np.clip(np.floor(fx).astype(int), 0, N.shape[1] - 2)
    ty, tx = np.clip(fy - i, 0, 1), np.clip(fx - j, 0, 1)
    n = (N[i, j] * (1 - ty) * (1 - tx) + N[i + 1, j] * ty * (1 - tx) + N[i, j + 1] * (1 - ty) * tx
         + N[i + 1, j + 1] * ty * tx)
    if np.isnan(n).any():
        raise ValueError("outside KNGeoid18")
    return n


def geoid_n_map(x, y, crs: str, step_m: float = 25.0) -> np.ndarray:
    """N at map coordinates. The geoid is smooth (cm over a site), so it is evaluated on a `step_m` grid over
    the points and interpolated from there: fast for 10^8 points."""
    from rasterio.warp import transform

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return np.zeros(0)
    x0, y0 = np.floor(x.min() / step_m) * step_m, np.floor(y.min() / step_m) * step_m
    nx = int((x.max() - x0) // step_m) + 2
    ny = int((y.max() - y0) // step_m) + 2
    gx, gy = np.meshgrid(x0 + np.arange(nx) * step_m, y0 + np.arange(ny) * step_m)
    lon, lat = transform(base_crs(crs), "EPSG:4326", gx.ravel().tolist(), gy.ravel().tolist())
    G = geoid_n(lon, lat).reshape(ny, nx)
    fx, fy = (x - x0) / step_m, (y - y0) / step_m
    i, j = np.clip(fy.astype(int), 0, ny - 2), np.clip(fx.astype(int), 0, nx - 2)
    ty, tx = fy - i, fx - j
    return (G[i, j] * (1 - ty) * (1 - tx) + G[i + 1, j] * ty * (1 - tx) + G[i, j + 1] * (1 - ty) * tx
            + G[i + 1, j + 1] * ty * tx)


def base_crs(crs: str) -> str:
    return str(crs).replace(SUFFIX, "")


def is_orthometric_crs(crs) -> bool:
    return SUFFIX in str(crs or "")


def target(cfg: dict) -> str | None:
    return cfg.get("height_datum")


def reg_crs(cfg: dict) -> str | None:
    """The map CRS with the height datum: 'EPSG:5186+KNGeoid18' for orthometric results, else as stored."""
    crs = cfg.get("crs")
    if crs and target(cfg) == ORTHOMETRIC and not is_orthometric_crs(crs):
        return crs + SUFFIX
    return crs


def las_datum(header) -> str:
    """'orthometric' when the LAS declares a geoid / vertical datum in its WKT (DJI Terra), else 'ellipsoidal'."""
    for v in header.vlrs:
        if getattr(v, "record_id", None) == 2112:
            wkt = getattr(v, "string", None) or bytes(getattr(v, "record_data", b"")).decode("ascii", "ignore")
            if "VERT_CS" in wkt or "geoid" in wkt.lower():
                return ORTHOMETRIC
    return "ellipsoidal"


ORTHOMETRIC_NAME_WORDS = ("kngeoid", "해발", "orthometric")


def file_datum(path) -> str:
    """Height datum of an input file: a LAS says so in its WKT; another export (OBJ, PLY) is orthometric when a
    .prj beside it declares a vertical CS / geoid (Metashape with the KNGeoid18 compound CS,
    scripts/kngeoid_to_geotiff.py) or its name says so, else ellipsoidal (Metashape's default, WGS84 heights)."""
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() in (".las", ".laz"):
        import laspy

        with laspy.open(str(path)) as f:
            return las_datum(f.header)
    prj = path.with_suffix(".prj")
    if prj.exists():
        wkt = prj.read_text(encoding="utf-8", errors="ignore")
        if "VERT_CS" in wkt or "geoid" in wkt.lower():
            return ORTHOMETRIC
    if any(w in path.name.lower() for w in ORTHOMETRIC_NAME_WORDS):
        return ORTHOMETRIC
    return "ellipsoidal"


def to_orthometric(xyz: np.ndarray, crs: str) -> np.ndarray:
    """Ellipsoidal -> orthometric heights (a copy)."""
    out = np.array(xyz, dtype=np.float64, copy=True)
    out[:, 2] -= geoid_n_map(out[:, 0], out[:, 1], crs)
    return out


def match_datum(xyz: np.ndarray, reference: np.ndarray, crs: str) -> float:
    """Height shift (0 or -N) that puts `xyz` (a mesh, a cloud of unknown datum) on `reference`: whichever of the
    two leaves the smaller median distance between them."""
    from scipy.spatial import cKDTree

    a = xyz[:: max(1, len(xyz) // 50_000)]
    ref = reference[:: max(1, len(reference) // 200_000)]
    tree = cKDTree(ref)
    n = float(np.median(geoid_n_map(a[:, 0], a[:, 1], crs)))
    d0 = np.median(tree.query(a)[0])
    d1 = np.median(tree.query(a - [0, 0, n])[0])
    return -n if d1 < d0 else 0.0
