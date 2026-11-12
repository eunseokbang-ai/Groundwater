"""Site-folder input: point the pipeline at one folder of survey data instead of writing a YAML by hand.

Files are recognised by extension and name; subfolders are searched, file names are free:

    <site folder>/
        *.las *.laz *.ply *.pcd     photogrammetry point cloud (the largest one is the default)
        *.obj                       mesh (used only when no point cloud is chosen; lighter files first)
        lidar / 라이다 ... in the path  LiDAR point cloud: its rear layer fills the netted / vegetated parts
                                    of the wall (lidar.py; the photogrammetry cloud stays the main input)
        thermal / 열화상 / h20t in the path   thermal data -- orthomosaic (.tif), point cloud or mesh: listed
                                    apart, never taken as the photogrammetry cloud (palette colours, not RGB)
        *.csv with dip, dip_direction columns            field joint measurements (validation)
        wall_line.csv               optional wall plan line: columns kind,x,y -- kind=wall rows in order
                                    left-to-right seen from the pit, one kind=pit row anywhere on the open side
        구간/<name>/                 sections of a site whose wall turns (a quarry pit's U), all sharing the data
                                    above: each holds its own wall_line.csv, section_area.csv (+ section_buffer.csv,
                                    analysis_region.csv) and gets its own 분석결과 (build_config(section=...))

A section can be analysed from a LiDAR scan alone (no photogrammetry of it): the LiDAR is then the main cloud
(main_cloud_options: the clouds whose extent covers the section, photogrammetry first), read for the section's area
only.

Analysis parameters come from the template config (configs/example_site.yaml) with `auto_scale` on, so the
size-dependent ones are set from the measured point spacing (see pipeline.apply_auto_scale). The config used
is written next to the results (분석결과/analysis_config.yaml) so a run can be repeated or tuned later.
"""
import csv
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "example_site.yaml"
OUTPUT_DIRNAME = "분석결과"
SECTIONS_DIRNAME = "구간"
WALL_LINE_NAME = "wall_line.csv"

CLOUD_EXT = {".las", ".laz", ".ply", ".pcd"}
LIDAR_WORDS = ("lidar", "라이다", "yellowscan", "vx15", "zenmuse_l", "l3_")
THERMAL_WORDS = ("thermal", "열화상", "h20t", "tir")


def _has(path: Path, words) -> bool:
    s = str(path).lower()
    return any(w in s for w in words)


def _is_field_csv(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8-sig") as f:
            header = [c.strip().lower() for c in next(csv.reader(f), [])]
    except (OSError, UnicodeDecodeError):
        return False
    return "dip" in header and "dip_direction" in header


def scan(folder: str | Path) -> dict:
    """Candidate input files in `folder`, each list largest first (results folders are skipped)."""
    folder = Path(folder)
    found = {"point_cloud": [], "lidar": [], "mesh": [], "thermal": [], "field": [], "wall_line": None}
    if not folder.is_dir():
        return found
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).parts
        # results (분석결과, and renamed copies such as 분석결과_이전_...) and section folders hold no input data
        if not p.is_file() or any(q.startswith(OUTPUT_DIRNAME) for q in rel[:-1]) or SECTIONS_DIRNAME in rel:
            continue
        ext = p.suffix.lower()
        if ext in CLOUD_EXT | {".obj", ".tif", ".tiff"} and _has(p.relative_to(folder), THERMAL_WORDS):
            found["thermal"].append(p)
        elif ext in CLOUD_EXT:
            found["lidar" if _has(p.relative_to(folder), LIDAR_WORDS) else "point_cloud"].append(p)
        elif ext == ".obj":
            found["mesh"].append(p)
        elif p.name.lower() == WALL_LINE_NAME:
            # the folder's own line, not one in a sub-folder (drafts of other sections, copies)
            if found["wall_line"] is None or len(p.parts) < len(found["wall_line"].parts):
                found["wall_line"] = p
        elif ext == ".csv" and _is_field_csv(p):
            found["field"].append(p)
    for k, v in found.items():
        if isinstance(v, list):
            v.sort(key=lambda q: q.stat().st_size, reverse=True)
    # meshes: the largest one a viewer can still handle first (a decimated copy rather than the full model)
    found["mesh"].sort(key=lambda q: (q.stat().st_size > HEAVY_MESH, -q.stat().st_size))
    # LiDAR: the folder's own scan first (a sub-folder may hold a larger scan of the whole site)
    found["lidar"].sort(key=lambda q: len(q.relative_to(folder).parts))
    return found


HEAVY_MESH = 2_000_000_000     # bytes of OBJ text (~20 M triangles): too heavy for the 3D views


LARGE_CLOUD = 150_000_000      # points; the whole file is read into memory before downsampling


def cloud_summary(path: str | Path) -> dict | None:
    """Header-only facts about a LAS/LAZ (point count, extent, colour, CRS) plus warnings; None for other formats."""
    path = Path(path)
    if path.suffix.lower() not in (".las", ".laz"):
        return None
    import laspy

    with laspy.open(str(path)) as f:
        h = f.header
        n = int(h.point_count)
        lo, hi = h.mins, h.maxs
        has_rgb = {"red", "green", "blue"} <= set(h.point_format.dimension_names)
        crs = las_epsg(h)
    ext = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
    warn = []
    if abs(lo[0]) <= 180 and abs(hi[0]) <= 180 and abs(lo[1]) <= 90 and ext[0] < 1 and ext[1] < 1:
        warn.append("좌표가 경위도(도) 단위로 보입니다. 미터 단위 투영좌표(예: EPSG:5186)로 내보낸 점군이 필요합니다.")
    if not has_rgb:
        warn.append("점에 색(RGB)이 없어 입면 영상·트레이스 교차검증은 건너뜁니다 (절리군 추출은 가능).")
    if n > LARGE_CLOUD:
        big = f"{n / 1e8:.1f}억" if n >= 1e8 else f"{n / 1e4:,.0f}만"
        warn.append(f"점이 {big} 개로 많습니다. 메모리가 부족하면 분석할 암사면만 잘라 낸 LAS를 쓰세요.")
    return {"points": n, "extent_m": ext, "rgb": has_rgb, "crs": crs, "warnings": warn}


def las_epsg(header) -> str | None:
    """'EPSG:nnnn' of a LAS header: the GeoTIFF keys' projected (3072) or geographic (2048) CS code -- read
    directly, so no pyproj is needed (Metashape writes these keys); else the WKT through laspy if it can."""
    for v in header.vlrs:
        if getattr(v, "record_id", None) == 34735 and hasattr(v, "geo_keys"):
            keys = {k.id: k.value_offset for k in v.geo_keys}
            for kid in (3072, 2048):
                if 1024 <= keys.get(kid, 0) < 32767:
                    return f"EPSG:{keys[kid]}"
    for v in header.vlrs:                    # OGC WKT (DJI Terra: COMPD_CS[PROJCS[...], VERT_CS[...]])
        if getattr(v, "record_id", None) == 2112:
            wkt = getattr(v, "string", None) or bytes(getattr(v, "record_data", b"")).decode("ascii", "ignore")
            proj = wkt.split("VERT_CS")[0]
            if "PROJCS" in proj:
                codes = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', proj)
                if codes:
                    return f"EPSG:{codes[-1]}"
    try:
        crs = header.parse_crs()
        return crs.to_string() if crs is not None else None
    except Exception:  # noqa: BLE001
        return None


def read_wall_line(path: str | Path):
    """(polyline [[x, y], ...] or None, viewer point [x, y] or None) from a wall_line.csv."""
    wall, pit = [], None
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v for k, v in row.items()}
            xy = [float(row["x"]), float(row["y"])]
            if row.get("kind", "wall").strip().lower() == "pit":
                pit = xy
            else:
                wall.append(xy)
    return (wall if len(wall) >= 2 else None), pit


def sections(folder: str | Path) -> list[str]:
    """Names of the sections of a site (구간/<name>/ holding a wall_line.csv)."""
    d = Path(folder) / SECTIONS_DIRNAME
    return sorted(p.name for p in d.iterdir() if (p / WALL_LINE_NAME).exists()) if d.is_dir() else []


def section_dir(folder: str | Path, section: str | None) -> Path:
    return Path(folder) / SECTIONS_DIRNAME / section if section else Path(folder)


def las_coverage(path: str | Path, polygon) -> float:
    """Share of the polygon's bounding box (map x, y) inside the LAS/LAZ header extent (1.0 for other formats)."""
    import numpy as np

    if Path(path).suffix.lower() not in (".las", ".laz") or not polygon:
        return 1.0
    import laspy

    with laspy.open(str(path)) as f:
        lo, hi = np.asarray(f.header.mins[:2]), np.asarray(f.header.maxs[:2])
    a = np.asarray(polygon, float)
    plo, phi = a.min(axis=0), a.max(axis=0)
    inter = np.clip(np.minimum(hi, phi) - np.maximum(lo, plo), 0, None)
    return float(inter.prod() / max((phi - plo).prod(), 1e-9))


def main_cloud_options(found: dict, polygon=None, min_coverage: float = 0.8) -> list:
    """Clouds that can be the main input of a section: photogrammetry first, then LiDAR, those covering the section
    polygon (map x, y) first."""
    clouds = found["point_cloud"] + found["lidar"]
    if not polygon:
        return clouds
    return sorted(clouds, key=lambda p: las_coverage(p, polygon) < min_coverage)        # stable: order kept


def build_config(folder: str | Path, point_cloud=None, mesh=None, field=None, lidar=None, thermal=None,
                 wall_line=None, auto_scale=True, vegetation=True, exg_threshold=None,
                 min_patch_area=None, region=None, use_region_polygon=True, friction_deg=None,
                 trace_min_length=None, planar_dir=None, clean_ground=True, clean_vegetation_faces=True,
                 clean_netting=True, clean_flat=True, use_lidar=True, section_overlap=True, fixed_sets=None,
                 section=None, exclude_wall_parallel=False, clean_above_ground=True) -> dict:
    """Pipeline config for `folder` (or its section `section`): template parameters + the chosen files + auto
    scale."""
    folder = Path(folder)
    sdir = section_dir(folder, section)
    if section and wall_line is None and (sdir / WALL_LINE_NAME).exists():
        wall_line = sdir / WALL_LINE_NAME
    cfg = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    s = lambda p: str(p) if p else None  # noqa: E731
    cfg["site_name"] = folder.name + (f" · {section}" if section else "")
    cfg["section_name"] = section
    cfg["paths"] = {"point_cloud": s(point_cloud), "mesh": s(mesh), "lidar_point_cloud": s(lidar),
                    "thermal_orthomosaic": s(thermal), "field_measurements": s(field),
                    "output_dir": str(sdir / OUTPUT_DIRNAME)}
    if clean_above_ground:                        # bare-earth DTM: points above the ground are trees (cleaning.above_ground)
        from .site_joints import find_ground

        g = find_ground(folder)
        cfg["paths"]["ground_dtm"] = s(g)
    polyline, pit = read_wall_line(wall_line) if wall_line else (None, None)
    cfg["wall_frame"] = {"polyline": polyline, "viewer_point": pit}
    cfg["auto_scale"] = bool(auto_scale)
    if min_patch_area is not None:
        cfg.setdefault("patches", {})["min_patch_area"] = float(min_patch_area)
    if exclude_wall_parallel:                     # wall-parallel normals left out of the set detection
        cfg["exposure"] = {"exclude_wall_parallel": True}
    if fixed_sets:                                # [(dip, dip direction), ...] the user pins as sets
        cfg.setdefault("orientation_sets", {})["fixed_sets"] = [[float(d), float(a)] for d, a in fixed_sets]
    # analysis region: chainage / elevation limits from the caller + the polygon drawn in the facade tool
    from .facade.region import BUFFER_FILE, REGION_FILE, SECTION_FILE, load_polygon, load_section

    region = dict(region or {})
    poly = load_polygon(sdir / REGION_FILE) if use_region_polygon else None
    if poly:
        region["polygon"] = poly
    sec = load_section(sdir / SECTION_FILE)
    buf = load_section(sdir / BUFFER_FILE) if section_overlap else None
    if sec and buf:                               # processed over the buffer, owning only its core
        region["map_polygon"], region["core_polygon"] = buf, sec
    elif sec or buf:
        region["map_polygon"] = sec or buf        # this section of the site only (plan polygon)
    cfg["region"] = region or None
    if friction_deg is not None:
        cfg.setdefault("kinematics", {})["friction_deg"] = float(friction_deg)
    if planar_dir:
        cfg["planar"] = {"folder": str(planar_dir), "resolution_m": 0.02}
    if trace_min_length is not None:
        cfg.setdefault("trace_mapping", {})["min_length_m"] = float(trace_min_length)
    cfg["vegetation"] = {"enabled": bool(vegetation)}
    cfg["cleaning"] = {"ground": bool(clean_ground), "vegetation_faces": bool(clean_vegetation_faces),
                       "netting": bool(clean_netting), "flat": bool(clean_flat), "lidar_behind": bool(use_lidar and lidar),
                       "above_ground": bool(clean_above_ground)}
    if exg_threshold is not None:
        cfg["vegetation"]["exg_threshold"] = float(exg_threshold)
    cfg["crs"] = folder_crs(folder) or cfg.get("crs")
    from . import heights

    if heights.available():                  # analysis and reports in orthometric height (해발고도, KNGeoid18)
        cfg["height_datum"] = heights.ORTHOMETRIC
    return cfg


def folder_crs(folder: Path) -> str | None:
    """The map CRS of the site: of the first GeoTIFF in a projected CRS (orthomosaic / DSM; wall-parallel planar
    exports are geocentric and are skipped), else of the first LAS/LAZ whose header carries one. Informational:
    all inputs must already share it."""
    import rasterio

    for f in sorted(Path(folder).rglob("*.tif")):
        if OUTPUT_DIRNAME in f.parts:
            continue
        try:
            with rasterio.open(f) as ds:
                if ds.crs is not None and ds.crs.is_projected:
                    auth = ds.crs.to_authority() or ds.crs.to_authority(confidence_threshold=20)
                    return ":".join(auth) if auth else ds.crs.to_string()
        except Exception:  # noqa: BLE001
            continue
    import laspy

    for f in sorted(list(Path(folder).rglob("*.las")) + list(Path(folder).rglob("*.laz"))):
        if OUTPUT_DIRNAME in f.parts:
            continue
        try:
            with laspy.open(str(f)) as r:
                crs = las_epsg(r.header)
            if crs:
                return crs
        except Exception:  # noqa: BLE001
            continue
    return None


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
