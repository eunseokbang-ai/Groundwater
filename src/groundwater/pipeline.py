"""Phase 1 end-to-end orchestration: RGB photogrammetry point cloud/mesh ->
discontinuity extraction -> wall frame + facade orthophoto -> 2D trace
cross-check -> (optional) field-measurement validation -> outputs on disk for
the dashboard.

CLI usage:
    python -m groundwater.pipeline --config configs/example_site.yaml
    python -m groundwater.pipeline --folder <site folder>      (files found automatically, see site_folder)
"""
import copy
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .discontinuity.cleaning import (MASK_FLAT, MASK_NET, drop_faces, drop_masked_traces, flat_faces, ground_faces,
                                     mask_codes, surface_mask, vegetation_faces, canopy_ground_faces)
from .discontinuity.geometry import DiscontinuitySet
from .discontinuity.normals import estimate_local_geometry
from .discontinuity.segmentation import extract_discontinuities
from .discontinuity.kinematics import FRICTION_DEG, VERTICAL_DEG, face_windows, kinematic_check
from .discontinuity.trace_mapping import Trace2D, extract_traces, match_traces_to_patches
from .discontinuity.trace_sets import trace_families
from .discontinuity.vegetation import EXG_THRESHOLD, vegetation_mask
from .facade.frame import WallFrame, build_wall_frame
from .facade.orthophoto import FacadeAccumulator
from .facade.region import in_core, in_polygon, is_active, region_mask
from .io.mesh import load_mesh, mesh_to_point_cloud
from .io.point_cloud import crop_bbox, downsample, iter_las_chunks, las_point_count, load_point_cloud
from .io.raster import Raster, write_raster
from .validation.field_compare import compare_to_field, load_field_measurements, summarize_errors

FOOTPRINT_SAMPLE = 500
# auto_scale: analysis voxel never finer than this (drone deliverable = major joints, not mm cracks) and the
# cloud / facade image kept to a size a desktop handles
AUTO_MIN_VOXEL = 0.02
AUTO_MAX_POINTS = 3_000_000
AUTO_MAX_FACADE_PX = 40_000_000
# a LAS/LAZ above this is thinned evenly while reading (every k-th point); override with
# preprocessing.max_load_points. 150 M points ~ 7 GB once loaded
MAX_LOAD_POINTS = 150_000_000
MANUAL_TRACES = "manual_traces.csv"


@dataclass
class PipelineResult:
    config: dict
    point_cloud: object  # open3d.geometry.PointCloud (analysis resolution)
    point_labels: np.ndarray  # (N, 2): [set_id, patch_id] per point, -1 = unassigned
    discontinuity_sets: list[DiscontinuitySet]
    wall_frame: WallFrame
    facade_rgb: Raster | None
    facade_depth: Raster | None
    field_comparison: pd.DataFrame | None
    traces: list[Trace2D] = field(default_factory=list)
    trace_confirmed: dict = field(default_factory=dict)  # patch_id -> True / False / None (n/a)
    facade_sets: Raster | None = None  # set id of the front-most analysis point per facade cell (-1 none)
    trace_families: list = field(default_factory=list)  # trace_sets.TraceFamily
    face_windows: list = field(default_factory=list)  # kinematics.face_windows
    kinematics: pd.DataFrame | None = None  # kinematically possible failure modes per window
    timings: dict = field(default_factory=dict)  # stage -> seconds, of the run that made this result
    surface_mask: Raster | None = None  # cleaning.surface_mask: netting / flat ground left out (wall coords)


class _Clock:
    """Seconds per pipeline stage: clock.lap("name") books the time since the previous lap."""

    def __init__(self):
        self.t, self.laps = time.perf_counter(), {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(self.laps.get(name, 0.0) + now - self.t, 1)
        self.t = now


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _source_datum(path) -> str:
    """Height datum of an input file (heights.file_datum)."""
    from . import heights

    return heights.file_datum(path)


def _needs_geoid(cfg: dict, path) -> bool:
    from . import heights

    return heights.target(cfg) == heights.ORTHOMETRIC and _source_datum(path) != heights.ORTHOMETRIC


def _load_input_cloud(cfg: dict):
    import open3d as o3d

    from . import heights

    paths = cfg["paths"]
    if paths.get("point_cloud"):
        max_points = cfg.get("preprocessing", {}).get("max_load_points", MAX_LOAD_POINTS)
        n = las_point_count(paths["point_cloud"])
        box = _region_box(cfg) if n else None
        if box is not None:               # a section of a scan of the whole site: only its area, thinned to the limit
            from .io import point_cloud as PC

            pcd = load_point_cloud(paths["point_cloud"], max_points, bbox=box)
            cfg["load_thinning"] = dict(PC.load_stats)
        else:
            if n:
                cfg["load_thinning"] = {"points_total": n, "step": max(1, -(-n // max_points)) if max_points else 1}
            pcd = load_point_cloud(paths["point_cloud"], max_points)
        src = paths["point_cloud"]
    elif paths.get("mesh"):
        mesh = load_mesh(paths["mesh"])
        pcd = mesh_to_point_cloud(mesh)
        src = paths["mesh"]
    else:
        raise ValueError("Config must provide paths.point_cloud or paths.mesh")
    if heights.target(cfg) == heights.ORTHOMETRIC:
        if _needs_geoid(cfg, src):
            pcd.points = o3d.utility.Vector3dVector(heights.to_orthometric(np.asarray(pcd.points), cfg["crs"]))
            cfg["height_conversion"] = {"point_cloud": "ellipsoidal -> orthometric (KNGeoid18)"}
        else:
            cfg["height_conversion"] = {"point_cloud": "orthometric in the file"}
    return pcd


def _region_box(cfg: dict, margin: float = 1.0):
    """((xmin, ymin), (xmax, ymax)) of the section polygon (map coordinates), or None."""
    poly = (cfg.get("region") or {}).get("map_polygon")
    if not poly:
        return None
    a = np.asarray(poly, float)
    return a.min(axis=0) - margin, a.max(axis=0) + margin


def estimate_point_spacing(points: np.ndarray, n_seeds: int = 12, box: float = 1.0, seed: int = 0) -> float:
    """Median nearest-neighbour distance (m), measured in small boxes around random points so that it
    reflects the true density (a random subsample would not) without building a tree on the whole cloud."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    meds = []
    for i in rng.choice(len(points), min(n_seeds, len(points)), replace=False):
        c = points[i]
        for half in (box / 2, box * 1.5):
            # one coordinate at a time on the survivors (an N x 3 temporary per seed costs GBs at 10^8 points)
            sel = points[np.abs(points[:, 0] - c[0]) < half]
            sel = sel[(np.abs(sel[:, 1] - c[1]) < half) & (np.abs(sel[:, 2] - c[2]) < half)]
            if len(sel) >= 30:
                break
        if len(sel) >= 3:
            d, _ = cKDTree(sel).query(sel, k=2)
            meds.append(np.median(d[:, 1]))
    if not meds:
        raise ValueError("Point cloud too sparse to estimate the point spacing")
    return float(np.median(meds))


def apply_auto_scale(cfg: dict, points: np.ndarray, uv: np.ndarray | None = None) -> dict:
    """Set the size-dependent parameters from the point spacing s (ratios tuned on the demo wall, s = 4 cm):
    voxel = s (>= AUTO_MIN_VOXEL, coarser if the cloud would exceed AUTO_MAX_POINTS), normal radius 3.75
    voxels, patch link 2 voxels, trace tolerance 3 voxels, facade pixel 1.25 s (capped image size).

    The surface area is n s^2, or the facade area (occupied 10 cm cells of the wall coordinates `uv`) if smaller: a
    multi-return LiDAR has many points per surface cell (canopy volume, ground under the trees, net and rock), and
    n s^2 then overstates the area -- Iksan west wall 2026-10, LiDAR only: 15.8 cm voxels instead of 7 cm, and a
    set crossing the wall obliquely was lost."""
    s = estimate_point_spacing(points)
    area = len(points) * s * s                     # ~ surface area covered
    if uv is not None and len(uv):
        key = np.floor(uv[:, 0] / 0.1).astype(np.int64) * 10_000_000 + np.floor(uv[:, 1] / 0.1).astype(np.int64)
        facade = len(np.unique(key)) * 0.01
        cfg["facade_area_m2"] = round(float(facade), 1)
        area = min(area, facade)
    voxel = max(s, AUTO_MIN_VOXEL, float(np.sqrt(area / AUTO_MAX_POINTS)))
    applied = {
        "point_spacing_m": round(s, 4),
        "voxel_size": round(voxel, 4),
        "search_radius": round(3.75 * voxel, 4),
        "link_distance": round(2.0 * voxel, 4),
        "facade_resolution": round(max(1.25 * s, float(np.sqrt(area / AUTO_MAX_FACADE_PX))), 4),
        "trace_tolerance": round(3.0 * voxel, 4),
    }
    cfg.setdefault("preprocessing", {})["voxel_size"] = applied["voxel_size"]
    cfg.setdefault("normals", {})["search_radius"] = applied["search_radius"]
    cfg.setdefault("patches", {})["link_distance"] = applied["link_distance"]
    cfg.setdefault("facade", {})["resolution"] = applied["facade_resolution"]
    cfg.setdefault("trace_matching", {})["tolerance"] = applied["trace_tolerance"]
    # traces are detected on a >= 2 cm grid whatever the facade pixel (finer only resolves texture)
    cfg.setdefault("trace_mapping", {})["resolution_m"] = round(max(0.02, applied["facade_resolution"]), 4)
    cfg["auto_scale_applied"] = applied
    return applied


def run_pipeline(cfg: dict) -> PipelineResult:
    cfg = copy.deepcopy(cfg)
    clock = _Clock()
    cloud = _load_input_cloud(cfg)
    prep = cfg.get("preprocessing", {})
    cloud = crop_bbox(cloud, prep.get("crop_bbox"))
    clock.lap("load")

    # wall frame first: the analysis region and every facade product are in wall coordinates
    frame_cfg = cfg.get("wall_frame", {})
    pts = np.asarray(cloud.points)
    sample = pts if len(pts) <= 500_000 else pts[np.random.default_rng(0).choice(len(pts), 500_000, replace=False)]
    frame = build_wall_frame(sample, frame_cfg.get("polyline"), frame_cfg.get("viewer_point"))
    region = cfg.get("region")
    uvw_cloud = None
    if is_active(region):
        uvw_cloud = frame.to_wall(pts)
        inside = region_mask(uvw_cloud, region, pts)
        cfg["region_kept_fraction"] = round(float(inside.mean()), 4)
        cloud = cloud.select_by_index(np.flatnonzero(inside))
        uvw_cloud = uvw_cloud[inside]
        pts = np.asarray(cloud.points)
        if region.get("core_polygon") and len(pts):
            _section_core(cfg, region["core_polygon"], pts, uvw_cloud)
    if len(pts) == 0:
        raise ValueError("No points left inside the analysis region")
    clock.lap("frame")

    if cfg.get("auto_scale"):
        apply_auto_scale(cfg, pts, uvw_cloud[:, :2] if uvw_cloud is not None else frame.to_wall(pts)[:, :2])
        prep = cfg["preprocessing"]
    clock.lap("auto_scale")
    # green vegetation (by colour) is left out of the joint analysis; the facade image still shows it
    veg = cfg.get("vegetation", {})
    veg_on = veg.get("enabled", True) and cloud.has_colors()
    veg_thr = veg.get("exg_threshold", EXG_THRESHOLD)
    analysis_cloud = cloud
    lidar_path = (cfg.get("paths") or {}).get("lidar_point_cloud")
    use_lidar = bool(lidar_path) and cfg.get("cleaning", {}).get("lidar_behind", True) and Path(lidar_path).exists()
    veg_uv = (np.zeros((0, 2)), np.zeros((0, 2)))         # (all points, green points): wall u, v of a subsample
    if veg_on:
        plants = vegetation_mask(np.asarray(cloud.colors), veg_thr)
        cfg["vegetation_removed_fraction"] = round(float(plants.mean()), 4)
        if use_lidar:
            sub = np.arange(0, len(plants), 10)
            uv = frame.to_wall(np.asarray(cloud.points)[sub])[:, :2]
            veg_uv = (uv, uv[plants[sub]])
        analysis_cloud = cloud.select_by_index(np.flatnonzero(~plants))
        del plants
    pcd = downsample(analysis_cloud, prep.get("voxel_size", 0.02))
    del analysis_cloud
    # tree crowns above the ground (not green in a LiDAR's colours, or seen from behind): no rock stands in the air
    dtm = (cfg.get("paths") or {}).get("ground_dtm")
    dtm = dtm if dtm and cfg.get("cleaning", {}).get("above_ground", True) and Path(dtm).exists() else None
    if dtm:
        from .discontinuity.cleaning import above_ground

        air, info = above_ground(np.asarray(pcd.points), dtm)
        info["removed_fraction"] = round(float(air.mean()), 4) if len(air) else 0.0
        cfg["above_ground"] = info
        if info.get("skipped"):
            dtm = None
        elif air.any():
            pcd = pcd.select_by_index(np.flatnonzero(~air))
        del air
    clock.lap("voxel")
    # rockfall netting and near-horizontal ground (cleaning.surface_mask) are left out before the extraction:
    # a net draped parallel to the face would otherwise make a "joint set" of its own
    clean = cfg.get("cleaning", {})
    smask = None
    if clean.get("netting", True) or clean.get("flat", True):
        uvw0 = frame.to_wall(np.asarray(pcd.points))
        smask = surface_mask(uvw0, np.asarray(pcd.colors) if pcd.has_colors() else None,
                             netting=clean.get("netting", True) and pcd.has_colors(), flat=clean.get("flat", True))
        if smask is not None:
            bits = mask_codes(smask, uvw0[:, 0], uvw0[:, 1])
            cfg["masked_fraction"] = {"netting": round(float((bits & MASK_NET > 0).mean()), 4),
                                      "flat": round(float((bits & MASK_FLAT > 0).mean()), 4)}
            pcd = pcd.select_by_index(np.flatnonzero(bits == 0))
        del uvw0
        clock.lap("masks")
    # LiDAR (multi-return) behind the netting and the vegetation: the rock the photogrammetry cannot see
    if use_lidar and smask is not None:
        import open3d as o3d

        from . import lidar

        cells = lidar.lidar_cells(smask, veg_uv[1], veg_uv[0], MASK_FLAT, MASK_NET)
        keep_every = int(clean.get("lidar_keep_every", lidar.KEEP_EVERY))
        L = lidar.rock_behind(lidar_path, cfg, frame, cells, np.asarray(pcd.points), region, keep_every=keep_every,
                              dtm=dtm)
        n_base_out = 0
        if L["reference_on_face"] is not None and clean.get("base_face_span", True):
            # the base cloud outside the rock face's span: the ground and the trunks behind the wall top, the floor
            n_base_out = int((~L["reference_on_face"]).sum())
            if n_base_out:
                pcd = pcd.select_by_index(np.flatnonzero(L["reference_on_face"]))
        if len(L["xyz"]):
            lp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(L["xyz"], dtype=np.float64)))
            if L["rgb"] is not None and pcd.has_colors():
                lp.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(L["rgb"], dtype=np.float64))
            lp = downsample(lp, prep.get("voxel_size", 0.02))
            n_base = len(pcd.points)
            pcd = pcd + lp
            cfg["lidar_rock"] = {"file": str(lidar_path), "points_added": len(lp.points), "base_points": n_base,
                                 "base_points_outside_face_dropped": n_base_out,
                                 "cells_m2": round(L["n_cells"] * abs(smask.transform.a) ** 2, 1),
                                 "alignment_shift_m": [round(float(x), 3) for x in L["shift"]], "keep_every": keep_every,
                                 "rear_points_outside_face_dropped": L["n_outside_face_dropped"],
                                 "rear_points_gentle_kept_on_face": L.get("n_gentle_kept_on_face", 0),
                                 "rear_points_above_ground_dropped": L.get("n_above_ground_dropped", 0)}
        clock.lap("lidar")

    surface_variation = estimate_local_geometry(pcd, **cfg.get("normals", {}))
    clock.lap("normals")
    set_info = {}
    # optional: point normals parallel to the excavated wall (bench / blasted faces, or joints along them) left out
    # of the set detection, so the sets crossing the wall are not outweighed by the wall (discontinuity.exposure)
    expo = cfg.get("exposure", {})
    exclude = None
    if expo.get("exclude_wall_parallel"):
        from .discontinuity.exposure import PARALLEL_DEG, point_mask

        exclude = point_mask(frame.to_wall(np.asarray(pcd.points)), np.asarray(pcd.normals), frame,
                             float(expo.get("parallel_deg", PARALLEL_DEG)))
    discontinuity_sets, point_labels = extract_discontinuities(
        pcd,
        surface_variation=surface_variation,
        **cfg.get("orientation_sets", {}),
        **cfg.get("patches", {}),
        exclude=exclude,
        info=set_info,
    )
    # sets pinned by the user and density peaks that did not become sets (clustering.find_orientation_sets)
    cfg["set_detection"] = {"fixed_sets": [list(map(float, f)) for f in cfg.get("orientation_sets", {}).get("fixed_sets") or []],
                            "fixed_set_ids": set_info.get("fixed", []), "candidates": set_info.get("candidates", [])}
    if exclude is not None:
        cfg["set_detection"]["wall_parallel_points_left_out"] = int(exclude.sum())
    clock.lap("joints")
    # faces that are not joints of the wall (cleaning.py): the ground / water in front of the foot and ground /
    # benches reaching back into the slope now, the faces on vegetation once the facade photo exists
    link = cfg.get("patches", {}).get("link_distance", 0.05)
    uvw_pcd = frame.to_wall(np.asarray(pcd.points))
    removed = {}
    if clean.get("ground", True):
        drop = ground_faces(uvw_pcd, point_labels[:, 1], discontinuity_sets)
        discontinuity_sets = drop_faces(discontinuity_sets, point_labels, drop, np.asarray(pcd.normals), link)
        removed["ground_water"] = len(drop)
    if clean.get("flat", True):
        drop = flat_faces(uvw_pcd, point_labels[:, 1], discontinuity_sets)
        discontinuity_sets = drop_faces(discontinuity_sets, point_labels, drop, np.asarray(pcd.normals), link)
        removed["flat_ground"] = len(drop)
    if (region or {}).get("core_polygon"):
        # overlap buffer: faces were segmented whole across the section boundary; the section keeps those whose
        # centroid lies in its core, the neighbouring section the others
        drop = {p.patch_id for d in discontinuity_sets for p in d.patches
                if not in_polygon(region["core_polygon"], np.asarray(p.centroid)[None, :2])[0]}
        discontinuity_sets = drop_faces(discontinuity_sets, point_labels, drop, np.asarray(pcd.normals), link)
        removed["buffer_zone"] = len(drop)

    field_comparison = None
    field_path = cfg["paths"].get("field_measurements")
    if field_path and Path(field_path).exists():
        field_comparison = compare_to_field(discontinuity_sets, load_field_measurements(field_path))

    result = PipelineResult(
        config=cfg,
        point_cloud=pcd,
        point_labels=point_labels,
        discontinuity_sets=discontinuity_sets,
        wall_frame=frame,
        facade_rgb=None,
        facade_depth=None,
        field_comparison=field_comparison,
        surface_mask=smask,
    )
    if cloud.has_colors():
        _build_facade(result, cloud, uvw_cloud, uvw_pcd, region, veg_on, veg_thr)
    del cloud
    clock.lap("facade")

    if result.facade_rgb is not None and (cfg.get("planar") or {}).get("folder"):
        apply_planar(result, cfg["planar"]["folder"], cfg["planar"].get("resolution_m", 0.02))
        clock.lap("planar")
    if clean.get("vegetation_faces", True) and veg_on and result.facade_rgb is not None:
        drop = vegetation_faces(result.discontinuity_sets, frame, result.facade_rgb, result.facade_depth, veg_thr)
        under = canopy_ground_faces(result.discontinuity_sets, frame, result.facade_rgb, result.facade_depth, veg_thr)
        if drop | under:
            result.discontinuity_sets = drop_faces(result.discontinuity_sets, result.point_labels, drop | under,
                                                   np.asarray(pcd.normals), link)
            _after_cleaning(result, uvw_pcd)
        removed["vegetation"] = len(drop)
        removed["ground_under_trees"] = len(under - drop)
        clock.lap("cleaning")
    cfg["faces_removed"] = removed

    # kinematic screening against the local face (rock points of the analysis cloud)
    result.face_windows = [w for w in face_windows(np.asarray(pcd.points), frame, cfg.get("kinematics", {}).get("window_m", 5.0))
                           if in_core((w["u0"] + w["u1"]) / 2, cfg)]
    _kinematics(result)
    clock.lap("kinematics")
    result.timings = clock.laps
    return result


def drop_canopy_ground(result) -> int:
    """canopy_ground_faces on a saved result (made before that cleaning existed): the faces dropped, the set map,
    trace check, trace families and kinematic screening redone. Returns how many faces were dropped."""
    cfg = result.config
    veg = cfg.get("vegetation", {})
    if result.facade_rgb is None or not veg.get("enabled", True):
        return 0
    under = canopy_ground_faces(result.discontinuity_sets, result.wall_frame, result.facade_rgb, result.facade_depth,
                                veg.get("exg_threshold", EXG_THRESHOLD))
    if under:
        pcd = result.point_cloud
        result.discontinuity_sets = drop_faces(result.discontinuity_sets, result.point_labels, under,
                                               np.asarray(pcd.normals), cfg.get("patches", {}).get("link_distance", 0.05))
        _after_cleaning(result, result.wall_frame.to_wall(np.asarray(pcd.points)))
        _kinematics(result)
    cfg.setdefault("faces_removed", {})["ground_under_trees"] = len(under)
    return len(under)


def _kinematics(result) -> None:
    joints = {f"Set {d.set_id}": (d.mean_dip, d.mean_dip_direction) for d in result.discontinuity_sets}
    for f in result.trace_families:
        if f.linked_set is None and f.dip is not None:
            joints[f"트레이스 F{f.family_id}"] = (f.dip, f.dip_direction)
    kc = result.config.get("kinematics", {})
    result.kinematics = kinematic_check(result.face_windows, joints, kc.get("friction_deg", FRICTION_DEG),
                                        vertical_deg=kc.get("vertical_deg", VERTICAL_DEG))


def _section_core(cfg: dict, core, pts: np.ndarray, uvw: np.ndarray) -> None:
    """cfg["section"]: the core's chainage range (the points of the processed area inside the core polygon)."""
    m = in_polygon(core, pts)
    if not m.any():
        raise ValueError("No points inside the section core (section_area.csv)")
    lo, hi = np.percentile(uvw[m, 0], [0.05, 99.95])
    u = uvw[:, 0]
    cfg["section"] = {"core_chainage": [round(float(lo), 2), round(float(hi), 2)],
                      "buffer_before_m": round(float(lo - u.min()), 1), "buffer_after_m": round(float(u.max() - hi), 1),
                      "core_point_share": round(float(m.mean()), 4)}


def _keep_traces(result, traces: list) -> list:
    """Detected traces on rock (the set map), off netting / flat ground and in the section core; hand-drawn ones
    always."""
    traces = drop_masked_traces(drop_traces_off_rock(traces, result.facade_sets), result.surface_mask)
    if not traces:
        return traces
    mid = np.array([t.coords.mean(axis=0)[0] for t in traces])
    own = in_core(mid, result.config)
    return [t for t, o in zip(traces, own) if o or t.source == "manual"]


def drop_traces_off_rock(traces: list, facade_sets: Raster | None, radius_m: float = 0.5, min_cover: float = 0.05) -> list:
    """Keep traces whose midpoint has joint-set surfaces (the set map) on at least `min_cover` of a
    `radius_m` neighbourhood: lines in the tree canopy above a crest (dry leaves are not green, so the
    colour filter keeps them) have no planar facets around them. Hand-drawn traces are always kept."""
    if facade_sets is None:
        return traces
    a = facade_sets.array[0] >= 0
    res = abs(facade_sets.transform.a)
    k = max(1, int(round(radius_m / res)))
    import cv2

    cover = cv2.blur(a.astype(np.float32), (2 * k + 1, 2 * k + 1))
    out = []
    for t in traces:
        if t.source == "manual":
            out.append(t)
            continue
        m = t.coords.mean(axis=0)
        r, c = facade_sets.world_to_pixel(np.array([m[0]]), np.array([m[1]]))
        r, c = int(r[0]), int(c[0])
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and cover[r, c] >= min_cover:
            out.append(t)
    return out


def apply_planar(result: PipelineResult, folder, resolution_m: float = 0.02) -> PipelineResult:
    """Use a wall-parallel orthomosaic + DEM (facade.planar) for the facade: register it to the cloud if not
    done yet, re-colour the facade image from it and detect the traces on it (sharper, straight from the
    photos), then redo the cross-check, the trace families and the kinematic screening."""
    from .facade import planar

    cfg = result.config
    dem, ortho = planar.find_planar_pair(folder)
    if dem is None or ortho is None:
        raise ValueError(f"No planar DEM + orthomosaic in {folder}")
    from . import heights

    crs = heights.reg_crs(cfg)
    reg = planar.load_registration(folder, crs)
    if reg is None:
        reg = planar.register(dem, np.asarray(result.point_cloud.points), result.wall_frame)
        reg["crs"] = crs                         # fitted to this cloud: in its height datum
        planar.save_registration(folder, reg)
    result.facade_rgb, share = planar.facade_rgb_from_planar(result, ortho, reg["matrix"])
    veg = cfg.get("vegetation", {})
    tcfg = {k: v for k, v in cfg.get("trace_mapping", {}).items() if k != "resolution_m"}
    traces = planar.detect_traces_planar(result, dem, ortho, reg["matrix"], resolution_m=resolution_m,
                                         exg_threshold=veg.get("exg_threshold", EXG_THRESHOLD),
                                         veg_on=veg.get("enabled", True), **tcfg)
    out_dir = cfg["paths"].get("output_dir")
    manual = load_manual_traces(Path(out_dir) / MANUAL_TRACES, result.facade_rgb) if out_dir else []
    lift_traces(manual, result.facade_depth, result.wall_frame)
    result.traces = _keep_traces(result, traces) + manual
    result.trace_confirmed = _check_traces(result, **cfg.get("trace_matching", {}))
    result.trace_families = trace_families(result.traces, result.wall_frame, result.discontinuity_sets)
    cfg["planar"] = {"folder": str(folder), "resolution_m": resolution_m, "dem": dem.name, "ortho": ortho.name,
                     "registration": reg.get("source", "icp"), "registration_median_m": reg["median_residual_m"],
                     "facade_recoloured": round(share, 3)}
    if result.face_windows:
        _kinematics(result)
    return result


def _build_facade(result, cloud, uvw_cloud, uvw_pcd, region, veg_on, veg_thr) -> None:
    """Facade image (full density when the LAS was thinned on load), set map, traces and trace families."""
    cfg, frame = result.config, result.wall_frame
    fcfg = cfg.get("facade", {})
    res = fcfg.get("resolution", 0.02)
    thinning = cfg.get("load_thinning", {}).get("step", 1)
    src = cfg["paths"].get("point_cloud")
    streaming = thinning > 1 and fcfg.get("full_density", True) and src
    if uvw_cloud is None:
        pts = np.asarray(cloud.points)
        # streaming from the file: the thinned cloud only sets the image extent -- a subsample does
        uvw_cloud = frame.to_wall(pts[::max(1, len(pts) // 2_000_000)] if streaming else pts)
    acc = FacadeAccumulator.covering(uvw_cloud, res, margin=0.1 if streaming else res)
    if streaming:
        # drawn from every point of the file, not the thinned cloud: no speckle of empty pixels
        geoid = _needs_geoid(cfg, src)
        from . import heights

        box = _region_box(cfg)
        for xyz, rgb in iter_las_chunks(src):
            if box is not None:               # a section of a larger scan: skip the rest before the wall frame
                m = np.all((xyz[:, :2] >= box[0]) & (xyz[:, :2] <= box[1]), axis=1)
                xyz, rgb = xyz[m], (rgb[m] if rgb is not None else None)
            if geoid:
                xyz = heights.to_orthometric(xyz, cfg["crs"])
            uvw = frame.to_wall(xyz)
            keep = region_mask(uvw, region, xyz) if is_active(region) else np.ones(len(uvw), bool)
            acc.add(uvw[keep], rgb[keep] if rgb is not None else None)
        cfg["facade_full_density"] = True
    else:
        acc.add(uvw_cloud, np.asarray(cloud.colors))
    result.facade_rgb, result.facade_depth = acc.finish(fcfg.get("fill_iterations", 2))
    del acc

    result.facade_sets = _set_raster(result, uvw_pcd)

    from .facade import planar

    pl_folder = (cfg.get("planar") or {}).get("folder")
    if pl_folder and all(planar.find_planar_pair(pl_folder)):
        return                                 # traces come from the sharper planar rasters (apply_planar)
    depth = result.facade_depth.array[0]
    valid = np.isfinite(depth)
    if veg_on:  # no lineaments from leaf edges
        valid &= ~vegetation_mask(np.moveaxis(result.facade_rgb.array, 0, -1), veg_thr, scale=255.0)
    traces = extract_traces(result.facade_rgb, valid_mask=valid, depth=depth, **cfg.get("trace_mapping", {}))
    out_dir = cfg["paths"].get("output_dir")
    if out_dir:
        traces += load_manual_traces(Path(out_dir) / MANUAL_TRACES, result.facade_rgb)
    lift_traces(traces, result.facade_depth, frame)
    result.traces = _keep_traces(result, traces)
    result.trace_confirmed = _check_traces(result, **cfg.get("trace_matching", {}))
    result.trace_families = trace_families(result.traces, frame, result.discontinuity_sets)


def _set_raster(result, uvw_pcd) -> Raster:
    """Set id of the front-most analysis point per facade cell (cell = the analysis voxel, at least the pixel)."""
    cfg = result.config
    res = cfg.get("facade", {}).get("resolution", 0.02)
    tf = result.facade_depth.transform
    set_res = max(res, cfg.get("preprocessing", {}).get("voxel_size", res))
    shape = result.facade_depth.array.shape[1:]
    sets_acc = FacadeAccumulator(tf.c, tf.f, int(np.ceil(shape[0] * res / set_res)), int(np.ceil(shape[1] * res / set_res)),
                                 set_res, with_labels=True)
    sets_acc.add(uvw_pcd, labels=result.point_labels[:, 0].astype(np.int16))
    return sets_acc.label_raster()


def _after_cleaning(result, uvw_pcd) -> None:
    """Set map, trace cross-check and trace families again, once faces were left out of the sets."""
    cfg = result.config
    result.facade_sets = _set_raster(result, uvw_pcd)
    result.traces = _keep_traces(result, result.traces)
    result.trace_confirmed = _check_traces(result, **cfg.get("trace_matching", {}))
    result.trace_families = trace_families(result.traces, result.wall_frame, result.discontinuity_sets)


def lift_traces(traces: list, depth_raster: Raster, frame, window: int = 2) -> None:
    """World (x, y, z) end points of facade traces, from the median facade depth around each end."""
    d = depth_raster.array[0]
    h, w = d.shape
    for t in traces:
        ends = []
        for u, v in t.coords:
            r, c = depth_raster.world_to_pixel(np.array([u]), np.array([v]))
            r, c = int(r[0]), int(c[0])
            patch = d[max(r - window, 0):min(r + window + 1, h), max(c - window, 0):min(c + window + 1, w)]
            patch = patch[np.isfinite(patch)]
            if not len(patch):
                break
            ends.append([u, v, float(np.median(patch))])
        t.xyz = frame.to_world(np.array(ends)) if len(ends) == 2 else None


def load_manual_traces(path: Path, raster: Raster | None = None) -> list:
    """Hand-drawn traces (facade tool): CSV trace_id,u,v[,note], the vertices of each polyline in order."""
    if not Path(path).exists():
        return []
    df = pd.read_csv(path)
    out = []
    for tid, g in df.groupby("trace_id", sort=False):
        uv = g[["u", "v"]].to_numpy(float)
        for a, b in zip(uv[:-1], uv[1:]):
            coords = np.array([a, b])
            if raster is not None:
                rows, cols = raster.world_to_pixel(coords[:, 0], coords[:, 1])
                pix = np.column_stack([rows, cols]).astype(float)
            else:
                pix = np.zeros((2, 2))
            out.append(Trace2D(coords=coords, pixel_coords=pix, length_m=float(np.linalg.norm(b - a)),
                               group=100_000 + int(tid), source="manual"))
    return out


def _check_traces(result: PipelineResult, tolerance: float = 0.1, max_angle_deg: float = 20.0) -> dict:
    patches = [p for dset in result.discontinuity_sets for p in dset.patches]
    if not patches:
        return {}
    points = np.asarray(result.point_cloud.points)
    patch_ids = result.point_labels[:, 1]
    order = np.argsort(patch_ids, kind="stable")
    sorted_ids = patch_ids[order]
    rng = np.random.default_rng(0)

    footprints, expected = [], []
    for p in patches:
        lo, hi = np.searchsorted(sorted_ids, [p.patch_id, p.patch_id + 1])
        idx = order[lo:hi]
        if len(idx) > FOOTPRINT_SAMPLE:
            idx = rng.choice(idx, FOOTPRINT_SAMPLE, replace=False)
        footprints.append(result.wall_frame.to_wall(points[idx])[:, :2])
        u = result.wall_frame.to_wall(p.centroid)[0, 0]
        expected.append(result.wall_frame.apparent_trace_direction(p.normal, u))

    confirmed = match_traces_to_patches(result.traces, footprints, expected, tolerance, max_angle_deg)
    return {p.patch_id: c for p, c in zip(patches, confirmed)}


RESULT_PICKLE = "result.pkl"


def load_results(output_dir: str | Path) -> PipelineResult:
    """A saved run back in memory (result.pkl + the analysis cloud from classified_point_cloud.ply), so the
    dashboard can show it again without re-running."""
    import pickle

    import open3d as o3d

    class _Unpickler(pickle.Unpickler):
        # results saved by `python -m groundwater.pipeline` refer to __main__.PipelineResult
        def find_class(self, module, name):
            if module == "__main__" and name == "PipelineResult":
                return PipelineResult
            return super().find_class(module, name)

    import dataclasses

    output_dir = Path(output_dir)
    with open(output_dir / RESULT_PICKLE, "rb") as f:
        result = _Unpickler(f).load()
    for fld in dataclasses.fields(PipelineResult):        # results saved before a field was added
        if fld.name not in result.__dict__:
            default = fld.default_factory() if fld.default_factory is not dataclasses.MISSING else fld.default
            setattr(result, fld.name, None if default is dataclasses.MISSING else default)
    result.point_cloud = o3d.io.read_point_cloud(str(output_dir / "classified_point_cloud.ply"))
    _fix_map_crs(result)
    return result


def _fix_map_crs(result) -> None:
    """Results saved before site_folder.folder_crs skipped non-projected GeoTIFFs could record the geocentric CRS
    of a wall-parallel planar export (EPSG:4978) as the site's map CRS; the 3D overlays then land thousands of km
    away. Take the input cloud's own (LAS header) CRS instead, in memory."""
    crs = str(result.config.get("crs") or "")
    if crs not in ("EPSG:4978", "EPSG:4979", "EPSG:4326"):
        return
    src = (result.config.get("paths") or {}).get("point_cloud")
    if not src or not str(src).lower().endswith((".las", ".laz")) or not Path(src).exists():
        return
    import laspy

    from .site_folder import las_epsg

    with laspy.open(str(src)) as f:
        fixed = las_epsg(f.header)
    if fixed and fixed != crs:
        result.config["crs"] = fixed
        result.config["crs_fixed_from"] = crs


def save_results(result: PipelineResult, output_dir: str | Path) -> None:
    import dataclasses
    import pickle

    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = result.wall_frame
    # the exact settings of this run (incl. auto-scale values): load it as a config to repeat or tune the run
    (output_dir / "analysis_config.yaml").write_text(
        yaml.safe_dump(result.config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    sets_rows = []
    patches_rows = []
    for dset in result.discontinuity_sets:
        sets_rows.append(
            {
                "set_id": dset.set_id,
                "mean_dip": dset.mean_dip,
                "mean_dip_direction": dset.mean_dip_direction,
                "fisher_k": dset.fisher_k,
                "n_points": dset.n_points,
                "n_patches": len(dset.patches),
                "spacing_mean_m": dset.spacing_mean,
                "spacing_std_m": dset.spacing_std,
            }
        )
        for p in dset.patches:
            u, v, w = frame.to_wall(p.centroid)[0]
            patches_rows.append(
                {
                    "set_id": p.set_id,
                    "patch_id": p.patch_id,
                    "n_points": p.n_points,
                    "chainage_m": u,
                    "elevation_m": v,
                    "offset_m": w,
                    "centroid_x": p.centroid[0],
                    "centroid_y": p.centroid[1],
                    "centroid_z": p.centroid[2],
                    "dip": p.dip,
                    "dip_direction": p.dip_direction,
                    "roughness_rms_m": p.roughness_rms,
                    "persistence_length_m": p.persistence_length,
                    "area_m2": p.area,
                    "face_parallel": frame.apparent_trace_direction(p.normal, u) is None,
                    "trace_confirmed_2d": result.trace_confirmed.get(p.patch_id),
                }
            )

    pd.DataFrame(sets_rows).to_csv(output_dir / "discontinuity_sets.csv", index=False)
    pd.DataFrame(patches_rows).to_csv(output_dir / "discontinuity_patches.csv", index=False)

    with open(output_dir / "wall_frame.json", "w", encoding="utf-8") as f:
        json.dump({"polyline": frame.vertices.tolist(), "length_m": frame.length}, f, indent=2)
    if result.facade_rgb is not None:
        write_raster(output_dir / "facade_rgb.tif", result.facade_rgb)
        write_raster(output_dir / "facade_depth.tif", result.facade_depth)
        _save_facade_products(result, output_dir)
    # how each face lies to the excavated wall: parallel (bench / blasted face, or a joint along it), flat, cross
    from .discontinuity.exposure import classify_folder

    expo = result.config.get("exposure", {})
    classify_folder(output_dir, float(expo.get("parallel_deg", 20.0))).to_csv(
        output_dir / "discontinuity_patches.csv", index=False)

    if result.field_comparison is not None:
        result.field_comparison.to_csv(output_dir / "field_comparison.csv", index=False)
        with open(output_dir / "field_comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(summarize_errors(result.field_comparison), f, indent=2)

    colors = _colors_by_set(result.point_labels[:, 0])
    colored_pcd = o3d.geometry.PointCloud(result.point_cloud)
    colored_pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))
    o3d.io.write_point_cloud(str(output_dir / "classified_point_cloud.ply"), colored_pcd)

    np.save(output_dir / "point_labels.npy", result.point_labels)
    # everything but the Open3D cloud (stored above as classified_point_cloud.ply), for load_results
    with open(output_dir / RESULT_PICKLE, "wb") as f:
        pickle.dump(dataclasses.replace(result, point_cloud=None), f, protocol=pickle.HIGHEST_PROTOCOL)


FACADE_PREVIEW = "facade_preview.jpg"
FACADE_SETMAP = "facade_setmap.png"
FACADE_META = "facade_preview.json"


def _save_facade_products(result: PipelineResult, output_dir: Path) -> None:
    """Set map raster, traces / families / kinematics tables, the facade preview for the facade tool, and
    the HTML report."""
    from PIL import Image

    from . import report
    from .discontinuity.trace_sets import apparent_angle, families_table

    if result.facade_sets is not None:
        write_raster(output_dir / "facade_sets.tif", result.facade_sets)
    fam_of = {id(t): f.family_id for f in result.trace_families for t in f.members}
    rows = [{"source": t.source, "group": t.group, "family": fam_of.get(id(t)),
             "u1": t.coords[0, 0], "v1": t.coords[0, 1], "u2": t.coords[1, 0], "v2": t.coords[1, 1],
             "length_m": t.length_m, "apparent_angle_deg": apparent_angle(t)} for t in result.traces]
    pd.DataFrame(rows).to_csv(output_dir / "traces.csv", index=False)
    families_table(result.trace_families, result.wall_frame, result.discontinuity_sets).to_csv(
        output_dir / "trace_families.csv", index=False)
    pd.DataFrame(result.face_windows).to_csv(output_dir / "face_windows.csv", index=False)
    if result.kinematics is not None:
        result.kinematics.to_csv(output_dir / "kinematics.csv", index=False)

    rgb, ext = report.facade_preview(result)
    Image.fromarray(rgb).save(output_dir / FACADE_PREVIEW, quality=88)
    Image.fromarray(report.set_overlay(result, rgb.shape[:2]), "RGBA").save(output_dir / FACADE_SETMAP)
    (output_dir / FACADE_META).write_text(json.dumps({"u_min": ext[0], "u_max": ext[1], "v_min": ext[2],
                                                       "v_max": ext[3], "width": rgb.shape[1],
                                                       "height": rgb.shape[0]}), encoding="utf-8")
    try:
        save_hydro_tables(result, output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"Seepage / grouting tables not written: {e}")
    try:
        report.build_report(result, output_dir)
    except Exception as e:  # noqa: BLE001 -- a report problem must not lose the analysis results
        print(f"Report not written: {e}")


def save_hydro_tables(result: PipelineResult, output_dir: Path, smap=None, g=None) -> None:
    """seepage_zones.csv, seepage_cells.csv and grouting_*.csv (see groundwater.hydro)."""
    from . import hydro

    curtain = result.config.get("curtain") or {}
    smap = smap or hydro.seepage_potential(result)
    g = g or hydro.grouting_inputs(result, curtain.get("azimuth"))
    if smap is not None:
        smap.zones.to_csv(output_dir / "seepage_zones.csv", index=False)
        rows, cols = smap.classes.shape
        rr, cc = np.nonzero(smap.rock)
        cells = pd.DataFrame({"chainage_m": smap.u_min + (cc + 0.5) * smap.cell_m,
                              "elevation_m": smap.v_max - (rr + 0.5) * smap.cell_m,
                              "class": smap.classes[rr, cc],
                              **{k: smap.layers[k][rr, cc] for k in ("score", "p21", "openness", "connectivity", "wetness")}})
        cells.round(4).to_csv(output_dir / "seepage_cells.csv", index=False)
    if g:
        g["sets"].to_csv(output_dir / "grouting_sets.csv", index=False)
        g["principal"].to_csv(output_dir / "grouting_permeability_axes.csv", index=False)
        pd.DataFrame([{"case": "최적", **g["best"]}, {"case": "연직공", **g["vertical"]}]
                     + ([{"case": "커튼 면 안 최적", **g["in_plane"]}] if g.get("in_plane") else [])
                     ).to_csv(output_dir / "grouting_hole_orientation.csv", index=False)
        if "curtain" in g:
            g["curtain"].to_csv(output_dir / "grouting_curtain_check.csv", index=False)


def _colors_by_set(set_ids: np.ndarray) -> np.ndarray:
    from .report import SET_COLORS            # one palette for the report, the dashboard and the PLY

    colors = np.full((len(set_ids), 3), 0.6)  # gray for unassigned
    for sid in set(set_ids):
        if sid == -1:
            continue
        h = SET_COLORS[int(sid) % len(SET_COLORS)].lstrip("#")
        colors[set_ids == sid] = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return colors


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 1 discontinuity-extraction pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Path to a site YAML config")
    src.add_argument("--folder", help="Site data folder: input files found automatically, scale set from density")
    args = parser.parse_args()

    if args.folder:
        from .site_folder import build_config, scan

        found = scan(args.folder)
        pick = lambda k: found[k][0] if found[k] else None  # noqa: E731
        cfg = build_config(args.folder, point_cloud=pick("point_cloud"), mesh=pick("mesh"), field=pick("field"),
                           lidar=pick("lidar"), thermal=pick("thermal"), wall_line=found["wall_line"])
    else:
        cfg = load_config(args.config)
    result = run_pipeline(cfg)
    output_dir = cfg["paths"].get("output_dir", "data/processed/output")
    save_results(result, output_dir)
    if result.config.get("auto_scale_applied"):
        print("Auto scale:", result.config["auto_scale_applied"])
    for f in result.trace_families:
        tag = ("Set " + " or ".join(str(c) for c in f.candidate_sets)) if f.linked_set is not None else "trace-only"
        est = "" if f.dip is None else f", est. {f.dip:.0f}/{f.dip_direction:03.0f} (q {f.orientation_quality})"
        print(f"  Trace family F{f.family_id}: {f.apparent_angle:.0f} deg, {f.n_traces} traces, {f.total_length_m:.0f} m, {tag}{est}")
    if result.kinematics is not None and len(result.kinematics):
        print("Kinematics:", result.kinematics.groupby("mode").size().to_dict())

    print(f"Wall frame: {len(result.wall_frame.vertices)}-vertex polyline, {result.wall_frame.length:.1f} m of chainage")
    print(f"Extracted {len(result.discontinuity_sets)} discontinuity sets")
    for dset in result.discontinuity_sets:
        print(
            f"  Set {dset.set_id}: dip={dset.mean_dip:.1f} dip_dir={dset.mean_dip_direction:.1f} "
            f"n_patches={len(dset.patches)} spacing={dset.spacing_mean}"
        )
    if result.facade_rgb is None:
        print("Point cloud has no colors: facade orthophoto and trace cross-check skipped")
    else:
        states = list(result.trace_confirmed.values())
        print(
            f"Facade traces: {len(result.traces)} detected; patches confirmed={states.count(True)} "
            f"unconfirmed={states.count(False)} face-parallel(n/a)={states.count(None)}"
        )
    if result.field_comparison is not None:
        print("Field comparison summary:", summarize_errors(result.field_comparison))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    # run through the package module so pickled results name groundwater.pipeline.PipelineResult
    from groundwater.pipeline import main as _main

    _main()
