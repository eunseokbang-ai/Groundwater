"""Grouping points into discontinuity sets (by orientation) and patches (by connectivity).

Orientation sets follow the stereonet kernel-density approach of Riquelme et al.
(2014, Discontinuity Set Extractor): smooth the pole density on the sphere, take
its peaks as set orientations, and assign each pole to the nearest peak inside a
cone. Two properties matter on a quarry wall:

- Poles are axial (n and -n are the same plane), so every comparison uses
  |n . m|. Folding to a hemisphere splits sub-vertical sets in two.
- Peaks are isolated by non-maximum suppression, so noisy poles bridging two
  sets cannot chain them into one cluster the way DBSCAN does on rough rock.

Everything here is O(n) in the number of points.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import axial_mean

_ASSIGN_CHUNK = 1_000_000


def _chord(angle_rad: float) -> float:
    return 2.0 * np.sin(angle_rad / 2.0)


def hemisphere_grid(spacing_deg: float) -> np.ndarray:
    """Near-uniform nodes on the upper hemisphere (one node per axis), Fibonacci lattice."""
    spacing = np.radians(spacing_deg)
    n_full = max(int(round(4.0 * np.pi / spacing**2)), 8)
    i = np.arange(n_full) + 0.5
    z = 1.0 - 2.0 * i / n_full
    theta = np.pi * (1.0 + 5.0**0.5) * i
    r = np.sqrt(1.0 - z**2)
    nodes = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return nodes[z > 0]


def assign_to_axes(normals: np.ndarray, axes: np.ndarray, cone_angle_deg: float) -> np.ndarray:
    """Label each pole with its nearest axis, or -1 if it lies outside every cone."""
    labels = np.full(len(normals), -1, dtype=int)
    if len(axes) == 0:
        return labels
    min_cos = np.cos(np.radians(cone_angle_deg))
    for start in range(0, len(normals), _ASSIGN_CHUNK):
        cos = np.abs(normals[start:start + _ASSIGN_CHUNK] @ axes.T)
        best = cos.argmax(axis=1)
        inside = cos[np.arange(len(best)), best] >= min_cos
        labels[start:start + _ASSIGN_CHUNK] = np.where(inside, best, -1)
    return labels


def pole_density(
    normals: np.ndarray, nodes: np.ndarray, bandwidth_deg: float, neighbor_radius_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Axial Gaussian kernel density of poles evaluated at every grid node.

    Returns (density, pair_i, pair_j, pair_angle) where the pairs list every
    distinct node pair within max(3 * bandwidth, neighbor_radius) -- reused by the
    caller for peak detection.
    """
    g = len(nodes)
    axial_tree = cKDTree(np.vstack([nodes, -nodes]))
    _, nearest = axial_tree.query(normals)
    counts = np.bincount(nearest % g, minlength=g).astype(float)

    sigma = np.radians(bandwidth_deg)
    radius = max(3.0 * sigma, np.radians(neighbor_radius_deg))
    pairs = cKDTree(nodes).sparse_distance_matrix(axial_tree, _chord(radius), output_type="ndarray")
    pair_i, pair_j = pairs["i"], pairs["j"] % g
    pair_angle = 2.0 * np.arcsin(np.clip(pairs["v"] / 2.0, 0.0, 1.0))
    distinct = pair_i != pair_j
    pair_i, pair_j, pair_angle = pair_i[distinct], pair_j[distinct], pair_angle[distinct]

    in_kernel = pair_angle <= 3.0 * sigma
    weights = np.exp(-pair_angle[in_kernel] ** 2 / (2.0 * sigma**2)) * counts[pair_j[in_kernel]]
    density = counts + np.bincount(pair_i[in_kernel], weights=weights, minlength=g)
    return density, pair_i, pair_j, pair_angle


CANDIDATE_CONTRAST = 2.0      # a density peak at least this (x median) that is not a set is reported as a candidate
CANDIDATE_RADIUS_DEG = 10.0   # ... if it is the densest node within this
CANDIDATE_SEPARATION_DEG = 15.0
MAX_CANDIDATES = 5
FIXED_DUPLICATE_DEG = 10.0    # an automatic set this close to a user set is the same set


def plane_normal(dip: float, dip_direction: float) -> np.ndarray:
    """Upward unit normal (= pole) of a plane given as dip / dip direction (deg)."""
    d, a = np.radians(dip), np.radians(dip_direction)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def find_orientation_sets(
    normals: np.ndarray,
    grid_spacing_deg: float = 1.5,
    bandwidth_deg: float = 6.0,
    min_separation_deg: float = 20.0,
    cone_angle_deg: float = 25.0,
    min_set_fraction: float = 0.003,
    min_set_points: int = 30,
    min_peak_contrast: float = 3.0,
    max_sets: int | None = None,
    fixed_sets=None,
    info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect discontinuity sets as peaks of the pole density.

    A peak becomes a set only if (a) it is the densest node within
    `min_separation_deg`, (b) its density is at least `min_peak_contrast` times the
    median node density -- the median tracks the scattered background, so random
    pole noise does not produce sets -- and (c) at least
    max(min_set_points, min_set_fraction * n) poles are assigned to it.

    `fixed_sets` [(dip, dip direction), ...]: sets the user pins -- always sets, at that orientation (not moved to
    the mean of their poles), kept with min_set_points poles; an automatic set within FIXED_DUPLICATE_DEG of one is
    dropped as the same set.

    `info` (a dict, filled in): "fixed" -- the set indices of the pinned sets; "candidates" -- density peaks that did
    not become sets but stand out (at least CANDIDATE_CONTRAST x the median and the densest node within
    CANDIDATE_RADIUS_DEG, min_separation_deg away from every set): dip, dip_direction, contrast, the share of the
    poles within 15 deg and why it is not a set. A borderline set (Iksan west wall 2026-10: 48/047 at 2.6-2.7x on a
    ridge toward 69/044) then shows as a candidate instead of coming and going between runs.

    Returns (labels, axes): labels[i] is the set index of pole i (-1 = none);
    axes[k] is the unit mean pole of set k. Sets are ordered by support, largest
    first.
    """
    info = {} if info is None else info
    info["fixed"], info["candidates"] = [], []
    no_sets = (np.full(len(normals), -1, dtype=int), np.empty((0, 3)))
    if len(normals) == 0:
        return no_sets

    nodes = hemisphere_grid(grid_spacing_deg)
    density, pair_i, pair_j, pair_angle = pole_density(normals, nodes, bandwidth_deg, min_separation_deg)

    near = pair_angle <= np.radians(min_separation_deg)
    neighbor_max = np.zeros(len(nodes))
    np.maximum.at(neighbor_max, pair_i[near], density[pair_j[near]])
    background = np.median(density)
    is_peak = (density >= neighbor_max) & (density > 0) & (density >= min_peak_contrast * background)

    candidates = np.nonzero(is_peak)[0]
    candidates = candidates[np.argsort(-density[candidates])]
    min_sep_cos = np.cos(np.radians(min_separation_deg))
    fixed = np.array([plane_normal(d, a) for d, a in (fixed_sets or [])]).reshape(-1, 3)
    dup_cos = np.cos(np.radians(FIXED_DUPLICATE_DEG))
    peaks: list[int] = []
    for c in candidates:
        if all(abs(nodes[c] @ nodes[p]) < min_sep_cos for p in peaks) and not np.any(np.abs(fixed @ nodes[c]) >= dup_cos):
            peaks.append(int(c))
        if max_sets is not None and len(peaks) >= max_sets:
            break
    if not peaks and not len(fixed):
        return no_sets

    min_support = max(min_set_points, min_set_fraction * len(normals))
    axes = np.vstack([fixed, nodes[peaks]])
    pinned = np.r_[np.ones(len(fixed), bool), np.zeros(len(peaks), bool)]
    for _ in range(5):
        labels = assign_to_axes(normals, axes, cone_angle_deg)
        support = np.bincount(labels[labels >= 0], minlength=len(axes))
        keep = np.nonzero(np.where(pinned, support >= min_set_points, support >= min_support))[0]
        if len(keep) == 0:
            return no_sets
        refined = np.array([axes[k] if pinned[k] else axial_mean(normals[labels == k])[0] for k in keep])
        converged = len(keep) == len(axes) and np.all(np.abs(np.sum(refined * axes, axis=1)) > np.cos(np.radians(0.1)))
        axes, pinned = refined, pinned[keep]
        if converged:
            break

    labels = assign_to_axes(normals, axes, cone_angle_deg)
    support = np.bincount(labels[labels >= 0], minlength=len(axes))
    order = np.argsort(-support)
    axes, pinned = axes[order], pinned[order]
    remap = np.full(len(order) + 1, -1)
    remap[order] = np.arange(len(order))
    labels = remap[labels]  # -1 indexes the trailing -1 entry
    info["fixed"] = [int(k) for k in np.flatnonzero(pinned)]
    info["candidates"] = _candidates(normals, nodes, density, background, pair_i, pair_j, pair_angle, axes,
                                     min_separation_deg, min_peak_contrast)
    return labels, axes


def _candidates(normals, nodes, density, background, pair_i, pair_j, pair_angle, axes, min_separation_deg,
                min_peak_contrast) -> list:
    """Density peaks that are not sets (see find_orientation_sets, `info`)."""
    from .geometry import normal_to_dip_dipdir

    def local_max(radius):
        m = pair_angle <= np.radians(radius)
        nmax = np.zeros(len(nodes))
        np.maximum.at(nmax, pair_i[m], density[pair_j[m]])
        return nmax

    near_sep, near_cand = local_max(min_separation_deg), local_max(CANDIDATE_RADIUS_DEG)
    ok = (density >= near_cand) & (density >= CANDIDATE_CONTRAST * background) & (density > 0)
    if len(axes):
        ok &= np.max(np.abs(nodes @ axes.T), axis=1) < np.cos(np.radians(min_separation_deg))
    out, chosen = [], []
    sup_cos = np.cos(np.radians(CANDIDATE_SEPARATION_DEG))
    sample = normals[:: max(1, len(normals) // 500_000)]
    for c in np.flatnonzero(ok)[np.argsort(-density[ok])]:
        if any(abs(nodes[c] @ nodes[k]) >= sup_cos for k in chosen):
            continue
        chosen.append(int(c))
        dip, dd = normal_to_dip_dipdir(nodes[c])
        contrast = float(density[c] / background) if background > 0 else 0.0
        why = []
        if contrast < min_peak_contrast:
            why.append(f"밀도 {contrast:.1f}배 < 기준 {min_peak_contrast:g}배")
        if density[c] < near_sep[c]:
            m = (pair_i == c) & (pair_angle <= np.radians(min_separation_deg))
            j = pair_j[m][np.argmax(density[pair_j[m]])]
            d2, a2 = normal_to_dip_dipdir(nodes[j])
            why.append(f"{min_separation_deg:g}° 안에 더 진한 곳 {d2:.0f}/{a2:03.0f} ({density[j] / background:.1f}배)")
        out.append({"dip": round(dip, 1), "dip_direction": round(dd, 1), "contrast": round(contrast, 2),
                    "share": round(float(np.mean(np.abs(sample @ nodes[c]) >= np.cos(np.radians(15.0)))), 4),
                    "reason": "; ".join(why) or "지지 점 부족"})
        if len(out) >= MAX_CANDIDATES:
            break
    return out


_HALF_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def voxel_connected_components(points: np.ndarray, link_distance: float) -> np.ndarray:
    """Split points into spatially connected patches.

    Points are binned into cubes of side `link_distance`; points in the same or
    26-adjacent cubes are connected. Consequently points closer than
    `link_distance` are always linked, and points more than 2*sqrt(3)*link_distance
    apart are never directly linked -- so parallel joints further apart than that
    stay separate patches. Memory is O(n), unlike DBSCAN's neighbor lists.
    """
    if len(points) == 0:
        return np.empty(0, dtype=int)
    keys = np.floor((points - points.min(axis=0)) / link_distance).astype(np.int64) + 1
    dims = keys.max(axis=0) + 2
    codes = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    voxels, inverse = np.unique(codes, return_inverse=True)

    rows, cols = [], []
    for dx, dy, dz in _HALF_NEIGHBOR_OFFSETS:
        target = voxels + (dx * dims[1] + dy) * dims[2] + dz
        pos = np.clip(np.searchsorted(voxels, target), 0, len(voxels) - 1)
        hit = voxels[pos] == target
        rows.append(np.nonzero(hit)[0])
        cols.append(pos[hit])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(voxels), len(voxels)))
    _, voxel_component = connected_components(graph, directed=False)
    return voxel_component[inverse.ravel()]
