"""2D discontinuity-trace extraction from the facade orthophoto, and a
cross-check of the 3D plane-based extraction against it.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..io.raster import Raster


@dataclass
class Trace2D:
    coords: np.ndarray  # (2, 2) endpoint map coordinates; on a facade raster: (chainage, elevation)
    pixel_coords: np.ndarray  # (2, 2) [[row1, col1], [row2, col2]]
    length_m: float


def extract_traces(
    raster: Raster,
    valid_mask: np.ndarray | None = None,
    blur_sigma_px: float = 1.5,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_min_line_length_px: int = 20,
    hough_max_line_gap_px: int = 5,
) -> list[Trace2D]:
    """Detect lineaments (candidate discontinuity traces) with Canny + probabilistic Hough.

    The image is Gaussian-blurred first: cv2.Canny does no smoothing of its own,
    so without it pixel-level texture turns into a carpet of short spurious
    lines. A line needs as many edge-pixel votes as its minimum length. Edges
    within two pixels of a no-data area are discarded -- the boundary of missing
    data is a straight, high-contrast edge that is not a discontinuity.
    """
    gray = _to_grayscale(raster.array, valid_mask)
    if blur_sigma_px > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma_px)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if valid_mask is not None:
        near_nodata = cv2.dilate((~valid_mask).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        edges[near_nodata > 0] = 0

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=hough_min_line_length_px,
        minLineLength=hough_min_line_length_px, maxLineGap=hough_max_line_gap_px,
    )
    if lines is None:
        return []

    traces = []
    for col1, row1, col2, row2 in lines.reshape(-1, 4):
        x, y = raster.pixel_to_world(np.array([row1, row2]), np.array([col1, col2]))
        coords = np.column_stack([x, y])
        traces.append(
            Trace2D(
                coords=coords,
                pixel_coords=np.array([[row1, col1], [row2, col2]]),
                length_m=float(np.linalg.norm(coords[0] - coords[1])),
            )
        )
    return traces


def _to_grayscale(array: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """array is (bands, rows, cols). Contrast is stretched between the 1st and
    99th percentile of valid pixels, so a few extreme pixels cannot set the scale."""
    band = (array[0] if array.shape[0] == 1 else array[:3].mean(axis=0)).astype(np.float64)
    valid = np.ones(band.shape, dtype=bool) if valid_mask is None else valid_mask
    if not valid.any():
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(band[valid], [1, 99])
    band = np.where(valid, band, np.median(band[valid]))
    scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0 if hi > lo else np.zeros_like(band)
    return scaled.astype(np.uint8)


def match_traces_to_patches(
    traces: list[Trace2D],
    patch_footprints: list[np.ndarray],
    expected_directions: list[np.ndarray | None],
    tolerance: float,
    max_angle_deg: float = 20.0,
) -> list[bool | None]:
    """Does an independent 2D trace corroborate each 3D discontinuity patch?

    A patch is confirmed when some trace passes within `tolerance` of its
    footprint (its points in map coordinates) AND runs within `max_angle_deg`
    of the patch's expected trace direction. Proximity alone is meaningless on a
    rock face, where edges are everywhere. Patches with no expected direction
    (planes parallel to the wall) get None: the check does not apply to them.
    """
    if not traces:
        return [None if d is None else False for d in expected_directions]

    step = tolerance / 2.0
    samples, owners, directions = [], [], []
    for k, trace in enumerate(traces):
        a, b = trace.coords
        n = int(np.ceil(np.linalg.norm(b - a) / step)) + 1
        samples.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
        owners.append(np.full(n, k))
        directions.append((b - a) / max(np.linalg.norm(b - a), 1e-12))
    tree = cKDTree(np.vstack(samples))
    owners = np.concatenate(owners)
    directions = np.array(directions)
    min_cos = np.cos(np.radians(max_angle_deg))

    confirmed: list[bool | None] = []
    for footprint, expected in zip(patch_footprints, expected_directions):
        if expected is None:
            confirmed.append(None)
            continue
        hits = tree.query_ball_point(footprint, r=tolerance)
        nearby = np.unique(owners[np.concatenate([np.asarray(h, dtype=int) for h in hits])]) if len(footprint) else []
        confirmed.append(bool(len(nearby)) and bool(np.any(np.abs(directions[nearby] @ expected) >= min_cos)))
    return confirmed
