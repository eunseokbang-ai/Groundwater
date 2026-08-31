"""2D discontinuity-trace extraction from the orthomosaic, for cross-checking
against the 3D plane-based extraction (Phase 1, step 4 of the project plan).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ..io.raster import Raster


@dataclass
class Trace2D:
    world_coords: np.ndarray  # (2, 2) array: [[x1, y1], [x2, y2]]
    pixel_coords: np.ndarray  # (2, 2) array: [[row1, col1], [row2, col2]]
    length_m: float


def extract_traces(
    raster: Raster,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_min_line_length_px: int = 20,
    hough_max_line_gap_px: int = 5,
) -> list[Trace2D]:
    """Detect lineaments (candidate discontinuity traces) in the orthomosaic.

    Uses Canny edge detection followed by a probabilistic Hough transform --
    simple and dependency-light; swap for skimage's line-segment detector if
    trace quality on a specific site needs improvement.
    """
    gray = _to_grayscale(raster.array)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=hough_min_line_length_px,
        maxLineGap=hough_max_line_gap_px,
    )

    traces: list[Trace2D] = []
    if lines is None:
        return traces

    for line in lines.reshape(-1, 4):
        col1, row1, col2, row2 = line
        x, y = raster.pixel_to_world(np.array([row1, row2]), np.array([col1, col2]))
        world = np.column_stack([x, y])
        length_m = float(np.linalg.norm(world[0] - world[1]))
        traces.append(
            Trace2D(
                world_coords=world,
                pixel_coords=np.array([[row1, col1], [row2, col2]]),
                length_m=length_m,
            )
        )
    return traces


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    """array is (bands, rows, cols) as returned by rasterio."""
    if array.shape[0] == 1:
        band = array[0]
    else:
        band = array[:3].mean(axis=0)
    band = band.astype(np.float64)
    band -= band.min()
    max_val = band.max()
    if max_val > 0:
        band = band / max_val * 255.0
    return band.astype(np.uint8)


def match_traces_to_patches(traces: list[Trace2D], patch_xy: np.ndarray, tolerance_m: float) -> np.ndarray:
    """For each discontinuity patch centroid (xy), flag whether a 2D trace runs
    within `tolerance_m` of it -- a simple agreement check between the 3D
    geometric extraction and the independent 2D image-based extraction.

    Returns a boolean array aligned with patch_xy.
    """
    confirmed = np.zeros(len(patch_xy), dtype=bool)
    if not traces:
        return confirmed

    for i, xy in enumerate(patch_xy):
        for trace in traces:
            if _point_to_segment_distance(xy, trace.world_coords[0], trace.world_coords[1]) <= tolerance_m:
                confirmed[i] = True
                break
    return confirmed


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ab_len_sq = ab @ ab
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip((p - a) @ ab / ab_len_sq, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))
