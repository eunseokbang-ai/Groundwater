"""Facade orthophoto: rasterize a colored point cloud onto the wall frame.

Building the image from the same point cloud used for discontinuity extraction
makes pixel <-> 3D correspondence exact by construction, with no dependence on
how Pix4D/Metashape define their planar-orthomosaic axes.
"""
import cv2
import numpy as np
from rasterio.transform import Affine

from ..io.raster import Raster


def rasterize_facade(
    uvw: np.ndarray, colors: np.ndarray, resolution: float, fill_iterations: int = 2
) -> tuple[Raster, Raster]:
    """Return (rgb, depth) rasters in wall coordinates (map x = chainage, map y = elevation).

    Each pixel takes the point nearest the viewer (largest w), so vegetation or
    netting in front of the rock hides what is behind it, as in a photo. Depth
    holds that point's w; NaN marks pixels with no data. Holes smaller than a
    couple of pixels are filled from their neighbours.
    """
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    u_min, v_max = u.min(), v.max()
    cols = np.floor((u - u_min) / resolution).astype(np.int64)
    rows = np.floor((v_max - v) / resolution).astype(np.int64)
    n_rows, n_cols = int(rows.max()) + 1, int(cols.max()) + 1
    pixel = rows * n_cols + cols

    order = np.lexsort((-w, pixel))
    first = np.r_[True, pixel[order][1:] != pixel[order][:-1]]
    front = order[first]

    depth = np.full(n_rows * n_cols, np.nan, dtype=np.float32)
    depth[pixel[front]] = w[front]
    rgb = np.zeros((n_rows * n_cols, 3), dtype=np.float32)
    rgb[pixel[front]] = colors[front]
    depth = depth.reshape(n_rows, n_cols)
    rgb = rgb.reshape(n_rows, n_cols, 3)

    _fill_small_holes(depth, rgb, fill_iterations)

    transform = Affine(resolution, 0.0, u_min, 0.0, -resolution, v_max)
    rgb8 = np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return (
        Raster(array=np.moveaxis(rgb8, -1, 0), transform=transform, crs=None, nodata=None),
        Raster(array=depth[None], transform=transform, crs=None, nodata=np.nan),
    )


def _fill_small_holes(depth: np.ndarray, rgb: np.ndarray, iterations: int) -> None:
    """Fill empty pixels that have >= 5 of 8 neighbours filled (in place).

    The 5-neighbour rule fills speckle and thin gaps between projected points but
    never creeps along the straight outer edge of the data, where pixels have at
    most 3 filled neighbours.
    """
    def neighbour_sum(a):
        return cv2.boxFilter(a, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)

    for _ in range(iterations):
        valid = np.isfinite(depth)
        if valid.all():
            return
        weight = valid.astype(np.float32)
        count = neighbour_sum(weight)
        fill = ~valid & (count >= 5)
        if not fill.any():
            return
        depth[fill] = (neighbour_sum(np.where(valid, depth, 0.0).astype(np.float32)) / np.maximum(count, 1))[fill]
        for c in range(3):
            rgb[..., c][fill] = (neighbour_sum(rgb[..., c] * weight) / np.maximum(count, 1))[fill]
