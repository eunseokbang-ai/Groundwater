"""Facade orthophoto: rasterize a colored point cloud onto the wall frame.

Building the image from the same point cloud used for discontinuity extraction
makes pixel <-> 3D correspondence exact by construction, with no dependence on
how Pix4D/Metashape define their planar-orthomosaic axes.
"""
import cv2
import numpy as np
from rasterio.transform import Affine

from ..io.raster import Raster


class FacadeAccumulator:
    """Front-most-point facade raster built in chunks, so a cloud that does not fit in memory at full
    density (streamed from the LAS) can still be rendered: each pixel keeps the point nearest the viewer
    (largest w) over everything added, with its colour and, optionally, an integer label."""

    def __init__(self, u_min: float, v_max: float, n_rows: int, n_cols: int, resolution: float,
                 with_labels: bool = False):
        self.u_min, self.v_max, self.res = u_min, v_max, resolution
        self.n_rows, self.n_cols = n_rows, n_cols
        self.depth = np.full(n_rows * n_cols, -np.inf, dtype=np.float32)
        self.rgb = np.zeros((n_rows * n_cols, 3), dtype=np.float32)
        self.labels = np.full(n_rows * n_cols, -1, dtype=np.int16) if with_labels else None

    @classmethod
    def covering(cls, uvw: np.ndarray, resolution: float, with_labels: bool = False, margin: float = 0.0):
        u0, u1 = uvw[:, 0].min() - margin, uvw[:, 0].max() + margin
        v0, v1 = uvw[:, 1].min() - margin, uvw[:, 1].max() + margin
        return cls(u0, v1, int(np.floor((v1 - v0) / resolution)) + 1, int(np.floor((u1 - u0) / resolution)) + 1,
                   resolution, with_labels)

    def add(self, uvw: np.ndarray, colors: np.ndarray | None = None, labels: np.ndarray | None = None) -> None:
        cols = np.floor((uvw[:, 0] - self.u_min) / self.res).astype(np.int64)
        rows = np.floor((self.v_max - uvw[:, 1]) / self.res).astype(np.int64)
        ok = (cols >= 0) & (cols < self.n_cols) & (rows >= 0) & (rows < self.n_rows)
        pixel, w = rows[ok] * self.n_cols + cols[ok], uvw[ok, 2].astype(np.float32)
        # z-buffer by an unbuffered per-pixel maximum (10x faster than sorting the chunk by pixel and depth):
        # a point wins its pixel if it is the new front-most one and in front of what the pixel held before
        before = self.depth[pixel]
        np.maximum.at(self.depth, pixel, w)
        win = (w >= self.depth[pixel]) & (w > before)
        px, idx = pixel[win], np.flatnonzero(ok)[win]
        if colors is not None:
            self.rgb[px] = colors[idx]
        if self.labels is not None and labels is not None:
            self.labels[px] = labels[idx]

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.u_min, 0.0, -self.res, self.v_max)

    def finish(self, fill_iterations: int = 2) -> tuple[Raster, Raster]:
        depth = np.where(np.isfinite(self.depth), self.depth, np.nan).astype(np.float32).reshape(self.n_rows, self.n_cols)
        rgb = self.rgb.reshape(self.n_rows, self.n_cols, 3)
        _fill_small_holes(depth, rgb, fill_iterations)
        rgb8 = np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        return (
            Raster(array=np.moveaxis(rgb8, -1, 0), transform=self.transform, crs=None, nodata=None),
            Raster(array=depth[None], transform=self.transform, crs=None, nodata=np.nan),
        )

    def label_raster(self) -> Raster:
        return Raster(array=self.labels.reshape(1, self.n_rows, self.n_cols), transform=self.transform, crs=None,
                      nodata=-1)


def rasterize_facade(
    uvw: np.ndarray, colors: np.ndarray, resolution: float, fill_iterations: int = 2, **_ignored
) -> tuple[Raster, Raster]:
    """Return (rgb, depth) rasters in wall coordinates (map x = chainage, map y = elevation).

    Each pixel takes the point nearest the viewer (largest w), so vegetation or
    netting in front of the rock hides what is behind it, as in a photo. Depth
    holds that point's w; NaN marks pixels with no data. Holes smaller than a
    couple of pixels are filled from their neighbours.
    """
    acc = FacadeAccumulator.covering(uvw, resolution)
    acc.add(uvw, colors)
    return acc.finish(fill_iterations)


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
