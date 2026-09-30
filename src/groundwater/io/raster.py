"""Raster I/O via rasterio.

A Raster's transform maps pixels to map coordinates: projected x/y for a
georeferenced orthomosaic, or (chainage, elevation) for a facade raster, which
has no CRS.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class Raster:
    array: np.ndarray  # shape (bands, rows, cols)
    transform: Affine
    crs: object
    nodata: float | None

    def pixel_to_world(self, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = rasterio.transform.xy(self.transform, row, col, offset="center")
        return np.asarray(x), np.asarray(y)

    def world_to_pixel(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = rasterio.transform.rowcol(self.transform, x, y)
        return np.asarray(rows), np.asarray(cols)


def load_raster(path: str | Path) -> Raster:
    with rasterio.open(str(path)) as src:
        array = src.read()
        return Raster(array=array, transform=src.transform, crs=src.crs, nodata=src.nodata)


def write_raster(path: str | Path, raster: Raster) -> None:
    bands, rows, cols = raster.array.shape
    with rasterio.open(
        str(path), "w", driver="GTiff", height=rows, width=cols, count=bands,
        dtype=raster.array.dtype, crs=raster.crs, transform=raster.transform, nodata=raster.nodata,
    ) as dst:
        dst.write(raster.array)
