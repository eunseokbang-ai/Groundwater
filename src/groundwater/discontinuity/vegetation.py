"""Colour-based vegetation filter: green plants on a rock face are dropped before the joint analysis.

Excess green in chromatic coordinates, ExG = (2G - R - B) / (R + G + B), is ~0 for grey/brown rock and
clearly positive for leaves, grass and moss. Tested on the Ulsan river cliff (facade image): 0.06 takes the
green canopy and moss mats while bare rock stays. Green must also be the strongest channel: yellow-brown
(iron-stained) rock has little blue and a positive ExG too, but red >= green. Dry brown vegetation is not
green and is left to the planarity filter (it is not planar); very dark points are kept because their
chromaticity is unreliable.
"""
import numpy as np

EXG_THRESHOLD = 0.06
MIN_BRIGHTNESS = 0.12        # R + G + B (0..3 scale) below which a point is too dark to judge


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """ExG of (N, 3) or (..., 3) colours in 0..1 (or 0..255; the index is scale-free)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    s = rgb.sum(axis=-1)
    return (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) / np.maximum(s, 1e-6)


def vegetation_mask(rgb: np.ndarray, threshold: float = EXG_THRESHOLD, scale: float = 1.0) -> np.ndarray:
    """True where a colour is vegetation. `scale` is the colour range maximum (1 for 0..1, 255 for 8-bit)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    bright = rgb.sum(axis=-1) / scale >= MIN_BRIGHTNESS
    out = np.zeros(rgb.shape[:-1], dtype=bool)
    # in blocks: a 130 M-point cloud would otherwise allocate several float temporaries of full size
    flat_rgb, flat_out, flat_bright = rgb.reshape(-1, 3), out.reshape(-1), bright.reshape(-1)
    for i in range(0, len(flat_rgb), 10_000_000):
        sl = slice(i, i + 10_000_000)
        c = flat_rgb[sl]
        flat_out[sl] = flat_bright[sl] & (excess_green(c) > threshold) & (c[:, 1] > c[:, 0]) & (c[:, 1] > c[:, 2])
    return out
