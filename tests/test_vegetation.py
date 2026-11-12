import numpy as np

from groundwater.discontinuity.vegetation import vegetation_mask


def test_green_plants_masked_rock_kept():
    colours = np.array([
        [0.60, 0.60, 0.60],   # grey rock
        [0.55, 0.42, 0.30],   # brown rock
        [0.47, 0.42, 0.26],   # yellow-brown (iron-stained) rock: positive ExG but red >= green
        [0.25, 0.45, 0.20],   # leaves
        [0.40, 0.48, 0.30],   # moss
        [0.02, 0.05, 0.01],   # too dark to judge -> kept
    ])
    assert vegetation_mask(colours).tolist() == [False, False, False, True, True, False]


def test_8bit_scale_matches_unit_scale():
    c = np.random.default_rng(0).random((1000, 3))
    # rounding to 8 bit may flip a point sitting exactly on the threshold, nothing more
    assert (vegetation_mask(c) != vegetation_mask(np.round(c * 255), scale=255.0)).mean() < 0.01
