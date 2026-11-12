import numpy as np

from groundwater.terrain.hydrology import (fill_depressions, flow_accumulation, flow_directions, strahler_order,
                                          trace_downstream, upstream_mask)


def v_valley(h=40, w=41):
    """Valley running south (rows grow southward) with its axis at the middle column."""
    rows, cols = np.indices((h, w))
    return 100 - rows * 0.5 + np.abs(cols - w // 2) * 1.0


def test_valley_routes_to_axis_and_outlet():
    z = v_valley()
    valid = np.ones_like(z, bool)
    filled = fill_depressions(z, valid)
    code = flow_directions(filled, valid, 5.0)
    acc = flow_accumulation(filled, code, valid)
    # the axis cell at the bottom collects (almost) the whole grid
    assert acc[-1, 20] > 0.9 * z.size / 2
    path = trace_downstream(code, 0, 0)
    assert path[-1][1] in (19, 20, 21) or path[-1][0] == z.shape[0] - 1


def test_pit_is_filled_and_drains():
    z = v_valley()
    z[20, 20] -= 30          # deep pit on the axis
    valid = np.ones_like(z, bool)
    filled = fill_depressions(z, valid)
    assert filled[20, 20] >= z[21, 20]         # raised to its spill level (lowest neighbour)
    code = flow_directions(filled, valid, 5.0)
    assert code[20, 20] != 0                   # no internal sink left
    acc = flow_accumulation(filled, code, valid)
    assert acc[-1, 20] > acc[19, 20]


def test_strahler_and_upstream():
    z = v_valley()
    valid = np.ones_like(z, bool)
    filled = fill_depressions(z, valid)
    code = flow_directions(filled, valid, 5.0)
    acc = flow_accumulation(filled, code, valid)
    streams = acc >= 20
    order = strahler_order(code, streams, filled)
    assert order.max() >= 2
    seed = np.zeros_like(valid)
    seed[-1, 20] = True
    up = upstream_mask(code, seed)
    assert up.sum() == acc[-1, 20]
