"""Seepage potential under rockfall netting (groundwater.hydro): the cells under the net are a ranking of their
own, by the groove depth of the analysed rock (the LiDAR layer behind the net), not by the net's front surface."""
from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

from groundwater import hydro
from groundwater.discontinuity.cleaning import MASK_NET
from groundwater.io.raster import Raster
from groundwater.viz import seepage_view

PX = 0.05                 # facade pixel (m)
LENGTH, HEIGHT = 20.0, 10.0
NET_FROM = 10.0           # the net hangs over chainage 10..20
GROOVES = (14.0, 16.0)    # behind the net, the rock is grooved between these chainages


class _Frame:
    """A straight wall along x: u = x, v = z, w = y."""

    def to_wall(self, xyz):
        xyz = np.atleast_2d(xyz)
        return np.column_stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]])

    def to_world(self, uvw):
        uvw = np.atleast_2d(uvw)
        return np.column_stack([uvw[:, 0], uvw[:, 2], uvw[:, 1]])


def _raster(a, res=PX):
    return Raster(array=a[None] if a.ndim == 2 else a, transform=Affine(res, 0, 0.0, 0, -res, HEIGHT), crs=None,
                  nodata=None)


@pytest.fixture
def wall():
    H, W = int(HEIGHT / PX), int(LENGTH / PX)
    u = (np.arange(W) + 0.5) * PX
    v = HEIGHT - (np.arange(H) + 0.5) * PX
    U, V = np.meshgrid(u, v)
    # the rock: flat, with 10 cm wide, 12 cm deep grooves every 40 cm in one band behind the net -- and shallow
    # ones (3 cm) in a band of the bare part, so that the bare ranking has something to rank too
    rock = np.zeros((H, W), np.float32)
    rock[(U >= GROOVES[0]) & (U < GROOVES[1]) & ((U * 100).astype(int) % 40 < 10)] = -0.12
    rock[(U >= 3.0) & (U < 5.0) & ((U * 100).astype(int) % 40 < 10)] = -0.03
    # what the facade shows: the rock on the bare part, a smooth net 0.5 m in front of it on the other
    front = np.where(U >= NET_FROM, 0.5, rock).astype(np.float32)
    grey = np.full((3, H, W), 128, np.uint8)
    mask = np.zeros((int(HEIGHT / 0.1), int(LENGTH / 0.1)), np.uint8)
    mask[:, int(NET_FROM / 0.1):] = MASK_NET
    pts = np.column_stack([U.ravel(), rock.ravel(), V.ravel()])          # world (x, y, z) = (u, w, v)
    traces = [SimpleNamespace(coords=np.array([[x, 2.0], [x, 8.0]]), length_m=6.0, source="auto", group=0)
              for x in (1.5, 2.5, 3.5, 4.5, 6.5)]
    return SimpleNamespace(
        facade_rgb=_raster(grey), facade_depth=_raster(front), facade_sets=_raster(np.zeros((H, W), np.int16)),
        surface_mask=_raster(mask, 0.1), point_cloud=SimpleNamespace(points=pts), wall_frame=_Frame(),
        traces=traces, trace_families=[], config={})


def test_netted_cells_are_ranked_on_their_own_by_the_rock_behind_the_net(wall):
    sm = hydro.seepage_potential(wall)
    cols = sm.classes.shape[1]
    u = (np.arange(cols) + 0.5) * sm.cell_m
    assert sm.under_net == "lidar" and sm.net is not None
    assert sm.net[:, u > NET_FROM].all() and not sm.net[:, u < NET_FROM].any()
    assert sm.rock.all()                                              # the whole wall is judged, in two rankings
    high_net = (sm.classes == 2) & sm.net
    assert high_net.any() and high_net[:, (u > GROOVES[0]) & (u < GROOVES[1])].sum() == high_net.sum()
    # the groove depth under the net is the rock's (cm), not the smooth net's (0)
    assert sm.layers["openness"][:, (u > GROOVES[0]) & (u < GROOVES[1])].min() > 0.01
    assert sm.layers["openness"][:, u > GROOVES[1]].max() < 0.005
    # each ranking has its own top cells: the net's take none of the bare rock's (this small wall has many equal
    # cells, so the share is not exactly 15 %)
    for cells in (sm.net, sm.rock & ~sm.net):
        assert 0.10 <= (sm.classes[cells] == 2).mean() <= 0.35
    z = sm.zones
    assert set(z["basis"]) == {hydro.BASIS_BARE, hydro.BASIS_NET}
    assert list(z["zone"]) == list(range(1, len(z) + 1))              # one numbering, the bare rock first
    assert z["basis"].tolist() == sorted(z["basis"].tolist(), key=[hydro.BASIS_BARE, hydro.BASIS_NET].index)
    assert z.loc[z["basis"] == hydro.BASIS_NET, "p21"].isna().all()   # no trace indicator under the net
    top = hydro.top_zones(z, 1)
    assert top["basis"].tolist() == [hydro.BASIS_BARE, hydro.BASIS_NET]


def test_other_treatments_of_the_netted_cells(wall):
    skip = hydro.seepage_potential(wall, under_net="skip")
    cols = skip.classes.shape[1]
    u = (np.arange(cols) + 0.5) * skip.cell_m
    assert skip.net is None and not skip.rock[:, u > NET_FROM].any() and skip.rock[:, u < NET_FROM].all()
    assert "basis" not in skip.zones
    front = hydro.seepage_potential(wall, under_net="front")          # as before: one ranking, the net's surface
    assert front.net is None and front.rock.all()
    assert front.layers["openness"][:, u > NET_FROM].max() == 0       # the grooves behind the net are not seen
    assert hydro.netting_cells(wall, front)[:, u > NET_FROM].all()
    lidar = hydro.seepage_potential(wall)
    bare = u < NET_FROM                                               # the bare rock: the same in 'skip' and 'lidar'
    assert (skip.classes[:, bare] == lidar.classes[:, bare]).all()
    with pytest.raises(ValueError):
        hydro.seepage_potential(wall, under_net="other")


def test_without_a_net_mask_nothing_changes(wall):
    wall.surface_mask = None
    sm = hydro.seepage_potential(wall)
    assert sm.net is None and sm.rock.all() and "basis" not in sm.zones


def test_the_two_rankings_are_drawn_apart(wall):
    sm = hydro.seepage_potential(wall)
    layers = seepage_view.grade_layers(sm)
    assert len(layers) == 4 and len({colour for _, _, colour in layers}) == 4
    for name, cells, _ in layers:
        assert (cells & sm.net).sum() == (cells.sum() if name.startswith("망 아래") else 0)
    assert np.isnan(seepage_view.layer_values(sm, "p21")[sm.net]).all()         # not measured under the net
    assert np.isfinite(seepage_view.layer_values(sm, "openness")[sm.net]).all()
    fig = seepage_view.map_figure(sm, "grade", np.zeros((40, 80, 3), np.uint8), [0, LENGTH, 0, HEIGHT])
    names = {t.name for t in fig.data if t.name}
    assert {"상 (상위 15%)", "망 아래 상 (라이다 홈 깊이 상위 15%)"} <= names
    assert len(seepage_view.grade_layers(hydro.seepage_potential(wall, under_net="front"))) == 2
