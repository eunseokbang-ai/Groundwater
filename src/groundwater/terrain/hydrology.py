"""Surface-water routing on a gridded DEM (D8): depression filling, flow directions, flow
accumulation (D8 and MFD), stream extraction with Strahler order, downstream tracing, upstream
catchments, outlet sub-basins and the topographic wetness index.

Direction codes: 0 = no outflow (outlet / nodata / edge), 1..8 = E, SE, S, SW, W, NW, N, NE
(row offsets grow southward, as in the raster).

The per-cell loops are compiled with numba, so regional DEMs (10^7 cells) run in seconds.
"""
import heapq

import numpy as np
from numba import njit

# code -> (drow, dcol); index 0 unused
D8 = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
DIST = [0, 1, 2 ** 0.5, 1, 2 ** 0.5, 1, 2 ** 0.5, 1, 2 ** 0.5]
_DR = np.array([d[0] for d in D8], np.int64)
_DC = np.array([d[1] for d in D8], np.int64)
_DIST = np.array(DIST)


# --- compiled kernels ---------------------------------------------------------------------------

@njit(cache=True)
def _priority_flood(z, valid, eps):
    h, w = z.shape
    filled = z.copy()
    closed = ~valid
    heap = [(0.0, 0, 0)]
    heap.pop()
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            seed = r == 0 or c == 0 or r == h - 1 or c == w - 1
            if not seed:
                for k in range(1, 9):
                    if not valid[r + _DR[k], c + _DC[k]]:
                        seed = True
                        break
            if seed:
                heapq.heappush(heap, (filled[r, c], r, c))
                closed[r, c] = True
    while len(heap) > 0:
        zc, r, c = heapq.heappop(heap)
        for k in range(1, 9):
            rr, cc = r + _DR[k], c + _DC[k]
            if 0 <= rr < h and 0 <= cc < w and not closed[rr, cc]:
                closed[rr, cc] = True
                if filled[rr, cc] <= zc:
                    filled[rr, cc] = zc + eps
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


@njit(cache=True)
def _accumulate(order, down, acc):
    for i in order:                       # upstream first
        j = down[i]
        if j >= 0 and acc[i] > 0:
            acc[j] += acc[i]
    return acc


@njit(cache=True)
def _topo_order(down):
    """Cells ordered upstream -> downstream (Kahn's algorithm on the receiver graph)."""
    n = down.size
    indeg = np.zeros(n, np.int64)
    for i in range(n):
        if down[i] >= 0:
            indeg[down[i]] += 1
    order = np.empty(n, np.int64)
    head = tail = 0
    for i in range(n):
        if indeg[i] == 0:
            order[tail] = i
            tail += 1
    while head < tail:
        i = order[head]
        head += 1
        j = down[i]
        if j >= 0:
            indeg[j] -= 1
            if indeg[j] == 0:
                order[tail] = j
                tail += 1
    return order[:tail]


@njit(cache=True)
def _propagate_up(order, down, values):
    """values[i] = values[down[i]] for unset cells (-1), walking downstream -> upstream."""
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        j = down[i]
        if values[i] < 0 and j >= 0:
            values[i] = values[j]
    return values


@njit(cache=True)
def _strahler(order, down, streams):
    n = down.size
    out = np.zeros(n, np.int16)
    top = np.zeros(n, np.int16)
    count = np.zeros(n, np.int16)
    for i in order:
        if not streams[i]:
            continue
        if top[i] == 0:
            o = 1
        else:
            o = top[i] + (1 if count[i] >= 2 else 0)
        out[i] = o
        j = down[i]
        if j >= 0 and streams[j]:
            if o > top[j]:
                top[j] = o
                count[j] = 1
            elif o == top[j]:
                count[j] += 1
    return out


@njit(cache=True)
def _mfd(filled, valid, cell, p, order):
    h, w = filled.shape
    acc = np.zeros(h * w)
    for i in range(h * w):
        if valid.ravel()[i]:
            acc[i] = 1.0
    fz = filled.ravel()
    wts = np.zeros(8)
    for i in order:                       # high to low
        a = acc[i]
        if a == 0:
            continue
        r, c = i // w, i % w
        tot = 0.0
        for k in range(1, 9):
            rr, cc = r + _DR[k], c + _DC[k]
            wts[k - 1] = 0.0
            if 0 <= rr < h and 0 <= cc < w and valid[rr, cc]:
                s = (fz[i] - fz[rr * w + cc]) / (_DIST[k] * cell)
                if s > 0:
                    wts[k - 1] = s ** p
                    tot += wts[k - 1]
        if tot > 0:
            for k in range(1, 9):
                if wts[k - 1] > 0:
                    acc[(r + _DR[k]) * w + c + _DC[k]] += a * wts[k - 1] / tot
    return acc.reshape(h, w)


# --- public API ---------------------------------------------------------------------------------

def fill_depressions(z: np.ndarray, valid: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-flood fill (Barnes et al. 2014, +epsilon) so every valid cell drains to the edge
    or to nodata. Returns the filled surface (nodata cells untouched)."""
    return _priority_flood(z.astype(np.float64), valid.astype(np.bool_), eps)


def flow_directions(filled: np.ndarray, valid: np.ndarray, cell: float) -> np.ndarray:
    """D8 steepest descent on a filled surface (uint8 codes, see module docstring)."""
    h, w = filled.shape
    zp = np.pad(np.where(valid, filled, np.inf), 1, constant_values=np.inf)
    best = np.zeros((h, w))
    code = np.zeros((h, w), np.uint8)
    for k in range(1, 9):
        dr, dc = D8[k]
        nb = zp[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        drop = (filled - nb) / (DIST[k] * cell)
        better = np.isfinite(nb) & (drop > best)
        best[better] = drop[better]
        code[better] = k
    code[~valid] = 0
    return code


def downstream_index(code: np.ndarray) -> np.ndarray:
    """Flat index of each cell's receiver (-1 where there is none)."""
    h, w = code.shape
    rows, cols = np.indices((h, w))
    rr, cc = rows + _DR[code], cols + _DC[code]
    idx = rr * w + cc
    idx[(code == 0) | (rr < 0) | (rr >= h) | (cc < 0) | (cc >= w)] = -1
    return idx.ravel().astype(np.int64)


def upstream_first(code: np.ndarray) -> np.ndarray:
    """Flat cell indices ordered so every cell comes before its receiver."""
    return _topo_order(downstream_index(code))


def flow_accumulation(filled: np.ndarray, code: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (itself included)."""
    acc = valid.ravel().astype(np.float64)
    return _accumulate(upstream_first(code), downstream_index(code), acc).reshape(code.shape)


def flow_accumulation_mfd(filled: np.ndarray, valid: np.ndarray, cell: float, p: float = 1.1) -> np.ndarray:
    """Multiple-flow-direction accumulation (Freeman 1991): each cell passes its flow to every lower
    neighbour in proportion to slope**p. Spreads flow over planar slopes instead of the parallel
    single-cell lines D8 draws there; use it for display, D8 for divides and channels."""
    order = np.argsort(-np.where(valid, filled, -np.inf).ravel(), kind="stable").astype(np.int64)
    return _mfd(filled.astype(np.float64), valid.astype(np.bool_), float(cell), float(p), order)


def strahler_order(code: np.ndarray, streams: np.ndarray, filled: np.ndarray = None) -> np.ndarray:
    return _strahler(upstream_first(code), downstream_index(code), streams.ravel()).reshape(code.shape)


def stream_segments(code: np.ndarray, streams: np.ndarray, order: np.ndarray, acc: np.ndarray):
    """Stream polylines as lists of (row, col), split at confluences. Yields (cells, order, max_acc)."""
    h, w = code.shape
    down = downstream_index(code)
    s = streams.ravel()
    idx = np.nonzero(s)[0]
    inflow = np.zeros(s.size, np.int16)
    j = down[idx]
    ok = (j >= 0)
    ok[ok] &= s[j[ok]]
    np.add.at(inflow, j[ok], 1)
    flat_order, flat_acc = order.ravel(), acc.ravel()
    for i in idx[inflow[idx] != 1]:        # heads (0) and confluences (>=2)
        cells = [i]
        j = down[i]
        while j >= 0 and s[j]:
            cells.append(j)
            if inflow[j] != 1:
                break
            j = down[j]
        yield [(k // w, k % w) for k in cells], int(flat_order[i]), float(flat_acc[cells[-1]])


def trace_downstream(code: np.ndarray, row: int, col: int, max_steps: int = 10 ** 6):
    path = [(row, col)]
    h, w = code.shape
    for _ in range(max_steps):
        k = code[row, col]
        if k == 0:
            break
        row, col = row + D8[k][0], col + D8[k][1]
        if not (0 <= row < h and 0 <= col < w):
            break
        path.append((row, col))
    return path


def upstream_mask(code: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Cells that drain into any seed cell (seeds included)."""
    lab = np.where(seeds.ravel(), 1, -1).astype(np.int64)
    down = downstream_index(code)
    lab = _propagate_up(upstream_first(code), down, lab)
    # cells whose path ends without reaching a seed keep -1; cells reaching a non-seed outlet too
    return (lab == 1).reshape(code.shape)


def terminal_outlet(code: np.ndarray) -> np.ndarray:
    """Flat index of the outlet each cell finally drains to."""
    down = downstream_index(code)
    lab = np.where(down < 0, np.arange(down.size), -1).astype(np.int64)
    return _propagate_up(upstream_first(code), down, lab).reshape(code.shape)


def subbasins(code: np.ndarray, acc: np.ndarray, valid: np.ndarray, min_cells: float) -> np.ndarray:
    """Sub-basin label per cell: the catchment of each stream link (>= min_cells) between
    confluences and of each outlet draining >= min_cells; -1 elsewhere (small edge catchments).
    Labels are the flat index of the link's outlet cell."""
    h, w = code.shape
    down = downstream_index(code)
    big = (acc.ravel() >= min_cells) & valid.ravel()
    inflow = np.zeros(down.size, np.int16)
    idx = np.nonzero(big)[0]
    j = down[idx]
    ok = j >= 0
    ok[ok] &= big[j[ok]]
    np.add.at(inflow, j[ok], 1)
    node = np.zeros(down.size, bool)
    node[idx[~ok]] = True                                  # big outlets (edge / nodata)
    jj = j[ok]
    node[idx[ok][inflow[jj] >= 2]] = True                  # last cell before a confluence
    lab = np.where(node, np.arange(down.size), -1).astype(np.int64)
    lab = _propagate_up(upstream_first(code), down, lab)
    return lab.reshape(h, w)


def downslope_azimuth(filled: np.ndarray, valid: np.ndarray, cell: float, code: np.ndarray = None,
                      sigma: float = 1.0) -> np.ndarray:
    """Downslope direction per cell for display, as uint8: 0..251 = azimuth 0..360 deg clockwise from
    north, 255 = none. Steepest descent of the slightly smoothed filled surface (smooth arrows instead
    of D8's eight directions); on filled flats, where the gradient vanishes, the D8 direction is used."""
    from scipy.ndimage import gaussian_filter

    zf = np.where(valid, filled, np.nan)
    fill = np.nanmean(zf) if np.isfinite(zf).any() else 0.0
    zs = gaussian_filter(np.where(valid, filled, fill), sigma)
    gy, gx = np.gradient(zs, cell)                 # gy along rows (southward), gx eastward
    east, north = -gx, gy                          # downslope vector
    mag = np.hypot(east, north)
    az = np.degrees(np.arctan2(east, north)) % 360
    if code is not None:
        d8_az = np.array([np.nan, 90, 135, 180, 225, 270, 315, 0, 45])[code]
        flat = (mag < 1e-3) & (code > 0)
        az = np.where(flat, d8_az, az)
        mag = np.where(flat, 1.0, mag)
    out = np.where(valid & (mag >= 1e-3) & np.isfinite(az), np.round(az * 252 / 360) % 252, 255)
    return out.astype(np.uint8)


def slope_radians(z: np.ndarray, cell: float) -> np.ndarray:
    gy, gx = np.gradient(z, cell)
    return np.arctan(np.hypot(gx, gy))


def wetness_index(acc: np.ndarray, slope: np.ndarray, cell: float) -> np.ndarray:
    """TWI = ln(a / tan b) with a = specific catchment area (m^2 per m of contour)."""
    a = np.maximum(acc, 1) * cell * cell / cell
    return np.log(a / np.maximum(np.tan(slope), 1e-3))


HYPSO = [(0.00, (40, 110, 70)), (0.15, (95, 160, 85)), (0.35, (190, 205, 115)), (0.55, (225, 200, 130)),
         (0.75, (180, 130, 85)), (0.90, (150, 120, 105)), (1.00, (240, 240, 240))]
# elevation tint as in common DEM viewers: blue lowland -> cyan -> green -> yellow -> orange summit
ELEV_RAMP = [(0.00, (58, 70, 182)), (0.09, (87, 129, 210)), (0.17, (113, 193, 230)), (0.25, (129, 208, 225)),
             (0.34, (132, 221, 193)), (0.42, (139, 225, 171)), (0.55, (184, 213, 148)), (0.63, (205, 196, 126)),
             (0.71, (232, 164, 82)), (0.80, (255, 130, 33)), (0.88, (242, 116, 35)), (1.00, (205, 92, 50))]


def color_relief(z: np.ndarray, valid: np.ndarray, cell: float, lo=None, hi=None, ramp=HYPSO) -> np.ndarray:
    """Hypsometric tint (default: green lowland -> brown upland -> white peaks; ELEV_RAMP: blue -> orange)
    modulated by a multi-directional hillshade. Returns RGBA uint8 (transparent outside `valid`)."""
    zv = z[valid]
    lo = np.percentile(zv, 1) if lo is None else lo
    hi = np.percentile(zv, 99.5) if hi is None else hi
    f = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([s for s, _ in ramp])
    cols = np.array([c for _, c in ramp], float)
    rgb = np.stack([np.interp(f, stops, cols[:, i]) for i in range(3)], -1)
    zf = np.where(valid, z, np.median(zv))
    hs = (hillshade(zf, cell, 315, 45) * 0.6 + hillshade(zf, cell, 45, 45) * 0.2 + hillshade(zf, cell, 270, 60) * 0.2)
    shade = 0.35 + 0.85 * hs
    out = np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8)
    return np.dstack([out, np.where(valid, 255, 0).astype(np.uint8)])


def hillshade(z: np.ndarray, cell: float, azimuth=315.0, altitude=45.0) -> np.ndarray:
    gy, gx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)
