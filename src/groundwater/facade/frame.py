"""Wall (facade) coordinate system for near-vertical rock faces.

A nadir map projection collapses a vertical wall into a thin strip, so every 2D
product of the wall -- facade orthophoto, trace map, thermal overlay -- lives in
a wall frame instead:

- u: chainage, metres along a plan-view polyline that follows the wall
- v: elevation (world z), metres -- directly comparable with borehole/ERT depths
- w: horizontal offset from the polyline, positive toward the viewer (pit side)

The polyline is ordered left-to-right as seen by someone standing in front of
the wall, so facade images read naturally. A curved pit wall is unrolled along
the polyline; a single straight wall section is the two-vertex case. The first
and last segments extend indefinitely, so points past the ends get u < 0 or
u > length instead of being squashed onto the end vertices.

Bends: with nearest-segment projection, the rock on the outer side of a bend (the face usually lies
metres behind a wall line drawn at the toe) all projects onto the vertex itself -- a 16 deg bend squeezes
~1.4 m of rock 5 m behind the line into one chainage, and the facade image shows a cut. With
`fillet=True` (used for user polylines) every bend is replaced by a circular arc, as large as the two
segments allow, so u runs continuously through it: points project onto the arc.
"""
import numpy as np

_CHUNK = 500_000
FILLET_MAX_RADIUS = 100.0       # m
FILLET_MAX_FRACTION = 0.45      # an arc's tangent length is at most this share of either neighbouring segment


def _threads() -> int:
    import os

    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _left(d):
    return np.array([-d[1], d[0]])


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class WallFrame:
    def __init__(self, vertices, fillet: bool = False):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2:
            raise ValueError("Wall polyline needs at least two [x, y] vertices")
        segments = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Wall polyline has repeated consecutive vertices")
        self.vertices = vertices
        self._fillet = fillet
        dirs = segments / lengths[:, None]
        parts = []                                       # dicts: kind, length, + geometry
        prev = vertices[0]
        for k in range(1, len(vertices) - 1):
            a, b = dirs[k - 1], dirs[k]
            theta = float(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
            if not fillet or abs(theta) < 1e-3:
                parts.append({"kind": 0, "p": prev, "d": a, "length": float(np.linalg.norm(vertices[k] - prev))})
                prev = vertices[k]
                continue
            half = np.tan(abs(theta) / 2)
            R = min(FILLET_MAX_RADIUS, FILLET_MAX_FRACTION * min(lengths[k - 1], lengths[k]) / half)
            T = R * half
            t1, t2 = vertices[k] - a * T, vertices[k] + b * T
            parts.append({"kind": 0, "p": prev, "d": a, "length": float(np.linalg.norm(t1 - prev))})
            s = float(np.sign(theta))
            centre = t1 + R * s * _left(a)
            parts.append({"kind": 1, "c": centre, "R": R, "r0": (t1 - centre) / R, "s": s, "sweep": abs(theta),
                          "a": a, "length": R * abs(theta)})
            prev = t2
        parts.append({"kind": 0, "p": prev, "d": dirs[-1], "length": float(np.linalg.norm(vertices[-1] - prev))})
        self._parts = parts
        self._c0 = np.concatenate([[0.0], np.cumsum([p["length"] for p in parts])])

    def __setstate__(self, state):
        """Frames pickled before arcs existed (results saved earlier) are rebuilt from their vertices with the
        plain nearest-segment rule they were computed with."""
        if "_parts" not in state:
            self.__init__(state["vertices"], fillet=False)
        else:
            self.__dict__.update(state)

    @property
    def length(self) -> float:
        return float(self._c0[-1])

    def reversed(self) -> "WallFrame":
        return WallFrame(self.vertices[::-1], fillet=self._fillet)

    def _project(self, i: int, x: np.ndarray, y: np.ndarray):
        """(squared distance, u, signed w) of points (x, y) on part i.

        Line: with rel = t d + n r (d the direction, r its right normal) the foot is at clip(t) and the offset is
        (t - clip(t)) d + n r, so d2 = (t - clip t)^2 + n^2 and the side is the sign of n. Arc: the foot is the radial
        projection, d2 = (rho - R)^2 and w = s (rho - R) (the right normal is s times the radial direction); points
        past either end of the arc are left to the neighbouring straight parts, which end at the arc's ends and so
        are at least as close (d2 = inf here)."""
        p = self._parts[i]
        last = len(self._parts) - 1
        if p["kind"] == 0:
            rx, ry = x - p["p"][0], y - p["p"][1]
            dx, dy = p["d"]
            t = rx * dx + ry * dy
            n = rx * dy - ry * dx
            tc = np.clip(t, -np.inf if i == 0 else 0.0, np.inf if i == last else p["length"])
            d2 = (t - tc) ** 2 + n * n
            w = np.sqrt(d2)
            w[n < 0] *= -1.0
            return d2, self._c0[i] + tc, w
        rx, ry = x - p["c"][0], y - p["c"][1]
        r0x, r0y = p["r0"]
        phi = p["s"] * np.arctan2(r0x * ry - r0y * rx, r0x * rx + r0y * ry)    # angle travelled from the start
        dr = np.hypot(rx, ry) - p["R"]
        d2 = np.where((phi >= 0.0) & (phi <= p["sweep"]), dr * dr, np.inf)
        return d2, self._c0[i] + p["R"] * phi, p["s"] * dr

    def to_wall(self, xyz: np.ndarray) -> np.ndarray:
        """World (x, y, z) -> wall (u, v, w): the nearest point of the wall line (its chainage u and signed
        offset w, + toward the viewer), v = z."""
        xyz = np.atleast_2d(xyz)
        out = np.empty((len(xyz), 3))
        out[:, 1] = xyz[:, 2]

        def block(start):
            x, y = xyz[start:start + _CHUNK, 0], xyz[start:start + _CHUNK, 1]
            if len(self._parts) == 1:
                _, u, w = self._project(0, x, y)
            else:
                proj = [self._project(i, x, y) for i in range(len(self._parts))]
                k = np.argmin(np.stack([d2 for d2, _, _ in proj]), axis=0)   # first part wins a tie, as before
                rows = np.arange(len(x))
                u = np.stack([q[1] for q in proj])[k, rows]
                w = np.stack([q[2] for q in proj])[k, rows]
            out[start:start + _CHUNK, 0] = u
            out[start:start + _CHUNK, 2] = w

        starts = range(0, len(xyz), _CHUNK)
        if len(starts) < 4:
            for s in starts:
                block(s)
        else:                                  # numpy releases the GIL: blocks of a large cloud in parallel
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=_threads()) as ex:
                list(ex.map(block, starts))
        return out

    def _part_at(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self._c0, u, side="right") - 1, 0, len(self._parts) - 1)

    def _frame_at(self, u: np.ndarray):
        """(point on the line, travel direction, right normal) at chainages u."""
        u = np.asarray(u, float)
        idx = self._part_at(u)
        pt = np.empty((len(u), 2))
        tan = np.empty((len(u), 2))
        for i in np.unique(idx):
            m = idx == i
            p = self._parts[i]
            t = u[m] - self._c0[i]
            if p["kind"] == 0:
                pt[m] = p["p"] + t[:, None] * p["d"]
                tan[m] = p["d"]
            else:
                phi = t / p["R"]
                # past either end of an arc (can only happen for inner parts by rounding) stay on its tangent
                phi_c = np.clip(phi, 0.0, p["sweep"])
                cs, sn = np.cos(p["s"] * phi_c), np.sin(p["s"] * phi_c)
                r0 = p["r0"]
                fr = np.column_stack([cs * r0[0] - sn * r0[1], sn * r0[0] + cs * r0[1]])
                tg = np.column_stack([-fr[:, 1], fr[:, 0]]) * p["s"]
                pt[m] = p["c"] + p["R"] * fr + (phi - phi_c)[:, None] * p["R"] * tg
                tan[m] = tg
        right = np.column_stack([tan[:, 1], -tan[:, 0]])
        return pt, tan, right

    def to_world(self, uvw: np.ndarray) -> np.ndarray:
        """Wall (u, v, w) -> world (x, y, z)."""
        uvw = np.atleast_2d(uvw)
        pt, _, right = self._frame_at(uvw[:, 0])
        xy = pt + uvw[:, 2:3] * right
        return np.column_stack([xy, uvw[:, 1]])

    def axes_at(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        """(along, wall_normal): horizontal unit vectors of the wall at chainage u.

        wall_normal points toward the viewer.
        """
        _, tan, right = self._frame_at(np.array([u]))
        return np.array([*tan[0], 0.0]), np.array([*right[0], 0.0])

    def apparent_trace_direction(self, normal: np.ndarray, u: float, min_angle_deg: float = 20.0) -> np.ndarray | None:
        """Unit (du, dv) direction in which a plane with this pole crosses the wall.

        The trace is the intersection of the joint plane with the wall, whose
        normal is taken as horizontal (exact for a vertical wall, a few degrees
        off for a leaning one). Returns None when the plane is within
        `min_angle_deg` of the wall itself: such a joint shows up as a face, not
        a line, so trace orientation says nothing about it.
        """
        along, wall_normal = self.axes_at(u)
        trace = np.cross(normal / np.linalg.norm(normal), wall_normal)
        if np.linalg.norm(trace) < np.sin(np.radians(min_angle_deg)):
            return None
        uv = np.array([trace @ along, trace[2]])
        return uv / np.linalg.norm(uv)


def build_wall_frame(points: np.ndarray, polyline=None, viewer_point=None) -> WallFrame:
    """Wall frame from a user polyline (bends rounded into arcs), or a straight line fitted to the points.

    Which side is "in front" is set, in order of preference, by `viewer_point`
    (any plan-view point on the open/pit side), by the polyline's own vertex
    order, or -- for a fitted straight wall only -- by assuming the crest lies
    further back than the toe (true for any wall that is not overhanging).
    """
    if polyline is not None:
        frame = WallFrame(polyline, fillet=True)
    else:
        xy = points[:, :2]
        centre = xy.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((xy - centre).T))
        direction = vecs[:, -1]
        proj = (xy - centre) @ direction
        frame = WallFrame([centre + direction * proj.min(), centre + direction * proj.max()])

    if viewer_point is not None:
        if frame.to_wall(np.array([[*viewer_point, 0.0]]))[0, 2] < 0:
            frame = frame.reversed()
    elif polyline is None:
        uvw = frame.to_wall(points)
        if np.std(uvw[:, 2]) > 0 and np.std(uvw[:, 1]) > 0 and np.corrcoef(uvw[:, 2], uvw[:, 1])[0, 1] > 0:
            frame = frame.reversed()
    return frame
