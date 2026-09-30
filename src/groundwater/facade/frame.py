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
"""
import numpy as np

_CHUNK = 500_000


class WallFrame:
    def __init__(self, vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2:
            raise ValueError("Wall polyline needs at least two [x, y] vertices")
        segments = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Wall polyline has repeated consecutive vertices")

        self.vertices = vertices
        self._starts = vertices[:-1]
        self._lengths = lengths
        self._dirs = segments / lengths[:, None]
        self._rights = np.column_stack([self._dirs[:, 1], -self._dirs[:, 0]])
        self._chainage = np.concatenate([[0.0], np.cumsum(lengths)])

    @property
    def length(self) -> float:
        return float(self._chainage[-1])

    def reversed(self) -> "WallFrame":
        return WallFrame(self.vertices[::-1])

    def _segment_bounds(self, s: int) -> tuple[float, float]:
        lower = -np.inf if s == 0 else 0.0
        upper = np.inf if s == len(self._lengths) - 1 else self._lengths[s]
        return lower, upper

    def to_wall(self, xyz: np.ndarray) -> np.ndarray:
        """World (x, y, z) -> wall (u, v, w)."""
        xyz = np.atleast_2d(xyz)
        out = np.empty((len(xyz), 3))
        for start in range(0, len(xyz), _CHUNK):
            xy = xyz[start:start + _CHUNK, :2]
            best_d2 = np.full(len(xy), np.inf)
            best_seg = np.zeros(len(xy), dtype=int)
            best_t = np.zeros(len(xy))
            for s in range(len(self._lengths)):
                rel = xy - self._starts[s]
                t = np.clip(rel @ self._dirs[s], *self._segment_bounds(s))
                d2 = np.sum((rel - t[:, None] * self._dirs[s]) ** 2, axis=1)
                better = d2 < best_d2
                best_d2[better], best_seg[better], best_t[better] = d2[better], s, t[better]
            rel = xy - self._starts[best_seg]
            side = np.where(np.einsum("ij,ij->i", rel, self._rights[best_seg]) < 0, -1.0, 1.0)
            out[start:start + _CHUNK, 0] = self._chainage[best_seg] + best_t
            out[start:start + _CHUNK, 1] = xyz[start:start + _CHUNK, 2]
            out[start:start + _CHUNK, 2] = side * np.sqrt(best_d2)
        return out

    def _segment_at(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self._chainage, u, side="right") - 1, 0, len(self._lengths) - 1)

    def to_world(self, uvw: np.ndarray) -> np.ndarray:
        """Wall (u, v, w) -> world (x, y, z)."""
        uvw = np.atleast_2d(uvw)
        s = self._segment_at(uvw[:, 0])
        t = uvw[:, 0] - self._chainage[s]
        xy = self._starts[s] + t[:, None] * self._dirs[s] + uvw[:, 2:3] * self._rights[s]
        return np.column_stack([xy, uvw[:, 1]])

    def axes_at(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        """(along, wall_normal): horizontal unit vectors of the wall at chainage u.

        wall_normal points toward the viewer.
        """
        s = int(self._segment_at(np.array([u]))[0])
        return np.array([*self._dirs[s], 0.0]), np.array([*self._rights[s], 0.0])

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
    """Wall frame from a user polyline, or a straight line fitted to the points.

    Which side is "in front" is set, in order of preference, by `viewer_point`
    (any plan-view point on the open/pit side), by the polyline's own vertex
    order, or -- for a fitted straight wall only -- by assuming the crest lies
    further back than the toe (true for any wall that is not overhanging).
    """
    if polyline is not None:
        frame = WallFrame(polyline)
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
