"""Joint planes inside the rock behind the face, inferred from the surface analysis, for the 3D scene.

Two kinds, kept apart because they mean different things:

* observed, extended ("관측 연장") -- every mapped trace and every exposed joint-set facet continued into
  the rock in its own plane. A trace (a joint cutting the face) becomes a boxy half-ellipse: its face trace,
  continued inward along the joint plane by `depth_factor` x trace length (a joint is roughly as deep as it is
  long; `max_depth` caps it); an exposed facet becomes a disc of its persistence (facets within 30 deg of
  the face are left out: their extension would only repaint the face; so are flat ones at the foot of the
  wall -- ground, a bench, the water surface). Plane orientation: the
  trace's set / trace family, else normal to the face. These are real joints, only their extent is assumed.
* statistical ("통계 절리망") -- a Poisson disc network per joint set in a slab `depth_m` behind the face:
  intensity P32 = 1 / spacing, disc diameter = persistence. Positions are random; it shows what the rock
  mass is like in general, not where a particular joint is.

Every polygon is clipped to the rock: radially, it stops where it would come out in front of the facade
(the front-most surface per chainage / elevation) or leave the mapped wall.
`coverage` ray-casts the grouting holes against the polygons: which joints a hole crosses and which
high-seepage joints no hole reaches.
"""
from dataclasses import dataclass

import numpy as np

from ..hydro import _pole


@dataclass
class Joint:
    name: str              # set / family name, as in hydro.joint_sets_for_flow ("Set 1", "트레이스 F2", ...)
    poly: np.ndarray       # (k, 3) world polygon, star-shaped around poly[0] (fan triangulated)
    pole: np.ndarray
    source: str            # "trace" | "facet" | "dfn"
    seepage_class: int     # class of the seepage cell at its face position (2 상, 1 중, 0 하, -1 none / dfn)
    score: float           # seepage score there (nan for dfn)
    size_m: float


class RockTest:
    """Is a world point inside the rock? Behind the front-most facade surface at its chainage / elevation."""

    def __init__(self, result, cell_m: float = 0.1, tol_m: float = 0.15, max_flat: float = 1.5):
        import cv2
        from scipy import ndimage

        self.frame = result.wall_frame
        d = result.facade_depth
        a = d.array[0].astype(np.float32)
        px = abs(d.transform.a)
        k = max(1, int(round(cell_m / px)))
        H, W = a.shape[0] // k, a.shape[1] // k
        blk = a[:H * k, :W * k].reshape(H, k, W, k)
        with np.errstate(all="ignore"):
            front = np.nanmax(blk, axis=(1, 3))                    # w points toward the viewer: max = front
        valid = np.isfinite(front)
        # the facade is a projection along the wall normal: ground or a bank in front of the toe shows up with
        # a large w, and the air above it would pass as "behind the face". Cells whose surface is flatter
        # than ~35 deg (|dw/dv| > 1.5) are not wall.
        f0 = np.where(valid, front, np.nanmedian(front) if valid.any() else 0).astype(np.float32)
        f0 = cv2.GaussianBlur(f0, (0, 0), 2.0)
        dwdv = np.abs(np.gradient(f0, px * k, axis=0))
        valid &= dwdv < max_flat
        # small gaps (vegetation, shadows) take the nearest face value; the outline of the wall stays
        closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(bool)
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        self.front = np.where(closed, front[idx[0], idx[1]], np.nan)
        self.u0, self.v0, self.cell = d.transform.c, d.transform.f, px * k
        self.tol = tol_m

    def face_w(self, u, v):
        r = ((self.v0 - np.asarray(v)) / self.cell).astype(int)
        c = ((np.asarray(u) - self.u0) / self.cell).astype(int)
        ok = (r >= 0) & (r < self.front.shape[0]) & (c >= 0) & (c < self.front.shape[1])
        out = np.full(np.shape(u), np.nan)
        out[ok] = self.front[r[ok], c[ok]]
        return out

    def inside(self, xyz):
        uvw = self.frame.to_wall(np.asarray(xyz).reshape(-1, 3))
        fw = self.face_w(uvw[:, 0], uvw[:, 1])
        return np.isfinite(fw) & (uvw[:, 2] <= fw + self.tol)


def _clip_star(centre, dirs, radii, rock: RockTest, steps: int = 6):
    """Polygon centre + r_i dirs_i with each r_i cut where the ray first leaves the rock."""
    f = np.linspace(1.0 / steps, 1.0, steps)
    pts = centre[None, None] + (f[None, :, None] * radii[:, None, None]) * dirs[:, None, :]
    ins = rock.inside(pts.reshape(-1, 3)).reshape(len(dirs), steps)
    first_out = np.where(ins.all(axis=1), steps, np.argmin(ins, axis=1))
    r = np.where(first_out > 0, f[np.maximum(first_out - 1, 0)], 0.0) * radii
    return np.vstack([centre, centre + r[:, None] * dirs]), r


def _seep_at(smap, u, v):
    if smap is None:
        return -1, np.nan
    r = int((smap.v_max - v) / smap.cell_m)
    c = int((u - smap.u_min) / smap.cell_m)
    if 0 <= r < smap.classes.shape[0] and 0 <= c < smap.classes.shape[1]:
        return int(smap.classes[r, c]), float(smap.layers["score"][r, c])
    return -1, np.nan


def observed(result, smap=None, depth_factor: float = 1.0, max_depth: float = 5.0, min_trace_m: float = 0.5,
             facets: bool = True, min_facet_m2: float = 0.3, rock: RockTest | None = None) -> list[Joint]:
    from .grouting3d import _trace_planes

    rock = rock or RockTest(result)
    frame = result.wall_frame
    planes = _trace_planes(result)
    out = []
    th = np.linspace(0, np.pi, 13)
    cs, sn = np.cos(th), np.sin(th)
    bx = np.sign(cs) * np.abs(cs) ** 0.5           # superellipse n = 4: boxy, keeps the trace edge whole
    by = np.abs(sn) ** 0.5
    for tr in result.traces:
        if tr.length_m < min_trace_m or tr.xyz is None:
            continue
        p0, p1 = tr.xyz
        mid = 0.5 * (p0 + p1)
        t = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        um, vm = 0.5 * (tr.coords[0] + tr.coords[1])
        _, outward = frame.axes_at(float(um))
        inward = -outward
        name, n = planes.get(id(tr), ("방향 미정", None))
        n = n if n is not None else np.cross(t, inward)
        n = n / max(np.linalg.norm(n), 1e-9)
        m = inward - (inward @ n) * n
        if np.linalg.norm(m) < 0.3:                # plane ~ parallel to the face: nothing to continue inward
            continue
        m /= np.linalg.norm(m)
        t = t - (t @ n) * n
        t /= np.linalg.norm(t)
        a = 0.5 * tr.length_m * 1.1
        depth = float(np.clip(depth_factor * tr.length_m, 0.3, max_depth))
        vec = (a * bx)[:, None] * t + (depth * by)[:, None] * m
        rad = np.linalg.norm(vec, axis=1)
        dirs = vec / np.maximum(rad, 1e-9)[:, None]
        # the polygon starts on the trace line: its centre is the trace midpoint, a little inside the rock
        poly, r = _clip_star(mid + 0.02 * m, dirs, rad, rock)
        if (r > 0.05).sum() < 4:
            continue
        cl, sc = _seep_at(smap, um, vm)
        out.append(Joint(name, poly, n, "trace", cl, sc, tr.length_m))
    if facets:
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts = np.asarray(result.point_cloud.points)
        toe = float(np.percentile(frame.to_wall(pts[::max(1, len(pts) // 200_000)])[:, 1], 1))
        for d in result.discontinuity_sets:
            n = _pole(d.mean_dip, d.mean_dip_direction)
            for p in d.patches:
                if getattr(p, "area", 1.0) < min_facet_m2:
                    continue
                pn = np.asarray(p.normal, float)
                pn = pn / max(np.linalg.norm(pn), 1e-9)
                uv = frame.to_wall(np.asarray(p.centroid, float)[None])[0]
                if abs(pn @ frame.axes_at(float(uv[0]))[1]) > np.cos(np.radians(30)):
                    continue                       # facet ~ parallel to the face: its "extension" is the face itself
                if abs(pn[2]) > np.cos(np.radians(20)) and uv[1] < toe + 1.0:
                    continue                       # flat at the foot of the wall: ground, a bench or water, not a joint
                a = np.cross(pn, [0.0, 0.0, 1.0]) if abs(pn[2]) < 0.95 else np.cross(pn, [1.0, 0.0, 0.0])
                a /= np.linalg.norm(a)
                b = np.cross(pn, a)
                dirs = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
                R = float(np.clip(0.5 * p.persistence_length * depth_factor, 0.3, max_depth))
                c = np.asarray(p.centroid, float)
                poly, r = _clip_star(c, dirs, np.full(len(dirs), R), rock)
                if (r > 0.05).sum() < 4:
                    continue
                cl, sc = _seep_at(smap, uv[0], uv[1])
                out.append(Joint(f"Set {d.set_id}", poly, pn, "facet", cl, sc, p.persistence_length))
    return out


def statistical(result, sets, depth_m: float = 5.0, radius_scale: float = 1.0, max_per_set: int = 40000,
                seed: int = 0, rock: RockTest | None = None) -> tuple[list[Joint], dict]:
    """Poisson disc network per set in the slab `depth_m` behind the face. Returns (joints, generated share per
    set -- 1 unless `max_per_set` capped it). All of them enter `coverage`; `mesh_traces` draws a subset."""
    rock = rock or RockTest(result)
    rng = np.random.default_rng(seed)
    valid = np.isfinite(rock.front)
    rr, cc = np.nonzero(valid)
    face_area = valid.sum() * rock.cell ** 2
    volume = face_area * depth_m
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    out, shown = [], {}
    for s in sets.itertuples():
        P32 = 1.0 / max(float(s.spacing_m), 0.05)
        R = 0.5 * float(s.persistence_m) * radius_scale
        n_true = int(P32 * volume / (np.pi * R * R))
        n = min(n_true, max_per_set)
        shown[s.name] = n / max(n_true, 1)
        if n == 0:
            continue
        pick = rng.integers(0, len(rr), n)
        u = rock.u0 + (cc[pick] + rng.random(n)) * rock.cell
        v = rock.v0 - (rr[pick] + rng.random(n)) * rock.cell
        w = rock.front[rr[pick], cc[pick]] - rng.random(n) * depth_m
        centres = result.wall_frame.to_world(np.column_stack([u, v, w]))
        pn = _pole(s.dip, s.dip_direction)
        a = np.cross(pn, [0.0, 0.0, 1.0]) if abs(pn[2]) < 0.95 else np.cross(pn, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(pn, a)
        dirs = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
        # all discs of the set clipped in one rock test (one transform of n x 12 x 4 points, not n of them)
        steps = 4
        f = np.linspace(1.0 / steps, 1.0, steps)
        P = centres[:, None, None, :] + (f[None, None, :, None] * R) * dirs[None, :, None, :]
        ins = rock.inside(P.reshape(-1, 3)).reshape(len(centres), len(dirs), steps)
        first_out = np.where(ins.all(axis=2), steps, np.argmin(ins, axis=2))
        r = np.where(first_out > 0, f[np.maximum(first_out - 1, 0)], 0.0) * R            # (n, 12)
        polys = np.concatenate([centres[:, None, :], centres[:, None, :] + r[:, :, None] * dirs[None]], axis=1)
        for poly, rad in zip(polys, r):
            if (rad > 0.05).sum() >= 4:
                out.append(Joint(s.name, poly, pn, "dfn", -1, np.nan, 2 * R))
    return out, shown


def _triangles(joints):
    V, F, owner = [], [], []
    k = 0
    for i, j in enumerate(joints):
        m = len(j.poly) - 1
        closed = np.linalg.norm(j.poly[1] - j.poly[-1]) < 1e-6 or j.source != "trace"
        V.append(j.poly)
        idx = [(k, k + 1 + q, k + 1 + (q + 1) % m) for q in range(m if closed else m - 1)]
        F += idx
        owner += [i] * len(idx)
        k += len(j.poly)
    return (np.vstack(V) if V else np.zeros((0, 3))), np.array(F, int).reshape(-1, 3), np.array(owner, int)


def coverage(holes: list, joints: list[Joint]) -> dict:
    """Which joints each hole crosses (ray against the joint polygons, within the hole length)."""
    import open3d as o3d

    V, F, owner = _triangles(joints)
    hits_per_hole = [[] for _ in holes]
    hit = np.zeros(len(joints), bool)
    if len(F) and holes:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        rays = np.array([np.hstack([h["collar"], h["dir"]]) for h in holes], np.float32)
        res = scene.list_intersections(o3d.core.Tensor(rays))
        split, t, prim = res["ray_splits"].numpy(), res["t_hit"].numpy(), res["primitive_ids"].numpy()
        for i, h in enumerate(holes):
            sl = slice(split[i], split[i + 1])
            ok = t[sl] <= h["length"]
            js = sorted(set(owner[prim[sl][ok]].tolist()))
            pts = {}
            for jj, tt in zip(owner[prim[sl][ok]], t[sl][ok]):
                pts.setdefault(int(jj), float(tt))
            hits_per_hole[i] = [(j, pts[j]) for j in js]
            hit[js] = True
    high = np.array([j.source != "dfn" and j.seepage_class == 2 for j in joints], bool)
    return {"per_hole": hits_per_hole, "hit": hit, "high": high,
            "n_high": int(high.sum()), "n_high_hit": int((high & hit).sum())}


def joint_groups(joints: list[Joint], colour_of, shade: str = "seepage", max_draw_dfn: int = 1500) -> list[dict]:
    """Joints grouped for drawing by (name, observed / statistical): indices, legend label and an RGB colour per
    joint (set colour, lighter for low seepage / small joints). A statistical set with more than
    `max_draw_dfn` discs is thinned to a random subset for drawing (share in the label)."""
    groups = {}
    for i, j in enumerate(joints):
        kind = j.source if j.source in ("dfn", "major") else "obs"
        groups.setdefault((j.name, kind), []).append(i)
    rng = np.random.default_rng(1)
    sizes = np.array([j.size_m for j in joints]) if joints else np.zeros(0)
    s_lo, s_hi = (np.percentile(sizes, [10, 90]) if len(sizes) else (0, 1))
    out = []
    for (name, kind), idx in groups.items():
        label = f"{name} " + {"obs": "관측 절리 연장", "dfn": "통계 절리망 (위치 임의)", "major": "주요 절리면"}[kind]
        if kind == "dfn" and len(idx) > max_draw_dfn:
            label += f" — {max_draw_dfn / len(idx):.0%}만 표시"
            idx = sorted(rng.choice(idx, max_draw_dfn, replace=False).tolist())
        sub = [joints[i] for i in idx]
        base = np.array(_hex_rgb(colour_of(name)), float)
        if shade == "seepage" and kind == "obs":
            f = np.clip(np.nan_to_num(np.array([j.score for j in sub]), nan=0.0), 0, 1)   # 0 low .. 1 high seepage
        elif shade == "size":
            f = np.clip((np.array([j.size_m for j in sub]) - s_lo) / max(s_hi - s_lo, 1e-9), 0, 1)
        else:
            f = np.full(len(sub), 0.35 if kind == "dfn" else 0.7)
        mix = 0.55 * (1 - f)                                                  # lighter = lower
        col = base[None] * (1 - mix[:, None]) + 255 * mix[:, None]
        out.append({"name": name, "kind": kind, "label": label, "idx": idx, "rgb": col.astype(int)})
    return out


def mesh_traces(joints: list[Joint], colour_of, opacity: float = 0.5, shade: str = "seepage", origin=np.zeros(3),
                max_draw_dfn: int = 1500):
    """Plotly Mesh3d traces, one per joint group (`joint_groups`)."""
    import plotly.graph_objects as go

    out = []
    for g in joint_groups(joints, colour_of, shade, max_draw_dfn):
        V, F, owner = _triangles([joints[i] for i in g["idx"]])
        fc = [f"rgb({r},{gg},{b})" for r, gg, b in g["rgb"][owner]]
        out.append(go.Mesh3d(x=V[:, 0] - origin[0], y=V[:, 1] - origin[1], z=V[:, 2], i=F[:, 0], j=F[:, 1],
                             k=F[:, 2], facecolor=fc, opacity=opacity, name=g["label"], showlegend=True,
                             hoverinfo="name", flatshading=True, lighting=dict(ambient=0.8, diffuse=0.3),
                             legendgroup=g["label"]))
    return out


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
