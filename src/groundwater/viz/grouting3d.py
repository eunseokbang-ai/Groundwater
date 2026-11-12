"""Virtual grouting holes drawn on the 3D model of the face (Plotly, also exported as a standalone HTML).

Four layouts:
* "face"  -- holes collared on the rock face in rows at given elevations, every `spacing` m of chainage,
             driven into the rock with the chosen trend/plunge (the horizontal part is turned into the rock);
* "crest" -- a grout curtain: holes collared along a line `offset` m behind the face (median face position at
             that chainage, not the wall line) at the crest, every
             `spacing` m, with the chosen trend/plunge (vertical or inclined in the curtain plane);
* network  -- holes designed on the joint network inside the rock (`network_layout`): collars in the
             high-seepage cells, direction and dip chosen so each hole crosses as many of the extended joints
             (and set intersections) still uncovered as possible;
* targeted -- shallow crack injection: holes on the mapped traces inside the high-seepage cells, each aimed
             at the joint plane a set depth behind its face trace (`targeted_layout`).

Along every hole the joints it is expected to cross are drawn as small discs in the set's orientation, at
the spacing the set has along the hole (s / |cos angle|): the same statistics as the hole-orientation numbers
of the grouting tab, made visible. These are expected positions, not mapped joints.

With a mesh of the site (`SiteMesh`, e.g. Metashape OBJ in map coordinates) the scene is fully 3D
(overhangs, the crest), crest collars sit exactly on the ground (ray straight down), and every hole is
ray-cast against the rock surface: a hole that leaves the rock before its full length (breaks out through
the face or an overhang) is flagged and the exit point marked.
"""
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go

from ..hydro import _line, _pole
from ..report import SET_COLORS

FAMILY_EXTRA = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
ROW_STEP_M = 5.0            # face layout: vertical distance between hole rows
MIN_CREST_PLUNGE = 45.0     # crest (curtain) holes flatter than this make no sense -- they stay at crest level


_HEX2 = np.array([f"{i:02x}" for i in range(256)], dtype="<U2")


def _hex_colours(C: np.ndarray) -> list:
    """'#rrggbb' strings for an (n, 3) uint8 array, vectorised (a Python f-string per vertex costs seconds)."""
    C = np.asarray(C, np.uint8)
    return np.char.add(np.char.add(np.char.add("#", _HEX2[C[:, 0]]), _HEX2[C[:, 1]]), _HEX2[C[:, 2]]).tolist()


def _set_color(name: str, i: int) -> str:
    if name.startswith("Set "):
        try:
            return SET_COLORS[int(name.split()[1]) % len(SET_COLORS)]
        except ValueError:
            pass
    return FAMILY_EXTRA[i % len(FAMILY_EXTRA)]


def _face_w(result, u: float, v: float) -> float | None:
    d = result.facade_depth
    r, c = d.world_to_pixel(np.array([u]), np.array([v]))
    a = d.array[0]
    win = a[max(r[0] - 20, 0):r[0] + 20, max(c[0] - 20, 0):c[0] + 20]
    return float(np.nanmedian(win)) if np.isfinite(win).any() else None


def _trend_plunge_down(d: np.ndarray) -> tuple[float, float]:
    """Trend / plunge of a hole direction (plunge positive downward, negative = rising hole)."""
    return float(np.degrees(np.arctan2(d[0], d[1])) % 360), float(np.degrees(np.arcsin(-d[2])))


def _into_rock(direction: np.ndarray, outward: np.ndarray) -> np.ndarray:
    if direction[:2] @ outward[:2] > 1e-9:          # pointing out of the face: turn the horizontal part round
        direction = direction.copy()
        direction[:2] *= -1
    return direction


@dataclass
class SiteMesh:
    scene: object          # open3d.t.geometry.RaycastingScene of the cropped full-resolution mesh
    display: tuple         # (vertices, faces, colours uint8) decimated for drawing
    n_faces: int


def load_site_mesh(path, result, margin_m: float = 3.0, behind_m: float = 25.0, display_voxel_m: float = 0.1,
                   hide_vegetation: bool = False, display_front_m: float = 40.0) -> SiteMesh:
    """Read a mesh in map coordinates and prepare
    * ray casting: cropped to the analysed wall (chainage / elevation of the analysis cloud +- margin, from
      `behind_m` behind the face to a few m in front), green (vegetation) triangles removed -- a collar must not
      land on a bush in front of the face, nor a hole "break out" through the back of one;
    * drawing: the whole mesh around the wall (also the ground in front, up to `display_front_m`), decimated
      to `display_voxel_m`, with its vegetation unless `hide_vegetation`.
    """
    import open3d as o3d

    from ..discontinuity.vegetation import vegetation_mask

    mesh = o3d.io.read_triangle_mesh(str(path))
    V = np.asarray(mesh.vertices)
    pts = np.asarray(result.point_cloud.points)
    from .. import heights

    if heights.target(result.config) == heights.ORTHOMETRIC and len(V):
        dz = heights.match_datum(V, pts, result.config["crs"])     # a Metashape OBJ is ellipsoidal
        if dz:
            V = V + [0.0, 0.0, dz]
            mesh.vertices = o3d.utility.Vector3dVector(V)
    uvw_p = result.wall_frame.to_wall(pts[::max(1, len(pts) // 200_000)])
    lo, hi = np.percentile(uvw_p, 0.5, axis=0), np.percentile(uvw_p, 99.5, axis=0)
    uvw = result.wall_frame.to_wall(V)

    def crop(m_u, m_front):
        keep = ((uvw[:, 0] > lo[0] - m_u) & (uvw[:, 0] < hi[0] + m_u) & (uvw[:, 1] > lo[1] - 3 * margin_m) &
                (uvw[:, 1] < hi[1] + 2 * margin_m) & (uvw[:, 2] > lo[2] - behind_m) & (uvw[:, 2] < hi[2] + m_front))
        sub = mesh.select_by_index(np.flatnonzero(keep))
        sub.remove_unreferenced_vertices()
        return sub

    ray = crop(margin_m, margin_m)
    if ray.has_vertex_colors():
        green = vegetation_mask(np.asarray(ray.vertex_colors))
        ray.remove_triangles_by_mask(green[np.asarray(ray.triangles)].any(axis=1))
        ray.remove_unreferenced_vertices()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ray))

    disp = crop(3 * margin_m, display_front_m).simplify_vertex_clustering(voxel_size=display_voxel_m)
    disp.remove_degenerate_triangles()
    disp.remove_unreferenced_vertices()
    col = (np.asarray(disp.vertex_colors) * 255).astype(np.uint8) if disp.has_vertex_colors() else \
        np.full((len(disp.vertices), 3), 150, np.uint8)
    F = np.asarray(disp.triangles)
    if hide_vegetation and disp.has_vertex_colors():
        green = vegetation_mask(col.astype(float), scale=255.0)
        F = F[~green[F].all(axis=1)]                      # only fully green triangles: no speckled holes
    return SiteMesh(scene=scene, display=(np.asarray(disp.vertices), F, col), n_faces=len(ray.triangles))


def _cast(scene, origins, dirs):
    import open3d as o3d

    rays = o3d.core.Tensor(np.hstack([origins, dirs]).astype(np.float32))
    return scene.cast_rays(rays)["t_hit"].numpy()


def ground_below(site: SiteMesh, xy: np.ndarray, z_from: float) -> np.ndarray:
    """Elevation of the LAST surface hit straight down from z_from at each (x, y): the ground, below any tree
    canopy the mesh also contains (the first hit would put a crest collar on a tree top). NaN where none."""
    import open3d as o3d

    o = np.column_stack([xy, np.full(len(xy), z_from)])
    rays = o3d.core.Tensor(np.hstack([o, np.tile([0.0, 0.0, -1.0], (len(xy), 1))]).astype(np.float32))
    hits = site.scene.list_intersections(rays)
    split = hits["ray_splits"].numpy()
    t = hits["t_hit"].numpy()
    out = np.full(len(xy), np.nan)
    for i in range(len(xy)):
        if split[i + 1] > split[i]:
            out[i] = z_from - t[split[i]:split[i + 1]].max()
    return out


def check_breakout(holes: list, site: SiteMesh, start_m: float = 0.3) -> None:
    """Mark holes that leave the rock before their full length (first surface hit along the hole)."""
    if not holes:
        return
    o = np.array([h["collar"] + start_m * h["dir"] for h in holes])
    d = np.array([h["dir"] for h in holes])
    t = _cast(site.scene, o, d) + start_m
    for h, ti in zip(holes, t):
        h["breakout_m"] = float(ti) if np.isfinite(ti) and ti < h["length"] else None


def hole_layout(result, mode: str, trend: float, plunge: float, spacing: float, length: float,
                rows=None, offset: float = 5.0, u_margin: float = 2.0, site: SiteMesh | None = None) -> list[dict]:
    frame = result.wall_frame
    pts = np.asarray(result.point_cloud.points)
    uvw = frame.to_wall(pts[::max(1, len(pts) // 400_000)])
    u_lo, u_hi = np.percentile(uvw[:, 0], [1, 99])
    v_lo, v_hi = np.percentile(uvw[:, 1], [1, 99])
    base = _line(trend, plunge)
    holes = []
    for u in np.arange(u_lo + u_margin, u_hi - u_margin + 1e-6, spacing):
        _, outward = frame.axes_at(u)
        if mode == "crest":
            near = uvw[np.abs(uvw[:, 0] - u) < max(1.0, spacing / 2)]
            if not len(near):
                continue
            top = float(np.percentile(near[:, 1], 98))
            face_w = float(np.median(near[:, 2]))          # the face itself, not the wall line (often drawn at the toe)
            c = frame.to_world(np.array([[u, top, face_w - offset]]))[0]
            if site is not None:                           # collar on the actual ground of the crest
                z = ground_below(site, c[None, :2], top + 30.0)[0]
                if not np.isfinite(z):
                    continue
                c[2] = z
            collars = [c]
        else:
            collars = []
            # default: rows every ROW_STEP_M over the whole height of the face, not one row at the toe
            for v in (rows or list(np.arange(v_lo + 2.0, v_hi - 1.0, ROW_STEP_M))):
                if site is not None:                       # collar on the mesh: ray from in front, into the face
                    start = frame.to_world(np.array([[u, v, 20.0]]))[0]
                    t = _cast(site.scene, start[None], -outward[None])[0]
                    if np.isfinite(t):
                        collars.append(start - t * outward)
                    continue
                w = _face_w(result, u, v)
                if w is not None:
                    collars.append(frame.to_world(np.array([[u, v, w]]))[0])
        for c in collars:
            holes.append({"collar": c, "dir": _into_rock(base, outward), "length": length, "u": float(u)})
    if site is not None:
        check_breakout(holes, site)
    return holes


def _trace_planes(result) -> dict:
    """id(trace) -> (name, pole) of the joint plane behind each trace: the linked 3D set, else the
    trace-only family's estimated orientation; traces without either get none (plane assumed normal to the face)."""
    sets = {d.set_id: d for d in result.discontinuity_sets}
    out = {}
    for f in result.trace_families:
        if f.linked_set is not None and f.linked_set in sets:
            d = sets[f.linked_set]
            plane = (f"Set {d.set_id}", _pole(d.mean_dip, d.mean_dip_direction))
        elif f.dip is not None:
            plane = (f"트레이스 F{f.family_id}", _pole(f.dip, f.dip_direction))
        else:
            continue
        for t in f.members:
            out[id(t)] = plane
    return out


def _aimed_hole(p, m, side, outward, depth, tan, min_plunge, extra_m, site):
    """Hole from the face beside trace point p to the joint plane `depth` m behind it (along m)."""
    inward = -outward
    target = p + depth * m
    d_face = max(float((target - p) @ inward), 0.1)
    collar = p + side * d_face / tan
    to_t = target - collar
    rise = float(np.linalg.norm(to_t[:2])) * np.tan(np.radians(min_plunge)) + to_t[2]  # > 0: too flat / rising
    if rise > 0:
        up = np.array([0.0, 0.0, 1.0]) - outward * outward[2]
        collar = collar + rise * up / max(up[2], 0.3)
    if site is not None:                                   # onto the actual rock surface
        start = collar + 3.0 * outward
        th = _cast(site.scene, start[None], inward[None])[0]
        if np.isfinite(th) and th < 6.0:
            collar = start + th * inward
    to_t = target - collar
    dist = float(np.linalg.norm(to_t))
    if dist < 0.2:
        return None
    return {"collar": collar, "dir": to_t / dist, "length": dist + extra_m, "target": target, "joint_at_m": dist}


def targeted_layout(result, smap, classes=(2,), min_trace_m: float = 1.0, spacing: float = 1.0,
                    joint_depth: float = 1.0, angle_deg: float = 45.0, extra_m: float = 0.5,
                    min_plunge: float = 10.0, site: SiteMesh | None = None) -> list[dict]:
    """Holes only where water is likely to come out: on the mapped traces inside the seepage cells of
    `classes` (2 = 상, 1 = 중), aimed at the joint itself.

    Same geometry as crack injection: along a trace every `spacing` m a hole is collared on the face beside
    the trace (alternating sides) and drilled at `angle_deg` to the face so that it meets the joint plane
    `joint_depth` m behind the face trace (measured in the joint plane, into the rock); it runs `extra_m` past
    the joint. The plane behind a trace is its set / family orientation, else normal to the face.
    Holes dip down by at least `min_plunge` so the grout stays in: beside a flat trace the collar goes above
    it, beside a steep one the sides alternate and the collar is raised along the face.
    """
    frame = result.wall_frame
    planes = _trace_planes(result)
    cls = smap.classes
    rows, cols = cls.shape
    tan = np.tan(np.radians(angle_deg))
    holes, used = [], []
    for k, tr in enumerate(result.traces):
        if tr.length_m < min_trace_m or tr.xyz is None:
            continue
        n_s = max(1, int(tr.length_m // spacing))
        s = (np.arange(n_s) + 0.5) / n_s
        uv = tr.coords[0] + s[:, None] * (tr.coords[1] - tr.coords[0])
        r = np.clip(((smap.v_max - uv[:, 1]) / smap.cell_m).astype(int), 0, rows - 1)
        c = np.clip(((uv[:, 0] - smap.u_min) / smap.cell_m).astype(int), 0, cols - 1)
        name, n = planes.get(id(tr), ("수동 트레이스" if tr.source == "manual" else "트레이스", None))
        t_dir = tr.xyz[1] - tr.xyz[0]
        t_dir = t_dir / max(np.linalg.norm(t_dir), 1e-9)
        for j in np.flatnonzero(np.isin(cls[r, c], classes)):
            p = tr.xyz[0] + s[j] * (tr.xyz[1] - tr.xyz[0])
            if any(np.linalg.norm(p - q) < 0.7 * spacing for q in used):
                continue                                   # crossing / overlapping traces: one hole is enough
            _, outward = frame.axes_at(float(uv[j, 0]))
            inward = -outward
            nn = n if n is not None else np.cross(t_dir, inward)
            nn = nn / max(np.linalg.norm(nn), 1e-9)
            m = inward - (inward @ nn) * nn                # down the joint plane, into the rock
            if np.linalg.norm(m) < 0.3:                    # joint nearly parallel to the face: go straight in
                m = inward
            m /= np.linalg.norm(m)
            side = np.cross(outward, t_dir)
            side /= max(np.linalg.norm(side), 1e-9)
            flat = abs(side[2]) > 0.3
            if flat:                                       # flat trace: collar above it, hole drilled downward
                side *= np.sign(side[2])
            elif len(holes) % 2:
                side = -side
            # on a stepped face the point behind the trace can lie in air (a ledge, a recess): with a mesh the
            # other side and deeper targets are tried and the first hole that stays in the rock is kept
            options = [(f, sg) for f in (1.0, 1.5, 2.0) for sg in ((1,) if flat else (1, -1))] if site else [(1.0, 1)]
            best = None
            for f, sg in options:
                h = _aimed_hole(p, m, side * sg, outward, f * joint_depth, tan, min_plunge, extra_m, site)
                if h is None:
                    continue
                if site is not None:
                    check_breakout([h], site)
                if best is None or (best.get("breakout_m") is not None and h.get("breakout_m") is None):
                    best = h
                if h.get("breakout_m") is None:
                    break
            if best is None:
                continue
            used.append(p)
            best.update({"u": float(uv[j, 0]), "v": float(uv[j, 1]), "joint": name, "pole": nn, "trace": k,
                         "score": float(smap.layers["score"][r[j], c[j]])})
            holes.append(best)
    return holes


def network_layout(result, smap, joints, sets, classes=(2,), length: float = 8.0, max_holes: int = 30,
                   min_gain: float = 1.0, min_gap: float = 2.0, plunges=(10, 20, 30, 45, 60),
                   turns=(-60, -40, -20, 0, 20, 40, 60), high_w: float = 1.0, other_w: float = 0.3,
                   set_bonus: float = 0.5, stat_w: float = 0.2, site: SiteMesh | None = None, rock=None) -> list[dict]:
    """Holes designed on the joint network inside the rock: where to collar, which way and how steep.

    Candidates: a collar on the face in every seepage cell of `classes` x a direction (turned up to 60 deg
    off the face normal, plunging 10-60 deg), `length` m long, kept only if it stays in the rock. Each is
    ray-cast against the joint planes extended into the rock (joints3d.observed): a joint in a high-seepage
    cell counts `high_w`, another one `other_w`; crossing joints of several sets adds `set_bonus` per extra
    set (the hole meets their intersections -- the channels water follows), and the joints beyond the mapped
    extension add `stat_w` x the expected number from the set spacings (Terzaghi, flow-weighted).
    Greedy: take the best hole, the joints it crosses count no more, repeat until a hole would add less than
    `min_gain` or `max_holes` are placed; collars stay `min_gap` m apart.
    """
    import open3d as o3d

    from ..hydro import crossing_weights, hole_frequency
    from . import joints3d

    frame = result.wall_frame
    rock = rock or joints3d.RockTest(result)
    rr, cc = np.nonzero(np.isin(smap.classes, classes))
    if not len(rr) or not joints:
        return []
    u = smap.u_min + (cc + 0.5) * smap.cell_m
    v = smap.v_max - (rr + 0.5) * smap.cell_m
    fw = rock.face_w(u, v)
    ok = np.isfinite(fw)
    u, v, fw, rr, cc = u[ok], v[ok], fw[ok], rr[ok], cc[ok]
    collars = frame.to_world(np.column_stack([u, v, fw]))
    outs = np.array([frame.axes_at(float(x))[1] for x in u])
    if site is not None:                                       # collars onto the actual rock surface
        start = collars + 3.0 * outs
        t = _cast(site.scene, start, -outs)
        hit = np.isfinite(t) & (t < 6.0)
        collars[hit] = start[hit] - t[hit, None] * outs[hit]
    share = crossing_weights(sets) if len(sets) else []
    cand_c, cand_d, cand_tp = [], [], []
    for i in range(len(u)):
        tr_in = float(np.degrees(np.arctan2(-outs[i, 0], -outs[i, 1])))
        for dt in turns:
            for p in plunges:
                cand_c.append(i)
                cand_d.append(_line(tr_in + dt, p))
                cand_tp.append(((tr_in + dt) % 360, p))
    cand_c, cand_d = np.array(cand_c), np.array(cand_d)
    # stays in the rock over its length (a little slack for the rough face near the collar)
    ts = np.arange(0.5, length + 1e-6, 0.5)
    P = collars[cand_c][:, None, :] + ts[None, :, None] * cand_d[:, None, :]
    inside = rock.inside(P.reshape(-1, 3)).reshape(len(cand_c), len(ts))
    good = inside[:, 1:].mean(axis=1) >= 0.95
    if site is not None:
        # the facade test cannot see the air under an overhang: with a mesh, a hole that meets the surface again
        # before its full length leaves the rock and is dropped
        t_out = _cast(site.scene, collars[cand_c] + 0.3 * cand_d, cand_d) + 0.3
        good &= ~(np.isfinite(t_out) & (t_out < length))
    cand_c, cand_d = cand_c[good], cand_d[good]
    cand_tp = [tp for tp, g in zip(cand_tp, good) if g]
    if not len(cand_c):
        return []
    # which joints each candidate crosses
    V, F, owner = joints3d._triangles(joints)
    scene_ = o3d.t.geometry.RaycastingScene()
    scene_.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    rays = np.hstack([collars[cand_c], cand_d]).astype(np.float32)
    res = scene_.list_intersections(o3d.core.Tensor(rays))
    split, th, prim = res["ray_splits"].numpy(), res["t_hit"].numpy(), res["primitive_ids"].numpy()
    weight = np.array([0.0 if j.source == "dfn" else (high_w if j.seepage_class in classes else other_w) for j in joints])
    names = [j.name for j in joints]
    hits = []
    for k in range(len(cand_c)):
        sl = slice(split[k], split[k + 1])
        hits.append(set(owner[prim[sl][th[sl] <= length]].tolist()))
    stat = {tp: stat_w * hole_frequency(sets, share, *tp) * length for tp in set(cand_tp)} if len(sets) else {}
    covered = np.zeros(len(joints), bool)
    chosen, holes = [], []
    while len(holes) < max_holes:
        best, best_gain = None, min_gain
        taken = collars[[c for c in chosen]] if chosen else np.zeros((0, 3))
        for k in range(len(cand_c)):
            if len(taken) and np.min(np.linalg.norm(taken - collars[cand_c[k]], axis=1)) < min_gap:
                continue
            new = [j for j in hits[k] if not covered[j] and weight[j] > 0]
            g = float(weight[new].sum()) if new else 0.0
            n_sets = len({names[j] for j in new if names[j] != "방향 미정"})   # unknown orientation is no set
            g += set_bonus * max(0, n_sets - 1) + stat.get(cand_tp[k], 0.0)
            if g > best_gain and new:
                best, best_gain = k, g
        if best is None:
            break
        new = [j for j in hits[best] if not covered[j] and weight[j] > 0]
        covered[new] = True
        i = cand_c[best]
        chosen.append(i)
        trend, plunge = cand_tp[best]
        holes.append({"collar": collars[i], "dir": cand_d[best], "length": length, "u": float(u[i]), "v": float(v[i]),
                      "network": True, "trend": trend, "plunge": plunge, "gain": round(best_gain, 2),
                      "new_high": int(sum(weight[j] >= high_w for j in new)),
                      "sets": ", ".join(sorted({names[j] for j in hits[best]})),
                      "score": float(smap.layers["score"][rr[i], cc[i]])})
    if site is not None:
        check_breakout(holes, site)
    return holes


def predicted_crossings(hole: dict, sets) -> list[tuple]:
    """(position xyz, set name, pole) of the joints a hole is expected to cross; for a targeted hole the
    mapped joint it is aimed at."""
    if "target" in hole:
        return [(hole["target"], hole["joint"], hole["pole"])]
    out = []
    for r in sets.itertuples():
        n = _pole(r.dip, r.dip_direction)
        c = abs(n @ hole["dir"])
        if c < 0.05:
            continue
        step = r.spacing_m / c
        for t in np.arange(step / 2, hole["length"], step):
            out.append((hole["collar"] + t * hole["dir"], r.name, n))
    return out


def _discs(items, radius: float, origin):
    """One Mesh3d per set: a 12-gon disc per crossing, in the joint plane."""
    meshes = {}
    for p, name, n in items:
        a = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.95 else np.cross(n, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = p + radius * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b)
        m = meshes.setdefault(name, {"v": [], "f": []})
        k = sum(len(x) for x in m["v"])
        m["v"].append(np.vstack([p, ring]) - origin)
        m["f"] += [(k, k + 1 + i, k + 1 + (i + 1) % 12) for i in range(12)]
    return meshes


def scene(result, holes, sets, surface=None, disc_radius: float = 0.5, title: str = "",
          surface_name: str = "암벽 표면", surface_opacity: float = 1.0, cut_chainage=None, joints=None,
          joint_opacity: float = 0.5, joint_shade: str = "seepage", show_predicted: bool = True,
          surface_points: int = 300_000, seepage=None, seepage_classes=(2, 1),
          seepage_opacity: float = 0.55) -> tuple[go.Figure, list]:
    """Plotly figure and a per-hole table.

    cut_chainage (u0, u1): the surface is left out over that chainage range -- a window into the rock (in the
    dashboard's viewer, viz/view3d, a lower `surface_opacity` also shows the rock behind the face; Plotly's own
    WebGL view draws overlapping translucent triangles out of order and turns the face into speckle). Without a `surface` the wall is drawn as `surface_points` points of the cloud in
    the facade photo's colours, at `surface_opacity` like a surface. joints: joints3d.Joint list drawn inside the rock; the holes are ray-cast
    against them (which joints each hole crosses, which high-seepage joints no hole reaches). seepage: a
    hydro.SeepageMap draped on the face (`seepage_classes`, 2 = 상, 1 = 중) with its best zones numbered, to compare
    where the holes go with where water may come out.
    """
    frame = result.wall_frame
    if surface is not None:
        V, F, C = surface
        origin = np.array([np.median(V[:, 0]), np.median(V[:, 1]), 0.0])
        if cut_chainage is not None:
            uc = frame.to_wall(V[F].mean(axis=1))[:, 0]
            F = F[(uc < cut_chainage[0]) | (uc > cut_chainage[1])]
        Vf = (V - origin).astype(np.float32)               # 32-bit arrays: half the HTML size, plenty of precision
        Fi = F.astype(np.int32)
        surf = dict(type="mesh3d", x=Vf[:, 0], y=Vf[:, 1], z=V[:, 2].astype(np.float32), i=Fi[:, 0], j=Fi[:, 1], k=Fi[:, 2],
                         vertexcolor=_hex_colours(C), flatshading=False,
                         lighting=dict(ambient=0.75, diffuse=0.55, specular=0.03, roughness=0.9, fresnel=0.05),
                         lightposition=dict(x=5e4, y=5e4, z=1e5),
                         name=surface_name, hoverinfo="skip", showlegend=True, opacity=surface_opacity)
    else:
        pts = np.asarray(result.point_cloud.points)
        idx = np.random.default_rng(0).choice(len(pts), min(int(surface_points), len(pts)), replace=False)
        P = pts[idx]
        origin = np.array([np.median(P[:, 0]), np.median(P[:, 1]), 0.0])
        col = np.full((len(P), 3), 150)
        if result.facade_rgb is not None:
            uv = frame.to_wall(P)
            r_, c_ = result.facade_rgb.world_to_pixel(uv[:, 0], uv[:, 1])
            ok = (r_ >= 0) & (r_ < result.facade_rgb.array.shape[1]) & (c_ >= 0) & (c_ < result.facade_rgb.array.shape[2])
            col[ok] = np.moveaxis(result.facade_rgb.array[:, r_[ok], c_[ok]], 0, -1)
        if cut_chainage is not None:
            uc = frame.to_wall(P)[:, 0]
            keep = (uc < cut_chainage[0]) | (uc > cut_chainage[1])
            P, col = P[keep], col[keep]
        surf = dict(type="scatter3d", x=(P[:, 0] - origin[0]).astype(np.float32), y=(P[:, 1] - origin[1]).astype(np.float32),
                    z=P[:, 2].astype(np.float32), mode="markers", marker=dict(size=1.6, color=_hex_colours(col)),
                    name="암벽 (점군)", hoverinfo="skip", opacity=surface_opacity)
    traces = [surf]
    if seepage is not None:
        from . import setmap3d

        traces += setmap3d.seepage_traces(result, seepage, origin, seepage_classes, seepage_opacity)

    cov = None
    if joints:
        from . import joints3d

        cov = joints3d.coverage(holes, joints)
    xs, ys, zs, labels = [], [], [], []
    table, crossings = [], []
    for i, h in enumerate(holes):
        end = h["collar"] + h["length"] * h["dir"]
        cr = predicted_crossings(h, sets) if show_predicted else []
        crossings += cr
        counts = {}
        for _, name, _ in cr:
            counts[name] = counts.get(name, 0) + 1
        row = {"hole": i + 1, "chainage_m": round(h["u"], 1), "collar_z": round(float(h["collar"][2]), 1),
               "n_joints": len(cr), **counts}
        if "target" in h:
            tp = _trend_plunge_down(h["dir"])
            row = {"hole": i + 1, "target_joint": h["joint"], "chainage_m": round(h["u"], 1),
                   "elevation_m": round(h["v"], 1), "seepage_score": round(h["score"], 2),
                   "trend": round(tp[0]), "plunge": round(tp[1]), "length_m": round(h["length"], 2),
                   "joint_at_m": round(h["joint_at_m"], 2)}
        if h.get("network"):
            row = {"hole": i + 1, "chainage_m": round(h["u"], 1), "elevation_m": round(h["v"], 1),
                   "trend": round(h["trend"]), "plunge": round(h["plunge"]), "length_m": round(h["length"], 1),
                   "new_high_joints": h["new_high"], "sets_crossed": h["sets"], "seepage_score": round(h["score"], 2)}
        if "breakout_m" in h:
            row["breaks_out_at_m"] = None if h["breakout_m"] is None else round(h["breakout_m"], 1)
        if cov is not None:
            hits = cov["per_hole"][i]
            row["crosses_mapped"] = sum(joints[j].source != "dfn" for j, _ in hits)
            row["crosses_statistical"] = sum(joints[j].source == "dfn" for j, _ in hits)
            row["crosses_high_seepage"] = sum(bool(cov["high"][j]) for j, _ in hits)
        table.append(row)
        for p in (h["collar"], end):
            xs.append(p[0] - origin[0])
            ys.append(p[1] - origin[1])
            zs.append(p[2])
        xs.append(None)
        ys.append(None)
        zs.append(None)
        labels.append(f"공 {i + 1}: 절리 {len(cr)}개")
    traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#7f1d1d", width=7), name="그라우팅 공",
                               hoverinfo="skip"))
    col = np.array([h["collar"] for h in holes]) if holes else np.zeros((0, 3))
    traces.append(go.Scatter3d(x=col[:, 0] - origin[0], y=col[:, 1] - origin[1], z=col[:, 2], mode="markers",
                               marker=dict(size=3.5 if seepage is None else 5.5, color="#fde047",
                                           line=dict(color="black", width=1)),
                               text=labels, hoverinfo="text", name="공 입구",
                               meta="ontop" if seepage is not None else None))      # over the draped map
    out_pts = np.array([h["collar"] + h["breakout_m"] * h["dir"] for h in holes if h.get("breakout_m") is not None])
    if len(out_pts):
        traces.append(go.Scatter3d(x=out_pts[:, 0] - origin[0], y=out_pts[:, 1] - origin[1], z=out_pts[:, 2],
                                   mode="markers", marker=dict(size=7, color="#dc2626", symbol="x"),
                                   name="암반 밖으로 나가는 지점", hoverinfo="name"))
    tr = [t for t in result.traces if t.xyz is not None]
    if tr:                                             # the mapped joints on the face, to judge what the holes aim at
        tx, ty, tz = [], [], []
        for t in tr:
            tx += [t.xyz[0, 0] - origin[0], t.xyz[1, 0] - origin[0], None]
            ty += [t.xyz[0, 1] - origin[1], t.xyz[1, 1] - origin[1], None]
            tz += [t.xyz[0, 2], t.xyz[1, 2], None]
        traces.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines", line=dict(color="#67e8f9", width=3),
                                   name="매핑된 트레이스 (표면 절리)", hoverinfo="name"))
    targeted = any("target" in h for h in holes)
    if targeted:                                       # the mapped traces the holes are aimed at
        tx, ty, tz = [], [], []
        for k in dict.fromkeys(h["trace"] for h in holes if "trace" in h):
            e = result.traces[k].xyz
            tx += [e[0, 0] - origin[0], e[1, 0] - origin[0], None]
            ty += [e[0, 1] - origin[1], e[1, 1] - origin[1], None]
            tz += [e[0, 2], e[1, 2], None]
        traces.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines", line=dict(color="#06b6d4", width=9),
                                   name="주입 대상 절리 (트레이스)", hoverinfo="name"))
    names = list(dict.fromkeys(r.name for r in sets.itertuples()))
    if joints:
        from . import joints3d

        def colour_of(name):
            return "#9ca3af" if name not in names else _set_color(name, names.index(name))

        traces += joints3d.mesh_traces(joints, colour_of, joint_opacity, joint_shade, origin)
        P = np.array([holes[i]["collar"] + t * holes[i]["dir"] for i, hh in enumerate(cov["per_hole"]) for _, t in hh])
        if len(P):
            traces.append(go.Scatter3d(x=P[:, 0] - origin[0], y=P[:, 1] - origin[1], z=P[:, 2], mode="markers",
                                       marker=dict(size=3, color="white", line=dict(color="black", width=1)),
                                       name="공이 절리를 지나는 점", hoverinfo="name"))
        miss = [j for j, (hi_, h_) in enumerate(zip(cov["high"], cov["hit"])) if hi_ and not h_]
        if miss:
            lx, ly, lz = [], [], []
            for j in miss:
                q = joints[j].poly
                q = np.vstack([q[1:], q[1:2]]) if joints[j].source != "trace" else np.vstack([q, q[:1]])
                lx += list(q[:, 0] - origin[0]) + [None]
                ly += list(q[:, 1] - origin[1]) + [None]
                lz += list(q[:, 2]) + [None]
            traces.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#dc2626", width=4),
                                       name=f"공이 닿지 않는 유출 '상' 절리 ({len(miss)})", hoverinfo="name"))
    for name, m in _discs(crossings, disc_radius, origin).items():
        v = np.vstack(m["v"])
        f = np.array(m["f"])
        idx = names.index(name) if name in names else None
        traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=_set_color(name, idx) if idx is not None else "#9ca3af", opacity=0.85,
                                name=f"{name} {'공이 만나는 위치' if targeted else '예상 관통'}",
                                showlegend=True, hoverinfo="name"))
    _, outward = frame.axes_at(frame.length / 2)
    eye = np.array([outward[0], outward[1], 0.0]) * 1.4 + np.array([0, 0, 0.5])
    # traces are built from checked numpy arrays; Plotly's per-element validation of ~10^6 colour strings and
    # coordinates would take most of the drawing time
    fig = go.Figure(traces, skip_invalid=False, _validate=False)
    fig.update_layout(
        title=title, height=760, margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"),
        scene=dict(aspectmode="data", xaxis_title=f"동 (m, +{origin[0]:.0f})", yaxis_title=f"북 (m, +{origin[1]:.0f})",
                   zaxis_title="표고 (m)", camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))))
    if cov is not None:
        obs = np.array([j.source != "dfn" for j in joints], bool)
        fig.layout.meta = {"high_joints": cov["n_high"], "high_joints_hit": cov["n_high_hit"],
                           "joints": int(obs.sum()), "joints_hit": int((cov["hit"] & obs).sum()),
                           "dfn": int((~obs).sum()), "dfn_hit": int((cov["hit"] & ~obs).sum())}
    return fig, table
