"""The joint-set map off the facade: on the point cloud in 3D, draped on the tiled model, and continued into the
rock behind the face ("face mapping" read inward), with sections through that rock.

Everything here is in wall coordinates, the frame the facade map already uses: x = chainage u (along the wall),
z = elevation v, and depth d = -w (metres behind the wall line, + into the rock). The wall is unrolled, so a
section "at depth d" or "at chainage u" means the same thing all along a bent wall.

* `cloud_sample` / `cloud_figure` -- the analysis cloud, points of a joint set in its colour, the rest in the
  facade photo's colour (the cloud itself carries the set colours, not the photo).
* `interior_figure` -- the joint planes inferred from the exposed face (joints3d.observed: every mapped trace and
  every exposed facet continued in its own plane, clipped to the rock; optionally the statistical network)
  as 3D surfaces, with the face points for context and the section planes.
* `plan_section` (horizontal, x-y), `parallel_section` (a surface d m behind the face, x-z) and `cross_section`
  (one chainage, y-z) -- the same joint surfaces cut by a plane, for what is hard to see in 3D.
* `tiles_overlay` -- the set map as an image draped on the front surface, for the CesiumJS tiled-model page.
"""
import numpy as np
import plotly.graph_objects as go

from .. import report

PHOTO_DIM = 0.85            # photo colour of the points outside any set, a little darker than the set colours
SET_MIX = 0.8               # share of the set colour in a set point (the rest is the photo, which keeps the texture)


def depth_of(uvw):
    return -uvw[:, 2]


def set_colour(sid: int) -> str:
    return report.SET_COLORS[int(sid) % len(report.SET_COLORS)]


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def facade_rgb_at(result, u, v) -> np.ndarray:
    """(n, 3) uint8 photo colour of the facade at chainage / elevation (grey off the image)."""
    fr = result.facade_rgb
    out = np.full((len(u), 3), 150, np.uint8)
    if fr is None:
        return out
    t = fr.transform
    r = ((v - t.f) / t.e).astype(int)
    c = ((u - t.c) / t.a).astype(int)
    H, W = fr.array.shape[1:]
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    out[ok] = np.moveaxis(fr.array[:, r[ok], c[ok]], 0, -1)
    return out


def cloud_sample(result, max_points: int = 250_000, seed: int = 0) -> dict:
    """A fixed random sample of the analysis cloud in wall coordinates with set ids and photo colours."""
    pts = np.asarray(result.point_cloud.points)
    n = len(pts)
    idx = np.arange(n) if n <= max_points else np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    uvw = result.wall_frame.to_wall(pts[idx])
    sid = np.asarray(result.point_labels[idx, 0], int)
    return {"uvw": uvw, "set": sid, "photo": facade_rgb_at(result, uvw[:, 0], uvw[:, 1])}


def dense_face_sample(result) -> dict:
    """The ~3 M point wall-coordinate sample the sections use (section._profile_points) with photo colours."""
    from .section import _profile_points

    uvw = _profile_points(result)
    return {"uvw": uvw, "set": np.full(len(uvw), -1), "photo": facade_rgb_at(result, uvw[:, 0], uvw[:, 1])}


def _hex(C) -> list:
    from .grouting3d import _hex_colours

    return _hex_colours(np.clip(C, 0, 255).astype(np.uint8))


def _in_range(uvw, u_range):
    if u_range is None:
        return np.ones(len(uvw), bool)
    return (uvw[:, 0] >= u_range[0]) & (uvw[:, 0] <= u_range[1])


class Disp:
    """Wall coordinates (u, v, w) -> the 3D view's (x, y, z).

    Unrolled (no frame): x = chainage u, y = depth -w, z = elevation v. Where the wall line bends (an arc) these
    axes turn with it, so a flat joint plane shows bent. True plan (`frame` given): map coordinates turned so that
    the wall direction at chainage `u_mid` runs along x (x = u_mid there, straight distances), y = straight depth
    behind the wall line at u_mid, z = elevation -- flat planes stay flat."""

    def __init__(self, frame=None, u_mid: float = 0.0):
        self.frame = frame
        self.u_mid = float(u_mid)
        if frame is not None:
            t, o = frame.axes_at(self.u_mid)
            self.t2 = np.asarray(t[:2], float) / np.linalg.norm(t[:2])
            self.o2 = np.asarray(o[:2], float) / np.linalg.norm(o[:2])
            self.origin = frame.to_world(np.array([[self.u_mid, 0.0, 0.0]]))[0][:2]

    @property
    def plan(self) -> bool:
        return self.frame is not None

    def __call__(self, uvw) -> np.ndarray:
        p = np.asarray(uvw, float).reshape(-1, 3)
        if self.frame is None:
            return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])
        X = self.frame.to_world(p)
        d = X[:, :2] - self.origin
        return np.column_stack([d @ self.t2 + self.u_mid, -(d @ self.o2), X[:, 2]])

    def polyline(self, corners_uvw, n: int = 40) -> np.ndarray:
        """A closed / open outline given in wall coordinates, subdivided so that it follows the wall in true plan."""
        c = np.asarray(corners_uvw, float)
        f = np.linspace(0, 1, n, endpoint=False)
        pts = [a + f[:, None] * (b - a) for a, b in zip(c[:-1], c[1:])] + [c[-1:]]
        return self(np.vstack(pts))

    def titles(self):
        if self.frame is None:
            return "측점 x (m)", "깊이 y (m, + 암반 안쪽)", "표고 z (m)"
        return "벽 방향 거리 x (m, 직선)", "깊이 y (m, 직선, + 암반 안쪽)", "표고 z (m)"


UNROLLED = Disp()


def _scene(fig, height, eye=(0.0, -1.6, 0.55), disp: Disp = UNROLLED):
    tx, ty, tz = disp.titles()
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(itemsizing="constant", orientation="h", y=1.0, yanchor="bottom"), uirevision="keep",
        scene=dict(aspectmode="data", xaxis_title=tx, yaxis_title=ty, zaxis_title=tz, dragmode="orbit",
                   # in front of the wall, looking in: chainage runs left to right as on the map
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1))))
    return fig


def cloud_figure(sample: dict, sets: list, u_range=None, show_other: bool = True, size: float = 1.6,
                 height: int = 820, disp: Disp = UNROLLED) -> go.Figure:
    """Point cloud: points of each joint set in the set colour (over the photo), the rest in photo colour."""
    uvw, sid, photo = sample["uvw"], sample["set"], sample["photo"].astype(float)
    keep = _in_range(uvw, u_range)
    traces = []
    if show_other:
        m = keep & (sid < 0)
        P = disp(uvw[m])
        traces.append(dict(type="scatter3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="기타 암면·식생 (사진 색)",
                                   marker=dict(size=size, color=_hex(photo[m] * PHOTO_DIM)), hoverinfo="skip"))
    for d in sets:
        m = keep & (sid == d.set_id)
        if not m.any():
            continue
        col = SET_MIX * _rgb(set_colour(d.set_id))[None] + (1 - SET_MIX) * photo[m]
        P = disp(uvw[m])
        traces.append(dict(
            type="scatter3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            name=f"Set {d.set_id}  {d.mean_dip:.0f}°/{d.mean_dip_direction:03.0f}°",
            marker=dict(size=size + 0.3, color=_hex(col)), hoverinfo="skip"))
    return _scene(go.Figure(traces, skip_invalid=False, _validate=False), height, disp=disp)


# --- joint surfaces in wall coordinates ------------------------------------------------------------------------

def joint_mesh(result, joints) -> dict:
    """All joint polygons as one triangle mesh in wall coordinates (V (n, 3) u v w, F (m, 3), owner (m,))."""
    from .joints3d import _triangles

    V, F, owner = _triangles(joints)
    return {"V": result.wall_frame.to_wall(V) if len(V) else np.zeros((0, 3)), "F": F, "owner": owner}


def _group_mesh(mesh, idx):
    """Compact (V, F) of the triangles owned by the joints `idx`."""
    sel = np.isin(mesh["owner"], idx)
    F = mesh["F"][sel]
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return mesh["V"][used], inv.reshape(-1, 3)


def _groups(joints, colour_of, max_draw_dfn):
    from .joints3d import joint_groups

    return joint_groups(joints, colour_of, shade="none", max_draw_dfn=max_draw_dfn)


def face_surface(result, u_range=None, max_cells: int = 600):
    """The front surface of the wall in the chainage window as a coloured triangle mesh in wall coordinates
    (u, v, w): the front-most facade point per grid cell (5-10 cm, at most `max_cells` along the
    wall) with the facade photo's colour. No triangle across a step in depth (overhang lip, wall foot), where it
    would hang as a long sliver. Returns (V (n, 3) float32, F (m, 3), C (n, 3) uint8) or None."""
    d, fr = result.facade_depth, result.facade_rgb
    if d is None:
        return None
    t = d.transform
    px = abs(t.a)
    dep = d.array[0]
    H, W = dep.shape
    u0_all, u1_all = t.c, t.c + W * px
    lo, hi = u_range if u_range is not None else (u0_all, u1_all)
    lo, hi = max(lo, u0_all), min(hi, u1_all)
    if hi <= lo:
        return None
    step = max(0.05, (hi - lo) / max_cells)
    k = max(1, int(round(step / px)))
    step = k * px
    c0, c1 = int((lo - t.c) / px) // k * k, min(W, int(np.ceil((hi - t.c) / px)))
    Hk, Wk = H // k, (c1 - c0) // k
    if Hk < 2 or Wk < 2:
        return None
    blk = dep[:Hk * k, c0:c0 + Wk * k].reshape(Hk, k, Wk, k)
    with np.errstate(all="ignore"):
        front = np.nanmax(blk, axis=(1, 3))
    rr = np.arange(Hk) * k + k // 2
    cc = c0 + np.arange(Wk) * k + k // 2
    if fr is not None:
        C = np.moveaxis(fr.array[:, rr[:, None], cc[None, :]], 0, -1).reshape(-1, 3)
    else:
        C = np.full((Hk * Wk, 3), 160, np.uint8)
    gu = (t.c + (cc + 0.5) * px)[None, :].repeat(Hk, 0)
    gv = (t.f - (rr + 0.5) * px)[:, None].repeat(Wk, 1)
    V = np.column_stack([gu.ravel(), gv.ravel(), front.ravel()])
    ok = np.isfinite(front).ravel()
    idx = np.arange(Hk * Wk).reshape(Hk, Wk)
    a, b, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    F = np.column_stack([np.r_[a, b], np.r_[c, c], np.r_[b, e]])
    F = F[ok[F].all(axis=1)]
    dw = V[F, 2]
    F = F[dw.max(axis=1) - dw.min(axis=1) < 4 * step]
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return V[used], inv.reshape(-1, 3), C[used].astype(np.uint8)


def interior_figure(sample: dict, joints, mesh, colour_of, u_range=None, opacity: float = 0.75,
                    face_mode: str = "surface", face_opacity: float = 0.35, face_points: int = 500_000,
                    surface=None, planes: dict | None = None, height: int = 820, opacity_of=None,
                    disp: Disp = UNROLLED) -> go.Figure:
    """Joint planes behind the face as surfaces (one per set / kind), the face for context -- a see-through
    surface (`surface` from face_surface) or `face_points` points of `sample` -- and the current section planes
    as thin outlines."""
    traces = []
    uvw = sample["uvw"]
    keep = _in_range(uvw, u_range)
    lo_u, hi_u = (u_range if u_range is not None else (uvw[:, 0].min(), uvw[:, 0].max()))
    if face_mode == "surface" and surface is not None:
        V, F, C = surface
        V = disp(V).astype(np.float32)
        traces.append(dict(type="mesh3d", x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], vertexcolor=_hex(C),
                                opacity=face_opacity, name="암면 (표면)", hoverinfo="skip", showlegend=True))
    elif face_mode == "points":
        idx = np.nonzero(keep)[0]
        idx = idx[:: max(1, len(idx) // face_points)]
        P = disp(uvw[idx]).astype(np.float32)
        traces.append(dict(type="scatter3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="암면 (점군)",
                                   marker=dict(size=1.6, color=_hex(sample["photo"][idx])), opacity=face_opacity,
                                   hoverinfo="skip"))
    inside = np.ones(len(joints), bool)
    if u_range is not None and len(mesh["F"]):
        # a joint is kept if any of its triangles reaches into the chainage window
        vin = (mesh["V"][:, 0] >= lo_u) & (mesh["V"][:, 0] <= hi_u)
        inside[:] = False
        inside[np.unique(mesh["owner"][vin[mesh["F"]].any(axis=1)])] = True
    for g in _groups(joints, colour_of, 1500):
        idx = [i for i in g["idx"] if inside[i]]
        if not idx:
            continue
        V, F = _group_mesh(mesh, idx)
        V = disp(V)
        dfn = g["kind"] == "dfn"
        traces.append(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                                color=colour_of(g["name"]), name=g["label"],
                                opacity=opacity * (0.55 if dfn else 1.0) * (opacity_of(g["name"]) if opacity_of else 1.0),
                                showlegend=True, hoverinfo="name", flatshading=True,
                                lighting=dict(ambient=0.75, diffuse=0.45, specular=0.05)))
    if planes:
        z_lo, z_hi = float(uvw[keep, 1].min()), float(uvw[keep, 1].max())
        d_lo, d_hi = float(depth_of(uvw[keep]).min()), float(depth_of(uvw[keep]).max()) + planes.get("depth_shown", 8.0)
        box = []                               # outlines in wall coordinates (u, v, w = -depth)
        if planes.get("v") is not None:
            v = planes["v"]
            box.append(([(lo_u, v, -d_lo), (hi_u, v, -d_lo), (hi_u, v, -d_hi), (lo_u, v, -d_hi), (lo_u, v, -d_lo)], "수평 단면"))
        if planes.get("u") is not None:
            u = planes["u"]
            box.append(([(u, z_lo, -d_lo), (u, z_lo, -d_hi), (u, z_hi, -d_hi), (u, z_hi, -d_lo), (u, z_lo, -d_lo)], "횡단면"))
        for corners, name in box:
            X = disp.polyline(corners)
            traces.append(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="lines",
                                       line=dict(color="#111827", width=5, dash="dash"), name=f"{name} 위치", hoverinfo="name"))
    # from the front, off to one side and above: the joints are behind the face, an oblique view sees past it
    return _scene(go.Figure(traces, skip_invalid=False, _validate=False), height, eye=(-1.0, -1.35, 0.9), disp=disp)


# --- sections ------------------------------------------------------------------------------------------------

def cut_segments(mesh, s: np.ndarray):
    """Where the triangle mesh crosses the zero level of `s` (a value per vertex): segment end points (k, 2, 3)
    in wall coordinates and the joint each segment belongs to (k,)."""
    F = mesh["F"]
    if not len(F):
        return np.zeros((0, 2, 3)), np.zeros(0, int)
    V = mesh["V"]
    sv = s[F]                                                             # (m, 3)
    pos = sv > 0
    cross = pos.any(axis=1) & ~pos.all(axis=1)
    F, sv = F[cross], sv[cross]
    owner = mesh["owner"][cross]
    pts = np.full((len(F), 3, 3), np.nan)
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        sa, sb = sv[:, a], sv[:, b]
        hit = (sa > 0) != (sb > 0)
        t = np.where(hit, sa / np.where(hit, sa - sb, 1.0), np.nan)
        pts[:, k] = V[F[:, a]] + t[:, None] * (V[F[:, b]] - V[F[:, a]])
    ok = np.isfinite(pts[..., 0])
    good = ok.sum(axis=1) == 2                                           # a plane cuts a triangle on two edges
    first = np.argsort(~ok[good], axis=1, kind="stable")[:, :2]          # the two crossed edges, in order
    seg = np.take_along_axis(pts[good], first[:, :, None], axis=1)
    return seg, owner[good]


def _segment_traces(seg, owner, joints, colour_of, cols, max_dfn=10 ** 9):
    """Plotly line traces of the cut segments, one per joint group; `cols` picks the 2 plot coordinates."""
    out = []
    for g in _groups(joints, colour_of, max_dfn):
        m = np.isin(owner, g["idx"])
        if not m.any():
            continue
        s = seg[m]
        xy = np.full((len(s), 3, 2), np.nan)
        xy[:, 0] = np.column_stack([cols[0](s[:, 0]), cols[1](s[:, 0])])
        xy[:, 1] = np.column_stack([cols[0](s[:, 1]), cols[1](s[:, 1])])
        xy = xy.reshape(-1, 2)
        dfn = g["kind"] == "dfn"
        out.append(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", name=g["label"].split(" — ")[0],
                              line=dict(color=colour_of(g["name"]), width=1.4 if dfn else 3, dash="dot" if dfn else "solid"),
                              opacity=0.65 if dfn else 0.95, hoverinfo="name", connectgaps=False))
    return out


# where the other two sections cut, drawn on each section: a dark dashed line (the joint sets have the colours)
# with a badge in the colour of that section's card in the dashboard
REF_COLOURS = {"A": "#1d4ed8", "B": "#15803d", "C": "#c2410c"}
REF_LINE = "#334155"


def _ref_line(fig, x, y, tag: str, text: str, where: str = "end", anchor=("right", "bottom")):
    """Dashed line (x, y; NaN breaks it) for section `tag`, labelled at its first or last point (the badge's
    `anchor` corner on that point, so that it stays inside the plot)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if not len(ok):
        return
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=REF_LINE, width=1.6, dash="dash"),
                             name=f"{tag} 단면 위치", showlegend=False, hoverinfo="name", connectgaps=False))
    k = ok[-1] if where == "end" else ok[0]
    fig.add_annotation(x=float(x[k]), y=float(y[k]), text=f"<b>{tag}</b> {text}", showarrow=False,
                       bgcolor=REF_COLOURS[tag], font=dict(color="white", size=11), borderpad=3,
                       xanchor=anchor[0], yanchor=anchor[1])


def _section_layout(fig, title, xt, yt, height=640, xr=None, yr=None):
    """1:1 section; with both ranges given the plot box shrinks to fit them (constrain="domain") instead of one
    axis stretching to keep the scale."""
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#f9fafb",
                      legend=dict(itemsizing="constant"),
                      xaxis=dict(title=xt, zeroline=False, range=xr, constrain="domain" if xr else None),
                      yaxis=dict(title=yt, zeroline=False, scaleanchor="x", scaleratio=1, range=yr,
                                 constrain="domain" if yr else None))
    return fig


def plan_section(sample, rock, joints, mesh, colour_of, v0: float, u_range=None, slab_m: float = 0.15,
                 depth_shown: float = 8.0, frame=None, refs: dict | None = None) -> go.Figure:
    """Horizontal section at elevation v0 (looking down; the face at the bottom, the rock above it).

    With `frame` (the wall frame) it is drawn in true plan: map coordinates turned so that the wall direction at
    the middle of the window runs left to right, distances straight. Without it, in wall coordinates (chainage
    along the wall line, depth normal to it) -- where the wall line bends (an arc) those axes turn with it and a
    straight joint shows as a curve.

    `refs` {"u": cross-section chainage, "d": wall-parallel section depth}: where those sections cut this one."""
    uvw = sample["uvw"]
    lo, hi = u_range if u_range is not None else (uvw[:, 0].min(), uvw[:, 0].max())
    if frame is not None:
        mid = 0.5 * (lo + hi)
        t_, o_ = frame.axes_at(float(mid))
        t2 = np.asarray(t_[:2], float) / np.linalg.norm(t_[:2])
        o2 = np.asarray(o_[:2], float) / np.linalg.norm(o_[:2])
        origin = frame.to_world(np.array([[mid, v0, 0.0]]))[0][:2]

        def to_xy(p):
            X = frame.to_world(np.asarray(p, float).reshape(-1, 3))[:, :2] - origin
            return X @ t2 + mid, -(X @ o2)
    else:
        def to_xy(p):
            p = np.asarray(p, float).reshape(-1, 3)
            return p[:, 0], -p[:, 2]
    traces = []
    us = np.arange(lo, hi + 0.1, 0.1)
    fw = rock.face_w(us, np.full(len(us), v0))
    ok = np.isfinite(fw)
    xr = yr = None
    if ok.any():
        fx, fy = to_xy(np.column_stack([us, np.full(len(us), v0), np.nan_to_num(fw)]))
        bx, by = to_xy(np.column_stack([us, np.full(len(us), v0), np.nan_to_num(fw) - depth_shown]))
        xs, ys = [], []
        # rock body: from the face back `depth_shown`, per run of chainage where the face exists
        runs = np.split(np.arange(len(us)), np.nonzero(np.diff(ok.astype(int)))[0] + 1)
        for r in runs:
            if not ok[r[0]]:
                continue
            xs += list(fx[r]) + list(bx[r][::-1]) + [fx[r][0], None]
            ys += list(fy[r]) + list(by[r][::-1]) + [fy[r][0], None]
        traces.append(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(180,150,110,0.18)", line=dict(width=0),
                                 name="암반 (단면)", hoverinfo="skip"))
        allx, ally = np.r_[fx[ok], bx[ok]], np.r_[fy[ok], by[ok]]
        xr = [float(allx.min()), float(allx.max())]
        yr = [float(ally.min()) - 1.5, float(ally.max()) + 1.0]
    sl = (np.abs(uvw[:, 1] - v0) < slab_m) & (uvw[:, 0] >= lo) & (uvw[:, 0] <= hi)
    if sl.any():
        px_, py_ = to_xy(uvw[sl])
        traces.append(go.Scattergl(x=px_, y=py_, mode="markers", name="암면 (점군 단면)",
                                   marker=dict(size=2, color="#6b7280"), hoverinfo="skip"))
    seg, own = cut_segments(mesh, mesh["V"][:, 1] - v0)
    inside = (seg[:, :, 0].max(axis=1) >= lo) & (seg[:, :, 0].min(axis=1) <= hi) if len(seg) else np.zeros(0, bool)
    traces += _segment_traces(seg[inside], own[inside], joints, colour_of,
                              (lambda p: to_xy(p)[0], lambda p: to_xy(p)[1]))
    fig = go.Figure(traces)
    if refs and ok.any():
        if refs.get("d") is not None:                 # the face peeled back by d, at this elevation
            run = np.where(ok, fw - refs["d"], np.nan)
            bx, by = to_xy(np.column_stack([us, np.full(len(us), v0), np.nan_to_num(run)]))
            _ref_line(fig, np.where(ok, bx, np.nan), np.where(ok, by, np.nan), "B", f"깊이 {refs['d']:.2f} m")
        u0 = refs.get("u")
        if u0 is not None and lo <= u0 <= hi:          # the cross section: the wall normal at that chainage
            f0 = rock.face_w(np.array([u0]), np.array([v0]))[0]
            f0 = float(f0) if np.isfinite(f0) else float(np.nanmedian(fw))
            ws = np.linspace(f0 + 1.0, f0 - depth_shown, 30)
            cx, cy = to_xy(np.column_stack([np.full(len(ws), u0), np.full(len(ws), v0), ws]))
            _ref_line(fig, cx, cy, "C", f"측점 {u0:.2f} m", anchor=("left", "top"))
    if frame is not None:
        return _section_layout(fig, f"수평 단면 (실제 평면): 표고 {v0:.1f} m — 구간 가운데의 벽 방향을 가로로 놓은 지도",
                               "벽 방향 거리 x (m, 직선 · 구간 가운데 = 그 측점)", "깊이 y (m, 직선, + 암반 안쪽)", xr=xr, yr=yr)
    return _section_layout(fig, f"수평 단면 (측점 기준): 표고 {v0:.1f} m — 벽 기준선을 따라 펼친 좌표",
                           "측점 x (m)", "깊이 y (m, + 암반 안쪽)", xr=[lo, hi], yr=yr)


def parallel_section(result, rock, joints, mesh, colour_of, depth: float, show_traces: bool = True,
                     refs: dict | None = None, u_range=None) -> go.Figure:
    """Section along the wall, `depth` m behind the face (x-z): the face peeled back by `depth`. Where a joint
    crosses this surface; the face traces (thin grey) show where the same joints come out at the face.
    `refs` {"v": plan-section elevation, "u": cross-section chainage}: where those sections cut this one.
    `u_range` limits the chainage shown (default: the whole wall)."""
    V = mesh["V"]
    fw = rock.face_w(V[:, 0], V[:, 1]) if len(V) else np.zeros(0)
    s = np.where(np.isfinite(fw), V[:, 2] - (fw - depth), np.nan)
    m2 = dict(mesh, V=V)
    seg, own = cut_segments(m2, np.nan_to_num(s, nan=1.0))
    traces = []
    # rock region: where the face (and so this surface) exists
    valid = np.isfinite(rock.front)
    H, W = valid.shape
    img = np.where(valid, 1.0, np.nan)
    traces.append(go.Heatmap(z=img[::-1], x0=rock.u0 + rock.cell / 2, dx=rock.cell, y0=rock.v0 - H * rock.cell + rock.cell / 2,
                             dy=rock.cell, colorscale=[[0, "rgba(180,150,110,0.22)"], [1, "rgba(180,150,110,0.22)"]],
                             showscale=False, hoverinfo="skip", name="암반"))
    if show_traces and result.traces:
        tr = [t for t in result.traces if t.length_m >= 0.5]
        seg2 = np.full((len(tr), 3, 2), np.nan)
        seg2[:, :2] = np.array([t.coords for t in tr]) if tr else np.zeros((0, 2, 2))
        xy = seg2.reshape(-1, 2)
        traces.append(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color="#6b7280", width=1),
                                   opacity=0.6, name="암면 트레이스 (참고: 깊이 0)", hoverinfo="skip", connectgaps=False))
    traces += _segment_traces(seg, own, joints, colour_of, (lambda p: p[:, 0], lambda p: p[:, 1]))
    fig = go.Figure(traces)
    xr = [rock.u0, rock.u0 + W * rock.cell] if u_range is None else [float(u_range[0]), float(u_range[1])]
    yr = [rock.v0 - H * rock.cell, rock.v0]
    if refs:
        if refs.get("v") is not None:
            _ref_line(fig, xr, [refs["v"]] * 2, "A", f"표고 {refs['v']:.2f} m", where="start", anchor=("left", "bottom"))
        if refs.get("u") is not None:
            _ref_line(fig, [refs["u"]] * 2, yr[::-1], "C", f"측점 {refs['u']:.2f} m", where="start",
                      anchor=("left", "top"))
    return _section_layout(fig, f"벽면 평행 단면 (x-z): 암면에서 {depth:.1f} m 안쪽 — 암면을 그만큼 벗겨낸 모습 (벽 전체)",
                           "측점 x (m)", "표고 z (m)", xr=xr, yr=yr)


def cross_section(sample, joints, mesh, colour_of, u0: float, slab_m: float = 0.15,
                  depth_shown: float = 8.0, refs: dict | None = None, rock=None) -> go.Figure:
    """Section at one chainage (y-z), looking along the wall: the face on the left, the rock to the right.
    `refs` {"v": plan-section elevation, "d": wall-parallel section depth}: where those sections cut this one
    (the depth from `rock`'s face, as the wall-parallel section itself, else from the face points here)."""
    uvw = sample["uvw"]
    sl = np.abs(uvw[:, 0] - u0) < slab_m
    traces = []
    xr = yr = None
    if sl.sum() > 20:
        s = uvw[sl]
        vb = np.floor(s[:, 1] / 0.25).astype(int)
        order = np.unique(vb)
        front = np.array([np.percentile(-s[vb == b, 2], 5) for b in order])   # face = smallest depth
        vv = (order + 0.5) * 0.25
        traces.append(go.Scatter(x=np.r_[front, [front.max() + depth_shown] * 2],
                                 y=np.r_[vv, [vv[-1], vv[0]]], fill="toself", fillcolor="rgba(180,150,110,0.18)",
                                 line=dict(width=0), name="암반 (단면)", hoverinfo="skip"))
        traces.append(go.Scattergl(x=-s[:, 2], y=s[:, 1], mode="markers", marker=dict(size=2, color="#6b7280"),
                                   name="암면 (점군 단면)", hoverinfo="skip"))
        xr = [float(front.min()) - 1.5, float(front.max()) + depth_shown + 1.0]
        yr = [float(vv.min()) - 0.5, float(vv.max()) + 0.5]
    seg, own = cut_segments(mesh, mesh["V"][:, 0] - u0)
    traces += _segment_traces(seg, own, joints, colour_of, (lambda p: -p[:, 2], lambda p: p[:, 1]))
    fig = go.Figure(traces)
    if refs and xr is not None:
        if refs.get("d") is not None:
            if rock is not None:
                fwv = rock.face_w(np.full(len(vv), u0), vv)
                dx = np.where(np.isfinite(fwv), -fwv + refs["d"], np.nan)
            else:
                dx = front + refs["d"]
            _ref_line(fig, dx, vv, "B", f"깊이 {refs['d']:.2f} m", anchor=("left", "top"))
        if refs.get("v") is not None:
            _ref_line(fig, xr, [refs["v"]] * 2, "A", f"표고 {refs['v']:.2f} m", where="start", anchor=("left", "bottom"))
    return _section_layout(fig, f"횡단면 (y-z): 측점 {u0:.1f} m — 왼쪽이 암면, 오른쪽으로 갈수록 암반 안쪽",
                           "깊이 y (m, + 암반 안쪽)", "표고 z (m)", xr=xr, yr=yr)


# --- the set map image ----------------------------------------------------------------------------------------

def setmap_rgba(result, max_px: int = 3000, alpha: int = 150):
    """(RGBA uint8 set-colour image of the facade, transparent off any set, at most `max_px`; pixel size u, v)."""
    import cv2

    lab = result.facade_sets.array[0].astype(np.int16)
    h, w = lab.shape
    k = max(1.0, max(h, w) / max_px)
    if k > 1:
        lab = cv2.resize(lab, (max(1, round(w / k)), max(1, round(h / k))), interpolation=cv2.INTER_NEAREST)
    out = np.zeros(lab.shape + (4,), np.uint8)
    for s in range(int(lab.max()) + 1):
        m = lab == s
        out[m, :3] = _rgb(set_colour(s)).astype(np.uint8)
        out[m, 3] = alpha
    t = result.facade_sets.transform
    return out, abs(t.a) * w / lab.shape[1], abs(t.e) * h / lab.shape[0]


def setmap_blend(result, max_px: int = 3000) -> np.ndarray:
    """The facade photo with the set colours over it (RGB uint8, top row first), as in the report figure."""
    rgb, _ = report.facade_preview(result, max_px)
    ov = report.set_overlay(result, rgb.shape[:2]).astype(float) / 255
    out = rgb.astype(float) / 255 * (1 - ov[..., 3:4]) + ov[..., :3] * ov[..., 3:4]
    out = (out * 255).astype(np.uint8)
    out[~np.isfinite(_resize_nearest(result.facade_depth.array[0], rgb.shape[:2]))] = 255
    return out


def _resize_nearest(a, shape):
    import cv2

    return cv2.resize(a.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _drape_grid(result, u0: float, v1: float, U: float, Vh: float, step_m: float, offset_m: float):
    """A triangle grid lying on the front surface over chainage u0..u0+U, elevation v1-Vh..v1 (every `step_m`,
    lifted `offset_m` toward the viewer): (grid u, grid v, world xyz per vertex, triangles), or None.

    The surface is the front-most facade point per grid cell (ground and benches included, unlike the rock test,
    which leaves flat ground out); no skin across a step in depth (overhang lip, wall foot onto the ground)."""
    us = np.arange(u0 + step_m / 2, u0 + U, step_m)
    vs = np.arange(v1 - step_m / 2, v1 - Vh, -step_m)
    if len(us) < 2 or len(vs) < 2:
        return None
    gu, gv = np.meshgrid(us, vs)
    d = result.facade_depth
    dt = d.transform
    k = max(1, int(round(step_m / abs(dt.a))))
    dep = d.array[0].astype(np.float32)
    Hk, Wk = dep.shape[0] // k, dep.shape[1] // k
    with np.errstate(all="ignore"):
        front = np.nanmax(dep[:Hk * k, :Wk * k].reshape(Hk, k, Wk, k), axis=(1, 3))
    rr = ((dt.f - gv.ravel()) / (abs(dt.e) * k)).astype(int)
    cc = ((gu.ravel() - dt.c) / (abs(dt.a) * k)).astype(int)
    inside = (rr >= 0) & (rr < Hk) & (cc >= 0) & (cc < Wk)
    fw = np.full(rr.shape, np.nan)
    fw[inside] = front[rr[inside], cc[inside]]
    bad = ~np.isfinite(fw)
    if bad.all():
        return None
    xyz = result.wall_frame.to_world(np.column_stack([gu.ravel(), gv.ravel(), np.where(bad, 0, fw) + offset_m]))
    H, W = gu.shape
    idx = np.arange(H * W).reshape(H, W)
    a, b, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.column_stack([np.r_[a, b], np.r_[c, c], np.r_[b, e]])
    tris = tris[~bad[tris].any(axis=1)]
    wt = np.where(bad, 0, fw)[tris]
    tris = tris[wt.max(axis=1) - wt.min(axis=1) < 4 * step_m]
    return gu, gv, xyz, tris


def _image_overlay(result, rgba, u0, v1, du, dv, name, kind, step_m, offset_m, colors=None) -> dict | None:
    """An RGBA image in wall coordinates (top-left at chainage u0, elevation v1; pixel du x dv) draped on the face,
    as an overlay entry for tiles3d.scene_json (positions in ECEF, PNG data URI). `colors`: its colour key, shown
    as swatches next to its name in the viewer."""
    import base64

    import cv2

    from .tiles3d import _to_ecef

    U, Vh = rgba.shape[1] * du, rgba.shape[0] * dv
    g = _drape_grid(result, u0, v1, U, Vh, step_m, offset_m)
    if g is None:
        return None
    gu, gv, xyz, tris = g
    st = np.column_stack([(gu.ravel() - u0) / U, 1 - (v1 - gv.ravel()) / Vh])     # t up, like the image
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        return None
    from .. import heights

    crs = heights.reg_crs(result.config) or "EPSG:5187"
    return {"name": name, "kind": kind, "eye": None, "target": None,
            "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
            "positions": _to_ecef(xyz, crs).round(3).ravel().tolist(),
            "st": st.astype(np.float32).round(5).ravel().tolist(), "indices": tris.ravel().tolist(),
            "colors": list(colors or [])}


def tiles_overlay(result, step_m: float = 0.1, offset_m: float = 0.12, max_px: int = 4096) -> dict | None:
    """The set map draped on the front surface, as an overlay entry for tiles3d.scene_json."""
    if result.facade_sets is None:
        return None
    rgba, du, dv = setmap_rgba(result, max_px, alpha=255)
    t = result.facade_sets.transform
    sets = sorted(d.set_id for d in result.discontinuity_sets)
    return _image_overlay(result, rgba, t.c, t.f, du, dv, "절리군 위치 지도", "setmap", step_m, offset_m,
                          colors=[set_colour(s) for s in sets])


# --- the seepage-potential map on the face (grouting views: where the holes go vs. where water may come out) ----

SEEPAGE_COLOURS = {2: "#dc2626", 1: "#f59e0b"}        # 상, 중 -- as the seepage map figure
SEEPAGE_NAMES = {2: "유출 가능성 상 (상위 15%)", 1: "유출 가능성 중 (다음 35%)"}


def seepage_rgba(smap, classes=(2, 1), alpha: int = 150) -> np.ndarray:
    """RGBA uint8 image of the seepage classes, one pixel per map cell (top row = highest elevation)."""
    out = np.zeros(smap.classes.shape + (4,), np.uint8)
    for k in classes:
        m = smap.classes == k
        out[m, :3] = _rgb(SEEPAGE_COLOURS[k]).astype(np.uint8)
        out[m, 3] = alpha
    return out


def seepage_tiles_overlay(result, smap, classes=(2, 1), step_m: float = 0.1, offset_m: float = 0.14) -> dict | None:
    """The seepage map draped on the face, as an overlay entry for tiles3d.scene_json (cells kept sharp)."""
    if smap is None:
        return None
    k = max(1, int(round(smap.cell_m / 0.05)))        # upscaled, nearest: the cell edges stay sharp on the model
    rgba = np.repeat(np.repeat(seepage_rgba(smap, classes, alpha=170), k, axis=0), k, axis=1)
    d = smap.cell_m / k
    return _image_overlay(result, rgba, smap.u_min, smap.v_max, d, d,
                          "유출 가능성 지도 (" + "·".join("상" if k == 2 else "중" for k in classes) + ")", "seepage",
                          step_m, offset_m, colors=[SEEPAGE_COLOURS[k] for k in classes])


def cell_drape_traces(result, u_min: float, v_max: float, cell_m: float, layers: list, origin,
                      opacity: float = 0.55, step_m: float = 0.25, offset_m: float = 0.12) -> list:
    """Plotly mesh traces of cell maps draped on the face: `layers` = [(name, bool (rows, cols) mask, colour)],
    one mesh per layer (toggled in the legend), each lifted a little more than the one before (where layers
    overlap, all stay visible). World coordinates minus `origin`."""
    if not layers:
        return []
    rows, cols = layers[0][1].shape
    # grid vertices on the cell edges (the grid puts them half a step in): every triangle lies in one cell
    step_m = cell_m / max(1, round(cell_m / step_m))
    out = []
    for n, (name, mask, colour) in enumerate(layers):
        g = _drape_grid(result, u_min - step_m / 2, v_max + step_m / 2, cols * cell_m + step_m, rows * cell_m + step_m,
                        step_m, offset_m + 0.03 * n)
        if g is None:
            return out
        gu, gv, xyz, tris = g
        uc, vc = gu.ravel()[tris].mean(axis=1), gv.ravel()[tris].mean(axis=1)
        r = np.floor((v_max - vc) / cell_m).astype(int)
        c = np.floor((uc - u_min) / cell_m).astype(int)
        ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        hit = np.zeros(len(tris), bool)
        hit[ok] = mask[r[ok], c[ok]]
        T = tris[hit]
        if not len(T):
            continue
        used = np.unique(T)
        remap = np.full(len(xyz), -1, np.int64)
        remap[used] = np.arange(len(used))
        Pm = (xyz[used] - origin).astype(np.float32)
        T = remap[T].astype(np.int32)
        out.append(dict(type="mesh3d", x=Pm[:, 0], y=Pm[:, 1], z=Pm[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2],
                        color=colour, opacity=opacity, name=name, hoverinfo="skip", showlegend=True))
    return out


def seepage_traces(result, smap, origin, classes=(2, 1), opacity: float = 0.55, step_m: float = 0.25,
                   offset_m: float = 0.12, zones: int = 10) -> list:
    """Plotly mesh traces of the seepage classes draped on the face (one per class, toggled in the legend) and
    the numbers of the `zones` best zones ("#n" as in the seepage map figure), world coordinates minus `origin`."""
    if smap is None:
        return []
    layers = [(SEEPAGE_NAMES[k], smap.classes == k, SEEPAGE_COLOURS[k]) for k in classes]
    out = cell_drape_traces(result, smap.u_min, smap.v_max, smap.cell_m, layers, origin, opacity, step_m, offset_m)
    if zones and smap.zones is not None and len(smap.zones):
        z = smap.zones.head(zones)
        Z = z[["x", "y", "z"]].to_numpy(float) - origin
        out.append(dict(type="scatter3d", x=Z[:, 0], y=Z[:, 1], z=Z[:, 2] + 0.8, mode="text",
                        text=[f"#{n}" for n in z["zone"]], name=f"유출 '상' 구역 번호 (상위 {len(z)})",
                        textfont=dict(color="#b91c1c", size=13), hoverinfo="text", showlegend=True))
    return out


# --- the main joint planes: a few persistent planes per set, through the rock block (block diagram) ---------------

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


def plane_tag(rank: int) -> str:
    """①, ②, ... for the main planes of a set, by rank (11th and on: (11), (12) ...)."""
    return CIRCLED[rank] if rank < len(CIRCLED) else f"({rank + 1})"


def _plane_axes(n):
    a = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.95 else np.cross(n, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def major_planes(result, sets, u_range=None, per_set: int = 2, tol_m=None, min_length_m: float = 0.0,
                 min_sep_m: float = 0.0) -> list:
    """The main (most persistent) joint planes of every set: the set's exposed facets and mapped traces that lie
    on one plane of the set's mean orientation -- the same offset along its normal, within `tol_m` (default a
    third of the set spacing, 0.25-1 m) -- grouped, ranked by their total length (facet persistence + trace
    length) and taken in that order: at most `per_set`, each with at least `min_length_m` of evidence, at least
    two tolerances and `min_sep_m` from the ones taken before (one surface's roughness is not a second plane).
    A set is a family of parallel planes some metres apart: with several per set those show up side by side.
    Each: name, rank, normal, a point on the plane, dip / dip direction, total length, number of pieces and the
    pieces' centres (world)."""
    from ..hydro import _pole
    from .grouting3d import _trace_planes

    frame = result.wall_frame
    pts = np.asarray(result.point_cloud.points)
    toe = float(np.percentile(frame.to_wall(pts[::max(1, len(pts) // 200_000)])[:, 1], 1))
    elems: dict = {}

    def add(name, c, w, uv):
        if u_range is not None and not (u_range[0] <= uv[0] <= u_range[1]):
            return
        elems.setdefault(name, []).append((np.asarray(c, float), float(w)))

    for d in result.discontinuity_sets:
        for p in d.patches:
            c = np.asarray(p.centroid, float)
            uv = frame.to_wall(c[None])[0]
            pn = np.asarray(p.normal, float) / max(np.linalg.norm(p.normal), 1e-9)
            if abs(pn[2]) > np.cos(np.radians(20)) and uv[1] < toe + 1.0:
                continue                                   # ground, a bench or water at the foot of the wall
            add(f"Set {d.set_id}", c, max(p.persistence_length, 0.1), uv)
    tp = _trace_planes(result)
    for t in result.traces:
        name = tp.get(id(t), (None, None))[0]
        if t.xyz is None or not name or not name.startswith("Set "):
            continue
        add(name, 0.5 * (t.xyz[0] + t.xyz[1]), t.length_m, 0.5 * (t.coords[0] + t.coords[1]))
    out = []
    for s in sets.itertuples():
        if s.name not in elems:
            continue
        n = _pole(s.dip, s.dip_direction)
        n /= np.linalg.norm(n)
        C = np.array([e[0] for e in elems[s.name]])
        W = np.array([e[1] for e in elems[s.name]])
        tol = tol_m or float(np.clip(float(s.spacing_m) / 3, 0.25, 1.0))
        dist = C @ n
        # the offsets carrying the most length within +-tol/2: a sliding window along the normal (single-linkage
        # grouping would chain a densely jointed set into one group), the best first, its neighbours suppressed
        order = np.argsort(dist)
        ds, ws = dist[order], np.r_[0.0, np.cumsum(W[order])]
        grid = np.arange(ds[0], ds[-1] + tol / 10, tol / 10)
        lo_i = np.searchsorted(ds, grid - tol / 2, "left")
        hi_i = np.searchsorted(ds, grid + tol / 2, "right")
        score = ws[hi_i] - ws[lo_i]
        chosen = []
        sep = max(2 * tol, float(min_sep_m))
        while len(chosen) < per_set and score.max() > 0:
            g0 = int(np.argmax(score))
            g = order[lo_i[g0]:hi_i[g0]]
            if score[g0] < min_length_m:
                break                                      # the rest carry less evidence still
            score[np.abs(grid - grid[g0]) < sep] = 0
            if len(g) < 2:
                continue
            d0 = float(np.average(dist[g], weights=W[g]))
            c_mean = np.average(C[g], axis=0, weights=W[g])
            chosen.append({"name": s.name, "rank": len(chosen), "normal": n, "offset": d0,
                           "point": c_mean - n * (c_mean @ n - d0), "dip": float(s.dip), "dd": float(s.dip_direction),
                           "length_m": float(W[g].sum()), "pieces": int(len(g)), "members": C[g]})
        out += chosen
    return out


def _runs(mask):
    """(start, stop) index pairs of the True runs of a 1D mask."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def major_model(result, rock, planes, u_range, v_range, depth_m: float = 8.0, extent: str = "block",
                margin_m: float = 3.0, cells: int = 150):
    """The main planes inside a rock block -- chainage `u_range` x elevation `v_range`, from the face back to
    `depth_m` behind its deepest point -- as one wall-coordinate mesh (the part inside the rock: behind the face),
    with the lines a block diagram shows: plane x plane intersections, where each plane meets the face and its
    traces on the block's back, sides and top. extent="observed" keeps each plane to its pieces' extent plus
    `margin_m` instead of the whole block. Joint objects (source "major") let the sections cut the planes."""
    from .joints3d import Joint

    frame = result.wall_frame
    u0, u1 = u_range
    v0, v1 = v_range
    gu, gv = np.meshgrid(np.arange(u0, u1 + 0.01, 0.25), np.arange(v0, v1 + 0.01, 0.25))
    fw = rock.face_w(gu.ravel(), gv.ravel())
    if not np.isfinite(fw).any():
        return None
    w_front, back = float(np.nanmax(fw)), float(np.nanmin(fw)) - depth_m
    centre = frame.to_world(np.array([[0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (w_front + back)]]))[0]
    L = float(np.linalg.norm([u1 - u0, v1 - v0, w_front - back])) * 1.15
    step = L / cells
    k = np.arange(-(cells // 2), cells // 2 + 1) * step
    S, T = np.meshgrid(k, k)
    H, W = S.shape
    idx = np.arange(H * W).reshape(H, W)
    a_, b_, c_, e_ = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    quads = np.column_stack([np.r_[a_, b_], np.r_[c_, c_], np.r_[b_, e_]])

    def in_box(uvw):
        return ((uvw[:, 0] >= u0) & (uvw[:, 0] <= u1) & (uvw[:, 1] >= v0) & (uvw[:, 1] <= v1) & (uvw[:, 2] >= back)
                & (uvw[:, 2] <= w_front + 0.05))

    joints, Vs, Fs, owners, labels, kept = [], [], [], [], [], []
    lines = {"face": [], "sides": []}
    for pl in planes:
        n = pl["normal"]
        a, b = _plane_axes(n)
        c0 = centre - n * ((centre - pl["point"]) @ n)
        X = c0 + S.ravel()[:, None] * a + T.ravel()[:, None] * b
        uvw = frame.to_wall(X)
        face = rock.face_w(uvw[:, 0], uvw[:, 1])
        box = in_box(uvw)
        if extent == "observed":
            M = pl["members"] - c0
            s_m, t_m = M @ a, M @ b
            box &= ((S.ravel() >= s_m.min() - margin_m) & (S.ravel() <= s_m.max() + margin_m)
                    & (T.ravel() >= t_m.min() - margin_m) & (T.ravel() <= t_m.max() + margin_m))
        rock_in = box & np.isfinite(face) & (uvw[:, 2] <= face + 0.03)
        F_box = quads[box[quads].all(axis=1)]
        F = quads[rock_in[quads].all(axis=1)]
        if not len(F):
            continue
        # where the plane comes out at the face, and its traces on the block's sides, back and top
        seg, _ = cut_segments({"V": uvw, "F": F_box, "owner": np.zeros(len(F_box), int)},
                              np.nan_to_num(uvw[:, 2] - face, nan=-1.0))
        lines["face"].append((pl["name"], seg))
        # the block's own faces lie on its boundary, so the whole plane grid is cut there and only the part of the
        # cut that is on that face of the block, in rock (and within the plane's extent) is kept
        full = {"V": uvw, "F": quads, "owner": np.zeros(len(quads), int)}
        side = []
        for axis, val in ((0, u0), (0, u1), (1, v1), (2, back)):
            sg, _ = cut_segments(full, uvw[:, axis] - val)
            if not len(sg):
                continue
            mid = sg.mean(axis=1)
            ok = np.ones(len(sg), bool)
            for ax2, lo2, hi2 in ((0, u0, u1), (1, v0, v1), (2, back, w_front + 0.05)):
                if ax2 != axis:
                    ok &= (mid[:, ax2] >= lo2 - 1e-6) & (mid[:, ax2] <= hi2 + 1e-6)
            fm = rock.face_w(mid[:, 0], mid[:, 1])
            ok &= np.isfinite(fm) & (mid[:, 2] <= fm + 0.03)
            if extent == "observed":
                M = pl["members"] - c0
                X_mid = frame.to_world(mid) - c0
                s_m, t_m = M @ a, M @ b
                sm_, tm_ = X_mid @ a, X_mid @ b
                ok &= ((sm_ >= s_m.min() - margin_m) & (sm_ <= s_m.max() + margin_m)
                       & (tm_ >= t_m.min() - margin_m) & (tm_ <= t_m.max() + margin_m))
            if ok.any():
                side.append(sg[ok])
        lines["sides"].append((pl["name"], np.concatenate(side) if side else np.zeros((0, 2, 3))))
        used, inv = np.unique(F.ravel(), return_inverse=True)
        off = sum(len(v) for v in Vs)
        Vs.append(uvw[used])
        Fs.append(inv.reshape(-1, 3) + off)
        owners.append(np.full(len(F), len(joints)))
        joints.append(Joint(pl["name"], np.zeros((1, 3)), n, "major", -1, np.nan, pl["length_m"]))
        kept.append(pl)
        P = uvw[used]
        # steep planes labelled at their top edge, gentle ones at their right end (centres would all bunch up)
        at = np.argmax(P[:, 1]) if abs(n[2]) < 0.7 else np.argmax(P[:, 0] - 0.2 * P[:, 2])
        labels.append((P[at],
                       f"{pl['name']}-{plane_tag(pl['rank'])}  {pl['dip']:.0f}°/{pl['dd']:03.0f}°"))
    if not joints:
        return None
    mesh = {"V": np.vstack(Vs), "F": np.vstack(Fs), "owner": np.concatenate(owners)}
    # plane x plane intersection lines inside the rock (planes of different sets)
    inter = []
    tt = np.arange(-L / 2, L / 2, step / 2)
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            p, q = kept[i], kept[j]
            if p["name"] == q["name"]:
                continue
            d = np.cross(p["normal"], q["normal"])
            if np.linalg.norm(d) < 0.2:
                continue
            d /= np.linalg.norm(d)
            A = np.vstack([p["normal"], q["normal"], d])
            x0 = np.linalg.solve(A, [p["normal"] @ p["point"], q["normal"] @ q["point"], d @ centre])
            uvw = frame.to_wall(x0 + tt[:, None] * d)
            face = rock.face_w(uvw[:, 0], uvw[:, 1])
            ok = in_box(uvw) & np.isfinite(face) & (uvw[:, 2] <= face + 0.03)
            for s0, s1 in _runs(ok):
                if s1 - s0 >= 2:
                    inter.append(uvw[s0:s1])
    _, outward = frame.axes_at(0.5 * (u0 + u1))
    return {"joints": joints, "mesh": mesh, "lines": lines, "intersections": inter, "labels": labels,
            "box": (u0, u1, v0, v1, back, w_front), "planes": kept, "outward": np.asarray(outward, float)}


def _disp(p):
    """wall (u, v, w) -> unrolled display (x = u, y = depth, z = v)"""
    return UNROLLED(p)


def _seg_xyz(segs, disp: Disp = UNROLLED):
    xy = np.full((len(segs), 3, 3), np.nan)
    xy[:, 0], xy[:, 1] = disp(segs[:, 0]), disp(segs[:, 1])
    return xy.reshape(-1, 3)


def major_figure(sample, model, colour_of, opacity: float = 0.55, face_mode: str = "points",
                 face_opacity: float = 0.35, face_points: int = 500_000, surface=None, planes: dict | None = None,
                 height: int = 820, disp: Disp = UNROLLED) -> go.Figure:
    """Block diagram of the main joint planes (major_model) with the face for context."""
    u0, u1, v0, v1, back, w_front = model["box"]
    # a plane within 30 deg of the face lies just behind it and would hide it: drawn fainter
    facing = {pl["name"]: abs(float(pl["normal"] @ model["outward"])) > np.cos(np.radians(30)) for pl in model["planes"]}
    fig = interior_figure(sample, model["joints"], model["mesh"], colour_of, (u0, u1), opacity, face_mode, face_opacity,
                          face_points, surface=surface, planes=planes, height=height,
                          opacity_of=lambda name: 0.4 if facing.get(name) else 1.0, disp=disp)
    by_set: dict = {}
    for name, sg in model["lines"]["face"]:
        if len(sg):
            by_set.setdefault(name, []).append(sg)
    for name, parts in by_set.items():
        X = _seg_xyz(np.concatenate(parts), disp)
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="lines", name=f"{name} 주요 절리면 ∩ 암면",
                                   line=dict(color=colour_of(name), width=4), hoverinfo="name"))
    first = True
    for name, sg in model["lines"]["sides"]:
        if not len(sg):
            continue
        X = _seg_xyz(sg, disp)
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="lines", name="블록 옆·윗·뒷면의 절리 흔적",
                                   legendgroup="block_sides", showlegend=first, line=dict(color=colour_of(name), width=5),
                                   hoverinfo="name"))
        first = False
    if model["intersections"]:
        X = np.vstack([np.vstack([disp(r), np.full((1, 3), np.nan)]) for r in model["intersections"]])
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="lines", name="주요 절리면 교선",
                                   line=dict(color="#111827", width=6), hoverinfo="name"))
    corners = [(u0, v0, w_front), (u1, v0, w_front), (u1, v0, back), (u0, v0, back),
               (u0, v1, w_front), (u1, v1, w_front), (u1, v1, back), (u0, v1, back)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    X = np.vstack([np.vstack([disp.polyline([corners[i], corners[j]]), (np.nan,) * 3]) for i, j in edges])
    fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="lines", name="암반 블록 (분석 범위)",
                               line=dict(color="#6b7280", width=2, dash="dash"), hoverinfo="name"))
    if model["labels"]:
        P = disp(np.array([p for p, _ in model["labels"]]))
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="text", text=[t for _, t in model["labels"]],
                                   textfont=dict(color="#111827"), name="이름표", hoverinfo="skip"))
    return fig
