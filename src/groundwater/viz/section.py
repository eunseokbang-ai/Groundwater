"""Cross-section through the wall at one chainage: the face profile, the joint planes inside the rock cut by
the section, and the grouting holes going in and the joints they cross -- the 3D scene read like a
geotechnical section drawing.

Section plane: chainage u = u0, spanned by the wall normal (w, + toward the viewer, 0 = wall line) and the
elevation. Joint polygons (joints3d.Joint) are intersected with that plane; holes within `band` m of it
are drawn projected onto it, with the points where they cross joints.
"""
import numpy as np
import plotly.graph_objects as go


def _cut(poly_uvw: np.ndarray, u0: float):
    """Segment where a planar polygon (closed ring, wall coordinates) crosses u = u0, or None."""
    q = np.vstack([poly_uvw, poly_uvw[:1]])
    s = q[:, 0] - u0
    pts = []
    for a, b, sa, sb in zip(q[:-1], q[1:], s[:-1], s[1:]):
        if sa == 0:
            pts.append(a)
        if sa * sb < 0:
            pts.append(a + (b - a) * (sa / (sa - sb)))
    if len(pts) < 2:
        return None
    P = np.array(pts)[:, 1:]                             # (v, w)
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return P[i], P[j]


_PROFILE_CACHE: dict = {}


def _profile_points(result):
    """Wall coordinates of a ~3 M point sample of the cloud, kept for the last two results (every slider
    move of the section would otherwise transform them again)."""
    key = id(result.point_cloud)
    if key not in _PROFILE_CACHE:
        pts = np.asarray(result.point_cloud.points)
        if len(_PROFILE_CACHE) >= 2:
            _PROFILE_CACHE.pop(next(iter(_PROFILE_CACHE)))
        _PROFILE_CACHE[key] = result.wall_frame.to_wall(pts[::max(1, len(pts) // 3_000_000)])
    return _PROFILE_CACHE[key]


def joint_u_ranges(frame, joints) -> np.ndarray:
    """(n, 2) chainage range of every joint polygon, from one transform of all their vertices."""
    if not joints:
        return np.zeros((0, 2))
    sizes = [len(j.poly) for j in joints]
    u = frame.to_wall(np.vstack([j.poly for j in joints]))[:, 0]
    ends = np.cumsum(sizes)
    starts = ends - np.array(sizes)
    return np.column_stack([np.minimum.reduceat(u, starts), np.maximum.reduceat(u, starts)])


def section_figure(result, u0: float, holes=None, joints=None, cov=None, colour_of=None, band: float = 1.0,
                   slab_m: float = 0.2, title: str = "", shade: str = "seepage", depth_shown: float = 15.0
                   ) -> go.Figure:
    from .joints3d import joint_groups

    frame = result.wall_frame
    uvw = _profile_points(result)
    sl = uvw[np.abs(uvw[:, 0] - u0) < slab_m]
    traces = []
    xr = None
    if len(sl) > 20:
        # rock body: from the front of the face profile (per 0.25 m of elevation) back into the rock
        vb = np.floor(sl[:, 1] / 0.25).astype(int)
        order = np.unique(vb)
        front = np.array([np.percentile(sl[vb == b, 2], 95) for b in order])
        vv = (order + 0.5) * 0.25
        back = front.min() - depth_shown
        traces.append(go.Scatter(x=np.r_[front, [back, back]], y=np.r_[vv, [vv[-1], vv[0]]], fill="toself",
                                 fillcolor="rgba(180,150,110,0.18)", line=dict(width=0), name="암반 (단면)",
                                 hoverinfo="skip"))
        xr = [back - 1.0, front.max() + 4.0]
    if len(sl):
        traces.append(go.Scattergl(x=sl[:, 2], y=sl[:, 1], mode="markers", marker=dict(size=2, color="#6b7280"),
                                   name="암면 (점군 단면)", hoverinfo="skip"))
    joints = joints or []
    colour_of = colour_of or (lambda name: "#6b7280")
    if joints:
        ring = [j.poly if j.source == "trace" else j.poly[1:] for j in joints]
        ur = joint_u_ranges(frame, joints)
        spans = (ur[:, 0] <= u0) & (ur[:, 1] >= u0)            # only joints the section actually cuts
        for g in joint_groups(joints, colour_of, shade, max_draw_dfn=10 ** 9):
            xs, ys = [], []
            for i in g["idx"]:
                if not spans[i]:
                    continue
                seg = _cut(frame.to_wall(ring[i]), u0)
                if seg is None:
                    continue
                (v1, w1), (v2, w2) = seg
                xs += [w1, w2, None]
                ys += [v1, v2, None]
            if xs:
                dfn = g["kind"] == "dfn"
                traces.append(go.Scatter(x=xs, y=ys, mode="lines", name=g["label"].split(" — ")[0],
                                         line=dict(color=colour_of(g["name"]), width=1.5 if dfn else 3,
                                                   dash="dot" if dfn else "solid"),
                                         opacity=0.6 if dfn else 0.95, hoverinfo="name"))
        if cov is not None:
            xs, ys = [], []
            for i, (hi, hit) in enumerate(zip(cov["high"], cov["hit"])):
                if hi and not hit and spans[i]:
                    seg = _cut(frame.to_wall(ring[i]), u0)
                    if seg is not None:
                        (v1, w1), (v2, w2) = seg
                        xs += [w1, w2, None]
                        ys += [v1, v2, None]
            if xs:
                traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#dc2626", width=6), opacity=0.45,
                                         name="공이 닿지 않는 유출 '상' 절리"))
    n_near = 0
    if holes:
        hx, hy, cx, cy, labels, px, py = [], [], [], [], [], [], []
        for i, h in enumerate(holes):
            ends = frame.to_wall(np.array([h["collar"], h["collar"] + h["length"] * h["dir"]]))
            t = np.linspace(0, 1, 20)
            seg = ends[0] + t[:, None] * (ends[1] - ends[0])
            if np.abs(seg[:, 0] - u0).min() > band:
                continue
            n_near += 1
            hx += [ends[0, 2], ends[1, 2], None]
            hy += [ends[0, 1], ends[1, 1], None]
            cx.append(ends[0, 2])
            cy.append(ends[0, 1])
            labels.append(f"공 {i + 1} (측점 {ends[0, 0]:.1f} m, 길이 {h['length']:.1f} m)")
            if cov is not None:
                for _, tt in cov["per_hole"][i]:
                    q = frame.to_wall((h["collar"] + tt * h["dir"])[None])[0]
                    px.append(q[2])
                    py.append(q[1])
        if hx:
            traces.append(go.Scatter(x=hx, y=hy, mode="lines", line=dict(color="#7f1d1d", width=5), name="그라우팅 공"))
            traces.append(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=10, color="#fde047",
                                     line=dict(color="black", width=1)), text=labels, hoverinfo="text", name="공 입구"))
        if px:
            traces.append(go.Scatter(x=px, y=py, mode="markers", marker=dict(size=8, color="white",
                                     line=dict(color="black", width=1.5)), name="공이 절리를 지나는 점"))
    fig = go.Figure(traces)
    fig.update_layout(
        title=title or f"단면: 측점 {u0:.1f} m (공은 ±{band:g} m 안의 것을 투영)", height=640,
        margin=dict(l=10, r=10, t=40, b=10), legend=dict(itemsizing="constant"),
        xaxis=dict(title="벽면 기준선에서 거리 w (m, + 앞쪽 / − 암반 안쪽)", zeroline=False, range=xr),
        yaxis=dict(title="표고 (m)", scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f9fafb")
    fig.layout.meta = {"holes_in_band": n_near}
    return fig
