"""Figures of the site-wide joint model (site_joints.py): 3D, plan section at an elevation, vertical section along
a line. Colours: the site-wide sets (viz.site3d.SITE_COLOURS)."""
import numpy as np

from .. import site_joints as SJ
from .site3d import SITE_COLOURS

RING_FRACTIONS = (0.35, 0.7, 1.0)          # a disc as rings of triangles, so the part above ground can be dropped


def _colour(g):
    return SITE_COLOURS[int(g) % len(SITE_COLOURS)]


def _disc_mesh(c, n, r, seg=24):
    """Vertices and triangles (i, j, k) of a disc as concentric rings."""
    a, b = SJ._basis(n)
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.cos(t)[:, None] * a + np.sin(t)[:, None] * b
    V = [np.asarray(c, float)[None]]
    for f in RING_FRACTIONS:
        V.append(c + r * f * ring)
    V = np.vstack(V)
    tri = [(0, 1 + k, 1 + (k + 1) % seg) for k in range(seg)]
    for q in range(len(RING_FRACTIONS) - 1):
        o0, o1 = 1 + q * seg, 1 + (q + 1) * seg
        for k in range(seg):
            k1 = (k + 1) % seg
            tri += [(o0 + k, o1 + k, o1 + k1), (o0 + k, o1 + k1, o0 + k1)]
    return V, np.array(tri)


def _mesh(discs, ground=None, keep_above: float = 0.5):
    """One Mesh3d's x, y, z, i, j, k for discs [(c, n, r)]; with `ground` (z grid, transform) the triangles whose
    centre is more than `keep_above` above the ground are left out (the disc continued underground only)."""
    Vs, Ts, o = [], [], 0
    for c, n, r in discs:
        V, T = _disc_mesh(c, n, r)
        if ground is not None:
            ctr = V[T].mean(axis=1)
            gz = SJ.ground_at(*ground, ctr[:, 0], ctr[:, 1])
            T = T[~(ctr[:, 2] > np.nan_to_num(gz, nan=-1e9) + keep_above)]
            if not len(T):
                continue
        Vs.append(V)
        Ts.append(T + o)
        o += len(V)
    if not Vs:
        return None
    V, T = np.vstack(Vs), np.vstack(Ts)
    return dict(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2])


def figure_3d(F, C, factor: float, ground=None, show_extended: bool = True, show_planes: bool = True,
              underground_only: bool = True, height: int = 760):
    """The observed faces (opaque), the faces extended `factor` times (translucent, underground only), the
    candidate continuous planes (outlined) and the ground (translucent surface)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if ground is not None:
        z, t = ground
        step = max(1, int(round(4.0 / abs(t.a))))
        zz = z[::step, ::step]
        xs = t.c + (np.arange(zz.shape[1]) * step + 0.5) * t.a
        ys = t.f + (np.arange(zz.shape[0]) * step + 0.5) * t.e
        fig.add_trace(go.Surface(x=xs, y=ys, z=zz, colorscale=[[0, "#a8a29e"], [1, "#e7e5e4"]], showscale=False,
                                 opacity=0.35, name="지표 (DTM)", hoverinfo="skip", showlegend=True))
    for g in sorted(F["group"].unique()):
        S = F[F["group"] == g]
        label, col = S["group_label"].iloc[0], _colour(g)
        m = _mesh([((r.x, r.y, r.z), (r.nx, r.ny, r.nz), r.r0) for r in S.itertuples()])
        if m:
            fig.add_trace(go.Mesh3d(**m, color=col, opacity=0.95, name=f"{label} 관측 ({len(S)})", showlegend=True,
                                    legendgroup=f"g{g}", hoverinfo="name", flatshading=True))
        if show_extended and factor > 1:
            m = _mesh([((r.x, r.y, r.z), (r.nx, r.ny, r.nz), r.r0 * factor) for r in S.itertuples()],
                      ground if underground_only else None)
            if m:
                fig.add_trace(go.Mesh3d(**m, color=col, opacity=0.28, name=f"{label} ×{factor:g} 연장 (지중)",
                                        showlegend=True, legendgroup=f"g{g}", hoverinfo="name", flatshading=True))
    if show_planes and C is not None and len(C):
        xs, ys, zs = [], [], []
        for _, p in C.iterrows():
            ring = SJ.plane_patch(F, p, 1.0)
            ring = np.vstack([ring, ring[:1]])
            xs += list(ring[:, 0]) + [None]
            ys += list(ring[:, 1]) + [None]
            zs += list(ring[:, 2]) + [None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#111827", width=4),
                                   name=f"연속 가능 불연속면 ({len(C)})", hoverinfo="skip"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), legend=dict(itemsizing="constant"),
                      scene=dict(aspectmode="data", xaxis_title="동 X (m)", yaxis_title="북 Y (m)", zaxis_title="표고 (m)",
                                 camera=dict(eye=dict(x=-0.4, y=1.3, z=0.9))))
    return fig


def _chords(F, factor, p0, m):
    seg = SJ.cut(F[["x", "y", "z"]].to_numpy(), F[["nx", "ny", "nz"]].to_numpy(), F["r0"].to_numpy() * factor, p0, m)
    return seg


FULL_R = 1000.0     # a candidate continuous plane carried "all the way": a disc this large
CLIP_TOL_M = 1.0    # kept up to this above the ground (the observed faces lie on it)


def _underground(seg, ground, step: float = 0.5):
    """(xs, ys, zs) polylines (NaN breaks) of the chords `seg` (n, 2, 3), cut to where they are underground: a point
    is kept when it is no more than CLIP_TOL_M above the ground. Without a ground, the chords as they are."""
    seg = seg[np.isfinite(seg[:, 0, 0])]
    if not len(seg):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if ground is None:
        nan = np.full((len(seg), 1), np.nan)
        return tuple(np.column_stack([seg[:, 0, k:k + 1], seg[:, 1, k:k + 1], nan]).ravel() for k in range(3))
    L = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    n = np.clip(np.ceil(L / step).astype(int) + 1, 2, 4000)
    xs, ys, zs = [], [], []
    for (p, q), k in zip(seg, n):
        f = np.linspace(0, 1, k)[:, None]
        P = p + f * (q - p)
        gz = SJ.ground_at(*ground, P[:, 0], P[:, 1])
        keep = np.isfinite(gz) & (P[:, 2] <= gz + CLIP_TOL_M)
        P[~keep] = np.nan
        xs += list(P[:, 0]) + [np.nan]
        ys += list(P[:, 1]) + [np.nan]
        zs += list(P[:, 2]) + [np.nan]
    return np.array(xs), np.array(ys), np.array(zs)


def _clip_seg(seg, lo, hi):
    """Segments (n, 2, 3) cut to the axis-aligned box lo..hi (Liang-Barsky on the coordinates given)."""
    out = np.full_like(seg, np.nan)
    for i, (p, q) in enumerate(seg):
        d = q - p
        t0, t1 = 0.0, 1.0
        for k in range(len(lo)):
            if abs(d[k]) < 1e-12:
                if p[k] < lo[k] or p[k] > hi[k]:
                    t0, t1 = 1.0, 0.0
                    break
                continue
            a_, b_ = (lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]
            t0, t1 = max(t0, min(a_, b_)), min(t1, max(a_, b_))
        if t0 < t1:
            out[i, 0], out[i, 1] = p + t0 * d, p + t1 * d
    return out


def _clip_box(seg, background):
    if background is None:
        return seg
    x0, x1, y0, y1 = background[1]
    return _clip_seg(seg, np.array([x0, y0, -1e9]), np.array([x1, y1, 1e9]))


def _clip_line(seg, a, t, L, zmin):
    """Chords in a vertical section cut to 0 <= distance <= L and above zmin."""
    lo = np.array([min(a[0], a[0] + t[0] * L), min(a[1], a[1] + t[1] * L), zmin]) - [0.01, 0.01, 0]
    hi = np.array([max(a[0], a[0] + t[0] * L), max(a[1], a[1] + t[1] * L), 1e9]) + [0.01, 0.01, 0]
    return _clip_seg(seg, lo, hi)


def _plane_chords(C, p0, m, min_span: float = 0.0):
    """Chords of the candidate continuous planes (members spanning >= min_span m) carried all the way (an
    assumption: drawn dashed)."""
    if C is not None and len(C):
        C = C[C["span_m"] >= min_span]
    if C is None or not len(C):
        return np.zeros((0, 2, 3)), np.zeros(0, int)
    seg = SJ.cut(C[["x", "y", "z"]].to_numpy(), C[["nx", "ny", "nz"]].to_numpy(), np.full(len(C), FULL_R), p0, m)
    return seg, C["group"].to_numpy()


def plan_figure(F, C, factor: float, z0: float, ground=None, background=None, title: str = "", height: int = 820,
                planes_full: bool = False, full_min_span: float = 0.0):
    """Plan section at elevation z0: where the (extended) discs cut it, coloured by set; the candidate continuous
    planes thicker; ground above z0 (rock / soil at that elevation) tinted, over the orthophoto `background`
    (img top row first, extent)."""
    import plotly.graph_objects as go

    from .seepage_view import _jpeg_uri

    fig = go.Figure()
    if background is not None:
        img, ext = background
        H, W = img.shape[:2]
        dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
        fig.add_trace(go.Image(source=_jpeg_uri(img), x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy,
                               hoverinfo="skip", opacity=0.75))
    if ground is not None:
        z, t = ground
        inside = np.where(np.isfinite(z) & (z > z0), 1.0, np.nan)[::-1]
        fig.add_trace(go.Heatmap(z=inside, x0=t.c + t.a / 2, dx=t.a, y0=t.f + (z.shape[0] - 0.5) * t.e, dy=-t.e,
                                 colorscale=[[0, "#fef3c7"], [1, "#fef3c7"]], showscale=False, opacity=0.35,
                                 hoverinfo="skip", name=f"표고 {z0:.0f} m에서 땅속"))
    seg = _chords(F, factor, np.array([0, 0, z0]), np.array([0, 0, 1.0]))
    on_plane = np.zeros(len(F), bool) if "plane" not in F else F["plane"].to_numpy() >= 0
    for g in sorted(F["group"].unique()):
        for strong in (False, True):
            sel = np.flatnonzero((F["group"].to_numpy() == g) & np.isfinite(seg[:, 0, 0]) & (on_plane == strong))
            if not len(sel):
                continue
            xs, ys, _ = _underground(seg[sel], ground)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_colour(g), width=4.5 if strong else 2),
                                     name=F[F["group"] == g]["group_label"].iloc[0] + (" · 연속 가능면" if strong else ""),
                                     hoverinfo="name"))
    if planes_full:
        pseg, pg = _plane_chords(C, np.array([0, 0, z0]), np.array([0, 0, 1.0]), full_min_span)
        for g in np.unique(pg):
            sel = np.flatnonzero((pg == g) & np.isfinite(pseg[:, 0, 0]))
            if len(sel):
                xs, ys, _ = _underground(_clip_box(pseg[sel], background), ground, 1.0)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_colour(g), width=1.5, dash="dash"),
                                         name=C[C["group"] == g]["group_label"].iloc[0] + " · 연속 가능면 끝까지 (가정)",
                                         hoverinfo="name", opacity=0.8))
        if background is not None:                    # the long lines must not widen the map
            fig.update_xaxes(range=[background[1][0], background[1][1]])
            fig.update_yaxes(range=[background[1][2], background[1][3]])
    fig.update_layout(title=title or f"수평 단면: 표고 {z0:.0f} m (×{factor:g} 연장)", height=height,
                      margin=dict(l=55, r=10, t=40, b=45), legend=dict(orientation="h", y=-0.08))
    fig.update_xaxes(title="동 X (m)", constrain="range")
    fig.update_yaxes(title="북 Y (m)", scaleanchor="x", scaleratio=1, constrain="range")
    return fig


def section_figure(F, C, factor: float, a, b, ground=None, title: str = "", height: int = 640, planes_full: bool = False,
                   full_min_span: float = 0.0):
    """Vertical section along a -> b: the chords of the (extended) discs in (distance, elevation), the ground
    profile and the rock / soil below it."""
    import plotly.graph_objects as go

    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    t = (b - a) / L
    m = np.array([-t[1], t[0], 0.0])
    seg = _chords(F, factor, np.r_[a, 0.0], m)
    fig = go.Figure()
    zmin = float(np.nanmin(F["z"])) - 30
    if ground is not None:
        s = np.linspace(0, L, max(50, int(L / 0.5)))
        gz = SJ.ground_at(*ground, a[0] + s * t[0], a[1] + s * t[1])
        fig.add_trace(go.Scatter(x=np.r_[s, s[::-1]], y=np.r_[gz, np.full(len(s), zmin)], fill="toself", mode="none",
                                 fillcolor="rgba(214,211,209,0.55)", name="땅속 (DTM 아래)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s, y=gz, mode="lines", line=dict(color="#57534e", width=2), name="지표 (DTM)"))
    on_plane = np.zeros(len(F), bool) if "plane" not in F else F["plane"].to_numpy() >= 0
    for g in sorted(F["group"].unique()):
        for strong in (False, True):
            sel = np.flatnonzero((F["group"].to_numpy() == g) & np.isfinite(seg[:, 0, 0]) & (on_plane == strong))
            if not len(sel):
                continue
            px, py, pz = _underground(seg[sel], ground)
            xs = (px - a[0]) * t[0] + (py - a[1]) * t[1]
            fig.add_trace(go.Scatter(x=xs, y=pz, mode="lines", line=dict(color=_colour(g), width=4 if strong else 1.8),
                                     name=F[F["group"] == g]["group_label"].iloc[0] + (" · 연속 가능면" if strong else ""),
                                     hoverinfo="name"))
    if planes_full:
        pseg, pg = _plane_chords(C, np.r_[a, 0.0], m, full_min_span)
        for g in np.unique(pg):
            sel = np.flatnonzero((pg == g) & np.isfinite(pseg[:, 0, 0]))
            if len(sel):
                px, py, pz = _underground(_clip_line(pseg[sel], a, t, L, zmin), ground, 1.0)
                xs = (px - a[0]) * t[0] + (py - a[1]) * t[1]
                fig.add_trace(go.Scatter(x=xs, y=pz, mode="lines", line=dict(color=_colour(g), width=1.5, dash="dash"),
                                         name=C[C["group"] == g]["group_label"].iloc[0] + " · 연속 가능면 끝까지 (가정)",
                                         hoverinfo="name", opacity=0.8))
        top = float(np.nanmax(SJ.ground_at(*ground, [a[0], b[0]], [a[1], b[1]]))) if ground is not None else float(F.z.max())
        fig.update_yaxes(range=[zmin, max(top, float(F.z.max())) + 15])
    for x, txt in ((0, "A"), (L, "B")):
        fig.add_annotation(x=x, y=1.02, yref="paper", text=txt, showarrow=False, font=dict(size=15))
    fig.update_layout(title=title or f"수직 단면 A→B (길이 {L:.0f} m, ×{factor:g} 연장)", height=height,
                      margin=dict(l=55, r=10, t=40, b=45), legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(title="A → B 거리 (m)", range=[0, L])
    fig.update_yaxes(title="표고 (m)", scaleanchor="x", scaleratio=1)
    return fig


# --- where the joint planes meet the ground (the expected trace on the surface) ---------------------------------

def ground_trace(c, n, r, ground, box=None) -> list:
    """Polylines (k, 3) where the disc (centre c, unit normal n, radius r) meets the ground (z grid, transform):
    the zero line of the ground's signed distance from the plane, on the ground grid, kept where the ground point
    is within r of the centre in the plane. `box` (x0, x1, y0, y1) limits the search (for large r)."""
    import contourpy

    z, t = ground
    c, n = np.asarray(c, float), np.asarray(n, float)
    x0, x1, y0, y1 = (c[0] - r, c[0] + r, c[1] - r, c[1] + r)
    if box is not None:
        x0, x1, y0, y1 = max(x0, box[0]), min(x1, box[1]), max(y0, box[2]), min(y1, box[3])
    c0, c1 = max(int(np.floor((x0 - t.c) / t.a)), 0), min(int(np.ceil((x1 - t.c) / t.a)) + 1, z.shape[1])
    r0, r1 = max(int(np.floor((y1 - t.f) / t.e)), 0), min(int(np.ceil((y0 - t.f) / t.e)) + 1, z.shape[0])
    if c1 - c0 < 2 or r1 - r0 < 2:
        return []
    X = t.c + (np.arange(c0, c1) + 0.5) * t.a
    Y = t.f + (np.arange(r0, r1) + 0.5) * t.e                  # north first
    G = z[r0:r1, c0:c1].astype(float)
    XX, YY = np.meshgrid(X, Y)
    D = np.stack([XX - c[0], YY - c[1], G - c[2]], axis=-1)
    d = D @ n
    off = np.linalg.norm(D - d[..., None] * n, axis=-1)
    bad = ~np.isfinite(d) | (off > r)
    if bad.all():
        return []
    dv = d[~bad]
    if dv.min() > 0 or dv.max() < 0:
        return []
    gen = contourpy.contour_generator(X, Y[::-1], np.ma.masked_array(np.nan_to_num(d), bad)[::-1],
                                      line_type="Separate")
    out = []
    for L in gen.lines(0.0):
        if len(L) >= 2:
            out.append(np.column_stack([L, SJ.ground_at(z, t, L[:, 0], L[:, 1])]))
    return out


def _lines_xy(polys):
    xs, ys = [], []
    for P in polys:
        xs += list(P[:, 0]) + [np.nan]
        ys += list(P[:, 1]) + [np.nan]
    return xs, ys


def disc_traces(F, factor: float, ground) -> dict:
    """{group: [polylines]} where the discs (radius r0 * factor) meet the ground."""
    out = {}
    for r in F.itertuples():
        out.setdefault(int(r.group), []).extend(ground_trace((r.x, r.y, r.z), (r.nx, r.ny, r.nz), r.r0 * factor, ground))
    return out


def plane_traces(C, ground, box, min_span: float = 0.0) -> dict:
    """{group: [polylines]} where the candidate continuous planes (members spanning >= min_span) carried all the
    way meet the ground within `box` (x0, x1, y0, y1)."""
    out = {}
    if C is None or not len(C):
        return out
    for r in C[C["span_m"] >= min_span].itertuples():
        out.setdefault(int(r.group), []).extend(ground_trace((r.x, r.y, r.z), (r.nx, r.ny, r.nz), FULL_R, ground, box))
    return out


def _strike_symbols(D, size: float):
    """Strike line (length `size`) and dip tick (toward the dip direction; none when near vertical) of each row
    of D (x, y, dip, dd)."""
    xs, ys = [], []
    for r in D.itertuples():
        a = np.radians(r.dd)
        dv = np.array([np.sin(a), np.cos(a)])              # dip direction, plan
        sv = np.array([dv[1], -dv[0]])                     # strike
        p = np.array([r.x, r.y])
        s0, s1 = p - sv * size / 2, p + sv * size / 2
        tk = p + dv * size * (0.3 if r.dip < 80 else 0.0)
        xs += [s0[0], s1[0], np.nan, p[0], tk[0], np.nan]
        ys += [s0[1], s1[1], np.nan, p[1], tk[1], np.nan]
    return xs, ys


def ortho_map(F, C, factor: float, ground=None, background=None, symbols: str = "planes", disc_lines: bool = True,
              full_lines: bool = False, full_min_span: float = 20.0, symbol_m: float = 6.0, height: int = 860):
    """The joints on the orthophoto as a geological map: strike-dip symbols (of the candidate continuous planes,
    'planes', or of every observed face, 'faces'), where the faces extended `factor` times meet the ground (the
    expected trace on the surface: solid) and where the candidate continuous planes carried all the way would
    (dashed, an assumption)."""
    import plotly.graph_objects as go

    from .seepage_view import _jpeg_uri

    fig = go.Figure()
    box = None
    if background is not None:
        img, ext = background
        H, W = img.shape[:2]
        dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
        fig.add_trace(go.Image(source=_jpeg_uri(img), x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy,
                               hoverinfo="skip"))
        box = tuple(ext)
    elif ground is not None:
        z, t = ground
        box = (t.c, t.c + z.shape[1] * t.a, t.f + z.shape[0] * t.e, t.f)
    label = {int(g): F[F["group"] == g]["group_label"].iloc[0] for g in F["group"].unique()}
    if ground is not None and full_lines:
        for g, polys in sorted(plane_traces(C, ground, box, full_min_span).items()):
            xs, ys = _lines_xy(polys)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_colour(g), width=1.6, dash="dash"),
                                     name=f"{label[g]} · 연속 가능면 끝까지의 지표 교선 (가정)", hoverinfo="name",
                                     opacity=0.9))
    if ground is not None and disc_lines:
        for g, polys in sorted(disc_traces(F, factor, ground).items()):
            xs, ys = _lines_xy(polys)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_colour(g), width=2.4),
                                     name=f"{label[g]} · ×{factor:g} 연장 면의 지표 교선", hoverinfo="name"))
    if symbols != "none":
        D = C if symbols == "planes" and C is not None and len(C) else F
        for g in sorted(D["group"].unique()):
            S = D[D["group"] == g]
            xs, ys = _strike_symbols(S, symbol_m)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="white", width=5), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_colour(g), width=2.6),
                                     name=f"{label[int(g)]} · 주향·경사 기호", legendgroup=f"s{g}", hoverinfo="skip"))
            if D is C:
                text = [f"연속 가능면 {r.id}: {r.dip:.0f}°/{r.dd:03.0f}° · 면 {r.n}개 · {r.span_m:.0f} m · {r.sections}"
                        for r in S.itertuples()]
            else:
                text = [f"{r.section} 면 {int(r.patch_id)}: {r.dip:.0f}°/{r.dd:03.0f}° · {r.area:.0f} m² · "
                        f"표고 {r.z:.0f} m" for r in S.itertuples()]
            fig.add_trace(go.Scatter(x=S["x"], y=S["y"], mode="markers", marker=dict(size=6, color=_colour(g),
                                     line=dict(color="white", width=1)), text=text, hoverinfo="text",
                                     legendgroup=f"s{g}", showlegend=False))
            if D is C or len(S) < 80:                      # the dip, at the end of the tick
                fig.add_trace(go.Scatter(x=S["x"] + np.sin(np.radians(S["dd"])) * symbol_m * 0.55,
                                         y=S["y"] + np.cos(np.radians(S["dd"])) * symbol_m * 0.55, mode="text",
                                         text=[f"{d:.0f}" for d in S["dip"]], textfont=dict(size=10, color=_colour(g)),
                                         hoverinfo="skip", legendgroup=f"s{g}", showlegend=False))
    if box is not None:
        fig.update_xaxes(range=[box[0], box[1]])
        fig.update_yaxes(range=[box[2], box[3]])
    fig.update_layout(title=f"정사영상 위 절리: 주향·경사 기호와 지표 교선 (×{factor:g} 연장)", height=height,
                      margin=dict(l=55, r=10, t=40, b=45), legend=dict(orientation="h", y=-0.08))
    fig.update_xaxes(title="동 X (m)", constrain="range")
    fig.update_yaxes(title="북 Y (m)", scaleanchor="x", scaleratio=1, constrain="range")
    return fig


# --- the model on the photo-textured tiled model (CesiumJS page, viz/wall_digitizer/tiles3d.html) ---------------

LINE_LIFT_M = 0.3          # ground traces drawn this far above the DTM


def tiles_scene(site, F, C, factor: float, ground, tiles_path, full_lines: bool = False,
                full_min_span: float = 20.0, model_alpha: float = 0.5, variant: str = "") -> dict:
    """The CesiumJS scene of the site joint model on the tiled model: the observed faces (discs of their area),
    the faces extended `factor` times (underground part, translucent), the outlines of the candidate continuous
    planes and the ground traces. The tiled model starts translucent (`model_alpha`) so what is underground shows."""
    from . import site3d
    from .tiles3d import SKY_MARGIN_M, _to_ecef, ground_top, sky_clip

    dirs = dict(site3d.section_results(site, variant))
    crs = site3d._crs(dirs[F["section"].iloc[0]])

    def E(P):
        return _to_ecef(np.asarray(P, float).reshape(-1, 3), crs).round(3)

    label = {int(g): F[F["group"] == g]["group_label"].iloc[0] for g in F["group"].unique()}
    scene = {"title": "지중 절리 (현장) — 실사 타일 모델 위", "tiles": str(tiles_path), "holes": [], "missed": [],
             "intro": f"관측된 절리면(같은 면적의 원판), ×{factor:g} 연장한 면의 땅속 부분(반투명), 연속 가능면의 범위(테두리), "
                      "면이 지표와 만나는 선. 땅속을 보려면 표면 불투명도를 낮추세요. 드래그 회전, 휠 확대, 더블클릭 = 회전 중심.",
             "joint_groups": [], "mesh_groups": [], "line_groups": [], "traces": [], "labels": [], "overlays": [],
             "wall": None, "crs": crs, "model_alpha": model_alpha}
    for g in sorted(label):
        S = F[F["group"] == g]
        col = _colour(g)
        polys = [E(SJ.disc((r.x, r.y, r.z), (r.nx, r.ny, r.nz), r.r0, 24)).ravel().tolist() for r in S.itertuples()]
        scene["joint_groups"].append({"label": f"{label[g]} 관측 면", "kind": "obs", "colour": col, "polys": polys,
                                      "colours": [col] * len(polys)})
    if factor > 1:
        for g in sorted(label):
            S = F[F["group"] == g]
            m = _mesh([((r.x, r.y, r.z), (r.nx, r.ny, r.nz), r.r0 * factor) for r in S.itertuples()], ground)
            if m:
                V = E(np.column_stack([m["x"], m["y"], m["z"]]))
                scene["mesh_groups"].append({"label": f"{label[g]} ×{factor:g} 연장 (땅속)", "colour": _colour(g),
                                             "alpha": 0.45, "show": True, "positions": V.ravel().tolist(),
                                             "indices": np.column_stack([m["i"], m["j"], m["k"]]).ravel().tolist()})
    if C is not None and len(C):
        for g in sorted(C["group"].unique()):
            rings = []
            for _, p in C[C["group"] == g].iterrows():
                ring = SJ.plane_patch(F, p, 1.0)
                rings.append(E(np.vstack([ring, ring[:1]])).ravel().tolist())
            scene["line_groups"].append({"label": f"{label[int(g)]} 연속 가능면 범위", "colour": _colour(g), "width": 3,
                                         "dash": False, "show": True, "lines": rings})
    if ground is not None:
        lift = np.array([0, 0, LINE_LIFT_M])
        for g, polys in sorted(disc_traces(F, factor, ground).items()):
            scene["line_groups"].append({"label": f"{label[g]} ×{factor:g} 연장 면의 지표 교선", "colour": _colour(g),
                                         "width": 2.5, "dash": False, "show": True,
                                         "lines": [E(P + lift).ravel().tolist() for P in polys if np.isfinite(P).all()]})
        if full_lines:
            z, t = ground
            box = (t.c, t.c + z.shape[1] * t.a, t.f + z.shape[0] * t.e, t.f)
            for g, polys in sorted(plane_traces(C, ground, box, full_min_span).items()):
                scene["line_groups"].append({"label": f"{label[g]} 연속 가능면 끝까지의 지표 교선 (가정)",
                                             "colour": _colour(g), "width": 2, "dash": True, "show": False,
                                             "lines": [E(P + lift).ravel().tolist() for P in polys
                                                       if np.isfinite(P).all()]})
    for name, S in F.groupby("section"):
        top = np.array([S["x"].median(), S["y"].median(), np.percentile(S["z"], 97) + 8])
        scene["labels"].append({"text": name, "p": E(top)[0].tolist()})
    A = F[["x", "y", "z"]].to_numpy()
    centre = np.median(A, axis=0)
    R = float(np.ptp(A[:, :2], axis=0).max())
    pit = np.mean([np.median(S[["x", "y", "z"]].to_numpy(), axis=0) for _, S in F.groupby("section")], axis=0)
    eye = centre + site3d._open_side(A, pit) * 0.8 * R + np.array([0, 0, 1.3 * R])
    scene["camera"] = {"eye": E(eye)[0].tolist(), "target": E(centre)[0].tolist()}
    top = ground_top(site, A)
    if top is not None:
        scene["sky"] = sky_clip(crs, centre[:2], top + SKY_MARGIN_M, top - 20, top + 150)
    return scene
