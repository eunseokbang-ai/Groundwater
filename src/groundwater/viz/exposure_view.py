"""Figures of the excavated-surface check (discontinuity/exposure.py): for the faces of one or more sections, how
much lies parallel to the excavated wall, near horizontal or across it -- by section and set, on stereonets, on the
elevation of each wall and in plan."""
import numpy as np
import pandas as pd

from ..discontinuity import exposure as X

ORDER = ["parallel", "flat", "cross"]


def load_sections(site, names, parallel_deg: float = X.PARALLEL_DEG, flat_deg: float = X.FLAT_DIP_DEG,
                  variant: str = "") -> pd.DataFrame:
    """The classified faces of the sections `names` (viz.site3d.section_results) with section, folder and the
    site-wide set label (sets of different sections within 20 deg are one set)."""
    from .site3d import section_results, site_sets

    chosen = [(n, d) for n, d in section_results(site, variant) if n in names]
    rows, sets_by = [], {}
    for n, d in chosen:
        P = X.classify_folder(d, parallel_deg, flat_deg)
        s = pd.read_csv(d / "discontinuity_sets.csv")
        s["area"] = s["set_id"].map(P.groupby("set_id")["area_m2"].sum()).fillna(0)
        sets_by[n] = s
        rows.append(P.assign(section=n, folder=str(d)))
    if not rows:
        return pd.DataFrame()
    P = pd.concat(rows, ignore_index=True)
    g = site_sets(sets_by)
    P["group"] = [g.get((s, int(i)), (-1, ""))[0] for s, i in zip(P["section"], P["set_id"])]
    P["set_label"] = [g.get((s, int(i)), (-1, f"Set {i}"))[1].split(" · ")[0] for s, i in zip(P["section"], P["set_id"])]
    return P


def table(P: pd.DataFrame) -> pd.DataFrame:
    """Per section and set: faces, area, area shares (%) and median roughness (mm) of the classes."""
    S = X.summary(P, by=("section", "set_label"))
    cols = {"section": "구간", "set_label": "절리군", "faces": "면 수", "area_m2": "면적 (m²)"}
    for c, name in X.CLASSES.items():
        cols[f"{c}_area_pct"] = f"{name} 면적 %"
    for c, name in X.CLASSES.items():
        cols[f"{c}_faces"] = f"{name} 면 수"
    for c, name in X.CLASSES.items():
        cols[f"{c}_rough_mm"] = f"{name} 거칠기 (mm)"
    return S[list(cols)].rename(columns=cols)


def bar_figure(P: pd.DataFrame, height: int = 380):
    """Stacked bars: the area of each section's sets by class."""
    import plotly.graph_objects as go

    S = X.summary(P, by=("section", "set_label"))
    x = [f"{s.split('_')[0]} · {lab}" for s, lab in zip(S["section"], S["set_label"])]
    fig = go.Figure()
    for c in ORDER:
        area = S["area_m2"] * S[f"{c}_area_pct"] / 100
        fig.add_trace(go.Bar(x=x, y=area, name=X.CLASSES[c], marker_color=X.CLASS_COLOURS[c],
                             text=[f"{p:.0f}%" if p >= 8 else "" for p in S[f"{c}_area_pct"]], textposition="inside",
                             hovertemplate="%{x}<br>" + X.CLASSES[c] + ": %{y:,.0f} m²<extra></extra>"))
    fig.update_layout(barmode="stack", height=height, margin=dict(l=55, r=10, t=30, b=40),
                      yaxis_title="면적 (m²)", legend=dict(orientation="h", y=1.12))
    return fig


def refind(P: pd.DataFrame, classes=("cross", "flat")) -> pd.DataFrame:
    """Orientation peaks of the faces of `classes` only, one pole per face (not by area: the exposure favours
    wall-parallel faces by area). Only orientations of the sets the run found can come out (faces exist only for
    those); to look for sets the wall outweighed, run the analysis with the wall-parallel normals left out."""
    from ..discontinuity.clustering import find_orientation_sets

    Q = P[P["exposure"].isin(classes)]
    if len(Q) < 10:
        return pd.DataFrame()
    n = X._pole(Q["dip"], Q["dip_direction"])
    lab, axes = find_orientation_sets(n, bandwidth_deg=8.0, min_set_points=8, min_set_fraction=0.03,
                                      min_peak_contrast=2.0)
    rows = []
    for k, a in enumerate(axes):
        a = a if a[2] >= 0 else -a
        m = lab == k
        rows.append({"방향 (경사/경사방향)": f"{np.degrees(np.arccos(np.clip(a[2], -1, 1))):.0f}/"
                                       f"{np.degrees(np.arctan2(a[0], a[1])) % 360:03.0f}",
                     "면 수": int(m.sum()), "면적 (m²)": round(float(Q["area_m2"].to_numpy()[m].sum()), 1),
                     "구간": "·".join(sorted({s.split("_")[0] for s in Q["section"].to_numpy()[m]}))})
    return pd.DataFrame(rows)


def stereonets_png(P: pd.DataFrame) -> bytes:
    """Three stereonets (lower hemisphere, poles): every face, the wall-parallel faces, the crossing and flat
    ones -- pole density by face count (not area), the poles coloured by class, the poles of the local wall
    (vertical planes facing the pit) as grey crosses on the rim."""
    from ..report import _png, _setup_mpl

    plt = _setup_mpl()
    import mplstereonet  # noqa: F401

    fig = plt.figure(figsize=(15, 5.6))
    panels = [("모든 면", ORDER), ("벽면 평행 (굴착면 또는 그 면을 따른 절리)", ["parallel"]),
              ("교차 · 수평 (벽면 평행 제외)", ["cross", "flat"])]
    rng = np.random.default_rng(0)
    wall = P["wall_az"].to_numpy()
    wall = wall[rng.choice(len(wall), min(400, len(wall)), replace=False)] if len(wall) else wall
    for i, (title, cls) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="stereonet")
        Q = P[P["exposure"].isin(cls)]
        for c in cls:
            R = Q[Q["exposure"] == c]
            if len(R):
                ax.pole(np.array(R["dip_direction"] - 90, float), np.array(R["dip"], float), "o", ms=2.2, color=X.CLASS_COLOURS[c], alpha=0.6,
                        label=f"{X.CLASSES[c]} ({len(R)})")
        if len(Q) >= 3:                           # density by face count, as lines over the poles
            ax.density_contour(np.array(Q["dip_direction"] - 90, float), np.array(Q["dip"], float),
                               measurement="poles", method="exponential_kamb", sigma=3, gridsize=150,
                               colors="#111827", linewidths=0.7, zorder=4)
        if len(wall):
            ax.pole(np.array(wall - 90, float), np.full(len(wall), 89.5), "x", ms=6, mew=1.4, color="#dc2626",
                    alpha=0.7, zorder=5, label="그 자리 벽면의 극점")
        a = Q["area_m2"].sum()
        ax.set_title(f"{title}\n면 {len(Q):,}개 · {a:,.0f} m² ({100 * a / max(P['area_m2'].sum(), 1e-9):.0f}%)",
                     fontsize=10, pad=18)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=2, frameon=False)
    fig.suptitle("굴착면과의 관계 — 등고선 = 극점 밀도 (면 개수 기준, 면적 아님), 빨간 × = 그 자리 벽면의 극점", fontsize=11)
    return _png(fig)


def _hover(R):
    return [f"{r.section} 면 {int(r.patch_id)} · {r.set_label}<br>{r.dip:.0f}°/{r.dip_direction:03.0f}° · "
            f"{r.area_m2:.1f} m² · 벽면과 {r.wall_angle_deg:.0f}°<br>거칠기 {1000 * r.roughness_rms_m:.0f} mm · "
            f"{X.CLASSES[r.exposure]}" for r in R.itertuples()]


def elevation_figure(P: pd.DataFrame, folder, height: int = 520):
    """The faces of one section on its wall elevation (chainage, elevation; the facade photo behind), coloured
    by class, size by area."""
    import json
    from pathlib import Path

    import plotly.graph_objects as go

    fig = go.Figure()
    folder = Path(folder)
    meta = folder / "facade_preview.json"
    if meta.exists() and (folder / "facade_preview.jpg").exists():
        from PIL import Image

        from .seepage_view import _jpeg_uri

        m = json.loads(meta.read_text(encoding="utf-8"))
        im = Image.open(folder / "facade_preview.jpg").convert("RGB")
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)))
        img = np.asarray(im)                              # top row first, as the orthophoto backgrounds
        dx, dy = (m["u_max"] - m["u_min"]) / img.shape[1], (m["v_max"] - m["v_min"]) / img.shape[0]
        fig.add_trace(go.Image(source=_jpeg_uri(img), x0=m["u_min"] + dx / 2, dx=dx, y0=m["v_min"] + dy / 2, dy=dy,
                               hoverinfo="skip", opacity=0.85))
        fig.update_xaxes(range=[m["u_min"], m["u_max"]])
        fig.update_yaxes(range=[m["v_min"], m["v_max"]])
    for c in ORDER:
        R = P[P["exposure"] == c]
        if not len(R):
            continue
        fig.add_trace(go.Scatter(x=R["chainage_m"], y=R["elevation_m"], mode="markers", name=X.CLASSES[c],
                                 marker=dict(size=np.clip(np.sqrt(R["area_m2"]) * 2.2, 4, 26), color=X.CLASS_COLOURS[c],
                                             line=dict(color="white", width=0.8), opacity=0.85),
                                 text=_hover(R), hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=55, r=10, t=30, b=40), legend=dict(orientation="h", y=1.1),
                      title=f"{P['section'].iloc[0]} — 입면 (점 크기 = 면적)")
    fig.update_xaxes(title="측점 (m)")
    fig.update_yaxes(title="표고 (m)", scaleanchor="x", scaleratio=1)
    return fig


def plan_figure(P: pd.DataFrame, background=None, height: int = 820):
    """The faces in plan on the orthophoto, coloured by class, size by area; the local wall normal at each face
    as a short grey tick toward the pit."""
    import plotly.graph_objects as go

    from .seepage_view import _jpeg_uri

    fig = go.Figure()
    if background is not None:
        img, ext = background
        H, W = img.shape[:2]
        dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
        fig.add_trace(go.Image(source=_jpeg_uri(img), x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy,
                               hoverinfo="skip"))
        fig.update_xaxes(range=[ext[0], ext[1]])
        fig.update_yaxes(range=[ext[2], ext[3]])
    for c in ORDER:
        R = P[P["exposure"] == c]
        if not len(R):
            continue
        fig.add_trace(go.Scatter(x=R["centroid_x"], y=R["centroid_y"], mode="markers", name=X.CLASSES[c],
                                 marker=dict(size=np.clip(np.sqrt(R["area_m2"]) * 1.6, 3, 20), color=X.CLASS_COLOURS[c],
                                             line=dict(color="white", width=0.6), opacity=0.85),
                                 text=_hover(R), hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=55, r=10, t=30, b=40), legend=dict(orientation="h", y=-0.06),
                      title="평면 — 굴착면과의 관계 (점 크기 = 면적)")
    fig.update_xaxes(title="동 X (m)", constrain="range")
    fig.update_yaxes(title="북 Y (m)", scaleanchor="x", scaleratio=1, constrain="range")
    return fig
