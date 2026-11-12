"""Site plan on the 2026.09 drone orthomosaic (NGII 2025 where it has no data) (16:9): cut-off walls and wells at their registered positions,
DEM-derived surface drainage, reported leachate releases and bypass routes to verify.

    .venv\\Scripts\\python.exe docs\\figures\\make_leachate_plan_ortho.py            (drone orthophoto background)
    .venv\\Scripts\\python.exe docs\\figures\\make_leachate_plan_ortho.py --relief   (0.5 m DTM colour hillshade,
                                                                                 writes leachate_plan_relief)

Inputs: 과거정사영상/(B060)정사영상_2025_*.tif (georeferenced from its map sheet, see
webapp/ortho_history/build_tiles.py), docs/figures/data/site_facilities.geojson (facility plan
registered to the orthophoto, ~1 m RMS) and DEM(5m)/*.tif (D8 surface routing).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from affine import Affine
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from svgkit import MUTED, TEXT, arrow, document, image, label, path, rect, save, text, text_width, wrap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "webapp" / "ortho_history")]
from groundwater.terrain import hydrology as hy  # noqa: E402
from build_tiles import find_sheets, georeference, read_metadata  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
NAME = "leachate_plan_ortho"
CRS = "EPSG:5186"
E0, E1, N0, N1 = 202600.0, 203500.0, 384600.0, 385240.0     # map extent (m)
MX, MY, MW, MH = 40, 168, 990, 702                            # map panel (px)
SX = MW / (E1 - E0)                                           # px per metre

COL = {"wall": "#ef4444", "pw": "#f59e0b", "mw": "#22c55e", "sump": "#2563eb", "leach": "#f97316",
       "stream": "#38bdf8", "path": "#ef4444", "hpath": "#c026d3", "catch": "#fb923c", "scope": "#c084fc"}


def px(x, y):
    return MX + (x - E0) * SX, MY + (N1 - y) * SX


def pts(xy):
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in (px(x, y) for x, y in xy))


DRONE_TILES = ROOT / "webapp" / "ortho_history" / "tiles" / "20260907"
HR_SITE = ROOT / "webapp" / "ortho_history" / "terrain" / "hr_site.geojson"
HR_DEM = ROOT / "DEM(5m)" / "dem(0.5m).tif"
RELIEF = "--relief" in sys.argv
RELIEF_RANGE = []                                             # (lo, hi) m of the colour ramp, set by relief_png()


def relief_png():
    """Background: hypsometric tint x multi-directional hillshade of the 0.5 m drone DTM, ramp stretched to the map."""
    scale = 2
    w, h = MW * scale, MH * scale
    with rasterio.open(HR_DEM) as ds:
        win = rasterio.windows.from_bounds(E0, N0, E1, N1, ds.transform).round_offsets().round_lengths()
        z = ds.read(1, window=win).astype(np.float64)
        t = ds.window_transform(win)
        nod = ds.nodata
    valid = np.isfinite(z) & (z > -1000) & ((z != nod) if nod is not None else True)
    lo, hi = np.percentile(z[valid], [1, 99.5])
    RELIEF_RANGE[:] = [float(lo), float(hi)]
    zf = np.where(valid, z, np.median(z[valid]))
    cr = hy.color_relief(zf, valid, t.a, lo, hi)
    grid = Affine((E1 - E0) / w, 0, E0, 0, -(N1 - N0) / h, N1)
    dst = np.zeros((3, h, w), np.float32)
    for i in range(3):
        reproject(cr[..., i].astype(np.float32), dst[i], src_transform=t, src_crs=CRS, dst_transform=grid, dst_crs=CRS,
                  resampling=Resampling.average)
    im = Image.fromarray(np.moveaxis(dst, 0, -1).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def relief_scale():
    """Elevation colour bar in the map's lower-right corner."""
    lo, hi = RELIEF_RANGE
    x, y, bw, bh = MX + MW - 290, MY + MH - 46, 260, 16
    stops = "".join(f'<stop offset="{s:.2f}" stop-color="rgb{tuple(c)}"/>' for s, c in hy.HYPSO)
    return "\n".join([
        f'<linearGradient id="hypso">{stops}</linearGradient>',
        rect(x - 10, y - 34, bw + 20, 78, rx=8, fill="white", fill_opacity=0.88),
        text(x, y - 10, "표고 (DTM 0.5 m)", 17, 700),
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="url(#hypso)" stroke="#6b7280" stroke-width="1"/>',
        text(x, y + bh + 20, f"{lo:.0f} m", 16, 400, TEXT),
        text(x + bw, y + bh + 20, f"{hi:.0f} m", 16, 400, TEXT, anchor="end"),
    ])


def ortho_png():
    """Background: the 2026.09 drone orthomosaic (from its tile pyramid), NGII 2025 where it has no data."""
    base = ngii_rgb()
    if DRONE_TILES.exists():
        from large_ortho import read_tiles
        drone = read_tiles(DRONE_TILES, (E0, N0, E1, N1), MW * 2, MH * 2).astype(np.float32)
        a = drone[..., 3:] / 255
        base = (drone[..., :3] * a + base * (1 - a)).astype(np.uint8)
    im = Image.blend(Image.fromarray(base), Image.new("RGB", (MW * 2, MH * 2), (255, 255, 255)), 0.1)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ngii_rgb():
    src = ROOT / "과거정사영상"
    sheet = next(s for s in find_sheets(src) if s["year"] == 2025)
    meta = read_metadata(src).get(2025)
    with rasterio.open(sheet["path"]) as ds:
        t = georeference(ds, sheet["sheet"], float(meta["지상표본거리"]) if meta else None)
        rgb = ds.read([1, 2, 3])
    scale = 2                                                  # 2x pixels for a crisp PNG
    w, h = MW * scale, MH * scale
    dst = np.zeros((3, h, w), np.uint8)
    grid = Affine((E1 - E0) / w, 0, E0, 0, -(N1 - N0) / h, N1)
    reproject(rgb, dst, src_transform=t, src_crs=CRS, dst_transform=grid, dst_crs=CRS, resampling=Resampling.average)
    return np.moveaxis(dst, 0, -1)


def drainage(liner_geom):
    from build_terrain import find_dem
    dem = find_dem()
    with rasterio.open(dem) as ds:
        z = ds.read(1).astype(float)
        t = ds.transform
    valid = z > 0
    from build_terrain import burn_corrections        # same stream-burn corrections as the viewer
    filled = hy.fill_depressions(burn_corrections(z, valid, t, log=lambda *_: None), valid)
    code = hy.flow_directions(filled, valid, t.a)
    acc = hy.flow_accumulation(filled, code, valid)
    area = acc * t.a * t.a
    streams = area >= 2e4
    order = hy.strahler_order(code, streams, filled)
    to_xy = lambda cells: [t * (c + 0.5, r + 0.5) for r, c in cells]
    segs = [(to_xy(cells), o) for cells, o, _ in hy.stream_segments(code, streams, order, acc) if len(cells) > 1]
    mask = rasterize([(liner_geom, 1)], out_shape=z.shape, transform=t).astype(bool) & valid
    up = hy.upstream_mask(code, mask)
    rr, cc = np.nonzero(mask)
    k = np.argmax(acc[rr, cc])
    outflow = to_xy(hy.trace_downstream(code, rr[k], cc[k]))
    length = sum(np.hypot(*np.subtract(b, a)) for a, b in zip(outflow, outflow[1:]))
    return segs, up, t, outflow, length, up.sum() * t.a * t.a / 1e4


def site_map():
    fc = json.loads((OUT_DIR / "data" / "site_facilities.geojson").read_text(encoding="utf-8"))
    feats = fc["features"]
    by = lambda kind: [f for f in feats if f["properties"]["kind"] == kind]
    pt = lambda name: next(f["geometry"]["coordinates"] for f in feats if f["properties"].get("name") == name)
    liner = by("hdpe_liner")[0]["geometry"]
    segs, up, dt, outflow, out_len, catch_ha = drainage(liner)

    out = [f'<clipPath id="mapClip"><rect x="{MX}" y="{MY}" width="{MW}" height="{MH}"/></clipPath>',
           f'<g clip-path="url(#mapClip)">', image(MX, MY, MW, MH, relief_png() if RELIEF else ortho_png())]
    # landfill catchment: 0.5 m DTM of the present ground when available, else the 5 m DEM
    rings = []
    if HR_SITE.exists():
        from rasterio.warp import transform as tc
        for f in json.loads(HR_SITE.read_text(encoding="utf-8"))["features"]:
            if f["properties"]["kind"] == "landfill_catchment":
                lon, lat = zip(*f["geometry"]["coordinates"][0])
                rings.append(list(zip(*tc("EPSG:4326", CRS, list(lon), list(lat)))))
                catch_ha = f["properties"]["area_ha"]
    if not rings:
        from rasterio.features import shapes
        rings = [geom["coordinates"][0] for geom, _ in shapes(up.astype(np.uint8), mask=up, transform=dt)]
    for ring in rings:
        out.append(f'<polygon points="{pts(ring)}" fill="{COL["catch"]}" fill-opacity="0.16" '
                   f'stroke="{COL["catch"]}" stroke-width="3" stroke-dasharray="9 6"/>')
    # streams (DEM), drawn under everything else; cut at the HDPE cover (the 5 m DEM still shows the
    # pre-landfill valley floor there)
    from shapely.geometry import LineString as _Line, shape as _shape
    cover = _shape(liner)
    cut = []
    for xy, o in segs:
        g = _Line(xy).difference(cover)
        cut += [(list(part.coords), o) for part in getattr(g, "geoms", [g]) if not part.is_empty and len(part.coords) > 1]
    for xy, o in cut:
        width = [0, 1.6, 2.6, 3.8, 5.2, 6.5][min(o, 5)]
        out.append(f'<polyline points="{pts(xy)}" fill="none" stroke="#0c4a6e" stroke-width="{width + 2.4}" '
                   f'stroke-opacity="0.55" stroke-linejoin="round"/>')
        out.append(f'<polyline points="{pts(xy)}" fill="none" stroke="{COL["stream"]}" stroke-width="{width}" '
                   'stroke-linejoin="round"/>')
    # survey scope: both flanks + the downstream (north / north-west) side
    from shapely.geometry import LineString, Point, shape
    from shapely.ops import unary_union
    walls = [shape(f["geometry"]).buffer(0) for f in by("cutoff_wall")]   # traced outlines may self-touch
    wells = [Point(f["geometry"]["coordinates"]) for f in by("pumping_well") + by("monitoring_well")]
    scope = unary_union([shape(liner).buffer(55), unary_union(walls + wells).convex_hull.buffer(90),
                         LineString(outflow[:60]).buffer(55)]).simplify(4)
    out.append(f'<polygon points="{pts(scope.exterior.coords)}" fill="none" stroke="{COL["scope"]}" '
               'stroke-width="3.5" stroke-dasharray="14 9"/>')
    # surface outflow path from the landfill (DEM)
    out.append(f'<polyline points="{pts(outflow)}" fill="none" stroke="#7f1d1d" stroke-width="7" stroke-opacity="0.6"/>')
    out.append(f'<polyline points="{pts(outflow)}" fill="none" stroke="{COL["path"]}" stroke-width="4"/>')
    # surface outflow path on the present ground (0.5 m DTM, farm-ditch culverts burned in)
    if HR_SITE.exists():
        from rasterio.warp import transform as tc
        f = next(f for f in json.loads(HR_SITE.read_text(encoding="utf-8"))["features"]
                 if f["properties"]["kind"] == "outflow_path")
        lon, lat = zip(*f["geometry"]["coordinates"])
        hp = list(zip(*tc("EPSG:4326", CRS, list(lon), list(lat))))
        out.append(f'<polyline points="{pts(hp)}" fill="none" stroke="white" stroke-width="8" stroke-opacity="0.8" stroke-linejoin="round"/>')
        out.append(f'<polyline points="{pts(hp)}" fill="none" stroke="{COL["hpath"]}" stroke-width="4.5" stroke-linejoin="round"/>')
        arr = np.array(hp)
        seg = np.cumsum(np.r_[0, np.hypot(*np.diff(arr, axis=0).T)])
        for s in (90, 230, 380, 520):
            if s + 10 < seg[-1]:
                i = np.searchsorted(seg, s)
                out.append(arrow("M{:.1f},{:.1f} L{:.1f},{:.1f}".format(*px(*arr[i]), *px(*arr[min(i + 3, len(arr) - 1)])),
                                 COL["hpath"], 4, marker="red"))
    # facilities
    for f in by("cutoff_wall"):
        line = f["properties"].get("centerline")
        if line:   # centre line, drawn wide enough to read at plan scale; dashed where only inferred
            dash = ' stroke-dasharray="12 6"' if f["properties"].get("source") == "dtm" else ""
            out.append(f'<polyline points="{pts(line)}" fill="none" stroke="#7f1d1d" stroke-width="8" stroke-linejoin="round"/>')
            out.append(f'<polyline points="{pts(line)}" fill="none" stroke="{COL["wall"]}" stroke-width="5" '
                       f'stroke-linejoin="round"{dash}/>')
        else:
            out.append(f'<polygon points="{pts(f["geometry"]["coordinates"][0])}" fill="{COL["wall"]}" stroke="#7f1d1d" '
                       'stroke-width="1.5"/>')
    for f in by("pumping_well"):
        x, y = px(*f["geometry"]["coordinates"])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{COL["pw"]}" stroke="#451a03" stroke-width="2"/>')
    for f in by("monitoring_well"):
        x, y = px(*f["geometry"]["coordinates"])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{COL["mw"]}" stroke="#052e16" stroke-width="2"/>')
    for f in by("collection_sump"):
        x, y = px(*f["geometry"]["coordinates"])
        out.append(f'<path d="M{x:.1f},{y - 8:.1f} l8,8 l-8,8 l-8,-8 z" fill="{COL["sump"]}" stroke="white" stroke-width="1.5"/>')

    # leachate routes: confirmed / to verify
    lc = COL["leach"]
    wW, wC, wE = pt("집수구-W"), pt("집수구-C"), pt("집수구-E")
    def a(xy, **kw):
        return arrow("M" + " L".join(f"{p[0]:.1f},{p[1]:.1f}" for p in (px(*q) for q in xy)), **kw)
    out.append(a([(wW[0] + 10, wW[1] - 25), (wW[0] - 35, wW[1] - 10), (wW[0] - 75, wW[1] + 30)],
                 color=lc, width=4.5, dash="11 7", marker="red"))
    out.append(a([(wE[0] - 5, wE[1] - 25), (wE[0] + 40, wE[1] + 5), (wE[0] + 70, wE[1] + 55)],
                 color=lc, width=4.5, dash="11 7", marker="red"))
    out.append(a([(wC[0] + 5, wC[1] - 40), (wC[0] - 5, wC[1] + 20), (wC[0] - 15, wC[1] + 60)],
                 color=lc, width=4.5, dash="4 6", marker="red"))
    out.append("</g>")
    out.append(rect(MX, MY, MW, MH, fill="none", stroke="#cbd5e1"))

    # labels
    lab = lambda x, y, rows, **kw: label(*px(x, y), rows, **kw)
    out += [
        lab(203250, 384900, [("매립지", 24, 700, "white"), ("HDPE 시트 복개", 19, 400, "#f3f4f6")],
            fill="#111827", opacity=0.75, stroke="#111827"),
        lab(203190, 385000, [("상류 연직차수벽", 20, 700, "#b91c1c"), ("L = 202 m", 18, 400, TEXT)], pad=6),
        lab(202978, 385190, [("하류 연직차수벽 (추정)", 20, 700, "#b91c1c"), ("L = 100 m · DTM 둔덕·고랑", 18, 400, TEXT)], pad=6),
        lab(202935, 384990, [("서북측 우회?", 20, 700, "#c2410c")], anchor="end", pad=6),
        lab(203260, 385070, [("동북측 우회?", 20, 700, "#c2410c")], pad=6),
        lab(203090, 385230, [("벽 하부 절리대 통과?", 20, 700, "#c2410c")], anchor="middle", pad=6),
        lab(202620, 385080, [("지표 유출 경로 (DEM 5 m)", 20, 700, "#b91c1c"),
                             (f"→ 서북쪽 → 큰 강 → 금강 방향, {out_len / 1000:.1f} km", 18, 400, TEXT)], pad=6),
        lab(202615, 384950, [("현재 지형 유출 경로 (DTM 0.5 m)", 20, 700, "#a21caf"),
                             ("진입로·농수로 따라 하류 차수벽 우회", 18, 400, TEXT)], pad=6),
        lab(203270, 384740, [(f"매립지 지표 유역 {catch_ha:.1f} ha", 20, 700, "#c2410c"),
                             ("(현재 지형, DTM 0.5 m)", 18, 400, TEXT)], pad=6),
        lab(203495, 384830, [("동쪽 채석장", 20, 700, TEXT)], anchor="end", pad=6),
    ]
    # north arrow + scale bar (100 m)
    out.append(path(f"M{MX + 40},{MY + 70} L{MX + 52},{MY + 34} L{MX + 64},{MY + 70} L{MX + 52},{MY + 60} Z",
                    fill="white", stroke="#111827", stroke_width=2))
    out.append(label(MX + 52, MY + 4, [("N", 20, 700, TEXT)], anchor="middle", pad=3))
    sb = 100 * SX
    x0, y0 = MX + 24, MY + MH - 30
    out.append(rect(x0 - 8, y0 - 30, sb + 70, 46, rx=6, fill="white", fill_opacity=0.85))
    out.append(rect(x0, y0, sb / 2, 8, fill="#111827"))
    out.append(rect(x0 + sb / 2, y0, sb / 2, 8, fill="white", stroke="#111827"))
    out.append(text(x0, y0 - 8, "0", 16, 700))
    out.append(text(round(x0 + sb, 1), y0 - 8, "100 m", 16, 700, anchor="middle"))
    if RELIEF:
        out.append(relief_scale())
    return "\n".join(out), out_len, catch_ha


def panel(out_len, catch_ha):
    x0, y0, w = 1066, 168, 494
    out = [rect(x0, y0, w, 702, rx=10, fill="white", stroke="#cbd5e1"), text(x0 + 20, y0 + 40, "범례", 25, 700)]
    rows = [
        ("wall", "연직차수벽 (상류 드론·하류 DTM 판독)"), ("pw", "양수정 PW-01~10"), ("mw", "관측정 MW-01~04"),
        ("sump", "집수구"), ("stream", "지표 하천망 (DEM 5 m, 2 ha↑)"), ("path", "매립지 지표 유출 경로 (DEM 5 m, 원지형)"),
        ("hpath", "매립지 지표 유출 경로 (DTM 0.5 m, 현재)"),
        ("catch", "매립지로 모이는 지표 유역 (DTM 0.5 m)"), ("dash", "우회 가능 경로 (검증 필요)"), ("scope", "지하 유출 조사 권장 범위"),
    ]
    y = y0 + 76
    for kind, name in rows:
        sx = x0 + 22
        if kind == "wall":
            out.append(rect(sx, y - 6, 44, 12, fill=COL["wall"], stroke="#7f1d1d"))
        elif kind in ("pw", "mw"):
            out.append(f'<circle cx="{sx + 22}" cy="{y}" r="7" fill="{COL[kind]}" stroke="#1f2937" stroke-width="2"/>')
        elif kind == "sump":
            out.append(f'<path d="M{sx + 22},{y - 8} l8,8 l-8,8 l-8,-8 z" fill="{COL["sump"]}"/>')
        elif kind == "stream":
            out.append(path(f"M{sx},{y} L{sx + 44},{y}", stroke="#0c4a6e", stroke_width=6))
            out.append(path(f"M{sx},{y} L{sx + 44},{y}", stroke=COL["stream"], stroke_width=3.5))
        elif kind == "path":
            out.append(path(f"M{sx},{y} L{sx + 44},{y}", stroke=COL["path"], stroke_width=4.5))
        elif kind == "hpath":
            out.append(path(f"M{sx},{y} L{sx + 44},{y}", stroke=COL["hpath"], stroke_width=4.5))
        elif kind == "catch":
            out.append(rect(sx, y - 11, 44, 22, fill=COL["catch"], fill_opacity=0.25, stroke=COL["catch"], stroke_width=2.5,
                            stroke_dasharray="6 4"))
        elif kind == "dash":
            out.append(arrow(f"M{sx},{y} L{sx + 44},{y}", COL["leach"], 4, dash="9 6", marker="red"))
        elif kind == "scope":
            out.append(rect(sx, y - 11, 44, 22, fill="none", stroke=COL["scope"], stroke_width=3, stroke_dasharray="8 5"))
        out.append(text(x0 + 82, y + 7, name, 20))
        y += 38
    y += 4
    out.append(path(f"M{x0 + 20},{y} L{x0 + w - 20},{y}", stroke="#e5e7eb", stroke_width=2))
    y += 34
    out.append(text(x0 + 20, y, "해석 요점", 23, 700))
    facts = [
        f"원지형(5 m)은 서북 계곡으로 차수벽 통과, 현재 지형(0.5 m)은 진입로·농수로로 우회 → 큰 강 → 금강 ({out_len / 1000:.1f} km)",
        "2017.7 제방 붕괴, 침출수 여러 방면 유출·중신천 물고기 폐사 (보도)",
        f"매립지로 모이는 빗물 {catch_ha:.1f} ha (0.5 m) — 분수계는 둘레 절벽 상단",
    ]
    for f in facts:
        out.append(f'<circle cx="{x0 + 28}" cy="{y + 21}" r="4" fill="{MUTED}"/>')
        for r in wrap(f, w - 64, 19):
            y += 27
            out.append(text(x0 + 40, y, r, 19, 400, TEXT))
        y += 5
    return "\n".join(out)


def build_svg():
    body_map, out_len, catch_ha = site_map()
    body = "\n".join([
        rect(0, 0, 1600, 900, fill="white"),
        text(40, 70, ("드론 DTM" if RELIEF else "정사영상") + " 기반 현장 평면도 — 차수시설과 침출수 확산 경로", 44, 700),
        text(40, 116, ("0.5 m 드론 DTM 컬러 음영기복도" if RELIEF else "2026.09.07 드론 정사영상") + " · 차수벽·HDPE: 드론 영상·0.5 m DTM 판독, 우물: 배치도 정합 · 지표수 DTM 0.5 m + DEM 5 m", 25, 400, MUTED),
        path("M40,140 L1560,140", stroke="#e5e7eb", stroke_width=2),
        body_map, panel(out_len, catch_ha),
        text(40, 892, "※ 우물·집수구: 배치도 정합(오차 수 m) · 하류 차수벽: DTM 둔덕·고랑으로 추정. 지표 흐름은 DEM 함몰지 채움 후 D8 계산, "
                      "지하 흐름은 미반영. 우회 경로는 조사로 확인할 가설.", 17, 400, MUTED),
    ])
    return document("", body)


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, "leachate_plan_relief" if RELIEF else NAME)
