"""Where rain falling around the landfill goes (16:9): drainage classes, small flow lines, flow arrows
and contours from the 5 m DEM on the 2025 orthophoto. Also copied to the viewer
(webapp/ortho_history/terrain/surface_watershed.png).

    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_terrain.py   (first, writes terrain/*)
    .venv\\Scripts\\python.exe docs\\figures\\make_surface_watershed.py
"""
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import Resampling, reproject

from svgkit import MUTED, TEXT, document, image, label, path, rect, save, text, wrap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "webapp" / "ortho_history")]
from groundwater.terrain import hydrology as hy  # noqa: E402
from build_tiles import find_sheets, georeference, read_metadata  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TERRAIN = ROOT / "webapp" / "ortho_history" / "terrain"
NAME = "leachate_surface_watershed"
CRS = "EPSG:5186"
MX, MY, MW, MH = 40, 168, 990, 702


def main_svg():
    meta = json.loads((TERRAIN / "meta.json").read_text(encoding="utf-8"))
    g = np.load(TERRAIN / "detail_grid.npz")
    cls, acc, code, z = g["cls"], g["acc"], g["code"], g["z"]
    t = Affine(*g["transform"])
    cell = t.a
    h, w = cls.shape
    E0, N1 = t.c, t.f
    E1, N0 = E0 + w * cell, N1 - h * cell
    sx, sy = MW / (E1 - E0), MH / (N1 - N0)
    px = lambda x, y: (MX + (x - E0) * sx, MY + (N1 - y) * sy)
    pts = lambda xy: " ".join(f"{a:.1f},{b:.1f}" for a, b in (px(x, y) for x, y in xy))
    classes = meta["detail"]["classes"]

    # orthophoto background
    src = ROOT / "과거정사영상"
    sheet = next(s for s in find_sheets(src) if s["year"] == 2025)
    om = read_metadata(src).get(2025)
    with rasterio.open(sheet["path"]) as ds:
        ot = georeference(ds, sheet["sheet"], float(om["지상표본거리"]) if om else None)
        rgb = ds.read([1, 2, 3])
    dst = np.zeros((3, MH * 2, MW * 2), np.uint8)
    reproject(rgb, dst, src_transform=ot, src_crs=CRS, dst_crs=CRS, resampling=Resampling.average,
              dst_transform=Affine((E1 - E0) / (MW * 2), 0, E0, 0, -(N1 - N0) / (MH * 2), N1))
    im = Image.blend(Image.fromarray(np.moveaxis(dst, 0, -1)), Image.new("RGB", (MW * 2, MH * 2), "white"), 0.25)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)

    out = [f'<clipPath id="mapClip"><rect x="{MX}" y="{MY}" width="{MW}" height="{MH}"/></clipPath>',
           '<g clip-path="url(#mapClip)">', image(MX, MY, MW, MH, buf.getvalue())]
    # drainage classes
    for geom, val in shapes(cls.astype(np.int16), mask=cls >= 0, transform=t):
        c = classes[int(val)]
        d = " ".join("M" + pts(ring).replace(" ", " L") + " Z" for ring in geom["coordinates"])
        out.append(path(d, fill=c["color"], fill_opacity=0.30, fill_rule="evenodd", stroke="white", stroke_width=2.2))
    # contours (5 m)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zz = np.where(z > 0, z, np.nan)
    xs = E0 + (np.arange(w) + 0.5) * cell
    ys = N1 - (np.arange(h) + 0.5) * cell
    cs = plt.contour(xs, ys, zz, levels=np.arange(np.floor(np.nanmin(zz) / 5) * 5, np.nanmax(zz) + 5, 5))
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) > 3:
                major = int(round(lev)) % 25 == 0
                out.append(f'<polyline points="{pts(seg)}" fill="none" stroke="#1f2937" '
                           f'stroke-opacity="{0.55 if major else 0.25}" stroke-width="{1.4 if major else 0.8}"/>')
    plt.close("all")
    # hillslope flow intensity (MFD): where rain gathers before it forms a channel
    out.append(image(MX, MY, MW, MH, (TERRAIN / "detail_mfd.png").read_bytes()).replace(
        "preserveAspectRatio=\"xMidYMid meet\"", "preserveAspectRatio=\"none\" style=\"image-rendering:auto\""))
    # downslope direction on the hillslopes (steepest descent of the smoothed surface, every 40 m)
    from scipy.ndimage import gaussian_filter
    zs = gaussian_filter(np.where(z > 0, z, np.nanmean(np.where(z > 0, z, np.nan))), 1.5)
    gy, gx = np.gradient(zs, cell)
    step = 8
    for r in range(step // 2, h, step):
        for c in range(step // 2, w, step):
            if cls[r, c] < 0 or g["mfd"][r, c] * cell * cell >= 3000:
                continue
            vx, vy = -gx[r, c], gy[r, c]          # map axes: x east, y north (rows grow south)
            n = np.hypot(vx, vy)
            if n < 0.02:
                continue
            x, y = t * (c + 0.5, r + 0.5)
            ux, uy = vx / n, vy / n
            tip = px(x + ux * 9, y + uy * 9)
            base = px(x - ux * 7, y - uy * 7)
            lx, ly = -uy * 5, ux * 5               # half base width (m), perpendicular
            b1, b2 = px(x - ux * 7 + lx, y - uy * 7 + ly), px(x - ux * 7 - lx, y - uy * 7 - ly)
            col = classes[int(cls[r, c])]["color"]
            out.append(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{b1[0]:.1f},{b1[1]:.1f} L{base[0]:.1f},{base[1]:.1f} '
                       f'L{b2[0]:.1f},{b2[1]:.1f} Z" fill="{col}" stroke="white" stroke-width="1"/>')
    # small flow lines
    flows = json.loads((TERRAIN / "detail_flows.geojson").read_text(encoding="utf-8"))
    from rasterio.warp import transform as tc
    for f in sorted(flows["features"], key=lambda f: f["properties"]["area_ha"]):
        lon, lat = zip(*f["geometry"]["coordinates"])
        fx, fy = tc("EPSG:4326", CRS, list(lon), list(lat))
        wdt = max(2.0, min(8, 2.0 + 2.6 * np.log10(f["properties"]["area_ha"] / 0.5)))
        out.append(f'<polyline points="{pts(zip(fx, fy))}" fill="none" stroke="white" stroke-width="{wdt + 2.2:.1f}" '
                   'stroke-linejoin="round" stroke-linecap="round" stroke-opacity="0.85"/>')
        out.append(f'<polyline points="{pts(zip(fx, fy))}" fill="none" stroke="{f["properties"]["color"]}" '
                   f'stroke-width="{wdt:.1f}" stroke-linejoin="round" stroke-linecap="round"/>')
    # landfill outline and cut-off walls
    fc = json.loads((OUT_DIR / "data" / "site_facilities.geojson").read_text(encoding="utf-8"))
    for f in fc["features"]:
        k = f["properties"]["kind"]
        if k == "hdpe_liner":
            out.append(f'<polygon points="{pts(f["geometry"]["coordinates"][0])}" fill="none" stroke="#111827" '
                       'stroke-width="3" stroke-dasharray="8 5"/>')
        elif k == "cutoff_wall":
            out.append(f'<polygon points="{pts(f["geometry"]["coordinates"][0])}" fill="#ef4444" stroke="#7f1d1d"/>')
    out.append("</g>")
    out.append(rect(MX, MY, MW, MH, fill="none", stroke="#cbd5e1"))

    # north arrow + scale bar
    out.append(path(f"M{MX + 40},{MY + 70} L{MX + 52},{MY + 34} L{MX + 64},{MY + 70} L{MX + 52},{MY + 60} Z",
                    fill="white", stroke="#111827", stroke_width=2))
    out.append(label(MX + 52, MY + 4, [("N", 20, 700, TEXT)], anchor="middle", pad=3))
    sb = 100 * sx
    x0, y0 = MX + 24, MY + MH - 30
    out.append(rect(x0 - 8, y0 - 30, sb + 70, 46, rx=6, fill="white", fill_opacity=0.85))
    out.append(rect(x0, y0, sb / 2, 8, fill="#111827"))
    out.append(rect(x0 + sb / 2, y0, sb / 2, 8, fill="white", stroke="#111827"))
    out.append(text(x0, y0 - 8, "0", 16, 700))
    out.append(text(round(x0 + sb, 1), y0 - 8, "100 m", 16, 700, anchor="middle"))
    # landfill label
    lf = next(f for f in fc["features"] if f["properties"]["kind"] == "hdpe_liner")["geometry"]["coordinates"][0]
    lx, ly = np.mean(lf, axis=0)
    out.append(label(*px(lx + 40, ly - 30), [("매립지", 22, 700, "white")], fill="#111827", opacity=0.8,
                     stroke="#111827", pad=6))

    # side panel
    x0, y0, pw = 1066, 168, 494
    out.append(rect(x0, y0, pw, 702, rx=10, fill="white", stroke="#cbd5e1"))
    out.append(text(x0 + 20, y0 + 40, "빗물이 최종적으로 가는 곳", 24, 700))
    y = y0 + 80
    for c in classes:
        out.append(rect(x0 + 22, y - 14, 40, 26, rx=4, fill=c["color"], fill_opacity=0.45, stroke=c["color"], stroke_width=2))
        rows = wrap(c["name"], pw - 180, 20, 700)
        for i, row in enumerate(rows):
            out.append(text(x0 + 76, y + 6 + i * 26, row, 20, 700))
        out.append(text(x0 + pw - 22, y + 6, f'{c["area_ha"]} ha', 20, 700, c["color"], anchor="end"))
        y += 26 * len(rows) + 18
    y += 4
    legend = [("mfd", "사면 흐름 강도 (빗물이 모이는 정도)"), ("line", "물길 (굵기 = 상류 면적, 0.5 ha↑)"),
              ("arrow", "사면 위 빗물 흐름 방향 (40 m 간격)"), ("contour", "등고선 5 m (굵은 선 25 m)"),
              ("liner", "매립지 HDPE 시트 경계"), ("wall", "연직차수벽")]
    for kind, name in legend:
        sx0 = x0 + 22
        if kind == "mfd":
            out.append(f'<defs><linearGradient id="mfdg"><stop offset="0" stop-color="#dbeafe" stop-opacity="0.2"/>'
                       f'<stop offset="1" stop-color="#1d4ed8" stop-opacity="0.9"/></linearGradient></defs>')
            out.append(rect(sx0, y - 9, 40, 18, rx=3, fill="url(#mfdg)", stroke="#94a3b8"))
        elif kind == "line":
            out.append(path(f"M{sx0},{y} L{sx0 + 40},{y}", stroke="#0d9488", stroke_width=5, stroke_linecap="round"))
        elif kind == "arrow":
            out.append(path(f"M{sx0 + 30},{y} L{sx0 + 12},{y - 6} L{sx0 + 16},{y} L{sx0 + 12},{y + 6} Z",
                            fill="#0d9488", stroke="white", stroke_width=1))
        elif kind == "contour":
            out.append(path(f"M{sx0},{y} L{sx0 + 40},{y}", stroke="#1f2937", stroke_width=1.4, stroke_opacity=0.6))
        elif kind == "liner":
            out.append(path(f"M{sx0},{y} L{sx0 + 40},{y}", stroke="#111827", stroke_width=3, stroke_dasharray="8 5"))
        elif kind == "wall":
            out.append(rect(sx0, y - 5, 40, 10, fill="#ef4444", stroke="#7f1d1d"))
        out.append(text(x0 + 76, y + 7, name, 19))
        y += 34
    y += 8
    out.append(path(f"M{x0 + 20},{y} L{x0 + pw - 20},{y}", stroke="#e5e7eb", stroke_width=2))
    y += 32
    out.append(text(x0 + 20, y, "해석 요점", 22, 700))
    lc = classes[0]
    notes = [
        f"매립지로 지표수가 모이는 범위는 {lc['area_ha']} ha — 매립지 남쪽 배후 사면이 대부분",
        "매립지 북쪽 끝으로 나간 물은 서쪽 골짜기를 따라 중신천 방향으로 흐름",
        "남서·북쪽 사면의 빗물은 매립지를 거치지 않고 다른 계곡으로 감",
    ]
    for n_ in notes:
        out.append(f'<circle cx="{x0 + 28}" cy="{y + 20}" r="4" fill="{MUTED}"/>')
        for r_ in wrap(n_, pw - 64, 18):
            y += 25
            out.append(text(x0 + 40, y, r_, 18, 400, TEXT))
        y += 4
    return out


def build_svg():
    defs = ('<marker id="tip" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>')
    body = "\n".join([
        rect(0, 0, 1600, 900, fill="white"),
        text(40, 70, "강우 시 매립지 주변 지표수 흐름 — 분수계와 작은 물길", 44, 700),
        text(40, 116, "DEM 5 m · 분수계 D8 + 사면 흐름 MFD · 2025 정사영상: 어디에 내린 비가 매립지로, 어디는 다른 계곡으로 가는가", 24, 400, MUTED),
        path("M40,140 L1560,140", stroke="#e5e7eb", stroke_width=2),
        *main_svg(),
        text(40, 892, "※ 지표 흐름만 해석(함몰지 채움 후 D8). 침투·지하 흐름, 벤치플륨관·우수차집관로 등 배수시설과 HDPE 위 흐름은 미반영. "
                      "DEM은 매립 전 지형일 수 있음.", 17, 400, MUTED),
    ])
    return document(defs, body)


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
    shutil.copyfile(OUT_DIR / f"{NAME}.png", TERRAIN / "surface_watershed.png")
    print(f"Copied to {TERRAIN / 'surface_watershed.png'}")
