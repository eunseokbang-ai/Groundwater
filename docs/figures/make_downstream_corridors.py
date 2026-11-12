"""Downstream cut-off wall and the two outflow corridors (16:9): plan view on the drone orthomosaic with
the 5 m and 0.5 m surface outflow paths, and the cross-valley profile along the wall line.

    .venv\\Scripts\\python.exe docs\\figures\\make_downstream_corridors.py

Inputs: site_facilities.geojson (wall traced on the 0.5 m DTM), webapp terrain/site.geojson (5 m path),
terrain/hr_site.geojson (0.5 m path), the drone tile pyramid and the two DEMs.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform as tc

from svgkit import MUTED, TEXT, arrow, document, image, label, path, pill, rect, save, text, wrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp" / "ortho_history"))
from large_ortho import read_tiles  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TERRAIN = ROOT / "webapp" / "ortho_history" / "terrain"
NAME = "downstream_corridors"
CRS = "EPSG:5186"
E0, E1, N0, N1 = 202755.0, 203105.0, 384925.0, 385205.0     # plan extent
MX, MY, MW, MH = 40, 170, 780, 624
SX = MW / (E1 - E0)
S0, S1 = -125.0, 185.0                                       # profile range along the wall line (m)
C5, CH, CW = "#dc2626", "#c026d3", "#111827"


def px(x, y):
    return MX + (x - E0) * SX, MY + (N1 - y) * SX


def pts(xy):
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in (px(x, y) for x, y in xy))


def outflow(fn):
    f = next(f for f in json.loads((TERRAIN / fn).read_text(encoding="utf-8"))["features"]
             if f["properties"]["kind"] == "outflow_path")
    lon, lat = zip(*f["geometry"]["coordinates"])
    return list(zip(*tc("EPSG:4326", CRS, list(lon), list(lat))))


def sample(dem, xy):
    with rasterio.open(dem) as ds:
        v = np.array([s[0] for s in ds.sample(xy)], float)
    v[v < -1000] = np.nan
    return v


def build_svg():
    fc = json.loads((OUT_DIR / "data" / "site_facilities.geojson").read_text(encoding="utf-8"))
    walls = {f["properties"]["name"][:2]: f["properties"]["centerline"] for f in fc["features"]
             if f["properties"]["kind"] == "cutoff_wall"}
    down = np.array(walls["하류"])
    a, b = down[0], down[1]
    d = (b - a) / np.linalg.norm(b - a)
    L = float(np.linalg.norm(b - a))
    A, B = a + d * S0, a + d * S1                             # profile line A-A'

    # --- plan --------------------------------------------------------------------------------
    rgba = read_tiles(ROOT / "webapp" / "ortho_history" / "tiles" / "20260907", (E0, N0, E1, N1), MW * 2, MH * 2)
    im = Image.blend(Image.fromarray(rgba[..., :3]), Image.new("RGB", (MW * 2, MH * 2), "white"), 0.15)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    out = [f'<clipPath id="mc"><rect x="{MX}" y="{MY}" width="{MW}" height="{MH}"/></clipPath>',
           '<g clip-path="url(#mc)">', image(MX, MY, MW, MH, buf.getvalue())]
    p5, ph = outflow("site.geojson"), outflow("hr_site.geojson")
    for line, col, w in [(p5, C5, 6), (ph, CH, 6)]:
        out.append(f'<polyline points="{pts(line)}" fill="none" stroke="white" stroke-width="{w + 4}" stroke-linejoin="round" stroke-opacity="0.85"/>')
        out.append(f'<polyline points="{pts(line)}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round"/>')
    # flow direction arrows on each path
    for line, col in [(p5, C5), (ph, CH)]:
        arr = np.array(line)
        seg = np.cumsum(np.r_[0, np.hypot(*np.diff(arr, axis=0).T)])
        for s in (60, 160, 260, 360):
            if s + 8 > seg[-1]:
                continue
            i = np.searchsorted(seg, s)
            q0, q1 = px(*arr[i]), px(*arr[min(i + 3, len(arr) - 1)])
            out.append(arrow(f"M{q0[0]:.1f},{q0[1]:.1f} L{q1[0]:.1f},{q1[1]:.1f}", col, 4, marker="red"))
    # walls
    for k, line in walls.items():
        dash = ' stroke-dasharray="12 6"' if k == "하류" else ""
        out.append(f'<polyline points="{pts(line)}" fill="none" stroke="{CW}" stroke-width="10" stroke-linejoin="round"/>')
        out.append(f'<polyline points="{pts(line)}" fill="none" stroke="#ef4444" stroke-width="6" stroke-linejoin="round"{dash}/>')
    # profile line A-A'
    out.append(f'<polyline points="{pts([A, B])}" fill="none" stroke="#facc15" stroke-width="3" stroke-dasharray="10 6"/>')
    for p_, t_ in [(A, "A"), (B, "A′")]:
        x, y = px(*p_)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="#facc15" stroke="{CW}" stroke-width="2"/>')
        out.append(text(round(x, 1), round(y + 7, 1), t_, 18, 700, CW, anchor="middle"))
    # second corridor + suggested monitoring point
    c2 = a + d * (-100)
    x, y = px(*c2)
    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="34" fill="{CH}" fill-opacity="0.18" stroke="{CH}" stroke-width="3" stroke-dasharray="6 4"/>')
    mon = (202850, 385060)
    x, y = px(*mon)
    out.append(f'<path d="M{x:.1f},{y - 14:.1f} l12,22 h-24 z" fill="#0ea5e9" stroke="white" stroke-width="2"/>')
    out.append("</g>")
    out.append(rect(MX, MY, MW, MH, fill="none", stroke="#cbd5e1", stroke_width=2))
    lab = lambda xy, rows, **kw: label(*px(*xy), rows, **kw)
    out += [
        lab((202990, 385180), [("① 5 m 경로: 원래 계곡", 20, 700, C5), ("→ 하류 차수벽 통과", 18, 400, TEXT)], pad=6),
        lab((202775, 384975), [("② 0.5 m 경로: 진입로 따라 우회", 20, 700, CH), ("성토로 계곡 바닥 대신 남서쪽으로", 18, 400, TEXT)], pad=6),
        lab((202905, 385000), [("제2 저지대 (진입로)", 19, 700, CH)], pad=5),
        lab((202862, 385052), [("수질 측정 제안 지점", 18, 700, "#0369a1")], pad=5),
        lab((203095, 385000), [("상류 차수벽", 19, 700, "#b91c1c")], anchor="end", pad=5),
        lab((202955, 385128), [("하류 차수벽 (추정)", 19, 700, "#b91c1c")], pad=5),
    ]
    x, y = px(*mon)
    sb = 50 * SX
    out.append(rect(MX + 16, MY + MH - 46, sb + 40, 34, rx=6, fill="white", fill_opacity=0.85))
    out.append(rect(MX + 26, MY + MH - 22, sb, 7, fill=CW))
    out.append(text(round(MX + 26 + sb, 1), MY + MH - 28, "50 m", 16, 700, anchor="end"))
    out.append(path(f"M{MX + 36},{MY + 62} L{MX + 48},{MY + 26} L{MX + 60},{MY + 62} L{MX + 48},{MY + 52} Z",
                    fill="white", stroke=CW, stroke_width=2))
    out.append(label(MX + 48, MY + 2, [("N", 18, 700, TEXT)], anchor="middle", pad=3))

    # --- profile ------------------------------------------------------------------------------
    s = np.arange(S0, S1 + 0.1, 1.0)
    xy = [tuple(a + d * k) for k in s]
    zh = sample(ROOT / "DEM(5m)" / "dem(0.5m).tif", xy)
    z5 = sample(next(p for p in (ROOT / "DEM(5m)").glob("*wide*.tif")), xy)
    GX, GY, GW, GH = 920, 226, 620, 316
    zlo, zhi = 20.0, 42.0
    gx = lambda k: GX + (k - S0) / (S1 - S0) * GW
    gy = lambda z: GY + GH - (z - zlo) / (zhi - zlo) * GH
    out.append(text(860, 186, "단면 A–A′ (하류 차수벽 선을 따라 연장, 남남서 → 북북동)", 22, 700))
    for z in range(20, 43, 5):
        out.append(path(f"M{GX},{gy(z):.1f} L{GX + GW},{gy(z):.1f}", stroke="#e5e7eb", stroke_width=1))
        out.append(text(GX - 10, round(gy(z) + 6, 1), str(z), 16, 400, MUTED, anchor="end"))
    out.append(text(GX - 10, GY + GH + 24, "m", 16, 400, MUTED, anchor="end"))
    for k in range(-100, 200, 50):
        out.append(text(round(gx(k), 1), GY + GH + 24, f"{k}", 16, 400, MUTED, anchor="middle"))
    out.append(text(GX + GW, GY + GH + 46, "벽 시작점부터 거리 (m)", 16, 400, MUTED, anchor="end"))
    # ground fill (0.5 m) and lines
    poly = " ".join(f"{gx(k):.1f},{gy(z):.1f}" for k, z in zip(s, zh) if np.isfinite(z))
    out.append(f'<polygon points="{gx(S0):.1f},{GY + GH} {poly} {gx(S1):.1f},{GY + GH}" fill="#e7d9bf"/>')
    out.append(f'<polyline points="{" ".join(f"{gx(k):.1f},{gy(z):.1f}" for k, z in zip(s, z5) if np.isfinite(z))}" '
               f'fill="none" stroke="#6b7280" stroke-width="2.5" stroke-dasharray="7 5"/>')
    out.append(f'<polyline points="{poly}" fill="none" stroke="#92400e" stroke-width="3"/>')
    # wall
    zw = np.nanmean(zh[(s >= 0) & (s <= L)])
    out.append(rect(gx(0), gy(zw), gx(L) - gx(0), GY + GH - gy(zw), fill="url(#wallhatch)", stroke="#b91c1c", stroke_width=2))
    out.append(label(round(gx(L / 2), 1), round(gy(zw) - 74, 1), [("하류 차수벽 64 m", 18, 700, "#b91c1c"),
                                                                 ("계곡 바닥 25.5 m (성토)", 16, 400, TEXT)], anchor="middle", pad=5))
    # annotations
    k2 = -100
    out.append(label(round(gx(-121), 1), round(gy(28.7) + 14, 1), [("제2 저지대", 17, 700, CH), ("28.7 m · 진입로", 15, 400, TEXT)],
                     pad=4))
    out.append(label(round(gx(-48), 1), round(gy(34.2) - 52, 1), [("낮은 능선 34 m", 17, 700, TEXT)], anchor="middle", pad=4))
    out.append(arrow(f"M{gx(-30):.1f},{gy(31.5):.1f} Q{gx(-48):.1f},{gy(26):.1f} {gx(-82):.1f},{gy(28.3):.1f}", CH, 3,
                     dash="6 4", marker="red"))
    out.append(text(round(gx(-40), 1), round(gy(22.3), 1), "벽 끝 우회?", 16, 700, CH, anchor="middle"))
    # legend
    ly = GY + GH + 70
    out.append(path(f"M{GX},{ly} L{GX + 36},{ly}", stroke="#92400e", stroke_width=3))
    out.append(text(GX + 44, ly + 6, "DTM 0.5 m (현재 지형)", 16))
    out.append(path(f"M{GX + 250},{ly} L{GX + 286},{ly}", stroke="#6b7280", stroke_width=2.5, stroke_dasharray="7 5"))
    out.append(text(GX + 294, ly + 6, "DEM 5 m (원지형, 바닥 23.7 m)", 16))

    # --- conclusions --------------------------------------------------------------------------
    y = 676
    out.append(text(860, y, "해석", 24, 700))
    notes = [
        ("주 계곡은 차수벽이 막음", "벽이 계곡 바닥 평탄면을 양쪽 사면 끝까지 가로지름"),
        ("깨끗한 지표수는 우회시킴", "계곡 바닥 성토(~2 m)로 빗물이 진입로 쪽으로 돌아 나감 → 침출수 처리량 감소"),
        ("남서쪽 제2 통로는 열려 있음", "낮은 능선(34 m) 하나 너머 저지대 — 지하 우회 여부는 관측정·ERT로 확인"),
    ]
    for t1, t2 in notes:
        y += 36
        out.append(f'<circle cx="870" cy="{y - 7}" r="5" fill="{CH}"/>')
        out.append(text(884, y, t1, 19, 700))
        for r_ in wrap(t2, 660, 17):
            y += 25
            out.append(text(884, y, r_, 17, 400, MUTED))
    defs = ('<pattern id="wallhatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<rect width="8" height="8" fill="#fecaca"/><path d="M0 0 V8" stroke="#ef4444" stroke-width="2.5"/></pattern>')
    body = [rect(0, 0, 1600, 900, fill="white"),
            text(40, 68, "하류 차수벽과 두 개의 유출 통로", 44, 700),
            text(40, 112, "2026.09.07 드론 정사영상 · 지표 유출 경로: DEM 5 m(원지형) vs DTM 0.5 m(현재 지형) · 단면: DTM 0.5 m", 23, 400, MUTED),
            path("M40,134 L1560,134", stroke="#e5e7eb", stroke_width=2), *out,
            text(40, 888, "※ 지표 흐름 해석(D8). 지하 흐름은 별도 확인 필요. 하류 차수벽 위치는 0.5 m DTM 둔덕·고랑으로 추정.", 16, 400, MUTED)]
    return document(defs, "\n".join(body))


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
