"""5 m vs 0.5 m surface drainage around the landfill (16:9): the same question - where does rain end
up - answered on the regional 5 m DEM and on the 0.5 m DTM of the present ground.

    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_terrain.py       (5 m, writes terrain/detail_*)
    .venv\\Scripts\\python.exe webapp\\ortho_history\\build_terrain_hr.py    (0.5 m, writes terrain/hr_*)
    .venv\\Scripts\\python.exe docs\\figures\\make_surface_watershed_hr.py
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
from rasterio.warp import Resampling, reproject, transform as tc

from svgkit import MUTED, TEXT, document, image, label, path, rect, save, text, wrap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "webapp" / "ortho_history")]
from groundwater.terrain import hydrology as hy  # noqa: E402
from build_tiles import find_sheets, georeference, read_metadata  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TERRAIN = ROOT / "webapp" / "ortho_history" / "terrain"
NAME = "leachate_surface_watershed_hr"
CRS = "EPSG:5186"
PW, PH, PY = 740, 566, 222              # each map panel


def main_svg():
    hrm = json.loads((TERRAIN / "hr_meta.json").read_text(encoding="utf-8"))
    m5 = json.loads((TERRAIN / "meta.json").read_text(encoding="utf-8"))
    g = np.load(TERRAIN / "hr_grid.npz")
    t = Affine(*g["transform"])
    fc = json.loads((OUT_DIR / "data" / "site_facilities.geojson").read_text(encoding="utf-8"))
    liner = next(f for f in fc["features"] if f["properties"]["kind"] == "hdpe_liner")["geometry"]["coordinates"][0]
    lx, ly = np.mean(liner, axis=0)
    half_w, half_h = 330.0, 330.0 * PH / PW
    E0, E1, N0, N1 = lx - half_w, lx + half_w, ly - half_h, ly + half_h

    # orthophoto background (shared)
    src = ROOT / "과거정사영상"
    sheet = next(s for s in find_sheets(src) if s["year"] == 2025)
    om = read_metadata(src).get(2025)
    with rasterio.open(sheet["path"]) as ds:
        ot = georeference(ds, sheet["sheet"], float(om["지상표본거리"]) if om else None)
        rgb = ds.read([1, 2, 3])
    dst = np.zeros((3, PH * 2, PW * 2), np.uint8)
    reproject(rgb, dst, src_transform=ot, src_crs=CRS, dst_crs=CRS, resampling=Resampling.average,
              dst_transform=Affine((E1 - E0) / (PW * 2), 0, E0, 0, -(N1 - N0) / (PH * 2), N1))
    im = Image.blend(Image.fromarray(np.moveaxis(dst, 0, -1)), Image.new("RGB", (PW * 2, PH * 2), "white"), 0.3)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    ortho = buf.getvalue()

    def panel(x0, title, sub, cls, ct, classes, flows, mfd_png, catch_ha, colour):
        sx = PW / (E1 - E0)
        px = lambda x, y: (x0 + (x - E0) * sx, PY + (N1 - y) * sx)
        pts = lambda xy: " ".join(f"{a:.1f},{b:.1f}" for a, b in (px(x, y) for x, y in xy))
        out = [f'<clipPath id="c{x0}"><rect x="{x0}" y="{PY}" width="{PW}" height="{PH}"/></clipPath>',
               f'<g clip-path="url(#c{x0})">', image(x0, PY, PW, PH, ortho)]
        for geom, v in shapes(cls.astype(np.int16), mask=cls >= 0, transform=ct):
            c = classes[int(v)]
            d = " ".join("M" + pts(ring).replace(" ", " L") + " Z" for ring in geom["coordinates"])
            op = 0.42 if int(v) == 0 else 0.2
            out.append(path(d, fill=c["color"], fill_opacity=op, fill_rule="evenodd", stroke="white", stroke_width=1.6))
        if mfd_png:
            out.append(image(x0, PY, PW, PH, mfd_png).replace('preserveAspectRatio="xMidYMid meet"', 'preserveAspectRatio="none"'))
        for f in sorted(flows, key=lambda f: f["properties"]["area_ha"]):
            lon, lat = zip(*f["geometry"]["coordinates"])
            fx, fy = tc("EPSG:4326", CRS, list(lon), list(lat))
            if max(fx) < E0 or min(fx) > E1 or max(fy) < N0 or min(fy) > N1:
                continue
            wd = max(1.4, min(6, 1.4 + 1.8 * np.log10(f["properties"]["area_ha"] / 0.2)))
            out.append(f'<polyline points="{pts(zip(fx, fy))}" fill="none" stroke="white" stroke-width="{wd + 2:.1f}" '
                       'stroke-linejoin="round" stroke-opacity="0.8"/>')
            out.append(f'<polyline points="{pts(zip(fx, fy))}" fill="none" stroke="{f["properties"]["color"]}" '
                       f'stroke-width="{wd:.1f}" stroke-linejoin="round"/>')
        out.append(f'<polygon points="{pts(liner)}" fill="none" stroke="#111827" stroke-width="2.5" stroke-dasharray="7 5"/>')
        for f in fc["features"]:
            if f["properties"]["kind"] == "cutoff_wall":
                out.append(f'<polygon points="{pts(f["geometry"]["coordinates"][0])}" fill="#ef4444" stroke="#7f1d1d"/>')
        out.append("</g>")
        out.append(rect(x0, PY, PW, PH, fill="none", stroke="#cbd5e1", stroke_width=2))
        out.append(text(x0, PY - 48, title, 27, 700, colour))
        out.append(text(x0, PY - 14, sub, 21, 400, MUTED))
        out.append(label(x0 + PW - 12, PY + 12, [(f"매립지로 유입 {catch_ha} ha", 24, 700, "#c2410c")], anchor="end", pad=8))
        # scale bar
        sb = 100 * sx
        bx, by = x0 + 18, PY + PH - 26
        out.append(rect(bx - 8, by - 28, sb + 60, 42, rx=6, fill="white", fill_opacity=0.85))
        out.append(rect(bx, by, sb, 7, fill="#111827"))
        out.append(text(round(bx + sb, 1), by - 7, "100 m", 16, 700, anchor="middle"))
        return out

    # 5 m panel from detail_grid.npz
    g5 = np.load(TERRAIN / "detail_grid.npz")
    t5 = Affine(*g5["transform"])
    flows5 = json.loads((TERRAIN / "detail_flows.geojson").read_text(encoding="utf-8"))["features"]
    classes5 = m5["detail"]["classes"]
    out = panel(40, "DEM 5 m (등고선 기반, 매립 전 지형 포함)", "자연 지형 기준 — 작은 물길·배수시설 미표현",
                g5["cls"], t5, classes5, flows5, None, classes5[0]["area_ha"], "#475569")
    # 0.5 m panel: classes from hr_grid, MFD intensity resampled to the panel
    mfd = g["mfd"]
    v = np.clip((np.log10(np.maximum(mfd, 1)) - np.log10(50)) / (np.log10(2e4) - np.log10(50)), 0, 1)
    rgba = np.zeros(v.shape + (4,), np.uint8)
    rgba[..., 0] = (200 * (1 - v)).astype(np.uint8)
    rgba[..., 1] = (230 - 60 * v).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = (np.where(g["valid"], v, 0) ** 1.2 * 235).astype(np.uint8)
    dst = np.zeros((4, PH * 2, PW * 2), np.uint8)
    reproject(np.moveaxis(rgba, -1, 0), dst, src_transform=t, src_crs=CRS, dst_crs=CRS, resampling=Resampling.max,
              dst_transform=Affine((E1 - E0) / (PW * 2), 0, E0, 0, -(N1 - N0) / (PH * 2), N1))
    buf = io.BytesIO()
    Image.fromarray(np.moveaxis(dst, 0, -1), "RGBA").save(buf, "PNG", optimize=True)
    flows_hr = json.loads((TERRAIN / "hr_flows.geojson").read_text(encoding="utf-8"))["features"]
    out += panel(820, "DTM 0.5 m (현재 지형: HDPE 면·벤치·도로·수로)", "작은 물길과 시설이 흐름을 바꿈",
                 g["cls"], t, hrm["classes"], flows_hr, buf.getvalue(), hrm["catchment_ha"], "#c2410c")
    return out, hrm, m5


def build_svg():
    body_map, hrm, m5 = main_svg()
    c5, ch = m5["site"]["catchment_ha"], hrm["catchment_ha"]
    body = [rect(0, 0, 1600, 900, fill="white"),
            text(40, 68, "강우 시 매립지 지표수 흐름 — DEM 5 m vs DTM 0.5 m", 42, 700),
            text(40, 112, "같은 질문(빗물이 어디로 가는가)을 해상도가 다른 두 지형 자료로 비교 · 2025 정사영상 배경", 24, 400, MUTED),
            path("M40,134 L1560,134", stroke="#e5e7eb", stroke_width=2), *body_map]
    notes = [
        f"매립지로 모이는 지표 유역: 5 m {c5} ha → 0.5 m {ch} ha. 현재 지형에서는 둘레 절벽 상단이 분수계 — 동쪽 절벽 너머 사면은 북쪽 골짜기로 빠짐",
        "0.5 m에서는 HDPE 면 위 흐름, 진입로 측구, 벤치를 따른 물길이 보임 — 우수 차집 시설의 실제 효과를 지형으로 확인 가능",
        "도로 아래 암거는 DTM에 없어 둑처럼 보임 → 필요하면 각인 보정 · 지표 흐름만 해석(지하 흐름 별도 확인)",
    ]
    y = 826
    for n in notes[:2]:
        body.append(f'<circle cx="48" cy="{y - 7}" r="5" fill="#c2410c"/>')
        body.append(text(62, y, wrap(n, 1490, 20)[0], 20, 400, TEXT))
        y += 30
    body.append(text(40, 892, "※ " + notes[2], 17, 400, MUTED))
    return document("", "\n".join(body))


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
    shutil.copyfile(OUT_DIR / f"{NAME}.png", TERRAIN / "surface_watershed_hr.png")
