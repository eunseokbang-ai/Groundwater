"""The landfill site 1978-2026 in 12 key years, one per major event (quarrying -> illegal landfilling -> detection
and leachate release -> enforced removal -> cutoff -> this project's drone survey), from the orthophoto viewer's layers:

  site_history_montage.jpg   -- 4 x 3 panels with year tags (also cut into the progress-summary card)
  docs/부지_변천_요약.pptx    -- one editable slide: the 12 years in time order with the event under each
                                (and site_history_summary.png when PowerPoint is installed)

Each year is resampled onto the same 640 m window (1 m/px) in EPSG:5186: aerial-photo captures with the
transform registered by build_tiles.py, NGII orthophotos with their sheet georeference, the drone orthomosaic with
its own. Events follow make_leachate_timeline.py (press reports 2016-2026).

    .venv\\Scripts\\python.exe docs\\figures\\make_site_history_montage.py

The slide needs python-pptx.
"""
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw, ImageFont
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
VIEWER = HERE.parents[1] / "webapp" / "ortho_history"
sys.path[:0] = [str(VIEWER)]
import captures as C  # noqa: E402
from build_tiles import ROOT, find_sheets, georeference, read_metadata  # noqa: E402

E0, N1, S, R = 202850, 385200, 640, 1.0       # 640 m window around the landfill, 1 m/px
OUT = HERE / "site_history_montage.jpg"
PPTX = HERE.parent / "부지_변천_요약.pptx"
CROP = (64, 32, 576, 544)                     # the slide's 512 m square around the pit (window pixels)

# phase -> (label, colour); events in time order: (layer id or year, tag, phase, event)
PHASES = {"pre": ("원지형", "#4d7c0f"), "quarry": ("채석", "#57534e"), "fill": ("불법매립", "#b91c1c"),
          "crisis": ("적발·유출", "#b45309"), "removal": ("행정대집행", "#1d4ed8"), "cutoff": ("차수·현장실증", "#15803d")}
EVENTS = [                                    # (key, tag, phase, [event, detail])
    (1978, "1978", "pre", ["원지형", "채석 전 산림"]),
    (1984, "1984", "quarry", ["채석 시작", "첫 굴착 흔적"]),
    (1992, "1992", "quarry", ["채석장 확장", "굴착 범위 확대"]),
    (2007, "2007", "quarry", ["채석 막바지", "깊은 굴착부 형성"]),
    (2011, "2011", "fill", ["불법매립 시작", "2011.2 ~ 2016.4"]),
    (2014, "2014", "fill", ["매립 진행", "물고기 폐사 2013·14"]),
    (2016, "2016", "crisis", ["적발", "원상복구 조치명령"]),
    (2018, "2018", "crisis", ["대량 유출 후 (2017)", "침출수 저류조·덮개"]),
    (2021, "2021", "removal", ["1차 행정대집행", "4.9만 t 처리"]),
    (2023, "2023", "removal", ["대집행 진행", "복구협약 2023.9"]),
    (2025, "2025", "cutoff", ["차수매트", "연직차수벽"]),
    ("20260907", "2026.9", "cutoff", ["본 과제", "1차 드론 조사"]),
]


def year_images():
    """{layer key: (640, 640, 3) uint8} for the EVENTS, all on the same window."""
    man = json.load(open(VIEWER / "layers.json", encoding="utf-8"))
    grid = Affine(R, 0, E0, 0, -R, N1)
    src = ROOT / "과거정사영상"
    meta = read_metadata(src)
    sheets = {s["year"]: s for s in find_sheets(src)}
    want = {k for k, *_ in EVENTS}
    out = {}
    for d in man["layers"]:
        key = d["id"] if d["kind"] == "drone" else d["year"]
        if key not in want or key in out:
            continue
        a = np.zeros((3, S, S), np.uint8)
        if d["kind"] == "capture":
            rgb = C.load_rgb(src / d["file"])
            rgb = C.clean_capture(rgb, C.icon_mask(rgb))
            reproject(np.moveaxis(rgb, -1, 0), a, src_transform=Affine(*d["transform"]), src_crs="EPSG:5186",
                      dst_transform=grid, dst_crs="EPSG:5186", resampling=Resampling.bilinear)
        elif d["kind"] == "drone":
            from rasterio.vrt import WarpedVRT

            with rasterio.open(src / d["file"]) as ds, WarpedVRT(ds, crs="EPSG:5186", transform=grid, width=S, height=S,
                                                                resampling=Resampling.average) as v:
                a = v.read([1, 2, 3], out_shape=(3, S, S))
        else:
            s = sheets[d["year"]]
            with rasterio.open(s["path"]) as ds:
                mm = meta.get(d["year"])
                t = georeference(ds, s["sheet"], float(mm["지상표본거리"]) if mm else None)
                reproject(ds.read([1, 2, 3]), a, src_transform=t, src_crs="EPSG:5186", dst_transform=grid,
                          dst_crs="EPSG:5186", resampling=Resampling.average)
        out[key] = np.moveaxis(a, 0, -1)
        print("read", key, flush=True)
    return out


def montage(images):
    font = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 34)
    sheet = Image.new("RGB", (4 * 405, 3 * 405), "white")
    for i, (key, tag, _, _) in enumerate(EVENTS):
        im = Image.fromarray(images[key]).resize((400, 400), Image.LANCZOS)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, int(dr.textlength(tag, font=font)) + 16, 46], fill=(17, 24, 39))
        dr.text((8, 2), tag, font=font, fill="white")
        sheet.paste(im, ((i % 4) * 405, (i // 4) * 405))
    sheet.save(OUT, quality=88)
    print(f"Wrote {OUT} ({len(EVENTS)} panels)")


def slide(images):
    """One slide: the 12 years in two rows of six (left to right in time), each with its event and phase colour."""
    from pptx import Presentation

    from make_module_linkage import Canvas, _e, export_png
    from svgkit import MUTED, TEXT, text_width

    prs = Presentation()
    prs.slide_width, prs.slide_height = 12192000, 6858000
    c = Canvas(prs, "history")
    c.rect(0, 0, 1600, 900, fill="#ffffff", name="배경")
    c.text(56, 82, "부지 변천 — 채석장에서 매립지, 차수까지", 46, True)
    c.text(56, 128, "1978–2026 주요 사건별 12시기 · 같은 범위 (약 510 m) · 흑백 항공사진 → 정사영상 → 드론", 26, False, MUTED,
           width=1496)
    c.path([(56, 152), (1552, 152)], "#e5e7eb", 2, head=None)
    w, gap = (1496 - 5 * 16) / 6, 16
    for i, (key, tag, phase, event) in enumerate(EVENTS):
        col, row = i % 6, i // 6
        x, y = 56 + col * (w + gap), 176 + row * 330
        im = Image.fromarray(images[key][CROP[1]:CROP[3], CROP[0]:CROP[2]]).resize((480, 480), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        pic = c.shapes.add_picture(io.BytesIO(buf.getvalue()), _e(x), _e(y), _e(w), _e(w))
        pic.name = f"{tag} 영상"
        colour = PHASES[phase][1]
        c.pill(x + 8, y + 24, tag, "#111827", "white", size=22)
        c.rect(x, y + w, w, 7, fill=colour, name=f"{tag} 단계")
        c.text(x, y + w + 38, event[0], 23, True, colour, width=w - 4)
        c.text(x, y + w + 70, event[1], 20, False, TEXT, width=w - 4)
    # phases, left to right
    x = 56
    for label, colour in PHASES.values():
        c.rect(x, 862, 30, 16, fill=colour, r=3, name=f"범례 {label}")
        c.text(x + 40, 878, label, 21, True, TEXT)
        x += 40 + text_width(label, 21, 700) + 36
    c.text(1552, 878, "사건: 언론 보도 종합 (2016–2026)", 19, False, MUTED, anchor="end")
    prs.save(PPTX)
    print("wrote", PPTX)
    if "--no-png" not in sys.argv:
        export_png(PPTX, ["site_history_summary"])


def main():
    images = year_images()
    montage(images)
    slide(images)


if __name__ == "__main__":
    main()
