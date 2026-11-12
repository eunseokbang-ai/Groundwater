"""One-slide progress summary (16:9) for the October interim report: what the drone module (WBS 2-1) did this
period. Written as an editable PowerPoint (docs/이번기간_수행내용.pptx: native text boxes and shapes, the thumbnails as
pictures) and, with PowerPoint installed, exported as progress_summary.png.

    .venv\\Scripts\\python.exe docs\\figures\\make_progress_summary.py

Thumbnails come from the other figures in this folder, two field images in data/ (drone photo and 3D model
screenshot of the 2026-09-07 survey) and the Bangudae rehearsal figures (make_bangudae_figures.py); numbers match
the plan / watershed figures. Needs python-pptx.
"""
import io
from pathlib import Path

from PIL import Image

from svgkit import MUTED, TEXT, wrap

OUT_DIR = Path(__file__).resolve().parent
DATA = OUT_DIR / "data"
NAME = "progress_summary"
TW, TH = 488, 150          # thumbnail size in slide units (rendered at 2x)

BLUE, TEAL, ORANGE, PURPLE = "#1d4ed8", "#0f766e", "#c2410c", "#7e22ce"


def cover(im, focus=(0.5, 0.5), box=None):
    """Crop `im` (optionally to a fractional box first) to the thumbnail aspect around `focus`; PNG bytes."""
    if box:
        w, h = im.size
        im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    w, h = im.size
    target = TW / TH
    if w / h > target:
        nw = int(h * target)
        x0 = min(max(int(focus[0] * w - nw / 2), 0), w - nw)
        im = im.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(w / target)
        y0 = min(max(int(focus[1] * h - nh / 2), 0), h - nh)
        im = im.crop((0, y0, w, y0 + nh))
    im = im.convert("RGB").resize((TW * 2, TH * 2), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


HISTORY = [("1978", 0, 0), ("1992", 2, 0), ("2011", 0, 1), ("2025", 2, 2)]   # (label, column, row) in the montage


def history_strip():
    """Montage cells side by side, cut to 120 m-wide-ish centre strips below the montage's own year tags."""
    m = Image.open(OUT_DIR / "site_history_montage.jpg")
    cw, gap = 300, 6
    strip = Image.new("RGB", (4 * cw + 3 * gap, 350), "white")
    for i, (_, c, r) in enumerate(HISTORY):
        x0, y0 = c * 405 + 70, r * 405 + 50
        strip.paste(m.crop((x0, y0, x0 + cw, y0 + 350)), (i * (cw + gap), 0))
    return cover(strip, focus=(0.5, 0.5))


def analysis_pair():
    """The real-data rehearsal (Bangudae): the joint-set stereonet and the seepage map of the wall, side by side."""
    bang = DATA / "bangudae"
    net = Image.open(bang / "stereonet.png").convert("RGB")
    net = net.resize((int(net.width * 292 / net.height), 292), Image.LANCZOS)
    sp = Image.open(bang / "seepage.png").convert("RGB")
    sp = sp.crop((int(0.06 * sp.width), 0, sp.width, int(0.86 * sp.height)))
    w = TW * 2 - net.width - 24
    sp = sp.resize((w, int(w * sp.height / sp.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (TW * 2, TH * 2), "white")
    canvas.paste(net, (8, 4))
    canvas.paste(sp, (net.width + 16, max(0, (TH * 2 - sp.height) // 2)))
    return cover(canvas)


def thumbnails():
    plan = Image.open(OUT_DIR / "leachate_plan_ortho.png")          # map panel ~ (40,168)-(1030,870) of 1600x900
    return [
        cover(Image.open(DATA / "summary_drone_photo.jpg"), focus=(0.5, 0.45)),
        cover(Image.open(DATA / "summary_3d_model.jpg"), focus=(0.45, 0.5), box=(0, 0, 0.96, 1)),
        history_strip(),
        cover(Image.open(OUT_DIR / "leachate_surface_watershed.png"), focus=(0.45, 0.45),
              box=(40 / 1600, 168 / 900, 1030 / 1600, 870 / 900)),
        cover(plan, box=(400 / 1600, 318 / 900, 980 / 1600, 496 / 900)),
        analysis_pair(),
    ]


CARDS = [
    ("1차 현장조사 (9.7)", BLUE, ["현장 답사 · 드론 동영상", "준광역 사진측량 · 라이다 (고도 150 m)"]),
    ("3차원 모델 · 정사영상", BLUE, ["매립지·암벽 3D 모델, DSM", "드론 정사영상 (2026.09.07)"]),
    ("과거 영상 이력 뷰어", TEAL, ["1973~2025 20개 시기 + 2026 드론", "굴착 1980 전후 → 매립 → 복구"]),
    ("지표수 해석 (5 m · 0.5 m)", TEAL, ["매립지 유역 4.9 ha (원지형 7.0 ha)", "유출: 서북 → 큰 강 → 금강 15.5 km"]),
    ("현장 평면도 · 차수시설 판독", ORANGE, ["상류 차수벽 202 m (드론 판독)", "하류 100 m (DTM 추정), 우회 경로 확인"]),
    ("불연속면 분석 프로그램", PURPLE, ["절리군 · 물길 지도 · 파괴 양상 · 천공안", "반구대 실자료 전 과정 검증 (6.5억 점)"]),
]


SUBTITLE = "드론 모듈 (WBS 2-1 UAV 기반 불연속면 규명) — 1차 조사 · 자료 구축 · 분석 도구"
NEXT = "암벽 근접비행(사진측량·라이다) + 열화상 동시 취득 → 절리 분석 → 물길 후보·천공안 제공 (2-4 · 2-3)"
PPTX = OUT_DIR.parent / "이번기간_수행내용.pptx"


def picture(c, x, y, w, h, png, radius=14, name=None):
    """A thumbnail with its top corners rounded like the card (a picture: crop or replace it in PowerPoint)."""
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.oxml.ns import qn

    from make_module_linkage import _e

    pic = c.shapes.add_picture(io.BytesIO(png), _e(x), _e(y), _e(w), _e(h))
    pic.auto_shape_type = MSO_SHAPE.ROUND_2_SAME_RECTANGLE
    geom = pic._element.spPr.find(qn("a:prstGeom"))
    av = geom.find(qn("a:avLst"))
    if av is None:
        av = geom.makeelement(qn("a:avLst"), {})
        geom.append(av)
    for g in list(av):
        av.remove(g)
    av.append(av.makeelement(qn("a:gd"), {"name": "adj1", "fmla": f"val {int(radius / h * 100000)}"}))
    av.append(av.makeelement(qn("a:gd"), {"name": "adj2", "fmla": "val 0"}))
    if name:
        pic.name = name
    return pic


def slide(prs):
    from make_module_linkage import Canvas

    c = Canvas(prs, "progress")
    c.rect(0, 0, 1600, 900, fill="#ffffff", name="배경")
    c.text(56, 82, "이번 기간 수행 내용", 46, True)
    c.text(56, 128, SUBTITLE, 26, False, MUTED, width=1260)
    c.pill(1552 - 20, 72, "2026. 9 ~ 10월", "#111827", "white", size=26, anchor="end")
    c.path([(56, 152), (1552, 152)], "#e5e7eb", 2, head=None)

    for i, ((title, color, lines), png) in enumerate(zip(CARDS, thumbnails())):
        col, row = i % 3, i // 3
        x, y = 56 + col * 504, 172 + row * 318
        c.rect(x, y, TW, 300, fill="#ffffff", line="#e5e7eb", lw=2, r=14, name=f"카드 {i + 1}")
        picture(c, x + 1, y + 1, TW - 2, TH - 1, png, name=f"사진 {i + 1}")
        c.rect(x, y + TH, TW, 5, fill=color, name=f"카드 {i + 1} 선")
        c.badge(x + 30, y + 30, str(i + 1), color, r=20, size=24)
        if i == 2:
            for k, (year, _, _) in enumerate(HISTORY):
                c.pill(x + (k + 1) * TW / 4 - 8, y + TH - 22, year, "#111827", "white", size=18, anchor="end")
        c.pill(x + TW - 20, y + TH + 36, "완료", "#dcfce7", "#15803d", size=20, anchor="end")
        size = 28 if len(title) <= 16 else 25
        c.text(x + 22, y + TH + 46, wrap(title, TW - 120, size, 700)[0], size, True, color, width=TW - 120)
        for k, s_ in enumerate(lines):
            cy = y + TH + 88 + k * 34
            c.rect(x + 26, cy - 12, 8, 8, fill=color, r=4, name="점")
            c.text(x + 44, cy, s_, 23, False, TEXT, width=TW - 60)

    y = 816
    c.rect(56, y, 1496, 64, fill="#fff7ed", line="#fdba74", lw=2, r=12, name="다음 단계")
    c.pill(76, y + 32, "다음 단계", ORANGE, "white", size=24)
    c.text(232, y + 42, NEXT, 25, True, "#9a3412", width=1300)
    return c


def main():
    import sys

    from pptx import Presentation

    from make_module_linkage import export_png

    prs = Presentation()
    prs.slide_width, prs.slide_height = 12192000, 6858000
    slide(prs)
    prs.save(PPTX)
    print("wrote", PPTX)
    if "--no-png" not in sys.argv:
        export_png(PPTX, [NAME])


if __name__ == "__main__":
    main()
