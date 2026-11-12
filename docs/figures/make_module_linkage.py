"""Technical-module linkage slides (16:9) from the project plan's WBS and schedule (사업계획서 9. 연구추진계획),
centred on the drone module (WBS 2-1 드론 불연속면 분석): which links are its own work, which are joint work and
which are data it leads in handing to other modules. Type sizes as in the plan's linkage chart (legible from afar).

  1  기술모듈 간 연계도        -- the eight flows, each tagged with the drone module's role (드론 제공 / 공동 / 타 모듈)
  2  드론 모듈 업무 범위       -- solo work, joint work and the data it hands over (to whom, what, when)
  3  추진 일정과 연계 시점      -- the schedule bars with the hand-offs, the drone module's in blue

Built from native PowerPoint shapes and text boxes (every box, arrow and label editable) into
docs/기술모듈_연계도.pptx; with PowerPoint installed the slides are also exported as module_linkage_<n>.png.

    .venv\\Scripts\\python.exe docs\\figures\\make_module_linkage.py

Needs python-pptx (pip install python-pptx). Coordinates are slide pixels on a 1600 x 900 canvas.
"""
import subprocess
import sys
from pathlib import Path

from svgkit import text_width

OUT_DIR = Path(__file__).resolve().parent
PPTX = OUT_DIR.parent / "기술모듈_연계도.pptx"

EMU_PX = 12192000 / 1600                     # EMU per slide pixel (13.333 in wide)
PT_PX = 0.6                                  # points per slide pixel
FONT = "Malgun Gothic"

TEXT, MUTED, LINE = "#111827", "#6b7280", "#e5e7eb"
ME = "#1d4ed8"                               # the drone module's work / the data it hands over
ME_FILL = "#eff6ff"
OTHER = "#6b7280"                            # links between other modules
TEAM = {"T1": ("#1d4ed8", "#eff6ff"), "T2": ("#c2410c", "#fff7ed"), "T3": ("#0f766e", "#f0fdfa")}
SLATE = ("#475569", "#f8fafc")
BUILD = ("#6b7280", "#f3f4f6")
POST = ("#7e22ce", "#faf5ff")
CIRCLED = "①②③④⑤⑥⑦⑧"


def _rgb(h):
    from pptx.dml.color import RGBColor

    return RGBColor.from_string({"white": "ffffff"}.get(h, h).lstrip("#"))


def _e(v):
    from pptx.util import Emu

    return Emu(int(round(v * EMU_PX)))


class Canvas:
    """One slide, drawn in slide pixels with native shapes."""

    def __init__(self, prs, warn_tag):
        self.slide = prs.slides.add_slide(prs.slide_layouts[6])
        self.shapes = self.slide.shapes
        self.tag = warn_tag

    # -- text -------------------------------------------------------------------
    def _font(self, run, size, bold, color):
        from pptx.oxml.ns import qn
        from pptx.util import Pt

        f = run.font
        f.size, f.bold, f.name = Pt(size * PT_PX), bold, FONT
        f.color.rgb = _rgb(color)
        rpr = run._r.get_or_add_rPr()
        for tag in ("a:ea", "a:cs"):                 # Korean glyphs take the east-Asian font
            el = rpr.find(qn(tag))
            if el is None:
                el = rpr.makeelement(qn(tag), {})
                rpr.append(el)
            el.set("typeface", FONT)

    def text(self, x, y, lines, size, bold=False, color=TEXT, anchor="start", width=None, gap=1.3, name=None):
        """Text whose first baseline is at y (as in SVG); `lines` a string or a list of lines (one paragraph each,
        `gap` x size apart); a run list [(s, bold, color), ...] makes one mixed line. Warns when wider than `width`."""
        from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
        from pptx.util import Pt

        lines = [lines] if isinstance(lines, (str, tuple)) else lines
        runs = [[ln] if isinstance(ln, tuple) else [(ln, bold, color)] if isinstance(ln, str) else ln for ln in lines]
        widths = [sum(text_width(s, size, 700 if b else 400) for s, b, _ in r) for r in runs]
        w = max(widths)
        if width and w > width + 1:
            print(f"  [{self.tag}] text {w:.0f} px > {width:.0f}: {''.join(s for s, _, _ in runs[widths.index(w)])}")
        bw = (width or w) + size * 0.4
        left = x if anchor == "start" else x - bw / 2 if anchor == "middle" else x - bw
        if anchor == "start":
            left -= size * 0.05
        L = size * gap
        top = y - size * 1.08 - (L - size * 1.3) * 0.5
        box = self.shapes.add_textbox(_e(left), _e(top), _e(bw), _e(L * len(runs) + size * 0.4))
        if name:
            box.name = name
        tf = box.text_frame
        tf.word_wrap, tf.auto_size = False, MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.TOP
        for i, r in enumerate(runs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"start": PP_ALIGN.LEFT, "middle": PP_ALIGN.CENTER, "end": PP_ALIGN.RIGHT}[anchor]
            p.line_spacing = Pt(L * PT_PX)
            for s, b, c in r:
                run = p.add_run()
                run.text = s
                self._font(run, size, b, c)
        return box

    # -- shapes -----------------------------------------------------------------
    def _line_style(self, shape, color, width, dash):
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        from pptx.util import Pt

        if color is None:
            shape.line.fill.background()
            return
        shape.line.color.rgb = _rgb(color)
        shape.line.width = Pt(width * PT_PX)
        if dash:
            shape.line.dash_style = {"dash": MSO_LINE_DASH_STYLE.DASH, "dot": MSO_LINE_DASH_STYLE.ROUND_DOT,
                                     "long": MSO_LINE_DASH_STYLE.LONG_DASH}[dash]

    def rect(self, x, y, w, h, fill=None, line=None, lw=2.0, r=0, dash=None, name=None, text=None, size=20,
             bold=True, color="white", align="center"):
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

        shp = self.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE,
                                    _e(x), _e(y), _e(w), _e(h))
        if r:
            shp.adjustments[0] = min(0.5, r / min(w, h))
        if fill:
            shp.fill.solid()
            shp.fill.fore_color.rgb = _rgb(fill)
        else:
            shp.fill.background()
        self._line_style(shp, line, lw, dash)
        shp.shadow.inherit = False
        if name:
            shp.name = name
        tf = shp.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        if text is not None:
            tf.word_wrap = False
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if align == "center" else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = text
            self._font(run, size, bold, color)
        return shp

    def pill(self, x, cy, s, fill, fg, size=20, stroke=None, anchor="start"):
        """Rounded label centred vertically on cy; returns its width."""
        w = text_width(s, size, 700) + size * 1.1
        h = size * 1.45
        left = x if anchor == "start" else x - w
        self.rect(left, cy - h / 2, w, h, fill=fill, line=stroke, lw=1.6, r=h / 2, text=s, size=size, color=fg)
        return w

    def badge(self, cx, cy, n, fill, fg="white", stroke="white", r=18, size=22):
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

        shp = self.shapes.add_shape(MSO_SHAPE.OVAL, _e(cx - r), _e(cy - r), _e(2 * r), _e(2 * r))
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
        self._line_style(shp, stroke, 2.5, None)
        shp.shadow.inherit = False
        shp.name = f"번호 {n}"
        tf = shp.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(CIRCLED.index(n) + 1) if n in CIRCLED else n
        self._font(run, size, True, fg)
        return shp

    def path(self, pts, color, width=3.0, dash=None, head="end", name=None):
        """Open polyline (one editable shape) with arrowheads at 'end', 'start', 'both' or None."""
        from pptx.oxml.ns import qn

        b = self.shapes.build_freeform(_e(pts[0][0]), _e(pts[0][1]), scale=1.0)
        b.add_line_segments([(_e(x), _e(y)) for x, y in pts[1:]], close=False)
        shp = b.convert_to_shape()
        shp.fill.background()
        self._line_style(shp, color, width, dash)
        shp.shadow.inherit = False
        if name:
            shp.name = name
        ln = shp.line._get_or_add_ln()
        for tag, on in (("a:headEnd", head in ("start", "both")), ("a:tailEnd", head in ("end", "both"))):
            if on:
                ln.append(ln.makeelement(qn(tag), {"type": "triangle", "w": "med", "len": "med"}))
        return shp


def header(c, title, subtitle, teams=True):
    c.rect(0, 0, 1600, 900, fill="#ffffff", name="배경")
    c.text(56, 82, title, 46, True)
    c.text(56, 128, subtitle, 26, False, MUTED, width=1496 if not teams else 1496)
    c.path([(56, 152), (1552, 152)], LINE, 2, head=None)
    if teams:
        x = 1552
        for team, name in [("T3", "T3 모니터링"), ("T2", "T2 차단 기술"), ("T1", "T1 원격탐사")]:
            x -= c.pill(x, 62, name, TEAM[team][0], "white", size=20, anchor="end") + 10


def module(c, x, y, w, h, title, colors, period=None, team=None, sub=None, title_size=26, solid=False, lw=2.5,
           dash=None):
    """A module box as in the plan's linkage chart: wrapped title, one short line, period and team pills."""
    from svgkit import wrap

    color, fill = colors
    c.rect(x, y, w, h, fill=color if solid else fill, line=color, lw=lw, r=12, dash=dash, name=title)
    ink = "#ffffff" if solid else (TEXT if color == BUILD[0] else color)
    rows = wrap(title, w - 28, title_size, 700)
    first = y + 10 + title_size * 1.3 - 4
    c.text(x + 14, first, rows, title_size, True, ink, width=w - 28, gap=1.3)
    if sub:
        c.text(x + 14, first + title_size * 1.3 * (len(rows) - 1) + 34, sub, 22, False, "#ffffff" if solid else TEXT,
               width=w - 28)
    px = x + 14
    if period:
        px += c.pill(px, y + h - 24, period, "white", color, size=20, stroke=None if solid else color) + 8
    if team:
        c.pill(px, y + h - 24, team, "#1e3a8a" if solid else color, "white", size=20)


# role of each link for the drone module: provided by it (thick blue), joint (dashed blue, two-way), other (grey)
GIVE = dict(color=ME, width=5)
JOINT = dict(color=ME, width=3.5, dash="dash")
OTH = dict(color="#374151", width=3.5)


def role_badge(c, cx, cy, n, role, r=18, size=22):
    if role == "give":
        c.badge(cx, cy, n, ME, r=r, size=size)
    elif role == "joint":
        c.badge(cx, cy, n, "#ffffff", fg=ME, stroke=ME, r=r, size=size)
    else:
        c.badge(cx, cy, n, "#111827", r=r, size=size)


# --- slide 1: the linkage network, by the drone module's role in each link --------------------------------
def slide_network(prs):
    c = Canvas(prs, "1")
    header(c, "기술모듈 간 연계도", "조사·진단 → 타겟 그라우팅 설계 → 시공 → 효과 검증 · 드론 모듈(2-1) 업무 구분")

    # 1. survey (the drone 3D is the drone module's part of it)
    c.rect(56, 250, 200, 350, fill=SLATE[1], line=SLATE[0], lw=2.5, r=12, name="1. 현황조사")
    c.text(70, 290, "1. 현황조사", 26, True, SLATE[0])
    c.pill(70, 320, "8–9월", "white", SLATE[0], size=20, stroke=SLATE[0])
    c.text(70, 382, "기존 자료 분석", 22, True, TEXT)
    c.text(70, 414, "예비 ERT·시추공", 22, False, MUTED)
    c.text(70, 470, "주변현황 조사", 22, True, TEXT)
    c.text(70, 502, "답사·드론 3D", 22, True, ME)

    # investigation group
    c.rect(312, 196, 430, 458, line="#94a3b8", lw=2.5, r=16, dash="dash", name="조사·진단 모듈")
    c.text(330, 186, "조사·진단 모듈", 22, True, MUTED)
    module(c, 328, 212, 330, 118, "2-1 드론 불연속면 분석", TEAM["T1"], "9–12월 · 4–7월", "T1", "절리군 · 물길 후보",
           solid=True, lw=4)
    module(c, 328, 366, 330, 118, "2-2 시계열 ERT", TEAM["T3"], "9월 · 12–1월 · 5–9월", "T3", "심부 플룸 가시화")
    module(c, 328, 520, 330, 118, "2-3 시추·시추공 탐사", TEAM["T3"], "12–4월", "T3", "투수성 파쇄대 확인")

    c.path([(493, 334), (493, 362)], head="both", name="③ 공동", **JOINT)
    role_badge(c, 533, 348, "③", "joint")
    c.path([(493, 488), (493, 516)], head="both", name="④ 타 모듈", **OTH)
    role_badge(c, 533, 502, "④", "other")
    c.path([(664, 286), (704, 286), (704, 579), (664, 579)], head="both", name="② 드론 제공", **GIVE)
    role_badge(c, 704, 432, "②", "give")

    module(c, 798, 300, 220, 250, "2-4 타겟 그라우팅 설계", TEAM["T2"], "11–5월", "T2", "최적 공법·현장실증")
    module(c, 1074, 360, 200, 130, "차수시스템 시공", BUILD, "3–5월", dash="dash")
    module(c, 1330, 250, 220, 130, "3-1 저감 효과 검증", TEAM["T3"], "3–9월", "T3", title_size=24)
    module(c, 1330, 470, 220, 130, "3-2 사후관리 방안", POST, "6–9월", title_size=24)

    # ①, ⑥: plain hand-overs; ⑤: the drone module's (blue, from 2-1) and T3's (grey) inputs to the design
    c.path([(260, 425), (306, 425)], head="end", name="① 공동", **JOINT)
    role_badge(c, 281, 385, "①", "joint")
    c.path([(662, 236), (880, 236), (880, 294)], head="end", name="⑤ 드론 제공", **GIVE)
    role_badge(c, 792, 236, "⑤", "give")
    c.path([(746, 470), (792, 470)], head="end", name="⑤ T3", **OTH)
    c.path([(1022, 425), (1068, 425)], head="end", name="⑥", **OTH)
    role_badge(c, 1046, 385, "⑥", "other")
    c.path([(1278, 425), (1300, 425), (1300, 315), (1324, 315)], head="end", name="시공 → 검증", **OTH)
    c.path([(1440, 384), (1440, 464)], head="end", name="⑧", **OTH)
    role_badge(c, 1480, 425, "⑧", "other")
    # ⑦: the drone re-survey after construction (4D ERT is T3's)
    c.path([(560, 208), (560, 172), (1440, 172), (1440, 244)], head="end", name="⑦ 드론 단독", **GIVE)
    role_badge(c, 985, 172, "⑦", "give")

    # the flows, each tagged with the drone module's role
    c.path([(56, 672), (1552, 672)], LINE, 2, head=None)
    tags = {"give": ("드론 제공", ME, "white", None), "solo": ("드론 단독", ME, "white", None),
            "joint": ("공동", "white", ME, ME), "other": ("타 모듈", "white", "#4b5563", "#9ca3af")}
    flows = [("①", "joint", "기존 자료·현장 여건 → 조사 계획"), ("②", "give", "후보 절리 → 시추 위치 · 시추공 절리 → 드론 검증"),
             ("③", "joint", "절리대 구조 ↔ ERT 플룸 분포 대조"), ("④", "other", "ERT 이상대 → 시추 위치 · 검층 → ERT 검증"),
             ("⑤", "give", "물길 후보·플룸 경로·파쇄대 → 주입공 위치·심도"), ("⑥", "other", "그라우팅 설계 → 차수시스템 시공"),
             ("⑦", "solo", "시공 전후 드론 재촬영 비교 (4D ERT는 T3)"), ("⑧", "other", "저감 효과 → 사후관리")]
    for i, (n, role, s) in enumerate(flows):
        col, row = divmod(i, 4)
        x, y = 56 + col * 760, 704 + row * 40
        role_badge(c, x + 16, y, n, "give" if role == "solo" else role, r=15, size=19)
        c.text(x + 44, y + 8, s, 22, False, TEXT, width=560)
        name, fill, fg, stroke = tags[role]
        c.pill(x + 44 + text_width(s, 22) + 14, y, name, fill, fg, size=17, stroke=stroke)
    # key: the drone module's box, then the three kinds of link
    c.rect(56, 860, 44, 24, fill=ME, r=5, name="범례 드론 모듈")
    c.text(112, 880, "드론 모듈 업무", 20, True, ME)
    x = 112 + text_width("드론 모듈 업무", 20, 700) + 48
    for (style, head), label in (((GIVE, "end"), "드론 모듈이 주도해 넘기는 자료"), ((JOINT, "both"), "공동 작업"),
                                 ((OTH, "end"), "타 모듈 간 연계")):
        c.path([(x, 872), (x + 44, 872)], head=head, **style)
        c.text(x + 56, 880, label, 20, True, ME if style is not OTH else "#374151")
        x += 56 + text_width(label, 20, 700) + 48
    return c


# --- slide 2: the drone module's scope -----------------------------------------------------------------
def slide_scope(prs):
    c = Canvas(prs, "2")
    header(c, "드론 모듈 업무 범위 (WBS 2-1)", "핵심은 드론 자료 기반 절리면 분석 · 다른 모듈과는 공동 해석과 자료 제공으로 연결")

    def card(x, w, title, solid):
        c.rect(x, 180, w, 610, fill=ME_FILL if solid else "#ffffff", line=ME, lw=3, r=14, name=title)
        c.rect(x, 180, w, 66, fill=ME if solid else "#dbeafe", r=14, name=title + " 머리")
        c.rect(x, 216, w, 30, fill=ME if solid else "#dbeafe", name=title + " 머리 아래")
        c.text(x + 24, 226, title, 28, True, "#ffffff" if solid else ME, width=w - 40)

    card(56, 470, "드론 모듈 단독 업무", True)
    for i, s in enumerate(["근접 입면 촬영 (RGB·LiDAR)", "점군·정사영상·DSM 처리", "절리군 · 주요 절리면 3D",
                           "유출 가능성 지도", "파괴 양상 (평면·쐐기·전도)", "표적 천공 배치안", "시공 후 재촬영 비교"]):
        y = 300 + i * 66
        c.badge(92, y - 8, str(i + 1), ME, r=16, size=19)
        c.text(122, y, s, 24, True, TEXT, width=390)

    card(565, 470, "공동 작업", False)
    for i, (who, s) in enumerate([("T1 황세호", "열화상 활성균열 판정"), ("T1 방민규", "드론 물리탐사 대조"),
                                   ("T3 ERT", "절리대 ↔ 플룸 대조 ③"), ("T3 시추", "시추공 절리로 검증 ②"),
                                   ("현황조사", "드론 3D·과거 영상 ①")]):
        y = 296 + i * 96
        c.pill(589, y - 8, who, "#ffffff", ME, size=19, stroke=ME)
        c.text(589, y + 40, s, 24, True, TEXT, width=420)

    card(1074, 478, "드론 모듈 주도 자료 제공", True)
    for i, (to, when, s) in enumerate([("→ 2-4 그라우팅", "11월~", "물길 후보·절리면·천공안"),
                                        ("→ 2-3 시추", "~12월", "후보 절리 → 시추 위치"),
                                        ("→ 3-1 효과 검증", "6–7월", "재촬영 전후 비교"),
                                        ("→ 3-2 사후관리", "필요 시", "파괴 양상 지도")]):
        y = 300 + i * 120
        c.text(1098, y, to, 26, True, ME, width=300)
        c.pill(1528, y - 9, when, "#ffffff", ME, size=19, stroke=ME, anchor="end")
        c.text(1098, y + 44, s, 24, False, TEXT, width=430)

    c.rect(56, 806, 1496, 64, fill="#fffbeb", line="#fcd34d", lw=1.5, r=10, name="합의 필요")
    c.text(80, 848, [[("합의 필요  ", True, "#b45309"),
                      ("열화상 '활성 물길' 판정 주체 (황세호) · 자료 전달 형식·좌표계 (T2·T3)", False, TEXT)]], 22, width=1450)
    return c


# --- slide 3: the schedule with the hand-offs, the drone module's in blue ------------------------------------
X0, X1, WEEKS = 400, 1552, 54
MONTHS = ["8월", "9월", "10월", "11월", "12월", "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월"]
ROW_TOP, ROW_H, BAR_H = 216, 68, 36


def wk(month, week):
    """Week index from the plan's first column (Aug wk3). month: 0 = Sep 2026 ... 12 = Sep 2027."""
    return 2 + month * 4 + (week - 1)


def xw(i):
    return X0 + (X1 - X0) * i / WEEKS


def row_mid(r):
    return ROW_TOP + ROW_H * r + ROW_H / 2


def slide_timeline(prs):
    c = Canvas(prs, "3")
    header(c, "추진 일정과 모듈 간 연계 시점", "사업계획서 추진일정 기준 · 파란 화살표 = 드론 모듈이 넘기는 자료")
    rows = ["1. 현황조사", "2-1 드론 불연속면", "2-3 시추·시추공 탐사", "2-2 시계열 ERT", "2-4 타겟 그라우팅 설계",
            "차수시스템 시공", "3-1 저감 효과 검증", "3-2 사후관리 방안", "보고회"]
    for r in range(len(rows)):
        if r == 1:
            c.rect(56, ROW_TOP + ROW_H * r, X1 - 56, ROW_H, fill="#dbeafe", name="드론 모듈 행")
        elif r % 2 == 0:
            c.rect(56, ROW_TOP + ROW_H * r, X1 - 56, ROW_H, fill="#f9fafb", name=f"행 {r}")
    c.text(xw(0), 176, "2026", 20, True, MUTED)
    c.text(xw(wk(4, 1)), 176, "2027", 20, True, MUTED)
    starts = [0] + [wk(m, 1) for m in range(13)]
    for i, name in enumerate(MONTHS):
        s, e = starts[i], starts[i + 1] if i + 1 < len(starts) else WEEKS
        c.text((xw(s) + xw(e)) / 2, 204, name, 22, True, TEXT, anchor="middle")
        c.path([(xw(s), ROW_TOP - 4), (xw(s), ROW_TOP + ROW_H * 9)], LINE, 1.5, head=None)
    for r, name in enumerate(rows):
        c.text(64, row_mid(r) + 9, name, 24, True, ME if r == 1 else TEXT, width=330)

    def bar(r, start, end, colors, caption=None, hatch=False):
        color, fill = colors
        y = row_mid(r) - BAR_H / 2
        c.rect(xw(start), y, xw(end) - xw(start), BAR_H, fill=fill if hatch else color, line=color if hatch else None,
               lw=2, r=8, dash="dash" if hatch else None, text=caption, size=22,
               color=TEXT if hatch else "#ffffff", name=caption or rows[r])

    bar(0, 0, wk(0, 4) + 1, SLATE)
    bar(1, wk(0, 2), wk(3, 4) + 1, TEAM["T1"], "1차: 설계 입력")
    bar(1, wk(7, 1), wk(10, 4) + 1, TEAM["T1"], "2차: 시공 전후")
    bar(2, wk(3, 1), wk(7, 4) + 1, TEAM["T3"], "시험시추·검층")
    bar(3, wk(0, 2), wk(0, 4) + 1, TEAM["T3"])
    bar(3, wk(3, 1), wk(4, 4) + 1, TEAM["T3"], "시공 전 기준")
    bar(3, wk(8, 1), wk(12, 4) + 1, TEAM["T3"], "4D 모니터링")
    bar(4, wk(2, 1), wk(8, 4) + 1, TEAM["T2"], "설계 → 현장실증")
    bar(5, wk(6, 1), wk(8, 4) + 1, BUILD, "시공", hatch=True)
    bar(6, wk(6, 1), wk(12, 4) + 1, TEAM["T3"], "유입량·ERT·재촬영 비교")
    bar(7, wk(9, 1), wk(12, 4) + 1, POST, "사후관리 방안")
    for start, end, name in [(0, wk(0, 1) + 1, "착수"), (wk(5, 1), wk(5, 4) + 1, "중간"), (wk(12, 1), wk(12, 4) + 1, "최종")]:
        bar(8, start, end, ("#111827", "#111827"), name)

    def handoff(idx, r_from, r_to, caption, side="right", mine=False, label_row=None):
        x = xw(idx)
        up = r_to < r_from
        y0 = row_mid(r_from) + (-1 if up else 1) * (BAR_H / 2 + 2)
        y1 = row_mid(r_to) + (1 if up else -1) * (BAR_H / 2 + 4)
        colour = ME if mine else "#111827"
        c.path([(x, y0), (x, y1)], colour, 5 if mine else 3.5, name=caption)
        w = text_width(caption, 22, 700) + 16
        ly = (y0 + y1) / 2 if abs(r_to - r_from) == 1 else row_mid(label_row if label_row is not None else r_from + 1)
        lx = x + 8 if side == "right" else x - 8 - w
        c.rect(lx, ly - 16, w, 32, fill=ME_FILL if mine else "#ffffff", line=colour, lw=2.5 if mine else 1.5, r=6,
               text=caption, size=22, color=colour)

    handoff(wk(2, 1), 1, 4, "후보 절리", side="left", mine=True, label_row=2)
    handoff(wk(3, 1) - 0.4, 1, 2, "시추 위치", mine=True)
    handoff(wk(4, 4) + 0.5, 3, 4, "플룸 경로")
    handoff(wk(7, 3) + 0.5, 2, 4, "투수성 파쇄대", side="left")
    handoff(wk(9, 3), 1, 6, "드론 재촬영", mine=True, label_row=2)
    handoff(wk(11, 3), 3, 6, "4D ERT", side="left")
    handoff(wk(6, 1) + 0.5, 4, 5, "주입공 배치")
    handoff(wk(9, 1) + 0.5, 6, 7, "저감 효과")
    c.text(56, 878, "행 순서는 연계 화살표가 겹치지 않도록 2-3을 2-2 위에 배치", 22, False, MUTED)
    return c


def export_png(pptx, names):
    """Each slide as <name>.png (3200 x 1800) through PowerPoint, when it is installed."""
    ps = f"""
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open('{pptx}', $true, $false, $false)
$i = 1
foreach ($n in @({", ".join(f"'{n}'" for n in names)})) {{
  $p.Slides.Item($i).Export((Join-Path '{OUT_DIR}' ($n + '.png')), 'PNG', 3200, 1800); $i++ }}
$p.Close()
if ($app.Presentations.Count -eq 0) {{ $app.Quit() }}
"""
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True, timeout=180)
        for n in names:
            print("wrote", OUT_DIR / f"{n}.png")
    except (OSError, subprocess.SubprocessError) as e:
        print("PNG export skipped (needs PowerPoint):", e)


def main():
    from pptx import Presentation

    prs = Presentation()
    prs.slide_width, prs.slide_height = 12192000, 6858000
    slide_network(prs)
    slide_scope(prs)
    slide_timeline(prs)
    prs.save(PPTX)
    print("wrote", PPTX)
    if "--no-png" not in sys.argv:
        export_png(PPTX, ["module_linkage_1", "module_linkage_scope", "module_linkage_2"])


if __name__ == "__main__":
    main()
