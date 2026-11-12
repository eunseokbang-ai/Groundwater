"""2nd interim deck (editable PowerPoint): Iksan U-shaped quarry wall, joint analysis from the LiDAR alone.
Native text boxes, tables and shapes (Malgun Gothic, large type as the 1st deck), the figures as pictures,
speaker notes on every slide. Numbers come from round2/summary.json; run make_lidar_round2_figures.py first.

    C:\\dokdo\\.venv\\Scripts\\python.exe docs\\figures\\make_round2_pptx.py      (any Python with python-pptx)

Writes docs/라이다_2차발표자료.pptx.
"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "round2"
REPO = HERE.parents[1]
OUT = HERE.parent / "라이다_2차발표자료.pptx"
PLAN = FIG / "sections_plan.png"          # (from the section drafts, scratch ubase/plot_deck.py)
FONT = "Malgun Gothic"
TEXT, MUTED, LINE = "1F2937", "6B7280", "E5E7EB"
ACCENT = "7C3AED"                                   # LiDAR (the 1st deck's sensor colour)
SEC = {"E": "1F77B4", "S": "D62728", "W": "2CA02C"}
SET_HEX = ["1F77B4", "D62728", "2CA02C", "FF7F0E", "9467BD", "8C564B"]     # report.SET_COLORS
W_IN, H_IN = 13.333, 7.5


def rgb(h):
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, data):
        self.d = data
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W_IN), Inches(H_IN)
        self.blank = self.prs.slide_layouts[6]
        self.slides = []

    # --- primitives -------------------------------------------------------------------------------------------
    def text(self, s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
        """runs: [paragraph, ...]; a paragraph is (text, size, bold, colour) or a list of such runs."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if name:
            tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.04)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        for i, para in enumerate(runs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            for t, size, bold, col in (para if isinstance(para, list) else [para]):
                r = p.add_run()
                r.text = t
                r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
                r.font.color.rgb = rgb(col)
            p.space_after = Pt(6)
        return tb

    def bullets(self, s, x, y, w, h, items, size=18, colour=ACCENT, gap=8):
        """items: str or (str, level); level 1 is indented and smaller."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            t, lvl = (it, 0) if isinstance(it, str) else it
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r0 = p.add_run()
            r0.text = ("• " if lvl == 0 else "– ")
            r0.font.name, r0.font.size, r0.font.bold = FONT, Pt(size - 2 * lvl), True
            r0.font.color.rgb = rgb(colour if lvl == 0 else MUTED)
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size = FONT, Pt(size - 2 * lvl)
            r.font.color.rgb = rgb(TEXT if lvl == 0 else MUTED)
            p.level = lvl
            p.space_after = Pt(gap)
        return tb

    def rect(self, s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
        r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid()
        r.fill.fore_color.rgb = rgb(fill)
        if line:
            r.line.color.rgb = rgb(line)
        else:
            r.line.fill.background()
        r.shadow.inherit = False
        return r

    def picture(self, s, path, x, y, w, h, align="center"):
        """The image fitted inside the box, aspect kept."""
        with Image.open(path) as im:
            iw, ih = im.size
        k = min(w / iw, h / ih)
        pw, ph = iw * k, ih * k
        px = x + (w - pw) / 2 if align == "center" else x
        py = y + (h - ph) / 2
        return s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))

    def table(self, s, x, y, w, rows, widths, size=14, header_fill=ACCENT, row_h=0.42, first_bold=False):
        n, m = len(rows), len(rows[0])
        gt = s.shapes.add_table(n, m, Inches(x), Inches(y), Inches(w), Inches(row_h * n)).table
        tot = sum(widths)
        for j, wd in enumerate(widths):
            gt.columns[j].width = Emu(int(Inches(w) * wd / tot))
        for i, row in enumerate(rows):
            gt.rows[i].height = Inches(row_h)
            for j, val in enumerate(row):
                cell = gt.cell(i, j)
                colour = None
                if isinstance(val, tuple):
                    val, colour = val
                cell.text = str(val)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.name, r.font.size = FONT, Pt(size)
                    r.font.bold = i == 0 or (first_bold and j == 0)
                    r.font.color.rgb = rgb("FFFFFF" if i == 0 else (colour or TEXT))
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(header_fill if i == 0 else ("F9FAFB" if i % 2 else "FFFFFF"))
        return gt

    def stat(self, s, x, y, w, big, label, colour=ACCENT, sub=None):
        self.rect(s, x, y, w, 1.55, "FFFFFF", line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.08
        self.rect(s, x, y, 0.09, 1.55, colour)
        self.text(s, x + 0.25, y + 0.12, w - 0.35, 0.75, [(big, 30, True, colour)])
        self.text(s, x + 0.25, y + 0.85, w - 0.35, 0.65, [(label, 14, False, TEXT)] + ([(sub, 12, False, MUTED)] if sub else []))

    # --- slide frame ------------------------------------------------------------------------------------------
    def slide(self, title, subtitle=None, colour=ACCENT, notes=""):
        s = self.prs.slides.add_slide(self.blank)
        self.rect(s, 0, 0, 0.12, H_IN, colour)
        self.text(s, 0.5, 0.32, 12.3, 0.7, [(title, 28, True, TEXT)], name="제목")
        if subtitle:
            self.text(s, 0.5, 0.98, 12.3, 0.45, [(subtitle, 16, False, MUTED)])
        ln = s.shapes.add_connector(1, Inches(0.5), Inches(1.45), Inches(W_IN - 0.5), Inches(1.45))
        ln.line.color.rgb = rgb(LINE)
        ln.line.width = Pt(1.5)
        self.slides.append(s)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def numbers(self):
        n = len(self.slides)
        for i, s in enumerate(self.slides[1:], start=2):
            self.text(s, W_IN - 1.6, H_IN - 0.45, 1.2, 0.3, [(f"{i} / {n}", 11, False, MUTED)], align=PP_ALIGN.RIGHT)
            self.text(s, 0.5, H_IN - 0.45, 8, 0.3, [("익산 폐석산 절벽 · 드론 모듈 (WBS 2-1) · 2차 발표", 11, False, MUTED)])

    def save(self):
        self.numbers()
        self.prs.save(OUT)
        print(f"wrote {OUT} ({len(self.slides)} slides)")


def set_rows(sec):
    rows = [["절리군", "경사/경사방향", "면 수", "면적 (m²)", "비율", "K", "간격 (m)"]]
    for x in sec["sets"]:
        rows.append([(f"Set {x['id']}", SET_HEX[x["id"] % len(SET_HEX)]), f"{x['dip']}° / {x['dd']:03d}°", f"{x['faces']:,}",
                     f"{x['area']:,}", f"{x['share']}%", x["k"], "-" if x["spacing"] is None else f"{x['spacing']:.1f}"])
    return rows


# --- slides ---------------------------------------------------------------------------------------------------
def s_title(D):
    s = D.prs.slides.add_slide(D.blank)
    D.slides.append(s)
    D.rect(s, 0, 0, W_IN, H_IN, "111827")
    D.rect(s, 0, 0, 0.18, H_IN, ACCENT)
    D.text(s, 0.9, 2.0, 11.5, 1.2, [("익산 폐석산 U자 절벽 — 라이다 기반 절리 분석", 38, True, "FFFFFF")])
    D.text(s, 0.9, 3.25, 11.5, 0.7, [("2차 발표 · 드론 모듈 (WBS 2-1)", 24, False, "C4B5FD")])
    D.text(s, 0.9, 4.6, 11.5, 1.4, [("자료: DJI Zenmuse L2 라이다, 2026-10-24 U자 전체 비행 (17.3억 점)", 16, False, "D1D5DB"),
                                   ("높이: 해발고도 (KNGeoid18) · 좌표: EPSG:5186", 16, False, "D1D5DB"),
                                   ("2026년 10월", 16, False, "9CA3AF")])
    s.notes_slide.notes_text_frame.text = (
        "1차 발표에서 워크플로와 반구대 사전 검증을 보였고, 이번에는 익산 현장 자료로 실제 분석한 결과입니다.\n"
        "열화상은 1~2주 뒤 추운 날 새벽에 다시 찍을 예정이고, 동측·남측은 사진측량 자료가 아직 없어서 이번 분석은 라이다만으로 했습니다.\n"
        "서측은 사진측량 결과가 있어 라이다만 결과와 비교해 신뢰도를 확인했습니다.")


def s_progress(D):
    s = D.slide("1차 발표 이후", "현장 취득 → 프로그램 개선 → 이번 발표 범위",
                notes="왼쪽은 10월에 새로 얻은 자료, 가운데는 그 자료를 쓰면서 고친 프로그램 기능, 오른쪽은 이번 발표 범위입니다.\n"
                      "라이다로 낙석방지망 뒤 암반을 복원하고, U자 벽을 구간으로 나눠 경계를 겹쳐 처리하며, "
                      "거친 면이 하나로 묶이지 않게 평면성 한계를 넣었습니다.")
    cols = [("현장 취득 (10월)", "1D4ED8", ["10/21 서측 근접 RGB 촬영 (사진측량)", "10/21 열화상 H20T (저녁, 그늘 지역)",
                                         "10/23 라이다 L2 서측", "10/24 라이다 L2 U자 전체 (17.3억 점, 62 GB)"]),
            ("프로그램 개선", ACCENT, ["낙석방지망·식생 뒤 암반: 라이다 뒤쪽 층", "높이 기준 해발고도로 통일 (KNGeoid18)",
                                   "U자 구간 나누기 + 경계 20 m 겹침", "평면성 한계 (한 면 = 한 평면)", "후보 절리군 표시 · 직접 지정"]),
            ("이번 발표", "C2410C", ["라이다만으로 E·S·W 세 구간 절리 분석", "서측: 사진측량 결과와 비교 검증",
                                  "열화상: 1~2주 뒤 추운 날 새벽 재촬영 후 결합"])]
    for i, (head, col, items) in enumerate(cols):
        x = 0.5 + i * 4.15
        D.rect(s, x, 1.75, 3.95, 4.9, "FFFFFF", line=LINE)
        D.rect(s, x, 1.75, 3.95, 0.62, col)
        D.text(s, x + 0.2, 1.8, 3.6, 0.55, [(head, 20, True, "FFFFFF")], anchor=MSO_ANCHOR.MIDDLE)
        D.bullets(s, x + 0.2, 2.55, 3.6, 4.0, items, size=17, colour=col)


def s_data(D):
    s = D.slide("자료 현황 — 이번 분석은 라이다만으로", "동측·남측은 사진측량 자료가 없고, 열화상은 다시 찍을 예정",
                notes="사진측량은 서측만 있어서 이번에는 서측 비교 검증에만 썼습니다.\n"
                      "열화상은 10/21 저녁 촬영이었는데 하루 종일 그늘인 벽이라 대비가 약했고, 추운 날 새벽에 다시 찍습니다.\n"
                      "라이다는 10/24 U자 전체 비행 자료 하나로 세 구간을 같은 조건에서 분석했습니다.")
    rows = [["자료", "범위", "규모", "이번 분석에서"],
            ["라이다 L2 (10/24)", "U자 전체", "17.3억 점 · 62 GB", ("주 자료 (E·S·W 모두)", ACCENT)],
            ["라이다 L2 (10/23)", "서측", "9.8억 점 · 35 GB", "사진측량 결과의 망 뒤 보강"],
            ["RGB 사진측량 (10/21)", "서측", "2.3억 점", "서측 비교 검증"],
            ["열화상 H20T (10/21)", "서측", "1,248장", ("제외 — 재촬영 예정", "C2410C")]]
    D.table(s, 0.6, 1.8, 12.1, rows, [3.2, 2.0, 2.8, 4.1], size=17, row_h=0.62)
    D.bullets(s, 0.6, 5.2, 12.1, 1.7, ["세 구간을 같은 센서·같은 날 자료로 분석해 구간끼리 바로 비교할 수 있음",
                                       "구간마다 그 범위의 점만 읽음 (구간 범위 점의 1/8, 망·식생 셀은 전체 점)"], size=17)


def s_quality(D):
    s = D.slide("라이다 자료 품질 — 절리 분석에 충분한가", "서측 시험 구간에서 사진측량과 비교",
                notes="평탄한 0.5 m 상자 안의 잡음은 라이다 12.8 mm, 사진측량 10.5 mm로 비슷하고, 분석 크기(5~7 cm)보다 훨씬 작습니다.\n"
                      "밀도는 라이다가 세 배 이상 높고, 맨 암반에서는 1/4로 줄여도 같은 절리군이 나왔습니다.\n"
                      "망 구간에서는 셀의 76%에서 망보다 15 cm 이상 뒤의 반사가 있어서 망 뒤 암반을 복원할 수 있었습니다.")
    stats = [("12.8 mm", "평면 위 잡음 (라이다)", "사진측량 10.5 mm"), ("35,000", "벽면 점 밀도 (점/m²)", "사진측량 10,000"),
             ("76 %", "망 셀 중 망 뒤 반사가 있는 곳", "15 cm 이상 뒤"), ("< 1 %", "5회 반사 한계에 걸린 점", "L3 16회는 큰 의미 없음")]
    for i, (big, lab, sub) in enumerate(stats):
        D.stat(s, 0.5 + i * 3.1, 1.8, 2.95, big, lab, sub=sub)
    rows = [["맨 암반 점 밀도 (점/m²)", "35,000 (전체)", "8,700 (1/4)", "2,200 (1/16)", "540 (1/64)"],
            ["찾은 절리군", "3개", ("3개 (같음)", "15803D"), ("1개 빠짐", "C2410C"), ("방향 틀어짐", "B91C1C")]]
    D.table(s, 0.5, 3.75, 12.3, rows, [3.2, 2.3, 2.3, 2.3, 2.3], size=16, row_h=0.6)
    D.bullets(s, 0.5, 5.25, 12.3, 1.6, ["맨 암반은 약 5,000~10,000 점/m²면 충분 — 망 뒤는 펄스가 많을수록 암반이 더 보임",
                                       "그래서 분석은 구간 점의 1/8, 망·식생 셀은 모든 점을 사용"], size=17)


def s_rear(D):
    s = D.slide("낙석방지망·식생 뒤 암반 복원", "다중 반사 라이다의 가장 뒤쪽 층만 남김",
                notes="파란 점은 망이나 잎에서 처음 돌아온 반사, 빨간 점은 마지막 반사입니다. 검은 선이 프로그램이 암반으로 남기는 뒤쪽 층입니다.\n"
                      "5 cm 셀마다 가장 뒤쪽 점에서 6 cm 안의 점만 남기고, 숲 바닥 같은 완만한 뒷면과 나무 아래 수평면은 뺍니다.\n"
                      "망이 암반에 바로 붙은 곳은 둘을 구분할 수 없다는 한계가 있습니다.")
    D.picture(s, FIG / "rear_layer.png", 0.4, 1.6, 7.6, 5.3)
    D.bullets(s, 8.2, 1.8, 4.7, 5.0, ["5 cm 셀마다 가장 뒤쪽 점에서 6 cm 안의 점 = 암반",
                                      "완경사(60° 미만) 뒷면은 숲 바닥 → 제외",
                                      "나무 아래 거의 수평인 면은 지면 → 제외",
                                      "망이 암반에 붙은 곳은 둘을 구분 못 함 (현장 확인 필요)"], size=17)


def s_sections(D):
    s = D.slide("U자 절벽 구간 나누기", "벽면이 바라보는 방향으로 E · S · W, 경계는 20 m씩 겹쳐 처리",
                notes="U자 벽 전체를 하나의 벽면 좌표로 펼치면 남측에서 방향이 크게 돌아 왜곡되므로, 바라보는 방향이 비슷한 구간으로 나눴습니다.\n"
                      "경계에서 딱 자르면 경계에 걸친 절리면이 잘립니다. 서측에서 21개 면, 전체 면적의 4.4%가 잘렸습니다.\n"
                      "그래서 경계 밖 20 m까지 함께 분석하고, 중심이 자기 구간에 있는 면만 그 구간 결과로 셉니다.")
    D.picture(s, PLAN, 0.3, 1.55, 6.6, 5.5)
    rows = [["구간", "발치 길이", "바라보는 방위"], [("E 동측 벽", SEC["E"]), "298 m", "226~280°"],
            [("S 남측 끝", SEC["S"]), "73 m", "283°→북→46°"], [("W 서측 벽", SEC["W"]), "290 m", "23~74°"]]
    D.table(s, 7.2, 1.8, 5.7, rows, [1.9, 1.5, 2.3], size=17, row_h=0.6)
    D.bullets(s, 7.2, 4.5, 5.7, 2.5, ["경계에서 딱 자르면 걸친 면이 잘림 (서측 21면, 면적 4.4%)",
                                      "경계 밖 20 m까지 분석, 중심이 구간 안인 면만 셈", "지도에서 겹침 구간은 회색"], size=17)


def s_height(D):
    s = D.slide("높이 기준 — 해발고도로 통일", "사진측량(타원체고)과 라이다(해발고도)의 약 24.5 m 차이를 맞춤",
                notes="Metashape 결과는 타원체고, DJI 라이다는 해발고도라 그대로 겹치면 약 24.5 m 어긋납니다.\n"
                      "국토지리정보원 KNGeoid18로 모든 자료를 해발고도로 통일했고, Metashape에서도 쓸 수 있도록 지오이드 파일과 좌표계 정의를 만들었습니다.\n"
                      "익산에서 타원체고 100 m가 해발 75.474 m로 바뀌는 것을 Metashape에서 확인했습니다.")
    D.stat(s, 0.6, 1.9, 3.8, "24.53 m", "익산의 지오이드고 N", sub="KNGeoid18 (국토지리정보원)")
    D.stat(s, 4.75, 1.9, 3.8, "100 → 75.47 m", "Metashape 변환 확인", sub="WGS84 타원체고 → 해발")
    D.stat(s, 8.9, 1.9, 3.8, "0 m", "자료 사이 높이 차이", sub="사진측량 · 라이다 · 열화상")
    D.bullets(s, 0.6, 3.9, 12.1, 3.0, ["분석·지도·보고서의 표고는 모두 해발고도 (지하수위·시추 자료와 바로 비교)",
                                       "라이다 LAS는 파일에 높이 기준이 들어 있어 자동 인식, 사진측량은 지오이드로 변환",
                                       "Metashape용 지오이드 파일(KNGeoid18.tif)과 좌표계(.prj) 작성 — 다음 처리부터 해발고도로 바로 출력"],
              size=18)


def s_improve(D):
    s = D.slide("분석 개선 — 한 면은 한 평면", "평면성 한계와 후보 절리군",
                notes="밀도 높은 라이다를 그대로 쓰면, 계단처럼 이어진 벽면 전체가 하나의 '면'으로 묶여 983 m²짜리 면이 생겼습니다.\n"
                      "평면에서 RMS 10 cm 넘게 벗어나는 면은 단차 앞뒤로 나눠, 이제 가장 큰 면이 109 m²이고 모든 면이 실제 평면입니다.\n"
                      "또 절리군 기준에 조금 못 미친 방향은 버리지 않고 후보로 보여 줘서, 현장 확인 후 직접 절리군으로 넣을 수 있습니다.")
    D.picture(s, FIG / "planarity.png", 0.3, 1.55, 8.6, 4.1)
    D.bullets(s, 9.1, 1.75, 4.0, 4.0, ["평면에서 RMS 10 cm 넘게 벗어나면 단차 앞·뒤로 나눔",
                                       "983 m² 덩어리 → 178개 평면", "망 위 매끈한 면 1,150 → 2,989 m² (서측)"], size=16)
    D.rect(s, 0.5, 5.85, 12.3, 1.05, "F5F3FF", line="DDD6FE")
    D.text(s, 0.7, 5.92, 12.0, 0.95, [[("후보 절리군: ", 16, True, ACCENT),
                                       ("기준(밀도 3배·20° 안 최고점)에 조금 못 미친 방향도 스테레오넷에 ◇로 표시하고, "
                                        "현장 확인 후 '절리군으로 추가'하면 안정성 검토까지 반영", 16, False, TEXT)]])


def s_next(D):
    s = D.slide("한계와 다음 단계", colour="C2410C",
                notes="라이다만 분석은 같은 날 같은 센서로 U자 전체를 볼 수 있는 장점이 있지만, 사진만큼 선명한 트레이스는 얻기 어렵습니다.\n"
                      "열화상을 추운 날 새벽에 다시 찍어 유출 후보와 절리를 결합하고, 동측·남측 사진측량이 들어오면 같은 방식으로 비교 검증하겠습니다.")
    D.text(s, 0.6, 1.7, 6.0, 0.5, [("한계", 20, True, "B91C1C")])
    D.bullets(s, 0.6, 2.25, 6.0, 4.6, ["망이 암반에 붙은 곳은 망과 암반 구분 불가",
                                       "라이다 색은 사진보다 흐려 트레이스(선형) 검출이 약함",
                                       "후보 절리군은 현장 확인 전까지 판정 보류",
                                       "운동학 검토는 구간별 대표 벽면 기준의 1차 선별"], size=17, colour="B91C1C")
    D.text(s, 6.9, 1.7, 6.0, 0.5, [("다음 단계", 20, True, "15803D")])
    D.bullets(s, 6.9, 2.25, 6.0, 4.6, ["열화상 재촬영: 1~2주 뒤 추운 날 해 뜨기 전",
                                       "열화상 이상 × 절리 매칭 → 유출 후보 목록",
                                       "동측·남측 사진측량 → 라이다 결과와 비교",
                                       "후보 절리군 · 수평 절리 현장 확인",
                                       "그라우팅 설계 입력 (2-4) 자료 정리"], size=17, colour="15803D")


def s_validate(D):
    a, b, cmp_ = D.d["W_photo"], D.d["W"], D.d["W_compare"]
    s = D.slide("검증 — 서측: 라이다만 vs 사진측량+라이다", "같은 벽을 두 가지 자료로 분석해 절리군 방향 비교",
                notes="서측은 사진측량 자료가 있어서 라이다만으로 한 결과와 비교했습니다.\n"
                      f"급경사 두 절리군은 1° 안팎으로 같고, 완경사 군은 약 {cmp_[2]['angle']:.0f}° 차이인데 경사가 낮아 경사방향이 잘 정해지지 않는 군입니다.\n"
                      "절리면 면적도 비슷해서 동측·남측의 라이다만 결과도 같은 신뢰도로 볼 수 있습니다. 다만 트레이스는 라이다 색이 흐려 훨씬 적게 잡힙니다.")
    D.picture(s, FIG / "stereo_W_photo.png", 0.3, 1.55, 4.3, 3.4)
    D.picture(s, FIG / "stereo_W.png", 4.6, 1.55, 4.3, 3.4)
    D.text(s, 0.3, 4.9, 4.3, 0.4, [("사진측량 + 라이다 보강", 15, True, TEXT)], align=PP_ALIGN.CENTER)
    D.text(s, 4.6, 4.9, 4.3, 0.4, [("라이다만", 15, True, ACCENT)], align=PP_ALIGN.CENTER)
    rows = [["사진측량", "라이다만", "차이"]] + [[c["a"], c["b"], (f"{c['angle']:.1f}°", "15803D" if c["angle"] < 5 else "C2410C")]
                                               for c in cmp_]
    D.table(s, 9.2, 1.7, 3.8, rows, [1.3, 1.3, 1.0], size=16, row_h=0.55)
    rows2 = [["", "사진측량", "라이다만"], ["절리면 면적 (m²)", f"{a['area_m2']:,}", f"{b['area_m2']:,}"],
             ["절리면 수", f"{a['faces']:,}", f"{b['faces']:,}"], ["트레이스", f"{a['traces']:,}", f"{b['traces']:,}"]]
    D.table(s, 9.2, 4.1, 3.8, rows2, [1.6, 1.1, 1.1], size=14, row_h=0.48, first_bold=True)
    D.bullets(s, 0.5, 5.5, 8.6, 1.5, ["급경사 두 절리군 방향 1° 안팎 일치, 절리면 면적 96%",
                                      "트레이스는 라이다 색이 흐려 약 1/6 — 선형 분석은 사진측량·열화상과 함께"], size=17)


def _section(D, key, title, subtitle, notes, tall=False):
    sec = D.d[key]
    s = D.slide(title, subtitle, colour=SEC[key], notes=notes)
    if tall:                                            # S: a tall facade, beside the stereonet and table
        D.picture(s, FIG / f"facade_{key}.png", 0.3, 1.55, 5.6, 5.45)
        D.picture(s, FIG / f"stereo_{key}.png", 6.0, 1.55, 4.0, 3.1)
        D.table(s, 6.1, 4.85, 6.9, set_rows(sec), [1.0, 1.7, 0.9, 1.1, 0.8, 0.6, 0.9], size=13, row_h=0.42)
        return s
    D.picture(s, FIG / f"facade_{key}.png", 0.3, 1.5, 12.7, 3.55)
    D.picture(s, FIG / f"stereo_{key}.png", 0.3, 5.0, 3.6, 2.05)
    D.table(s, 4.1, 5.15, 8.9, set_rows(sec), [1.0, 1.7, 0.9, 1.1, 0.8, 0.6, 0.9], size=13, row_h=0.4)
    return s


def s_E(D):
    e = D.d["E"]
    c = e["candidates"][0] if e["candidates"] else None
    _section(D, "E", "동측 벽 (E) — 라이다만", f"벽면 대부분이 {e['sets'][0]['dip']}°/{e['sets'][0]['dd']:03d}° 절리면 · 담당 측점 "
             f"{e['core_chainage'][0]:.0f}~{e['core_chainage'][1]:.0f} m",
             notes=f"동측 벽은 절리면 면적의 {e['sets'][0]['share']}%가 {e['sets'][0]['dip']}°/{e['sets'][0]['dd']:03d}° 한 절리군입니다. "
                   "벽 자체가 이 절리면을 따라 굴착된 것으로 보입니다.\n"
                   "위쪽 절반은 낙석방지망이 덮고 있고, 그 뒤 암반은 라이다 뒤쪽 층으로 복원했습니다.\n"
                   + (f"벽을 가로지르는 거의 수직인 {c['dip']}°/{c['dd']:03d}° 방향은 면으로는 기준(3배)에 조금 못 미친 후보({c['contrast']:.1f}배)이고, "
                      "입면에서는 트레이스 방향군으로 뚜렷하게 잡힙니다." if c else ""))


def s_S(D):
    sv = D.d["S"]
    _section(D, "S", "남측 끝 (S) — 라이다만", "기둥·오목부 포함, 벽 방향이 123° 돌아가는 구간",
             notes="남측 끝은 벽이 크게 돌아가는 곳이라 두 방향의 절리면이 번갈아 벽면을 이룹니다.\n"
                   f"{sv['sets'][0]['dip']}°/{sv['sets'][0]['dd']:03d}°는 동측 벽의 주 절리군과 같은 군이고, "
                   f"{sv['sets'][1]['dip']}°/{sv['sets'][1]['dd']:03d}°는 동측·서측에서는 후보로만 보이던 동서 주향의 수직 절리군입니다.\n"
                   "망이 넓게 덮여 있어 라이다 뒤쪽 층 보강 점이 가장 많은 구간입니다.", tall=True)


def s_W(D):
    w = D.d["W"]
    _section(D, "W", "서측 벽 (W) — 라이다만", f"담당 측점 {w['core_chainage'][0]:.0f}~{w['core_chainage'][1]:.0f} m · "
             "사진측량 결과와 같은 세 절리군",
             notes="서측 벽은 벽면과 나란한 87°/035° 군과 이를 비스듬히 가르는 87°/261° 군이 거의 반씩, 그리고 완경사 군이 있습니다.\n"
                   "사진측량 결과와 같은 세 군이고, 위쪽 숲 아래 지면은 수평면 제외로 걸러졌습니다.")


def s_synthesis(D):
    s = D.slide("U자 전체 종합 — 네 개의 절리군", "구간마다 다르게 보이던 절리군을 한 체계로 묶음",
                notes="세 구간의 절리군을 방향으로 묶으면 현장 전체에 네 개의 절리군이 있습니다.\n"
                      "서측 벽은 J1을 따라, 동측 벽은 J2를 따라 굴착되었고, 남측 끝에서는 J3와 J2가 번갈아 벽면을 이룹니다.\n"
                      "벽과 나란한 절리군은 면으로 넓게 드러나고, 벽을 가로지르는 절리군은 면보다 트레이스로 보입니다. "
                      "그래서 구간마다 면으로 잡히는 절리군이 다릅니다.")
    E, S_, W = D.d["E"], D.d["S"], D.d["W"]

    def find(sec, dip_lo, dip_hi, dds):
        for x in sec["sets"]:
            if dip_lo <= x["dip"] <= dip_hi and any(abs((x["dd"] - d + 180) % 360 - 180) <= 20 or
                                                     (x["dip"] >= 80 and abs((x["dd"] - d) % 360 - 180) <= 20) for d in dds):
                return f"{x['dip']}°/{x['dd']:03d}° ({x['share']}%)", TEXT
        for c in sec["candidates"]:
            if dip_lo <= c["dip"] <= dip_hi and any(abs((c["dd"] - d + 180) % 360 - 180) <= 20 or
                                                     (c["dip"] >= 80 and abs((c["dd"] - d) % 360 - 180) <= 20) for d in dds):
                return f"후보 {c['dip']}°/{c['dd']:03d}°", MUTED
        return "—", MUTED

    groups = [("J1  NW–SE 주향, 북동 경사 (급경사)", 75, 90, [35]), ("J2  N–S 주향, 서 경사 (급경사)", 75, 90, [262]),
              ("J3  E–W 주향, 수직", 80, 90, [171]), ("J4  완경사 (6~17°)", 0, 25, [318, 348, 35])]
    rows = [["절리군", "W 서측 벽", "S 남측 끝", "E 동측 벽"]]
    for name, lo, hi, dds in groups:
        rows.append([name] + [find(sec, lo, hi, dds) for sec in (W, S_, E)])
    rows[3][3] = (rows[3][3][0] + " · 트레이스", rows[3][3][1])
    D.table(s, 0.5, 1.75, 12.3, rows, [4.2, 2.7, 2.7, 2.7], size=16, row_h=0.62, first_bold=True)
    D.bullets(s, 0.5, 5.1, 12.3, 1.9, ["벽은 절리면을 따라 굴착됨: 서측 = J1, 동측 = J2, 남측 끝 = J3 + J2",
                                       "벽과 나란한 군은 넓은 면으로, 벽을 가로지르는 군은 트레이스·후보로 드러남",
                                       "J1·J2·J3 모두 거의 수직 → 수직 절리망이 지하수 통로의 주 후보, J4는 수평 연결"], size=17)


def s_stability(D):
    s = D.slide("안정성 1차 검토 (운동학)", "5 m 구간마다 벽면 방향과 절리군으로 파괴 형태 가능 여부 · 마찰각 30°",
                notes="그래프는 5 m 구간 중 각 파괴 형태가 운동학적으로 가능한 구간의 비율입니다.\n"
                      "주 절리군이 거의 수직이라 평면파괴는 드물고, 전도·쐐기가 일부 구간에서 가능합니다.\n"
                      "서측 사진측량 결과에서 벽면 밖으로 47° 기운 후보 방향(47°/046°)이 나왔습니다. 실제 절리로 확인되면 서측의 평면파괴 검토가 달라지므로 현장 확인이 필요합니다.")
    D.picture(s, FIG / "kinematics.png", 0.4, 1.6, 7.6, 4.3)
    rows = [["구간", "평면", "쐐기", "전도", "구간 수"]]
    for k, name in (("W", "W 서측 벽"), ("S", "S 남측 끝"), ("E", "E 동측 벽")):
        kin, n = D.d[k]["kinematics"], D.d[k]["windows"]
        rows.append([(name, SEC[k]), f"{kin['평면파괴']}%", f"{kin['쐐기파괴']}%", f"{kin['전도파괴']}%", n])
    D.table(s, 8.3, 1.8, 4.7, rows, [1.6, 0.8, 0.8, 0.8, 0.9], size=15, row_h=0.52, first_bold=True)
    D.bullets(s, 8.3, 4.1, 4.7, 2.9, ["주 절리군이 거의 수직 → 평면파괴는 드묾",
                                      "전도·쐐기는 일부 구간에서 가능",
                                      "서측 사진측량에서 나온 후보 47°/046°가 확인되면 평면파괴 재검토"], size=16)


RESULT_SLIDES = [s_validate, s_E, s_S, s_W, s_synthesis, s_stability]


def build(D):
    for f in (s_title, s_progress, s_data, s_quality, s_rear, s_sections, s_height, s_improve, *RESULT_SLIDES, s_next):
        f(D)


def main():
    data = json.loads((FIG / "summary.json").read_text(encoding="utf-8")) if (FIG / "summary.json").exists() else {}
    D = Deck(data)
    build(D)
    D.save()


if __name__ == "__main__":
    main()
