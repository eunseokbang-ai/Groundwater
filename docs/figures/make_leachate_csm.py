"""Conceptual cross-section of groundwater / leachate flow at the Iksan reclaimed
quarry, for presentations. Writes an SVG (editable in PowerPoint via
"Convert to Shape") and, if Microsoft Edge is installed, a 2x PNG.

    python docs/figures/make_leachate_csm.py

Depth figures (landfill up to 23 m, plume to ~35 m) follow the project plan
and the preliminary ERT; everything else is schematic, not to scale.
Vertical exaggeration: 7 px per metre below the landfill surface.
"""
import random
from pathlib import Path

from svgkit import H, W, arrow, curve, document, fmt, label, lines, path, save, text, text_width

OUT_DIR = Path(__file__).resolve().parent
NAME = "leachate_conceptual_model"

C = {
    "text": "#1f2937", "muted": "#6b7280",
    "soil": "#dcc294", "wrock": "#c4a577", "rock": "#a7a197", "joint": "#4b433c",
    "waste": "#6e6259", "hdpe": "#1d1d20", "cutoff": "#9aa5b1",
    "gw": "#1d6fc2", "leach": "#d9480f", "plume": "#f97316",
    "tree": "#5b8c3e", "tree2": "#79a95a", "trunk": "#6b4f3a",
}

LANDFILL_TOP = 440          # y of the landfill surface (HDPE cover)
PX_PER_M = 7
LANDFILL_BASE = LANDFILL_TOP + 23 * PX_PER_M   # 601
PLUME_BASE = LANDFILL_TOP + 35 * PX_PER_M      # 685
SECTION_BOTTOM = 798

# --- geometry (x right, y down) ----------------------------------------------
HILL = [(0, 196), (60, 180), (110, 172), (170, 175), (230, 182), (280, 190), (320, 200)]
WALL_FACE = [(320, 200), (326, 236), (322, 268), (330, 302), (327, 336), (334, 370), (331, 404), (338, 440)]
WALL_BURIED = [(338, 440), (343, 500), (350, 560), (360, 600)]
PIT_FLOOR = [(360, 600), (420, 603), (500, 601), (580, 603), (660, 599), (740, 596), (820, 591),
             (900, 584), (960, 576), (1020, 568), (1080, 559), (1140, 552)]
VALLEY = [(1140, 552), (1220, 553), (1300, 554), (1400, 553), (1460, 549), (1530, 545), (1600, 542)]

HILL_SOIL_BASE = [(0, 210), (60, 194), (110, 186), (170, 189), (230, 196), (280, 204), (322, 214)]
HILL_WROCK_BASE = [(0, 266), (60, 250), (110, 242), (170, 246), (230, 254), (280, 264), (324, 278)]
LOW_SOIL_BASE = [(360, 610), (420, 616), (500, 612), (580, 620), (660, 611), (740, 610), (820, 604), (900, 598),
                 (960, 592), (1020, 586), (1080, 580), (1140, 576), (1220, 592), (1300, 598), (1400, 600),
                 (1460, 600), (1530, 600), (1600, 600)]
LOW_WROCK_BASE = [(360, 655), (420, 660), (500, 655), (580, 662), (660, 652), (740, 650), (820, 646), (900, 640),
                  (960, 634), (1020, 628), (1080, 622), (1140, 618), (1220, 650), (1300, 665), (1400, 675),
                  (1460, 678), (1530, 680), (1600, 680)]

CUTOFF_X, CUTOFF_BOTTOM = 996, 662            # upstream wall at the landfill toe (L = 202 m)
CUTOFF2_X, CUTOFF2_BOTTOM = 1222, 646         # downstream wall added later (L = 100 m)
WELL_X = 1166                                 # leachate pumping well between the walls
PADDY_X0 = 1252
HILL_JOINTS = [((40, 245), (330, 350)), ((20, 300), (332, 410)), ((50, 380), (342, 488)), ((30, 440), (349, 548))]
PIT_FRACTURES = [((470, 601), (545, 775)), ((650, 599), (735, 780)), ((840, 589), (905, 760))]
FAINT_JOINTS = [((130, 245), (146, 470)), ((205, 256), (222, 560)), ((268, 268), (289, 430)), ((0, 520), (160, 560)),
                ((90, 600), (300, 660)), ((20, 690), (330, 760)), ((1180, 640), (1320, 700)),
                ((1350, 700), (1480, 780)), ((1500, 690), (1560, 790)), ((560, 700), (600, 790)),
                ((780, 700), (830, 790)), ((960, 690), (1010, 780))]


def interp_y(points, x):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# --- drawing ----------------------------------------------------------------------
def defs():
    return f"""
  <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#e3eef8"/><stop offset="1" stop-color="#f8fbfd"/>
  </linearGradient>
  <pattern id="soilDots" width="9" height="9" patternUnits="userSpaceOnUse">
    <rect width="9" height="9" fill="{C['soil']}"/><circle cx="2" cy="3" r="1.1" fill="#b89a67"/>
    <circle cx="6.5" cy="7" r="0.9" fill="#c9ab78"/></pattern>
  <pattern id="wrockDash" width="16" height="12" patternUnits="userSpaceOnUse">
    <rect width="16" height="12" fill="{C['wrock']}"/><path d="M2 4 l5 -1.5 M9 10 l5 -1.5" stroke="#a3845a" stroke-width="1.2"/></pattern>
  <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="6" height="6" fill="{C['cutoff']}"/><path d="M0 0 V6" stroke="#5b6470" stroke-width="1.4"/></pattern>
  <filter id="soft" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="5"/></filter>
  <clipPath id="landfillClip"><path d="{landfill_d()}"/></clipPath>
"""


def landfill_d():
    return (f"M{fmt((338, LANDFILL_TOP))} L{fmt((1000, LANDFILL_TOP))} L1140,548 L1140,552 "
            f"{curve(PIT_FLOOR[::-1])} {lines(WALL_BURIED[::-1][1:])} Z")


def ground():
    out = []
    rock = (f"M{fmt(HILL[0])} {curve(HILL)} {lines(WALL_FACE[1:] + WALL_BURIED[1:])} "
            f"{curve(PIT_FLOOR)} {curve(VALLEY)} L1600,{SECTION_BOTTOM} L0,{SECTION_BOTTOM} Z")
    out.append(path(rock, fill=C["rock"]))

    hill_wrock = (f"M{fmt(HILL[0])} {curve(HILL)} L326,236 L322,268 L{fmt(HILL_WROCK_BASE[-1])} "
                  f"{curve(HILL_WROCK_BASE[::-1])} Z")
    hill_soil = f"M{fmt(HILL[0])} {curve(HILL)} L{fmt(HILL_SOIL_BASE[-1])} {curve(HILL_SOIL_BASE[::-1])} Z"
    low_top = PIT_FLOOR + VALLEY[1:]
    low_wrock = (f"M{fmt(low_top[0])} {curve(PIT_FLOOR)} {curve(VALLEY)} L{fmt(LOW_WROCK_BASE[-1])} "
                 f"{curve(LOW_WROCK_BASE[::-1])} Z")
    low_soil = (f"M{fmt(low_top[0])} {curve(PIT_FLOOR)} {curve(VALLEY)} L{fmt(LOW_SOIL_BASE[-1])} "
                f"{curve(LOW_SOIL_BASE[::-1])} Z")
    out += [path(hill_wrock, fill="url(#wrockDash)"), path(low_wrock, fill="url(#wrockDash)"),
            path(hill_soil, fill="url(#soilDots)"), path(low_soil, fill="url(#soilDots)")]

    for a, b in FAINT_JOINTS:
        out.append(path(f"M{fmt(a)} L{fmt(b)}", stroke=C["joint"], stroke_width=1.2, stroke_opacity=0.45))
    for a, b in HILL_JOINTS + PIT_FRACTURES:
        out.append(path(f"M{fmt(a)} L{fmt(b)}", stroke=C["joint"], stroke_width=3.2, stroke_linecap="round"))
    return "\n".join(out)


def landfill():
    rng = random.Random(4)
    pieces = []
    palette = ["#5a4d45", "#857264", "#6a5e55", "#9a8878", "#4d433c", "#7b6d61"]
    for _ in range(520):
        x, y = rng.uniform(338, 1140), rng.uniform(LANDFILL_TOP, 606)
        w, h = rng.uniform(4, 15), rng.uniform(3, 8)
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="1.5" '
                      f'fill="{rng.choice(palette)}" transform="rotate({rng.uniform(-40, 40):.0f} {x:.0f} {y:.0f})"/>')
    return (path(landfill_d(), fill=C["waste"]) + '\n<g clip-path="url(#landfillClip)">' + "".join(pieces) + "</g>\n"
            + path(f"M338,{LANDFILL_TOP - 2} L1000,{LANDFILL_TOP - 2} L1062,487", fill="none", stroke=C["hdpe"],
                   stroke_width=6, stroke_linejoin="round")
            + path("M372,436 Q470,426 585,436 Z", fill="#a9d3f0")
            + f'<rect x="{CUTOFF_X}" y="{LANDFILL_TOP - 2}" width="12" height="{CUTOFF_BOTTOM - LANDFILL_TOP + 2}" '
              f'fill="url(#hatch)" stroke="#4b5563" stroke-width="1.2"/>')


def downstream_works():
    """Downstream cut-off wall and the pumping well between the two walls."""
    top2 = interp_y(VALLEY, CUTOFF2_X) - 2
    top_w = interp_y(PIT_FLOOR + VALLEY[1:], WELL_X)
    return "\n".join([
        f'<rect x="{CUTOFF2_X}" y="{top2:.0f}" width="12" height="{CUTOFF2_BOTTOM - top2:.0f}" '
        f'fill="url(#hatch)" stroke="#4b5563" stroke-width="1.2"/>',
        path(f"M{WELL_X},{top_w - 14:.0f} L{WELL_X},638", stroke="#374151", stroke_width=4),
        path(f"M{WELL_X},600 L{WELL_X},636", stroke="#374151", stroke_width=8, stroke_dasharray="3 3"),
        f'<rect x="{WELL_X - 9}" y="{top_w - 30:.0f}" width="18" height="16" rx="2" fill="#4b5563"/>',
        arrow(f"M{WELL_X + 14},{top_w + 30:.0f} L{WELL_X + 14},{top_w - 22:.0f}", C["leach"], 3, marker="red"),
    ])


def plume():
    tongue = ("M660,645 C850,650 1080,650 1300,612 L1362,598 L1300,632 "
              f"C1080,{PLUME_BASE} 850,{PLUME_BASE} 660,682 Z")
    lens = "M392,603 C520,655 860,650 985,572 L960,576 C800,600 560,610 392,603 Z"
    fingers = [((470, 601), (499, 650)), ((650, 599), (682, PLUME_BASE - 18)), ((840, 589), (868, 640))]
    body = [path(lens, fill=C["plume"]), path(tongue, fill=C["plume"])]
    body += [path(f"M{fmt(a)} L{fmt(b)}", stroke=C["plume"], stroke_width=36, stroke_linecap="round")
             for a, b in fingers]
    return f'<g opacity="0.5" filter="url(#soft)">{"".join(body)}</g>'


def water():
    out = []
    wt_segments = [
        [(0, 280), (80, 284), (160, 292), (240, 306), (330, 330)],
        [(345, 470), (500, 474), (700, 478), (900, 482), (1000, 486)],
        [(1008, 488), (1080, 520), (1140, 560), (1220, 572), (1320, 574), (1420, 572), (1520, 568), (1600, 566)],
    ]
    for seg in wt_segments:
        out.append(path(f"M{fmt(seg[0])} {curve(seg)}", fill="none", stroke=C["gw"], stroke_width=2.2,
                        stroke_dasharray="9 6"))
    for x, y in [(170, 293), (720, 478), (1350, 573)]:
        out.append(path(f"M{x - 8},{y - 13} L{x + 8},{y - 13} L{x},{y - 2} Z", fill=C["gw"]))

    # rain + infiltration
    cloud = [(130, 96, 22), (162, 84, 27), (200, 88, 25), (234, 97, 19), (178, 102, 22)]
    out += [f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="#cbd5e1"/>' for x, y, r in cloud]
    out += [f'<circle cx="{x}" cy="{y}" r="{r - 1.5}" fill="white"/>' for x, y, r in cloud]
    for x in range(116, 256, 16):
        out.append(path(f"M{x},{126} L{x - 8},{interp_y(HILL, x - 8) - 34:.0f}", stroke="#4f90cf", stroke_width=2.2,
                        stroke_dasharray="8 5"))
    for x in (70, 170, 250):
        top = interp_y(HILL, x) + 6
        out.append(arrow(f"M{x},{top:.0f} L{x},{top + 34:.0f}", C["gw"], 2.5))

    # groundwater along joints toward the wall
    for (a, b) in HILL_JOINTS:
        dx, dy = b[0] - a[0], b[1] - a[1]
        start = (a[0] + dx * 0.08, a[1] + dy * 0.08 - 7)
        end = (a[0] + dx * 0.93, a[1] + dy * 0.93 - 7)
        out.append(arrow(f"M{fmt(start)} L{fmt(end)}", C["gw"], 3))
    # seepage trickling down the exposed face
    for (x, y) in [(330, 350), (332, 410)]:
        out.append(path(f"M{x + 2},{y + 2} q-4,9 1,17 t1,16", fill="none", stroke=C["gw"], stroke_width=3,
                        stroke_linecap="round"))
        out.append(arrow(f"M{x + 4},{y} L{x + 22},{y + 16}", C["gw"], 2.6))
    # regional flow at depth (inferred)
    out.append(arrow(f"M40,730 C400,770 900,785 1300,760", C["gw"], 2.6, dash="10 7"))
    out.append(arrow("M1246,592 L1410,595", C["gw"], 2.6))
    return "\n".join(out)


def leachate_flows():
    d = [
        "M347,448 C400,470 480,500 560,545",
        "M346,491 C420,505 520,540 600,578",
        "M352,551 C390,560 420,572 452,590",
        "M478,612 L496,652",
        "M660,612 L680,664",
        "M846,600 L862,642",
        "M700,664 C900,674 1100,670 1330,618",
        "M1060,522 Q1110,556 1150,560",                 # shallow leachate toward the pumping well
        "M1184,600 C1200,640 1214,664 1246,658 C1280,650 1320,612 1380,568",   # under the downstream wall
    ]
    out = [arrow(s, C["leach"], 3.2, marker="red") for s in d]
    # residual seepage reaching the paddy drains (thin, dashed)
    out.append(arrow("M1236,566 L1400,566", C["leach"], 2.4, dash="7 5", marker="red"))
    return "\n".join(out)


def surface_features():
    out = []
    out.append(path(f"M{fmt(HILL[0])} {curve(HILL)}", fill="none", stroke=C["tree"], stroke_width=5))
    rng = random.Random(9)
    for x in [14, 40, 68, 96, 124, 152, 180, 208, 236, 264, 292]:
        y = interp_y(HILL, x)
        r = rng.uniform(9, 12)
        out.append(f'<rect x="{x - 1.5}" y="{y - 9:.0f}" width="3" height="10" fill="{C["trunk"]}"/>')
        out.append(f'<circle cx="{x}" cy="{y - 9 - r:.0f}" r="{r:.0f}" fill="{rng.choice([C["tree"], C["tree2"]])}"/>')

    # paddies: water film + rice
    out.append(path(f"M{PADDY_X0},552 L1410,551", stroke="#9fd0c0", stroke_width=6))
    for x in range(PADDY_X0 + 6, 1408, 11):
        out.append(path(f"M{x},{547} l-2,-9 M{x},{547} l2,-9", stroke="#4f8f3a", stroke_width=1.4))
    for x in (1300, 1366):
        out.append(path(f"M{x - 6},{552} q6,-7 12,0", fill="#b89a67"))
    out.append(path(f"M1410,{551} {curve(VALLEY[3:])}", fill="none", stroke=C["tree"], stroke_width=4))

    for x, base, roof in [(1478, 548, "#b4534b"), (1520, 546, "#8c5a3c"), (1562, 543, "#4b6c8a")]:
        out.append(f'<rect x="{x - 15}" y="{base - 22}" width="30" height="22" fill="#f3efe6" stroke="#6b7280"/>')
        out.append(path(f"M{x - 19},{base - 22} L{x},{base - 38} L{x + 19},{base - 22} Z", fill=roof))
        out.append(f'<rect x="{x - 4}" y="{base - 15}" width="8" height="8" fill="#9fc3e0"/>')
    out.append(f'<rect x="1436" y="{532}" width="10" height="16" fill="#6b7280"/>')
    out.append(path("M1441,548 L1441,650", stroke="#374151", stroke_width=3))
    out.append(path("M1433,650 L1449,650", stroke="#374151", stroke_width=3))
    return "\n".join(out)


def labels():
    t, m = C["text"], C["muted"]
    out = [
        text(40, 60, "폐석산 복구지 지하수·침출수 유동 개념 모식도", 46, 700),
        text(262, 108, "강우 → 지하수 함양", 25, 700, C["gw"]),
        label(12, 214, [("배후 산지 지하수", 22, 700, C["gw"])], pad=6),
        text(346, 236, "폐석산 암사면 (수직 절벽)", 27, 700),
        label(362, 270, [("① 암벽 절리·파쇄대를 통한 지하수 유출", 26, 700, t),
                         ("외수 유입 일 50~70 톤 (강우 시 최대 70~80 톤)", 22, 400, m)]),
        path("M366,336 L340,350", stroke=t, stroke_width=1.4),
        path("M400,348 L342,405", stroke=t, stroke_width=1.4),
        text(590, 414, "표면 차수막 (HDPE)", 23, 700),
        path("M700,420 L716,436", stroke=t, stroke_width=1.4),
        label(836, 352, [("연직 차수벽", 23, 700, t), ("상류 L=202 m · 하류 L=100 m(추가)", 20, 400, m)], pad=7),
        path("M940,418 L1000,440", stroke=t, stroke_width=1.4),
        path("M1110,418 L1226,550", stroke=t, stroke_width=1.4),
        text(640, 514, "매립 폐기물층", 32, 700, "white", anchor="middle"),
        text(640, 544, "광재류·폐주물사 등 · 불법매립 약 150만 t", 21, 400, "#f3f4f6", anchor="middle"),
        text(640, 570, "(2021~24년 약 29만 t 이적 처리)", 19, 400, "#e5e7eb", anchor="middle"),
        text(418, 472, "② 폐기물 통과 → 침출수 생성", 23, 700, "white"),
    ]
    # depth ruler (landfill surface = 0 m)
    x = 398
    out.append(path(f"M{x},{LANDFILL_TOP} L{x},{PLUME_BASE}", stroke="#111827", stroke_width=2))
    for y in (LANDFILL_TOP, LANDFILL_BASE, PLUME_BASE):
        out.append(path(f"M{x - 7},{y} L{x + 7},{y}", stroke="#111827", stroke_width=2))
    out.append(path(f"M{x + 7},{LANDFILL_BASE} L585,{LANDFILL_BASE}", stroke="#111827", stroke_width=1.2,
                    stroke_dasharray="4 4"))
    out.append(path(f"M{x + 7},{PLUME_BASE} L690,{PLUME_BASE}", stroke="#111827", stroke_width=1.2,
                    stroke_dasharray="4 4"))
    out.append(label(388, LANDFILL_BASE - 20, [("매립층 최대 23 m", 23, 700, t)], anchor="end", pad=7))
    out.append(label(388, PLUME_BASE - 22, [("침출수 플룸 심도 ~35 m", 23, 700, "#9a3412"),
                                            ("예비 ERT 저비저항대", 19, 400, m)], anchor="end", pad=7))

    out.append(label(700, 700, [("③ 절리대 따라 심부 확산·차수벽 하부 우회", 25, 700, "#9a3412"),
                                ("측방(동·서) 우회 가능성은 평면도 참조", 21, 400, m)]))
    out.append(text(40, 778, "지중 지하수 흐름 (추정)", 22, 700, C["gw"]))
    out.append(label(1158, 640, [("양수정", 20, 700, t)], anchor="end", pad=5))

    out.append(label(1180, 344, [("④ 하류 유출 → 농경지·하천", 26, 700, "#b91c1c"),
                                 ("침출수 비소: 지하수 기준의 1,600배", 21, 400, m),
                                 ("2017 범람 → 농수로·중신천, 물고기 폐사", 21, 400, m)]))
    out.append(path("M1330,452 L1330,504", stroke=t, stroke_width=1.2, stroke_dasharray="3 3"))
    out.append(text(1262, 530, "농경지 (논·밭)", 24, 700))
    out.append(text(1520, 494, "마을 (민가)", 24, 700, anchor="middle"))
    out.append(label(1590, 664, [("지하수 관정", 20, 700, t), ("1 km 내 약 100가구 음용", 19, 400, m)],
                     anchor="end", pad=6))
    out.append(text(1590, 592, "풍화토", 22, 700, "#6b5433", anchor="end"))
    out.append(text(1590, 648, "풍화암", 22, 700, "#5c4a33", anchor="end"))
    out.append(text(1590, 786, "연암", 22, 700, "#3f3a35", anchor="end"))
    return "\n".join(out)


def legend():
    x0, y0, w = 1068, 74, 514
    row = lambda i: y0 + 64 + i * 40
    h = row(4) - y0 + 30
    out = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="8" fill="white" fill-opacity="0.95" stroke="#cbd5e1"/>',
           text(x0 + 16, y0 + 34, "범례", 25, 700)]
    cx, tx, fs = x0 + 16, x0 + 66, 22
    out.append(arrow(f"M{cx},{row(0)} L{cx + 38},{row(0)}", C["gw"], 3))
    out.append(text(tx, row(0) + 7, "지하수 흐름", fs))
    out.append(arrow(f"M{cx},{row(1)} L{cx + 38},{row(1)}", C["leach"], 3, marker="red"))
    out.append(text(tx, row(1) + 7, "침출수 흐름", fs))
    out.append(path(f"M{cx},{row(2) + 4} L{cx + 40},{row(2) + 4}", stroke=C["gw"], stroke_width=2.2, stroke_dasharray="7 5"))
    out.append(path(f"M{cx + 12},{row(2) - 9} L{cx + 28},{row(2) - 9} L{cx + 20},{row(2) + 2} Z", fill=C["gw"]))
    out.append(text(tx, row(2) + 7, "지하수위", fs))
    out.append(f'<rect x="{cx}" y="{row(3) - 10}" width="40" height="20" rx="6" fill="{C["plume"]}" fill-opacity="0.5"/>')
    out.append(text(tx, row(3) + 7, "침출수 플룸 (ERT)", fs))
    out.append(path(f"M{cx},{row(4) + 7} L{cx + 40},{row(4) - 7}", stroke=C["joint"], stroke_width=3.2))
    out.append(text(tx, row(4) + 7, "절리·파쇄대", fs))

    cx2, tx2 = x0 + 270, x0 + 320
    swatches = [(C["waste"], "매립 폐기물"), ("url(#soilDots)", "풍화토"), ("url(#wrockDash)", "풍화암"), (C["rock"], "연암")]
    for i, (fill, name) in enumerate(swatches):
        out.append(f'<rect x="{cx2}" y="{row(i) - 10}" width="40" height="20" fill="{fill}" stroke="#6b7280" stroke-width="0.8"/>')
        out.append(text(tx2, row(i) + 7, name, fs))
    out.append(path(f"M{cx2},{row(4) - 5} L{cx2 + 18},{row(4) - 5}", stroke=C["hdpe"], stroke_width=5))
    out.append(f'<rect x="{cx2 + 24}" y="{row(4) - 11}" width="12" height="22" fill="url(#hatch)" stroke="#4b5563"/>')
    out.append(text(tx2, row(4) + 7, "차수막 / 차수벽", fs))
    return "\n".join(out)


def process_strip():
    steps = [("① 암벽 절리로 외수 유입", "#dbeafe", "#1e3a8a"),
             ("② 폐기물 통과·침출수 생성", "#ede3d9", "#57402d"),
             ("③ 심부 확산·차수벽 우회", "#ffedd5", "#9a3412"),
             ("④ 하류 농경지·하천 영향", "#fee2e2", "#991b1b")]
    out, x, y, w, h, gap = [], 40, 810, 350, 52, 36
    for i, (s, bg, fg) in enumerate(steps):
        size = 25
        while text_width(s, size, 700) > w - 24:
            size -= 1
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{bg}"/>')
        out.append(text(x + w / 2, round(y + h / 2 + size * 0.36, 1), s, size, 700, fg, anchor="middle"))
        if i < len(steps) - 1:
            out.append(arrow(f"M{x + w + 8},{y + h / 2} L{x + w + gap - 8},{y + h / 2}", "#374151", 3, marker="gray"))
        x += w + gap
    out.append(text(40, 890, "※ 개념 모식도(축척 없음). 매립층 두께·플룸 심도: 사업계획서·예비 ERT, 차수벽 길이: 시설 배치도, "
                             "피해·수치: 2016–2026 언론 보도. 지하 구조·흐름 경로는 개략적 표현.", 18, 400, C["muted"]))
    return "\n".join(out)


def build_svg():
    body = "\n".join([
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<rect width="{W}" height="{SECTION_BOTTOM}" fill="url(#sky)"/>',
        ground(), plume(), landfill(), downstream_works(), surface_features(), water(), leachate_flows(),
        labels(), legend(),
        path(f"M0,{SECTION_BOTTOM} L{W},{SECTION_BOTTOM}", stroke="#9ca3af", stroke_width=1),
        process_strip(),
    ])
    return document(defs(), body)


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
