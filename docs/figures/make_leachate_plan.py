"""Conceptual plan view of the Iksan reclaimed quarry (16:9, companion to the cross-section):
cut-off walls and pumping wells, observed leachate releases, and the bypass routes that a
front-only barrier leaves open (around the wall ends, beneath them along joints).

    .venv\\Scripts\\python.exe docs\\figures\\make_leachate_plan.py

Layout follows the site facility plan (upstream wall L = 202 m, downstream wall L = 100 m)
and press reports (2017: overflow in several directions, to the Jungsin stream and the
north-east neighbouring site). Not to scale; positions and directions are schematic.
"""
from pathlib import Path

from svgkit import MUTED, TEXT, arrow, document, label, path, rect, save, text, wrap

OUT_DIR = Path(__file__).resolve().parent
NAME = "leachate_plan_concept"

COL = {"forest": "#dfe9d3", "forest_dot": "#b7cda0", "yard": "#eadfc8", "rock": "#9ca3af", "waste": "#5e534b",
       "wall": "#dc2626", "gw": "#1d6fc2", "leach": "#d9480f", "scope": "#7c3aed", "paddy": "#cfe3b5",
       "channel": "#eab308", "well": "#f59e0b", "mw": "#16a34a"}

# North up (checked against the 2025 orthophoto): the landfill's wide end and both cut-off walls
# lie to the north, the narrow tip to the south; paddies and solar farms to the north-west.
LANDFILL = ("M470,400 L620,330 L800,390 C815,480 825,620 812,760 C806,810 796,835 786,846 "
            "C730,800 640,700 570,600 C520,530 482,460 470,400 Z")
ROCK_FACE = ("M470,400 C482,460 520,530 570,600 C640,700 730,800 786,846 C796,835 806,810 812,760 "
             "C825,620 815,480 800,390")
UP_WALL = [(440, 376), (620, 300), (832, 366)]
DOWN_WALL = [(250, 250), (320, 205), (400, 255)]
PUMP_WELLS = [(560, 352), (690, 342), (520, 290), (640, 262), (740, 300), (450, 300)]
MON_WELLS = [(200, 330), (470, 236), (900, 470), (360, 560)]


def defs():
    return f"""
  <pattern id="forest" width="22" height="22" patternUnits="userSpaceOnUse">
    <rect width="22" height="22" fill="{COL['forest']}"/><circle cx="6" cy="7" r="4" fill="{COL['forest_dot']}"/>
    <circle cx="16" cy="17" r="3.5" fill="{COL['forest_dot']}"/></pattern>
  <pattern id="paddy" width="14" height="14" patternUnits="userSpaceOnUse">
    <rect width="14" height="14" fill="{COL['paddy']}"/><path d="M3 10 l2 -5 l2 5 M9 12 l2 -5 l2 5" stroke="#7aa85a" stroke-width="1"/></pattern>
  <pattern id="waste" width="16" height="16" patternUnits="userSpaceOnUse">
    <rect width="16" height="16" fill="{COL['waste']}"/><path d="M0 16 L16 0" stroke="#6f645b" stroke-width="2"/></pattern>
"""


def pts(points):
    return " ".join(f"{x},{y}" for x, y in points)


def wall(points, width=7):
    return (f'<polyline points="{pts(points)}" fill="none" stroke="{COL["wall"]}" stroke-width="{width}" '
            'stroke-linejoin="round" stroke-linecap="round"/>')


def house(x, y, roof):
    return (f'<rect x="{x - 13}" y="{y - 12}" width="26" height="18" fill="#f3efe6" stroke="#6b7280"/>'
            + path(f"M{x - 16},{y - 12} L{x},{y - 26} L{x + 16},{y - 12} Z", fill=roof))


def site_map():
    out = [rect(40, 168, 990, 702, fill="url(#forest)", stroke="#cbd5e1")]
    # cleared yard north of the landfill (facilities, access road)
    out.append(path("M240,168 L860,168 L860,300 L980,330 L880,400 L832,366 L620,300 L440,376 L360,440 "
                    "L240,420 Z", fill=COL["yard"]))
    # paddies and village to the north-west, drain toward the stream
    out.append(path("M40,168 L240,168 L240,420 L200,470 L40,470 Z", fill="url(#paddy)"))
    for x, y, roof in [(90, 506, "#b4534b"), (130, 498, "#8c5a3c"), (170, 510, "#4b6c8a")]:
        out.append(house(x, y, roof))
    out.append(path("M250,340 C180,380 110,400 40,420", fill="none", stroke="#60a5fa", stroke_width=6,
                    stroke_linecap="round"))
    # neighbouring site to the north-east
    out.append(path("M880,182 C950,176 1016,196 1020,236 C1016,276 960,290 910,284 C876,272 862,214 880,182 Z",
                    fill="#e5ddd0", stroke="#a8a29e", stroke_width=2, stroke_dasharray="6 5"))
    # survey scope for the subsurface outflow: downstream fan (NW-N-NE) and both flanks
    # (drawn before the landfill, which hides the part of the outline that crosses it)
    out.append(path("M500,680 L340,430 C230,370 130,300 112,178 L1016,178 L1016,330 C962,420 920,520 900,680 Z",
                    fill=COL["scope"], fill_opacity=0.08, stroke=COL["scope"], stroke_width=3, stroke_dasharray="12 8"))
    out.append(path(ROCK_FACE, fill="none", stroke=COL["rock"], stroke_width=26, stroke_linecap="round"))
    out.append(path(LANDFILL, fill="url(#waste)", stroke="#3f3a35", stroke_width=2))
    out.append(wall(UP_WALL))
    out.append(wall(DOWN_WALL))
    for x, y in PUMP_WELLS:
        out.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{COL["well"]}" stroke="#7c2d12" stroke-width="2"/>')
    for x, y in MON_WELLS:
        out.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{COL["mw"]}" stroke="#14532d" stroke-width="2"/>')
    return "\n".join(out)


def flows():
    gw, lc = COL["gw"], COL["leach"]
    out = [
        # groundwater inflow from the hills south, east and west of the quarry
        arrow("M620,866 L700,808", gw, 4), arrow("M990,700 L842,690", gw, 4), arrow("M440,700 L590,652", gw, 4),
        # confirmed: leachate toward the wide (north) end, captured between the walls
        arrow("M650,560 L626,344", lc, 5, marker="red"),
        arrow("M590,276 C520,250 450,240 372,236", lc, 3.5, marker="red"),
        # 2017 overflow: to the north-east neighbouring site, and via the drain to the stream
        arrow("M712,226 L842,226", lc, 4, marker="red"),
        arrow("M300,232 C280,270 262,300 250,330", lc, 4, marker="red"),
        # possible bypass (unverified): around both ends of the upstream wall, beneath along joints
        arrow("M452,398 C400,420 340,400 300,340 C280,310 230,290 186,286", lc, 4, dash="11 8", marker="red"),
        arrow("M842,412 C900,400 926,360 930,300", lc, 4, dash="11 8", marker="red"),
        arrow("M620,322 C618,290 616,260 615,228", lc, 4, dash="4 7", marker="red"),
    ]
    return "\n".join(out)


def map_labels():
    out = [
        label(300, 820, [("배후 산지 지하수 유입", 22, 700, COL["gw"])], pad=6),
        label(706, 600, [("매립지", 26, 700, "white"), ("HDPE 시트 복개", 20, 400, "#f3f4f6")], anchor="middle",
              fill="#3f3a35", opacity=0.85, stroke="#3f3a35"),
        label(560, 742, [("굴착 암벽", 22, 700, TEXT), ("(드론·열화상 조사)", 19, 400, MUTED)], anchor="end", pad=6),
        label(1022, 190, [("동북측 인접 부지", 21, 700, TEXT), ("2017 범람 유입", 19, 400, COL["leach"])],
              anchor="end", pad=6),
        label(860, 440, [("상류 연직차수벽", 21, 700, COL["wall"]), ("L = 202 m", 19, 400, TEXT)], pad=6),
        label(240, 232, [("하류 연직차수벽", 21, 700, COL["wall"]), ("L = 100 m (추가)", 19, 400, TEXT)],
              anchor="end", pad=6),
        label(330, 470, [("서북측 우회?", 21, 700, COL["leach"])], pad=6),
        label(912, 300, [("동북측 우회?", 21, 700, COL["leach"])], anchor="end", pad=6),
        label(420, 176, [("벽 하부 절리대 통과?", 21, 700, COL["leach"])], pad=6),
        label(56, 180, [("농경지·마을", 21, 700, TEXT)], pad=6),
        label(56, 440, [("농수로 → 중신천 (방향 개략)", 21, 700, "#1d4ed8")], pad=6),
    ]
    # north arrow + scale note
    out.append(path("M988,830 L1000,794 L1012,830 L1000,820 Z", fill=TEXT))
    out.append(text(1000, 786, "N", 22, 700, anchor="middle"))
    out.append(text(1016, 860, "축척 없음 · 위치 개략", 19, 400, MUTED, anchor="end"))
    return "\n".join(out)



def panel():
    x0, y0, w = 1066, 168, 494
    out = [rect(x0, y0, w, 702, rx=10, fill="white", stroke="#cbd5e1")]
    out.append(text(x0 + 20, y0 + 40, "범례", 25, 700))
    rows = [
        ("swatch", "url(#waste)", "매립지 (HDPE 시트 복개)"),
        ("band", COL["rock"], "굴착 암벽 (노출 사면)"),
        ("wall", COL["wall"], "연직차수벽"),
        ("dot2", None, "양수정 · 관측정"),
        ("arrow", COL["gw"], "지하수 유입"),
        ("arrow", COL["leach"], "침출수 흐름 (보도·시설로 확인)"),
        ("dash", COL["leach"], "우회 가능 경로 (추정, 검증 필요)"),
        ("scope", COL["scope"], "지하 유출 조사 권장 범위"),
    ]
    y = y0 + 80
    for kind, color, name in rows:
        sx = x0 + 22
        if kind == "swatch":
            out.append(rect(sx, y - 12, 44, 24, fill=color, stroke="#3f3a35"))
        elif kind == "band":
            out.append(path(f"M{sx},{y} L{sx + 44},{y}", stroke=color, stroke_width=14, stroke_linecap="round"))
        elif kind == "wall":
            out.append(path(f"M{sx},{y + 6} L{sx + 22},{y - 6} L{sx + 44},{y + 6}", fill="none", stroke=color,
                            stroke_width=6, stroke_linejoin="round"))
        elif kind == "dot2":
            out.append(f'<circle cx="{sx + 10}" cy="{y}" r="8" fill="{COL["well"]}" stroke="#7c2d12" stroke-width="2"/>')
            out.append(f'<circle cx="{sx + 34}" cy="{y}" r="8" fill="{COL["mw"]}" stroke="#14532d" stroke-width="2"/>')
        elif kind == "arrow":
            out.append(arrow(f"M{sx},{y} L{sx + 44},{y}", color, 4, marker="red" if color == COL["leach"] else "blue"))
        elif kind == "dash":
            out.append(arrow(f"M{sx},{y} L{sx + 44},{y}", color, 4, dash="9 6", marker="red"))
        elif kind == "scope":
            out.append(rect(sx, y - 12, 44, 24, rx=4, fill=color, fill_opacity=0.1, stroke=color, stroke_width=2.5,
                            stroke_dasharray="7 5"))
        out.append(text(x0 + 82, y + 8, name, 22))
        y += 42

    # evidence box
    y += 10
    out.append(path(f"M{x0 + 20},{y} L{x0 + w - 20},{y}", stroke="#e5e7eb", stroke_width=2))
    y += 38
    out.append(text(x0 + 20, y, "근거 자료", 24, 700))
    facts = [
        "2017.7 제방 붕괴, 침출수 '여러 방면' 유출 (중신천·동북측 부지) — 언론 보도",
        "차수벽 상류 202 m · 하류 100 m, 양수정 10공 — 시설 배치도",
        "앞쪽(하류)만 차단 → 벽 끝·하부 우회는 미검증",
    ]
    for f in facts:
        rows_ = wrap(f, w - 64, 20)
        out.append(f'<circle cx="{x0 + 28}" cy="{y + 22}" r="4" fill="{MUTED}"/>')
        for r in rows_:
            y += 29
            out.append(text(x0 + 40, y, r, 20, 400, TEXT))
        y += 6
    return "\n".join(out)


def build_svg():
    body = "\n".join([
        rect(0, 0, 1600, 900, fill="white"),
        text(40, 70, "폐석산 복구지 개념 평면도 — 차수시설과 침출수 확산 경로", 44, 700),
        text(40, 116, "시설 배치도·언론 보도 기반 개략도: 확인된 유출(실선)과 검증이 필요한 우회 경로(점선)", 25, 400, MUTED),
        path("M40,140 L1560,140", stroke="#e5e7eb", stroke_width=2),
        site_map(), flows(), map_labels(), panel(),
        text(40, 892, "※ 개념도(축척 없음). 지형·시설 위치와 흐름 방향은 개략적이며, 우회 경로는 조사로 확인할 가설임.", 18, 400, MUTED),
    ])
    return document(defs(), body)


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
