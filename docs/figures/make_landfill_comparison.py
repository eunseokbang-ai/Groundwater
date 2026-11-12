"""Side-by-side cross-sections (16:9): an engineered sanitary landfill vs. this illegally filled quarry,
showing why leachate can migrate deep here. Schematic, not to scale.

    .venv\\Scripts\\python.exe docs\\figures\\make_landfill_comparison.py
"""
import random
from pathlib import Path

from svgkit import MUTED, TEXT, arrow, document, label, path, pill, rect, save, text, wrap

OUT_DIR = Path(__file__).resolve().parent
NAME = "landfill_comparison"

C = {"sky": "#eef5fb", "soil": "#dcc294", "rock": "#a7a197", "joint": "#4b433c", "waste": "#6e6259",
     "clay": "#9a6b3a", "membrane": "#111827", "gravel": "#cbd5e1", "gw": "#1d6fc2", "leach": "#d9480f",
     "plume": "#f97316", "good": "#15803d", "bad": "#b91c1c", "cover": "#86a46b"}

DEFS = f"""
  <pattern id="soil" width="9" height="9" patternUnits="userSpaceOnUse">
    <rect width="9" height="9" fill="{C['soil']}"/><circle cx="2" cy="3" r="1.1" fill="#b89a67"/></pattern>
  <pattern id="rock" width="26" height="18" patternUnits="userSpaceOnUse">
    <rect width="26" height="18" fill="{C['rock']}"/><path d="M0 9 H26 M13 0 V9 M0 18 V9 M26 18 V9" stroke="#959086" stroke-width="1"/></pattern>
  <pattern id="clay" width="10" height="10" patternUnits="userSpaceOnUse">
    <rect width="10" height="10" fill="{C['clay']}"/><path d="M0 5 H10" stroke="#7c5428" stroke-width="1.5"/></pattern>
  <pattern id="gravel" width="10" height="10" patternUnits="userSpaceOnUse">
    <rect width="10" height="10" fill="#e5e7eb"/><circle cx="3" cy="3" r="2" fill="#9ca3af"/><circle cx="8" cy="8" r="1.6" fill="#9ca3af"/></pattern>
  <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="6" height="6" fill="#9aa5b1"/><path d="M0 0 V6" stroke="#4b5563" stroke-width="1.4"/></pattern>
  <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="7"/></filter>
"""


def waste_fill(poly_d, clip_id, x0, x1, y0, y1, seed):
    rng = random.Random(seed)
    pal = ["#5a4d45", "#857264", "#6a5e55", "#9a8878", "#4d433c"]
    pieces = []
    for _ in range(int((x1 - x0) * (y1 - y0) / 260)):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        w, h = rng.uniform(4, 13), rng.uniform(3, 7)
        pieces.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="1.5" fill="{rng.choice(pal)}" '
                      f'transform="rotate({rng.uniform(-40, 40):.0f} {x:.0f} {y:.0f})"/>')
    return (f'<clipPath id="{clip_id}"><path d="{poly_d}"/></clipPath>' + path(poly_d, fill=C["waste"])
            + f'<g clip-path="url(#{clip_id})">{"".join(pieces)}</g>')


def panel_frame(x0, title, color):
    return [rect(x0, 160, 740, 560, rx=14, fill=C["sky"], stroke="#cbd5e1"),
            pill(x0 + 16, 190, title, color, "white", size=26)[0]]


def sanitary(x0=40):
    out = panel_frame(x0, "정상 위생매립장 — 바닥부터 막고 묻음", C["good"])
    X = lambda v: x0 + v
    gs = 360                                                   # original ground surface
    # ground layers
    out.append(rect(X(0), gs, 740, 720 - gs, fill="url(#rock)"))
    out.append(rect(X(0), gs, 740, 60, fill="url(#soil)"))
    out.append(f'<rect x="{X(0)}" y="{gs}" width="740" height="{720 - gs}" fill="none"/>')
    # pit with liner system
    pit = f"M{X(150)},{gs} L{X(250)},585 L{X(500)},585 L{X(600)},{gs} Z"
    clay = (f"M{X(126)},{gs} L{X(234)},612 L{X(516)},612 L{X(624)},{gs} L{X(600)},{gs} L{X(500)},585 "
            f"L{X(250)},585 L{X(150)},{gs} Z")
    out.append(path(clay, fill="url(#clay)"))
    out.append(path(pit, fill="url(#gravel)"))
    inner = f"M{X(160)},{gs} L{X(258)},568 L{X(492)},568 L{X(590)},{gs} Z"
    mound = (f"M{X(140)},{gs} Q{X(375)},{gs - 120} {X(610)},{gs} L{X(590)},{gs} L{X(492)},568 L{X(258)},568 "
             f"L{X(160)},{gs} Z")
    out.append(waste_fill(f"M{X(160)},{gs} Q{X(375)},{gs - 104} {X(590)},{gs} L{X(492)},568 L{X(258)},568 Z",
                          "w1", X(160), X(590), gs - 110, 570, 3))
    out.append(path(f"M{X(150)},{gs} L{X(250)},585 L{X(500)},585 L{X(600)},{gs}", fill="none",
                    stroke=C["membrane"], stroke_width=5, stroke_linejoin="round"))
    # final cover: soil + membrane + grass
    out.append(path(f"M{X(132)},{gs + 2} Q{X(375)},{gs - 128} {X(618)},{gs + 2}", fill="none", stroke=C["cover"],
                    stroke_width=16))
    out.append(path(f"M{X(140)},{gs - 6} Q{X(375)},{gs - 112} {X(610)},{gs - 6}", fill="none",
                    stroke=C["membrane"], stroke_width=4))
    # collection pipes + outlet to treatment
    for px in range(290, 470, 45):
        out.append(f'<circle cx="{X(px)}" cy="576" r="7" fill="white" stroke="#1f2937" stroke-width="2.5"/>')
    out.append(arrow(f"M{X(500)},578 L{X(560)},578 L{X(640)},{gs - 18} L{X(660)},{gs - 50}", C["leach"], 4,
                     marker="red"))
    out.append(rect(X(600), 228, 128, 52, rx=8, fill="white", stroke=C["leach"], stroke_width=2.5))
    out.append(text(X(664), 262, "처리시설", 22, 700, C["leach"], anchor="middle"))
    # rain shed off the cover
    for xx in (250, 330, 420, 500):
        out.append(path(f"M{X(xx)},228 L{X(xx - 8)},262", stroke="#4f90cf", stroke_width=2.5, stroke_dasharray="7 5"))
    out.append(arrow(f"M{X(300)},{gs - 66} Q{X(200)},{gs - 40} {X(150)},{gs - 12}", C["gw"], 3.5))
    out.append(arrow(f"M{X(450)},{gs - 66} Q{X(540)},{gs - 40} {X(596)},{gs - 12}", C["gw"], 3.5))
    # groundwater table, clean
    out.append(path(f"M{X(0)},650 C{X(250)},658 {X(500)},662 {X(740)},668", fill="none", stroke=C["gw"],
                    stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(arrow(f"M{X(60)},680 L{X(680)},694", C["gw"], 3.5))
    # monitoring wells
    for xx in (70, 690):
        out.append(path(f"M{X(xx)},{gs - 18} L{X(xx)},700", stroke="#374151", stroke_width=4))
        out.append(rect(X(xx) - 8, gs - 34, 16, 16, fill="#374151"))
    # labels
    out += [
        label(X(18), 256, [("최종 덮개 (흙+차수막)", 21, 700, TEXT)], pad=6),
        label(X(18), 470, [("바닥 차수층", 21, 700, "#7c5428"), ("점토 + 차수막", 19, 400, TEXT)], pad=6),
        label(X(726), 470, [("침출수 집수층", 21, 700, TEXT), ("+ 집수관", 19, 400, TEXT)], anchor="end", pad=6),
        label(X(370), 520, [("폐기물", 22, 700, "white")], anchor="middle", fill="#111827", opacity=0.6,
              stroke="#111827", pad=5),
        label(X(370), 612, [("지하수 — 폐기물과 분리, 깨끗하게 유지", 20, 700, C["gw"])], anchor="middle", pad=5),
        label(X(80), 548, [("관측정", 19, 700, TEXT)], pad=4),
    ]
    return out


def quarry(x0=820):
    out = panel_frame(x0, "이 현장 — 깨진 암반 위에 바로 매립", C["bad"])
    X = lambda v: x0 + v
    floor = 585                                                # quarry floor
    # rock mass with hill (left), cliff, pit, downstream ground (right)
    ground = (f"M{X(0)},250 L{X(120)},244 L{X(140)},250 L{X(146)},{floor} L{X(610)},{floor} "
              f"L{X(640)},470 L{X(740)},466 L{X(740)},720 L{X(0)},720 Z")
    out.append(path(ground, fill="url(#rock)"))
    out.append(path(f"M{X(0)},250 L{X(120)},244 L{X(140)},250 L{X(140)},270 L{X(0)},272 Z", fill="url(#soil)"))
    out.append(path(f"M{X(640)},470 L{X(740)},466 L{X(740)},492 L{X(640)},496 Z", fill="url(#soil)"))
    # joints in the cliff (dipping into the pit) and blast cracks / fractures below the floor
    for y in (300, 360, 420, 480, 540):
        out.append(path(f"M{X(0)},{y - 40} L{X(146)},{y + 14}", stroke=C["joint"], stroke_width=3))
    for xa, xb in [(200, 175), (290, 320), (380, 360), (470, 505), (560, 540)]:
        out.append(path(f"M{X(xa)},{floor} L{X(xb)},710", stroke=C["joint"], stroke_width=3))
    for xa in (230, 330, 420, 510):
        out.append(path(f"M{X(xa)},{floor} l-10,26 m10,-26 l12,22", stroke=C["joint"], stroke_width=1.6))
    # leachate plume: under the floor along fractures, and under/around the wall
    plume = [path(f"M{X(160)},{floor} L{X(610)},{floor} L{X(700)},560 L{X(730)},610 L{X(560)},700 L{X(480)},712 "
                  f"L{X(330)},712 L{X(200)},700 L{X(160)},640 Z", fill=C["plume"])]
    for xa, xb in [(290, 320), (380, 360), (470, 505)]:
        plume.append(path(f"M{X(xa)},{floor} L{X(xb)},712", stroke=C["plume"], stroke_width=36))
    out.append(f'<g opacity="0.45" filter="url(#soft)">{"".join(plume)}</g>')
    # waste right on the broken floor, up against the cliff
    wd = (f"M{X(146)},330 L{X(560)},330 L{X(640)},466 L{X(610)},{floor} L{X(146)},{floor} Z")
    out.append(waste_fill(wd, "w2", X(146), X(640), 330, floor, 7))
    # post-hoc HDPE cover, cut-off wall, pumping well
    out.append(path(f"M{X(146)},327 L{X(560)},327 L{X(642)},463", fill="none", stroke=C["membrane"],
                    stroke_width=6, stroke_linejoin="round"))
    out.append(rect(X(650), 452, 14, 170, fill="url(#hatch)", stroke="#4b5563"))
    out.append(path(f"M{X(520)},300 L{X(520)},575", stroke="#374151", stroke_width=4))
    out.append(path(f"M{X(520)},520 L{X(520)},575", stroke="#374151", stroke_width=9, stroke_dasharray="3 3"))
    out.append(rect(X(511), 284, 18, 16, fill="#374151"))
    out.append(arrow(f"M{X(536)},540 L{X(536)},310", C["leach"], 3, marker="red"))
    # groundwater: high in the hill, waste submerged
    out.append(path(f"M{X(0)},300 C{X(60)},304 {X(120)},318 {X(146)},350", fill="none", stroke=C["gw"],
                    stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(path(f"M{X(150)},470 L{X(645)},476", fill="none", stroke=C["gw"], stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(path(f"M{X(665)},482 L{X(740)},486", fill="none", stroke=C["gw"], stroke_width=2.5, stroke_dasharray="10 6"))
    # flows: inflow along joints, leachate down fractures, under and around the wall
    for y in (300, 360, 420):
        out.append(arrow(f"M{X(30)},{y - 34} L{X(150)},{y + 18}", C["gw"], 3.5))
    for xa, xb in [(290, 316), (380, 362), (470, 500)]:
        out.append(arrow(f"M{X(xa)},{floor + 6} L{X(xb)},{floor + 104}", C["leach"], 4, marker="red"))
    out.append(arrow(f"M{X(600)},{floor + 10} C{X(640)},650 {X(690)},650 {X(728)},600", C["leach"], 4, marker="red"))
    # labels
    out += [
        label(X(160), 236, [("사후 HDPE 덮개", 21, 700, TEXT)], pad=6),
        label(X(14), 446, [("외수 유입", 21, 700, C["gw"]), ("절리로 일 50~70 t", 19, 400, TEXT)], pad=6),
        label(X(380), 392, [("매립층 최대 23 m", 21, 700, "white")], anchor="middle", fill="#111827", opacity=0.6,
              stroke="#111827", pad=5),
        label(X(380), 494, [("폐기물이 지하수에 잠김", 19, 700, C["gw"])], anchor="middle", pad=4),
        label(X(726), 380, [("연직차수벽", 20, 700, TEXT), ("(사후, 하류만)", 18, 400, TEXT)], anchor="end", pad=5),
        label(X(500), 236, [("양수정", 20, 700, TEXT)], pad=5),
        label(X(20), 652, [("바닥·측면 차수층 없음", 21, 700, C["bad"]), ("발파 균열·절리로 직접 유출", 19, 400, TEXT)], pad=6),
        label(X(726), 664, [("침출수 심부 이동", 21, 700, C["bad"]), ("예비 ERT ~35 m", 19, 400, TEXT)], anchor="end", pad=6),
    ]
    return out


def bullets(x0, color, items):
    out, y = [], 754
    for s in items:
        out.append(f'<circle cx="{x0 + 10}" cy="{y - 7}" r="5" fill="{color}"/>')
        for r in wrap(s, 700, 22):
            out.append(text(x0 + 26, y, r, 22, 400, TEXT))
            y += 31
        y += 4
    return out


def build_svg():
    body = [rect(0, 0, 1600, 900, fill="white"),
            text(40, 68, "정상 매립장과 비교: 왜 이 현장은 침출수가 깊이 내려가는가", 44, 700),
            text(40, 112, "바닥·측면 차단과 침출수 집수가 없으면, 고인 침출수가 절리·발파 균열을 따라 심부로 이동", 24, 400, MUTED),
            path("M40,134 L1560,134", stroke="#e5e7eb", stroke_width=2)]
    body += sanitary() + quarry()
    body.append(arrow("M790,440 L812,440", "#374151", 4, marker="gray"))
    body += bullets(40, C["good"], [
        "바닥·측면 이중 차단 → 침출수가 땅속으로 못 내려감",
        "집수관으로 모아 처리, 바닥 수위를 얕게 유지",
        "최종 덮개로 빗물 유입 최소화 · 관측정으로 상시 감시",
    ])
    body += bullets(820, C["bad"], [
        "차단층 없이 깨진 암반에 직접 접촉 · 폐기물이 지하수에 잠김",
        "고인 침출수의 압력이 절리·균열로 전달 → 심부·측방 확산",
        "덮개·차수벽·양수는 사후 보완: 바닥과 지하 유입은 여전히 열림",
    ])
    body.append(text(40, 892, "※ 개념 모식도(축척 없음). 현장 수치: 사업계획서·예비 ERT. 정상 매립장은 일반적인 위생매립 구조를 단순화한 것.",
                     17, 400, MUTED))
    return document(DEFS, "\n".join(body))


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
