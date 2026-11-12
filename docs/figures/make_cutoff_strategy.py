"""Two strategies in fractured rock (16:9): a uniformly deep cut-off wall vs. targeted grouting of the
main inflow / outflow fractures plus hydraulic containment by pumping. Schematic, not to scale.

    .venv\\Scripts\\python.exe docs\\figures\\make_cutoff_strategy.py
"""
from pathlib import Path

from make_landfill_comparison import DEFS, C, bullets, panel_frame, waste_fill
from svgkit import MUTED, TEXT, arrow, document, label, path, rect, save, text

OUT_DIR = Path(__file__).resolve().parent
NAME = "cutoff_strategy_comparison"
GROUT = "#7c3aed"
FLOOR = 585


def site(X, clip_id, seed, fracture_width=3, inner_wt=470):
    """Common geometry: hill + cliff on the left, waste in the pit, downstream ground on the right."""
    out = [path(f"M{X(0)},250 L{X(120)},244 L{X(140)},250 L{X(146)},{FLOOR} L{X(610)},{FLOOR} "
                f"L{X(640)},470 L{X(740)},466 L{X(740)},720 L{X(0)},720 Z", fill="url(#rock)"),
           path(f"M{X(0)},250 L{X(120)},244 L{X(140)},250 L{X(140)},270 L{X(0)},272 Z", fill="url(#soil)"),
           path(f"M{X(640)},470 L{X(740)},466 L{X(740)},492 L{X(640)},496 Z", fill="url(#soil)")]
    for y in (300, 360, 420, 480, 540):
        out.append(path(f"M{X(0)},{y - 40} L{X(146)},{y + 14}", stroke=C["joint"], stroke_width=fracture_width))
    for xa, xb in [(200, 175), (290, 320), (380, 360), (470, 505), (560, 540), (620, 700)]:
        out.append(path(f"M{X(xa)},{FLOOR} L{X(xb)},712", stroke=C["joint"], stroke_width=fracture_width))
    wd = f"M{X(146)},330 L{X(560)},330 L{X(640)},466 L{X(610)},{FLOOR} L{X(146)},{FLOOR} Z"
    out.append(waste_fill(wd, clip_id, X(146), X(640), 330, FLOOR, seed))
    out.append(path(f"M{X(146)},327 L{X(560)},327 L{X(642)},463", fill="none", stroke=C["membrane"],
                    stroke_width=6, stroke_linejoin="round"))
    return out


def deep_wall(x0=40):
    X = lambda v: x0 + v
    out = panel_frame(x0, "A. 균일하게 깊은 차수벽", "#475569")
    out += site(X, "wa", 11)
    # very deep wall with a couple of defects
    out.append(rect(X(650), 452, 18, 240, fill="url(#hatch)", stroke="#4b5563"))
    for y in (545, 625):
        out.append(rect(X(650), y, 18, 9, fill=C["sky"]))
        out.append(arrow(f"M{X(630)},{y + 4} L{X(690)},{y + 4}", C["leach"], 3.5, marker="red"))
    # water level raised behind the wall
    out.append(path(f"M{X(0)},300 C{X(60)},304 {X(120)},318 {X(146)},350", fill="none", stroke=C["gw"],
                    stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(path(f"M{X(150)},420 L{X(645)},424", fill="none", stroke=C["gw"], stroke_width=3, stroke_dasharray="10 6"))
    out.append(path(f"M{X(670)},482 L{X(740)},486", fill="none", stroke=C["gw"], stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(arrow(f"M{X(560)},470 L{X(560)},428", C["gw"], 3, marker="blue"))
    # inflow unchanged; leachate down the fractures and under the wall tip
    for y in (300, 360, 420):
        out.append(arrow(f"M{X(30)},{y - 34} L{X(150)},{y + 18}", C["gw"], 3.5))
    for xa, xb in [(290, 316), (380, 362), (470, 500)]:
        out.append(arrow(f"M{X(xa)},{FLOOR + 6} L{X(xb)},{FLOOR + 104}", C["leach"], 4, marker="red"))
    out.append(arrow(f"M{X(600)},{FLOOR + 30} C{X(630)},718 {X(700)},718 {X(728)},680", C["leach"], 4, marker="red"))
    out += [
        label(X(14), 446, [("외수 유입 그대로", 20, 700, C["gw"])], pad=5),
        label(X(380), 372, [("벽 뒤 수위 상승", 20, 700, C["gw"])], anchor="middle", pad=5),
        label(X(726), 300, [("깊이↑ → 시추·주입·비용 급증", 19, 700, TEXT), ("암반 구간 시공 곤란", 18, 400, TEXT)],
              anchor="end", pad=6),
        label(X(624), 568, [("벽 결함으로 누수 →", 19, 700, C["bad"])], anchor="end", pad=4),
        label(X(20), 640, [("벽 아래 절리로 우회", 20, 700, C["bad"]), ("불투수층이 없어 끝이 열림", 18, 400, TEXT)], pad=6),
    ]
    return out


def targeted(x0=820):
    X = lambda v: x0 + v
    out = panel_frame(x0, "B. 타겟 정밀 차수 + 수리적 봉쇄", "#6d28d9")
    out += site(X, "wb", 12)
    # grouted zones on the main inflow joints (cliff) and main outflow fractures (floor, under the wall)
    for a, b in [((X(0), 296), (X(146), 354)), ((X(0), 380), (X(146), 434))]:
        out.append(path(f"M{a[0]},{a[1]} L{b[0]},{b[1]}", stroke=GROUT, stroke_width=16, stroke_opacity=0.55,
                        stroke_linecap="round"))
    for xa, xb in [(470, 505), (620, 700)]:
        out.append(path(f"M{X(xa)},{FLOOR} L{X(xb)},712", stroke=GROUT, stroke_width=16, stroke_opacity=0.55,
                        stroke_linecap="round"))
    # existing shallow wall
    out.append(rect(X(650), 452, 14, 170, fill="url(#hatch)", stroke="#4b5563"))
    # pumping well lowers the level inside below the outside level -> inward gradient
    out.append(path(f"M{X(520)},300 L{X(520)},575", stroke="#374151", stroke_width=4))
    out.append(path(f"M{X(520)},520 L{X(520)},575", stroke="#374151", stroke_width=9, stroke_dasharray="3 3"))
    out.append(rect(X(511), 284, 18, 16, fill="#374151"))
    out.append(arrow(f"M{X(536)},540 L{X(536)},310", C["leach"], 3, marker="red"))
    out.append(path(f"M{X(0)},300 C{X(60)},304 {X(120)},318 {X(146)},350", fill="none", stroke=C["gw"],
                    stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(path(f"M{X(150)},520 C{X(300)},522 {X(470)},540 {X(520)},560 C{X(560)},540 {X(620)},530 {X(645)},528",
                    fill="none", stroke=C["gw"], stroke_width=3, stroke_dasharray="10 6"))
    out.append(path(f"M{X(670)},482 L{X(740)},486", fill="none", stroke=C["gw"], stroke_width=2.5, stroke_dasharray="10 6"))
    out.append(arrow(f"M{X(728)},640 C{X(700)},700 {X(640)},690 {X(612)},630", C["gw"], 3.5))
    # reduced inflow (thin, dashed) and small residual leakage
    for y in (300, 420):
        out.append(arrow(f"M{X(30)},{y - 34} L{X(120)},{y + 6}", C["gw"], 2.2, dash="6 5"))
    out.append(arrow(f"M{X(290)},{FLOOR + 6} L{X(310)},{FLOOR + 70}", C["leach"], 2.5, dash="6 5", marker="red"))
    # monitoring: surface ERT line + a downstream monitoring well
    for xx in range(170, 560, 34):
        out.append(f'<path d="M{X(xx)},318 l6,-11 l6,11 z" fill="#0f766e"/>')
    out.append(path(f"M{X(706)},452 L{X(706)},700", stroke="#374151", stroke_width=4))
    out.append(rect(X(698), 436, 16, 16, fill="#374151"))
    out += [
        label(X(14), 446, [("① 유입 절리 그라우팅", 20, 700, GROUT), ("드론 2-1 → 설계 2-4", 18, 400, TEXT)], pad=5),
        label(X(170), 232, [("④ 시계열 ERT·관측정으로 효과 확인 (3-1)", 20, 700, "#0f766e")], pad=5),
        label(X(380), 470, [("③ 양수: 안쪽 수위↓", 20, 700, C["gw"]), ("바깥 → 안쪽 흐름 (내향 경사)", 18, 400, TEXT)],
              anchor="middle", pad=5),
        label(X(20), 640, [("② 주요 파쇄대 그라우팅", 20, 700, GROUT), ("시추·ERT로 위치 확인 (2-2·2-3)", 18, 400, TEXT)],
              pad=6),
    ]
    return out


def build_svg():
    body = [rect(0, 0, 1600, 900, fill="white"),
            text(40, 68, "균열 암반의 차수 전략 — 깊은 벽 하나보다 물길을 찾아 막고 퍼낸다", 42, 700),
            text(40, 112, "완전 차단이 아니라, 유출을 관리 가능한 수준으로 줄이고 측정으로 입증하는 것이 현실적 목표", 24, 400, MUTED),
            path("M40,134 L1560,134", stroke="#e5e7eb", stroke_width=2)]
    body += deep_wall() + targeted()
    body.append(arrow("M790,440 L812,440", "#374151", 4, marker="gray"))
    body += bullets(40, "#475569", [
        "깊을수록 비용·난도 급증, 암반에서는 그라우트 커튼으로 전환",
        "불투수층에 박히지 않으면 벽 아래로 우회 · 결함으로 누수",
        "하류만 막으면 벽 뒤 수위 상승 → 벽 끝·아래 우회 압력 증가",
    ])
    body += bullets(820, "#6d28d9", [
        "상류 유입 절리를 막아 침출수 발생량 자체를 줄임",
        "실제 물길인 주요 파쇄대만 집중 주입 → 같은 비용으로 큰 효과",
        "양수로 안쪽 수위를 낮게 유지 + 시공 전후 측정으로 보완",
    ])
    body.append(text(40, 892, "※ 개념 모식도(축척 없음). B는 본 과제 기술모듈(2-1 드론 절리 분석, 2-2·2-3 ERT·시추, 2-4 타겟 그라우팅, 3-1 효과 검증)의 연계를 단순화한 것.",
                     17, 400, MUTED))
    return document(DEFS, "\n".join(body))


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
