"""Site water-management scheme (16:9): clean-water and leachate lines of the current works, and the
paths they cannot stop, which the project addresses.

    .venv\\Scripts\\python.exe docs\\figures\\make_water_management.py

Facilities follow the site facility plan legend (HDPE 시트, 기존 우수차집관로, 벤치플륨관, 집수구,
우수조정지, 연직차수벽, 양수관정, 저류조, 관측정); figures from the project plan, the DEM analysis
and press reports.
"""
from pathlib import Path

from svgkit import MUTED, TEXT, arrow, document, path, pill, rect, save, text, text_width, wrap

OUT_DIR = Path(__file__).resolve().parent
NAME = "water_management_scheme"

BLUE = ("#1d6fc2", "#eff6ff")
RED = ("#b91c1c", "#fef2f2")
ORANGE = ("#c2410c", "#fff7ed")
PURPLE = ("#6d28d9", "#f5f3ff")


def box(x, y, w, h, title, sub, colors, dashed=False, title_size=25, sub_size=21):
    color, fill = colors
    out = [rect(x, y, w, h, rx=12, fill=fill, stroke=color, stroke_width=3,
                stroke_dasharray="10 7" if dashed else None)]
    t_rows = wrap(title, w - 28, title_size, 700)
    s_rows = [r for s in (sub if isinstance(sub, list) else [sub]) for r in wrap(s, w - 28, sub_size)] if sub else []
    total = len(t_rows) * title_size * 1.25 + (len(s_rows) * sub_size * 1.3 + 6 if s_rows else 0)
    cy = y + (h - total) / 2
    for r in t_rows:
        cy += title_size * 1.25
        out.append(text(x + w / 2, round(cy - 5, 1), r, title_size, 700, color, anchor="middle"))
    cy += 6
    for r in s_rows:
        cy += sub_size * 1.3
        out.append(text(x + w / 2, round(cy - 5, 1), r, sub_size, 400, TEXT, anchor="middle"))
    return "\n".join(out)


def row_header(y, label_, colors):
    color, _ = colors
    svg, _ = pill(40, y, label_, color, "white", size=22)
    return svg


def flow(x0, x1, y, colors, dashed=False):
    marker = {"#1d6fc2": "blue", "#b91c1c": "red", "#c2410c": "red", "#6d28d9": "gray"}[colors[0]]
    return arrow(f"M{x0 + 6},{y} L{x1 - 8},{y}", colors[0], 4, dash="10 7" if dashed else None, marker=marker)


def build_svg():
    W_, G = 330, 66                                   # box width, gap (arrow space)
    xs = [40 + i * (W_ + G) for i in range(4)]        # 40, 436, 832, 1228
    out = [rect(0, 0, 1600, 900, fill="white"),
           text(40, 68, "현장 물 관리 체계 — 깨끗한 물과 침출수를 나눠 처리", 44, 700),
           text(40, 112, "현재 시설은 지표수·침출수를 가두고 처리 · 본 과제는 그 시설로 막지 못하는 지하 유입·우회를 줄임", 24, 400, MUTED),
           path("M40,134 L1560,134", stroke="#e5e7eb", stroke_width=2)]

    # ① clean water
    out.append(row_header(160, "① 깨끗한 물 (우수) — 침투 전에 가로채 내보냄", BLUE))
    out.append(box(xs[0], 180, W_, 84, "절벽·배후 사면 빗물", "지표 유역 4.9 ha (0.5 m)", BLUE))
    out.append(box(xs[0], 276, W_, 84, "매립지 노면 빗물", "HDPE 차수막으로 침투 차단", BLUE))
    out.append(box(xs[1], 196, W_, 148, "우수차집관로·벤치플륨관", "절벽면–매립지 경계에서 흘러내리는 물을 가로챔", BLUE))
    out.append(box(xs[2], 196, W_, 148, "집수구 → 우수조정지", "깨끗한 물을 모아 둠", BLUE))
    out.append(box(xs[3], 196, W_, 148, "외부 방류", "자연 배수 또는 펌프", BLUE))
    out.append(arrow(f"M{xs[0] + W_ + 6},222 C{xs[0] + W_ + 36},222 {xs[1] - 40},260 {xs[1] - 8},264", BLUE[0], 4, marker="blue"))
    out.append(arrow(f"M{xs[0] + W_ + 6},318 C{xs[0] + W_ + 36},318 {xs[1] - 40},284 {xs[1] - 8},278", BLUE[0], 4, marker="blue"))
    out += [flow(xs[1] + W_, xs[2], 270, BLUE), flow(xs[2] + W_, xs[3], 270, BLUE)]

    # ② leachate
    out.append(row_header(396, "② 오염된 물 (침출수) — 차수벽 안에 가두고 퍼내서 처리", RED))
    yb, hb = 416, 124
    out.append(box(xs[0], yb, W_, hb, "침출수 생성", "빗물·지하수가 폐기물층 통과 (광재류·폐주물사, 비소)", RED))
    out.append(box(xs[1], yb, W_, hb, "연직차수벽 안쪽에 모임", "상류 L=202 m · 하류 L=100 m", RED))
    out.append(box(xs[2], yb, W_, hb, "양수정 PW-01~10", "안쪽 수위를 낮게 유지 → 넘침·우회 방지", RED))
    out.append(box(xs[3], yb, W_, hb, "저류조 → 처리 → 방류", "2026년 1.2만 t 처리 계획", RED))
    out += [flow(xs[i] + W_, xs[i + 1], yb + hb / 2, RED) for i in range(3)]
    out.append(text(xs[1] + W_ / 2 + 36, yb + hb + 30,
                    "관측정 MW-01~04: 안팎 수위·수질 감시 · 양수 중단 → 수위 상승 → 넘침·우회 (2017 범람)", 20, 400, MUTED))

    # ③ gaps -> project
    # right-aligned so the links between rows ② and ③ (on the left) stay clear
    yc, hc = 634, 184
    head3 = "③ 현재 시설로 막지 못하는 경로 → 본 과제가 보완"
    out.append(pill(1560 - text_width(head3, 22, 700) - 14, yc - 24, head3, ORANGE[0], "white", size=22)[0])
    out.append(box(xs[0], yc, W_, hc, "절리·파쇄대 지하 유입",
                   ["외수 일 50~70 t", "(강우 시 70~80 t)", "표면 시설로 차단 불가"], ORANGE, dashed=True))
    out.append(box(xs[1], yc, W_ * 2 + G, hc, "차수벽 끝·벽 아래 우회 → 하류",
                   ["서북·동북측 벽 끝, 절리대 따라 심도 ~35 m", "지표 유출 방향: 서북쪽 → 중신천"], ORANGE, dashed=True))
    out.append(box(xs[3], yc, W_, hc, "본 과제의 대응",
                   ["드론 절리 분석(2-1) → 유입 지점 타겟 그라우팅(2-4)", "ERT·관측정으로 우회 확인(2-2·3-1)"],
                   PURPLE, title_size=25, sub_size=19))
    # links: gap boxes feed the leachate line, and both lead to the project's response
    out.append(arrow(f"M{xs[0] + W_ / 2},{yc - 6} L{xs[0] + W_ / 2},{yb + hb + 8}", ORANGE[0], 4, dash="10 7", marker="red"))
    out.append(text(xs[0] + W_ / 2 + 14, yc - 22, "침출수 계속 생성", 19, 700, ORANGE[0]))
    out.append(text(xs[1] + W_ / 2 + 14, yc - 22, "우회 유출", 19, 700, ORANGE[0]))
    out.append(arrow(f"M{xs[1] + W_ / 2},{yb + hb + 8} L{xs[1] + W_ / 2},{yc - 6}", ORANGE[0], 4, dash="10 7", marker="red"))
    out.append(flow(xs[2] + W_, xs[3], yc + hc / 2, PURPLE))
    out.append(path(f"M{xs[0] + W_ / 2},{yc + hc + 4} L{xs[0] + W_ / 2},{yc + hc + 26} L{xs[3] + W_ / 2},{yc + hc + 26}",
                    fill="none", stroke=PURPLE[0], stroke_width=3, stroke_dasharray="6 6"))
    out.append(arrow(f"M{xs[3] + W_ / 2},{yc + hc + 26} L{xs[3] + W_ / 2},{yc + hc + 6}", PURPLE[0], 3, marker="gray"))

    out.append(text(40, 890, "※ 시설 구성은 현장 시설 배치도 범례 기준(이중벽관·저류조 역할은 확인 필요). 수치: 사업계획서·DEM 해석·언론 보도.",
                    17, 400, MUTED))
    return document("", "\n".join(out))


if __name__ == "__main__":
    save(build_svg(), OUT_DIR, NAME)
