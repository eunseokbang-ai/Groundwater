"""Press-coverage timeline of the Nangsan (Iksan) quarry leachate case, for the presentation (16:9):

  leachate_timeline_1 -- 2011-2026 timeline: six phases, key events, reported leachate releases
  leachate_timeline_2 -- removal progress (cumulative tonnes) and recurring issues / project implications

    .venv\\Scripts\\python.exe docs\\figures\\make_leachate_timeline.py

All figures are as reported in the press (익산신문, 익산시민뉴스, 한국일보, 경향신문, 오마이뉴스,
전북일보, 아주경제, 전북도민일보, 전민일보, 뉴스핌, 2016-2026); totals differ between outlets.
"""
from pathlib import Path

from svgkit import MUTED, TEXT, document, path, pill, rect, save, text, text_width, wrap

OUT_DIR = Path(__file__).resolve().parent
X0, X1, Y_FIRST, Y_LAST = 80, 1520, 2011, 2027      # time axis: 90 px per year
BAND_Y, BAND_H = 468, 44

PHASES = [  # (start, end, colour, light fill)
    (2011.08, 2016.33, "#57534e", "#f5f5f4"),
    (2016.33, 2017.00, "#b45309", "#fffbeb"),
    (2017.00, 2018.00, "#b91c1c", "#fef2f2"),
    (2018.00, 2021.00, "#475569", "#f8fafc"),
    (2021.00, 2025.00, "#1d4ed8", "#eff6ff"),
    (2025.00, 2026.90, "#15803d", "#f0fdf4"),
]
RELEASES = [2013.6, 2014.6, 2017.49, 2017.6, 2019.5]   # reported fish kills / leachate overflows


def tx(year):
    return X0 + (year - Y_FIRST) / (Y_LAST - Y_FIRST) * (X1 - X0)


def header(title, subtitle):
    return [rect(0, 0, 1600, 900, fill="white"), text(56, 82, title, 46, 700), text(56, 128, subtitle, 26, 400, MUTED),
            path("M56,152 L1552,152", stroke="#e5e7eb", stroke_width=2)]


def badge(cx, cy, n, color, r=17, size=22):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'
            + text(cx, round(cy + size * 0.36, 1), str(n), size, 700, "white", anchor="middle"))


def card(x, y, w, n, phase, period, title, bullets, size=24):
    """Phase card; bullets = [(text, emphasis)]. Returns (svg, height)."""
    color, fill = PHASES[n - 1][2], PHASES[n - 1][3]
    head = 54
    body = []
    cy = y + head + 8
    for s, strong in bullets:
        rows = wrap(s, w - 52, size, 700 if strong else 400)
        body.append(f'<circle cx="{x + 24}" cy="{cy + size * 0.95 - size * 0.36:.1f}" r="4" fill="{color}"/>')
        for row in rows:
            cy += size * 1.32
            body.append(text(x + 38, round(cy - 2, 1), row, size, 700 if strong else 400, color if strong else TEXT))
        cy += 6
    h = cy - y + 10
    out = [rect(x, y, w, h, rx=12, fill=fill, stroke=color, stroke_width=2.5),
           f'<path d="M{x},{y + 12} a12,12 0 0 1 12,-12 h{w - 24} a12,12 0 0 1 12,12 v{head - 12} h{-w} z" fill="{color}"/>',
           badge(x + 30, y + head / 2, n, "white", r=16),
           text(x + 30, round(y + head / 2 + 8, 1), str(n), 22, 700, color, anchor="middle")]
    tw = text_width(period, 20, 700) + 14
    out.append(text(x + 58, round(y + head / 2 + 10, 1), title, 27, 700, "white"))
    out.append(pill(x + w - tw - 14, y + head / 2, period, "white", color, size=20)[0])
    return "\n".join(out + body), h


CARDS = [  # (n, side, anchor year, period, title, bullets)
    (1, "up", 2013.7, "2011.2–2016.4", "불법매립", [
        ("허위 성적서로 지정폐기물 둔갑 매립", False),
        ("차수막·침출수 처리시설 없음", False),
        ("2013·14년 하천 물고기 집단 폐사", True)]),
    (2, "down", 2016.6, "2016", "적발·조치명령", [
        ("5~6월 환경부 중앙환경감시단 적발", False),
        ("10월 익산시 실태조사·민관협의회", False),
        ("12월 원상복구 조치명령 (업체 6곳)", False)]),
    (3, "up", 2017.5, "2017", "구속기소·대량 유출", [
        ("5.30 4명 구속기소 (약 20만 t)", False),
        ("6.27 저수조 범람 → 논·배수로 오염", True),
        ("7.10 제방 붕괴, 1주 새 3회 유출", True),
        ("중신천·동북측 부지로 확산", True)]),
    (4, "down", 2019.5, "2018–2020", "규모 확인·처리 정체", [
        ("총 매립량 143만~156만 t 보도", False),
        ("배출업체 45곳 중 31곳 소송", False),
        ("처리율 1.9% (2019.4)", False),
        ("2019년 유출 8건 보고", True),
        ("1 km 내 약 100가구 지하수 음용", True)]),
    (5, "up", 2023.0, "2021–2024", "행정대집행", [
        ("2021 1차 대집행 4.9만 t", False),
        ("2022 2차 5.4만 t (109억 원)", False),
        ("2023.9 복구협약 (정부·주민·업체)", False),
        ("2024년까지 5차, 누적 약 28.9만 t", False)]),
    (6, "down", 2025.9, "2025–2026", "책임 확정·오염 차단", [
        ("2025.7 법원, 배출업체 책임 인정", False),
        ("차수매트·연직차수벽 설치", False),
        ("하부 토양·지하수 정화", False),
        ("2026 정밀 차수 현장실증 (본 과제)", True)]),
]


def slide_timeline():
    out = header("익산 낭산 폐석산 침출수 사태 연표", "언론 보도로 본 2011–2026년 — 불법매립 적발에서 오염 차단·현장실증까지")
    w, gap = 470, 25
    xs = [56, 56 + w + gap, 56 + 2 * (w + gap)]
    up = [c for c in CARDS if c[1] == "up"]
    down = [c for c in CARDS if c[1] == "down"]

    # phase band with year labels
    for (s, e, color, _), in zip(PHASES):
        out.append(rect(tx(s), BAND_Y, tx(e) - tx(s), BAND_H, fill=color))
    for yr in range(Y_FIRST, Y_LAST):
        x = tx(yr)
        out.append(path(f"M{x:.1f},{BAND_Y} L{x:.1f},{BAND_Y + BAND_H}", stroke="white", stroke_width=1.5, stroke_opacity=0.6))
        if yr <= 2026:
            out.append(text(round(x + 45, 1), BAND_Y + 30, str(yr), 20, 700, "white", anchor="middle"))

    # cards + leader lines
    for row, y_card in [(up, 176), (down, 566)]:
        for (n, side, year, period, title, bullets), x in zip(row, xs):
            color = PHASES[n - 1][2]
            svg, h = card(x, y_card, w, n, PHASES[n - 1], period, title, bullets)
            ax = tx(year)
            if side == "up":
                y0, y1 = y_card + h, BAND_Y
            else:
                y0, y1 = y_card, BAND_Y + BAND_H
            cx = min(max(ax, x + 40), x + w - 40)
            out.append(path(f"M{cx:.1f},{y0:.1f} L{ax:.1f},{y1:.1f}", stroke=color, stroke_width=3, fill="none"))
            out.append(f'<circle cx="{ax:.1f}" cy="{y1:.1f}" r="7" fill="{color}" stroke="white" stroke-width="2.5"/>')
            out.append(svg)

    # reported leachate releases / fish kills, just under the band
    for yr in RELEASES:
        x = tx(yr)
        y = BAND_Y + BAND_H + 4
        out.append(f'<path d="M{x:.1f},{y + 18:.1f} l-8,-14 h16 z" fill="#dc2626" stroke="white" stroke-width="1.5"/>')

    lx = 56
    out.append(f'<path d="M{lx + 8},{862} l-8,-14 h16 z" fill="#dc2626"/>')
    out.append(text(lx + 24, 862, "침출수 유출·하천 피해 보도 시점", 22, 400, MUTED))
    out.append(text(1552, 862, "수치는 언론 보도 기준이며 매체·시점에 따라 다름", 22, 400, MUTED, anchor="end"))
    return document("", "\n".join(out))


# --- slide 2: removal progress and issues -------------------------------------------
PROGRESS = [  # (label, cumulative 만 t, note)
    ("2019.4", 0.29, "처리율 1.9%"),
    ("2021", 4.9, "1차 대집행"),
    ("2022", 10.6, "2차 (109억 원)"),
    ("2024", 28.9, "5차 누적"),
]


def slide_progress():
    out = header("불법매립 폐기물 처리 진척과 반복된 쟁점", "처리는 2021년 행정대집행 이후 가속 — 매립층 대부분은 여전히 현장에 남아 침출수 공급원")

    # bar chart
    cx0, cy0, cw, ch = 140, 230, 600, 470          # plot area (bottom at cy0 + ch)
    base = cy0 + ch
    out.append(text(56, 205, "누적 처리량 (만 t)", 26, 700))
    top = 36
    for v in (0, 10, 20, 30):
        y = base - v / top * ch
        out.append(path(f"M{cx0},{y:.1f} L{cx0 + cw},{y:.1f}", stroke="#e5e7eb" if v else "#9ca3af", stroke_width=1.5))
        out.append(text(cx0 - 12, round(y + 8, 1), str(v), 22, 400, MUTED, anchor="end"))
    bw = 92
    for i, (label, val, note) in enumerate(PROGRESS):
        x = cx0 + 40 + i * (cw - 80) / (len(PROGRESS) - 1) - bw / 2
        hgt = max(val / top * ch, 3)
        color = "#475569" if i == 0 else "#1d4ed8"
        out.append(rect(x, base - hgt, bw, hgt, rx=4, fill=color))
        out.append(text(round(x + bw / 2, 1), round(base - hgt - 42, 1), f"{val:g}", 30, 700, color, anchor="middle"))
        out.append(text(round(x + bw / 2, 1), round(base - hgt - 12, 1), note, 22, 400, MUTED, anchor="middle"))
        out.append(text(round(x + bw / 2, 1), base + 34, label, 24, 700, TEXT, anchor="middle"))

    # scale of what is left
    out.append(rect(56, 770, 700, 92, rx=12, fill="#fef2f2", stroke="#b91c1c", stroke_width=2))
    out.append(text(76, 808, "전체 매립량(보도) 143만~156만 t", 26, 700, "#b91c1c"))
    out.append(text(76, 846, "누적 처리 약 29만 t ≈ 전체의 약 2할 → 나머지는 차수 대상", 24, 400, TEXT))

    # right column: recurring issues
    rx0, rw = 820, 732
    out.append(text(rx0, 205, "반복된 쟁점", 26, 700))
    issues = [
        ("침출수 관리 실패", "차수·처리시설 없이 시작 → 장마철 범람·제방 붕괴 반복"),
        ("책임 분산", "업체 40여 곳·정부·지자체 간 소송 (2025 판결로 정리)"),
        ("처리 속도·재원", "처리율 1~4%대 정체 → 2021년 이후 국비로 가속"),
        ("주민 건강", "지하수 음용 가구, 하천수 농업용수, 물고기 폐사"),
    ]
    y = 226
    for i, (head_, desc) in enumerate(issues, 1):
        out.append(badge(rx0 + 18, y + 18, i, "#475569", r=17, size=20))
        out.append(text(rx0 + 46, y + 28, head_, 27, 700))
        yy = y + 28
        for row in wrap(desc, rw - 46, 24):
            yy += 34
            out.append(text(rx0 + 46, yy, row, 24, 400, MUTED))
        y = yy + 20

    # implications for the project
    box_y = y + 8
    lines = [
        ("2017년 유출은 남쪽만이 아닌 \"여러 방면\"", False),
        ("→ 측방(동·서) 유출 경로 조사 필요", True),
        ("이적 대상은 고농도 지정폐기물 일부", False),
        ("→ 남은 매립층이 장기 침출수 공급원", True),
    ]
    bh = 62 + len(lines) * 36
    out.append(rect(rx0, box_y, rw, bh, rx=12, fill="#eff6ff", stroke="#1d4ed8", stroke_width=2.5))
    out.append(text(rx0 + 22, box_y + 42, "본 과제(정밀 차수·확산 방지)에 주는 시사점", 27, 700, "#1d4ed8"))
    yy = box_y + 44
    for row, arrow_row in lines:
        yy += 36
        out.append(text(rx0 + 22 + (16 if arrow_row else 0), yy, row, 24, 700 if arrow_row else 400,
                        "#1d4ed8" if arrow_row else TEXT))
    out.append(text(1552, 862, "출처: 2016–2026년 언론 보도 종합 (수치는 매체·시점별 차이)", 22, 400, MUTED, anchor="end"))
    return document("", "\n".join(out))


if __name__ == "__main__":
    save(slide_timeline(), OUT_DIR, "leachate_timeline_1")
    save(slide_progress(), OUT_DIR, "leachate_timeline_2")
