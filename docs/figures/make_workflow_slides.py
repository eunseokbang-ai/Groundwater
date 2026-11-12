"""Workflow slides (16:9) for presentations: drone acquisition -> processing ->
program analysis -> interpretation, with the program's rehearsal on real data (Ulsan
Bangudae petroglyph cliff) before Iksan. Large type throughout (body ~30 px on a
1600x900 canvas) so every word stays legible when projected. The Bangudae slides
embed real program output: run make_bangudae_figures.py first (data/bangudae/).

Run with the project venv from any directory:

    .venv\\Scripts\\python.exe docs\\figures\\make_workflow_slides.py

Items marked "개발 예정" are not implemented yet (Phase 2); keep them marked
until they are, so the slides never overstate what the program does.
"""
import io
import os
import sys
from pathlib import Path

import numpy as np

from svgkit import MUTED, TEXT, arrow, document, image, label, path, pill, rect, save, text, text_width, wrap

OUT_DIR = Path(__file__).resolve().parent
REPO = OUT_DIR.parents[1]

TITLE, SUB, BODY, SMALL, NOTE = 46, 26, 30, 24, 22
STAGES = {
    1: ("① 취득", "#1d4ed8", "#dbeafe"),
    2: ("② 처리", "#0f766e", "#ccfbf1"),
    3: ("③ 분석", "#c2410c", "#ffedd5"),
    4: ("④ 해석", "#7e22ce", "#f3e8ff"),
}
PLAN_FILL, PLAN_TEXT = "#f3f4f6", "#6b7280"
DONE_FILL, DONE_TEXT = "#dcfce7", "#15803d"
SENSOR = {"rgb": "#1d4ed8", "lidar": "#7c3aed", "thermal": "#dc2626"}


# --- shared pieces --------------------------------------------------------------
def frame(title, subtitle, page, total, stage=None, note=None):
    out = [rect(0, 0, 1600, 900, fill="white")]
    if stage:
        out.append(rect(0, 0, 14, 900, fill=STAGES[stage][1]))
    out.append(text(56, 82, title, TITLE, 700))
    if subtitle:
        out.append(text(56, 128, subtitle, SUB, 400, MUTED))
    out.append(path("M56,152 L1552,152", stroke="#e5e7eb", stroke_width=2))
    out.append(text(1552, 878, f"{page} / {total}", NOTE, 400, MUTED, anchor="end"))
    if note:
        out.append(text(56, 878, note, NOTE, 400, MUTED))
    if stage:
        x = 1552
        for s in sorted(STAGES, reverse=True):
            name, color, _ = STAGES[s]
            x -= text_width(name, 20, 700) + 14
            out.append(pill(x, 62, name, color if s == stage else "#f3f4f6", "white" if s == stage else "#9ca3af", size=20)[0])
            x -= 10
    return out


def plan_pill(x, y, s="개발 예정", size=20):
    return pill(x, y, s, PLAN_FILL, PLAN_TEXT, size=size, stroke="#9ca3af", dash="4 3")


def bullets(x, y, items, width, color, size=BODY, gap=16):
    """items: [(text, status)] with status None / 'done' / 'plan'. Returns (svg, height)."""
    out, cy = [], y
    for s, status in items:
        planned = status == "plan"
        tag_w = text_width("개발 예정", 20, 700) + 24 if planned else 0
        rows = wrap(s, width - size - tag_w, size)
        mid = cy + size * 0.62
        r = size * 0.22
        if planned:
            out.append(f'<circle cx="{x + r}" cy="{mid:.1f}" r="{r:.1f}" fill="white" stroke="#9ca3af" stroke-width="3"/>')
        else:
            out.append(f'<circle cx="{x + r}" cy="{mid:.1f}" r="{r:.1f}" fill="{DONE_TEXT if status == "done" else color}"/>')
        for i, row in enumerate(rows):
            out.append(text(round(x + size * 0.9, 1), round(cy + size * (0.95 + 1.3 * i), 1), row, size, 400,
                            PLAN_TEXT if planned else TEXT))
        if planned:
            last = text_width(rows[-1], size)
            out.append(plan_pill(x + size * 0.9 + last + 12, cy + size * (0.6 + 1.3 * (len(rows) - 1)))[0])
        cy += size * 1.3 * len(rows) + gap
    return "\n".join(out), cy - y


def card(x, y, w, h, header, color, fill="white", header_h=66, size=32, dashed=False):
    r = 14
    stroke = "#9ca3af" if dashed else color
    head = "#9ca3af" if dashed else color
    return "\n".join([
        rect(x, y, w, h, rx=r, fill=fill, stroke=stroke, stroke_width=2.5, stroke_dasharray="8 5" if dashed else None),
        path(f"M{x},{y + r} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r} a{r},{r} 0 0 1 {r},{r} v{header_h - r} h-{w} z", fill=head),
        text(x + 22, round(y + header_h / 2 + size * 0.36, 1), header, size, 700, "white"),
    ])


def point_box(x, y, w, h, title, sub, color):
    """Rounded box with a coloured left bar, a bold title and one muted line."""
    out = [rect(x, y, w, h, rx=12, fill="white", stroke="#e5e7eb", stroke_width=2), rect(x, y, 12, h, fill=color)]
    out.append(text(x + 34, y + h / 2 - 6, title, BODY + 2, 700, color))
    out.append(text(x + 34, y + h / 2 + 34, sub, SMALL, 400, MUTED))
    return "\n".join(out)


def drone(x, y, s=1.0, color="#1f2937"):
    return "\n".join([
        path(f"M{x - 22 * s},{y} L{x + 22 * s},{y}", stroke=color, stroke_width=2.5 * s),
        f'<ellipse cx="{x - 22 * s}" cy="{y - 4 * s}" rx="{11 * s}" ry="{3 * s}" fill="{color}" fill-opacity="0.55"/>',
        f'<ellipse cx="{x + 22 * s}" cy="{y - 4 * s}" rx="{11 * s}" ry="{3 * s}" fill="{color}" fill-opacity="0.55"/>',
        rect(x - 9 * s, y - 5 * s, 18 * s, 11 * s, rx=3 * s, fill=color),
        f'<circle cx="{x}" cy="{y + 9 * s}" r="{3.5 * s}" fill="{color}"/>',
    ])


def banner(x, y, s, fg="#9a3412", bg="#ffedd5", stroke="#fdba74", size=SMALL):
    return label(x, y, [(s, size, 700, fg)], pad=10, fill=bg, stroke=stroke, opacity=1)


def slide_defs():
    return """
  <pattern id="net" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <path d="M0 0 H12 M0 0 V12" stroke="#2f9e44" stroke-width="1.6"/></pattern>
  <pattern id="rockTex" width="48" height="36" patternUnits="userSpaceOnUse">
    <rect width="48" height="36" fill="#b3aca2"/><path d="M5 10 l14 -4 M26 27 l12 -5 M36 7 l7 6" stroke="#9c958b" stroke-width="1.5"/></pattern>
"""


def stage_icon(stage, cx, cy):
    """Illustration for the bottom of each overview card, drawn around (cx, cy)."""
    out = []
    if stage == 1:
        out.append(rect(cx - 130, cy - 70, 48, 140, fill="url(#rockTex)", stroke="#6b6259"))
        out.append(path(f"M{cx + 40},{cy} L{cx - 82},{cy - 52} L{cx - 82},{cy + 52} Z", fill="#1d4ed8", fill_opacity=0.12,
                        stroke="#1d4ed8", stroke_width=1.5))
        out.append(drone(cx + 62, cy - 6, 1.9, "#1e293b"))
    elif stage == 2:
        rng = np.random.default_rng(3)
        for k, x in enumerate([cx - 138, cx - 44, cx + 50]):
            out.append(rect(x, cy - 44, 88, 88, rx=8, fill="#f8fafc", stroke="#94a3b8", stroke_width=1.5))
            if k == 0:
                for px, py in rng.uniform([x + 8, cy - 36], [x + 80, cy + 36], size=(80, 2)):
                    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="#78716c"/>')
            elif k == 1:
                pts = [(x + 10 + 23 * i, cy - 34 + 23 * j + (8 if (i + j) % 2 else 0)) for j in range(4) for i in range(4)]
                for j in range(3):
                    for i in range(3):
                        a, b, c, d = pts[j * 4 + i], pts[j * 4 + i + 1], pts[(j + 1) * 4 + i], pts[(j + 1) * 4 + i + 1]
                        out.append(path(f"M{a[0]},{a[1]} L{b[0]},{b[1]} L{d[0]},{d[1]} Z M{a[0]},{a[1]} L{c[0]},{c[1]} L{d[0]},{d[1]}",
                                        fill="none", stroke="#0f766e", stroke_width=1.4))
            else:
                ramp = ["#1e3a8a", "#2563eb", "#22c55e", "#facc15", "#f97316", "#dc2626"]
                for j in range(5):
                    for i in range(5):
                        out.append(rect(x + 7 + i * 15, cy - 37 + j * 15, 14, 14, fill=ramp[(i + 2 * j + i * j) % 6]))
    elif stage == 3:
        r = 66
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="#6b7280" stroke-width="2"/>')
        out.append(path(f"M{cx - r},{cy} L{cx + r},{cy} M{cx},{cy - r} L{cx},{cy + r}", stroke="#e5e7eb", stroke_width=1.5))
        rng = np.random.default_rng(5)
        for (ox, oy), color in [((-28, -24), "#1f77b4"), ((30, 12), "#2ca02c"), ((-10, 36), "#9467bd")]:
            for dx, dy in rng.normal(0, 6, size=(14, 2)):
                out.append(f'<circle cx="{cx + ox + dx:.1f}" cy="{cy + oy + dy:.1f}" r="3" fill="{color}"/>')
        out.append(path(f"M{cx - r + 5},{cy + 16} Q{cx},{cy - 58} {cx + r - 5},{cy - 20}", fill="none", stroke="#d62728", stroke_width=2.4))
    else:
        out.append(rect(cx - 120, cy - 68, 140, 136, fill="url(#rockTex)", stroke="#6b6259"))
        out.append(path(f"M{cx - 108},{cy - 54} L{cx + 8},{cy + 50}", stroke="#dc2626", stroke_width=4))
        for t in (0.2, 0.5, 0.8):
            hx, hy = cx - 108 + 116 * t, cy - 54 + 104 * t
            out.append(f'<circle cx="{hx:.0f}" cy="{hy:.0f}" r="8" fill="white" stroke="#7e22ce" stroke-width="3"/>')
            out.append(arrow(f"M{cx + 110},{hy:.0f} L{hx + 16:.0f},{hy:.0f}", "#7e22ce", 3, marker="gray"))
    return "\n".join(out)


# --- slides -----------------------------------------------------------------------
def s_overview(n, total):
    out = frame("드론 기반 암사면 분석 워크플로", "WBS 2-1 불연속 절리대 파악  →  2-4 타겟 그라우팅 설계 입력", n, total,
                note="③ 구현 완료 · 반구대 암각화 암벽 실자료로 사전 검증 · 라이다·열화상 융합은 개발 예정")
    cols = [
        (1, "① 자료 취득", ["전 둘레 근접비행", "RGB·라이다·열화상", "RTK·현장 실측"]),
        (2, "② 자료처리", ["3차원 점군·메시", "정사영상·온도 영상", "좌표계 정합", "과거 영상·DEM"]),
        (3, "③ 프로그램 분석", ["절리군·트레이스", "유출 가능성 지도", "그라우팅 설계", "반구대 사전 검증"]),
        (4, "④ 해석·활용", ["활성 물길 후보", "타겟 그라우팅 설계", "시공 전후 검증"]),
    ]
    w, gap, x0, y0, h = 343, 44, 56, 180, 540
    for i, (stage, header, items) in enumerate(cols):
        x = x0 + i * (w + gap)
        color = STAGES[stage][1]
        out.append(card(x, y0, w, h, header, color))
        svg, _ = bullets(x + 22, y0 + 92, [(s, None) for s in items], w - 34, color)
        out.append(svg)
        out.append(stage_icon(stage, x + w / 2, y0 + h - 100))
        if i < 3:
            out.append(arrow(f"M{x + w + 8},{y0 + h / 2} L{x + w + gap - 12},{y0 + h / 2}", "#374151", 6, marker="gray"))
    left, right = x0 + w / 2, x0 + 3 * (w + gap) + w / 2
    out.append(arrow(f"M{right},{y0 + h + 6} V800 H{left} V{y0 + h + 12}", "#475569", 4, dash="12 8", marker="gray"))
    out.append(label(800, 776, [("반복 촬영: 강우 전후 · 그라우팅 시공 전후", 28, 700, "#334155")], anchor="middle", pad=10, opacity=1))
    return document(slide_defs(), "\n".join(out))


def s_flight(n, total):
    blue = SENSOR["rgb"]
    out = frame("① 입면 근접비행 방식", "사람이 접근하기 어려운 수직 암벽을 정면에서 일정 거리로 촬영", n, total, stage=1)
    out.append(path("M56,196 L230,196 L240,280 L232,360 L246,440 L238,520 L250,600 L244,680 L256,780 L56,780 Z",
                    fill="url(#rockTex)", stroke="#6b6259", stroke_width=2))
    out.append(rect(256, 772, 640, 16, fill="#1d1d20"))
    out.append(path("M282,212 L282,766", stroke="#2f9e44", stroke_width=4, stroke_dasharray="3 6"))
    out.append(path("M700,500 L262,380 L262,620 Z", fill=blue, fill_opacity=0.12, stroke=blue, stroke_width=2))
    out.append(path("M700,330 L262,330 L262,520 Z", fill="none", stroke=blue, stroke_width=2, stroke_dasharray="8 6"))
    out.append(path("M700,670 L262,480 L262,700 Z", fill="none", stroke=blue, stroke_width=2, stroke_dasharray="8 6"))
    for yy, stop in [(410, 250), (455, 282), (500, 250), (545, 282), (590, 250)]:
        out.append(path(f"M688,502 L{stop},{yy}", stroke="#7c3aed", stroke_width=2, stroke_opacity=0.85))
        out.append(f'<circle cx="{stop}" cy="{yy}" r="5" fill="#7c3aed"/>')
    for yy in (330, 500, 670):
        out.append(drone(720, yy, 1.7))
    out.append(path("M262,218 L690,218", stroke=TEXT, stroke_width=2.5, marker_start="url(#ah-gray)", marker_end="url(#ah-gray)"))
    out.append(label(476, 170, [("D ≈ 20 m", 30, 700, TEXT)], anchor="middle", pad=6, opacity=1))
    out.append(text(296, 262, "방지망", SMALL, 700, "#2f7d32"))
    out.append(label(770, 540, [("라이다", SMALL, 700, "#7c3aed")], pad=6, opacity=1))

    boxes = [("벽면과 일정 거리 유지", "이격 D ≈ 20 m, 카메라 수평", blue),
             ("정면 + 경사 ±20° 촬영", "오버행·절리면 음영 보완", blue),
             ("사진 중복 80% / 70%", "진행 방향 / 비행 스트립 사이", blue)]
    for i, (t, s, c) in enumerate(boxes):
        out.append(point_box(950, 190 + i * 200, 602, 170, t, s, c))
    return document(slide_defs(), "\n".join(out))


def s_site(n, total):
    out = frame("① 현장 여건 대응", "낙석방지망·식생·붕락면이 있는 벽면에서 자료 품질 확보", n, total, stage=1)
    crest = "M56,214 L170,196 L290,206 L400,188 L540,200 L670,186 L800,198 L930,184 L1000,194 L1000,720 L56,720 Z"
    out.append(path(crest, fill="url(#rockTex)", stroke="#6b6259", stroke_width=2))
    for d in ["M110,230 L300,370", "M360,210 L580,390", "M640,200 L860,380", "M740,380 L960,560", "M130,420 L360,560",
              "M470,300 L476,700", "M900,210 L910,480", "M250,210 L258,380"]:
        out.append(path(d, stroke="#4b433c", stroke_width=3))
    for poly in ["M80,222 L390,196 L400,380 L86,406 Z", "M660,192 L930,186 L940,340 L668,356 Z"]:
        out.append(path(poly, fill="#8fd694", fill_opacity=0.2))
        out.append(path(poly, fill="url(#net)"))
    for cx, cy in [(500, 236), (530, 250), (840, 420), (866, 408), (200, 520), (600, 470)]:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#4d8f32" fill-opacity="0.9"/>')
    out.append(path("M720,720 Q750,610 830,596 Q910,610 940,720 Z", fill="#d8d1c6", stroke="#8a8175", stroke_width=2, stroke_dasharray="6 4"))
    for dx in range(0, 200, 18):
        out.append(rect(730 + dx, 704 - (dx % 3) * 5, 13, 10, rx=2, fill="#8a8175"))
    out.append(rect(56, 720, 944, 18, fill="#1d1d20"))
    for y in (300, 450, 600):
        out.append(path(f"M72,{y} L984,{y}", stroke=SENSOR["rgb"], stroke_width=2.5, stroke_dasharray="12 9"))
    for dx, dy in [(230, 300), (720, 300), (330, 450), (820, 450), (560, 600)]:
        out.append(drone(dx, dy, 1.1, "#1e293b"))
    out.append(label(92, 330, [("낙석방지망", 26, 700, "#2f7d32")], pad=7, opacity=0.95))
    out.append(label(560, 500, [("식생", 26, 700, "#2f7d32")], pad=7, opacity=0.95))
    out.append(label(640, 640, [("붕락면", 26, 700, "#7c2d12")], pad=7, opacity=0.95))

    items = [("방지망", "라이다 다중반사로 일부 투과", "마스킹 개발 예정"),
             ("식생", "평면성 필터로 자동 제외", None),
             ("붕락면", "근접 추가 촬영 · 고신뢰 절리면", None),
             ("가려진 구간", "추정 없이 '평가 불가' 표시", "개발 예정")]
    y = 186
    for title, desc, tag in items:
        out.append(text(1040, y + 34, title, BODY + 2, 700))
        if tag:
            out.append(plan_pill(1040 + text_width(title, BODY + 2, 700) + 14, y + 24, tag)[0])
        for k, row in enumerate(wrap(desc, 500, SMALL + 2)):
            out.append(text(1040, y + 76 + k * 34, row, SMALL + 2, 400, MUTED))
        y += 138
    return document(slide_defs(), "\n".join(out))


def sensor_icon(kind, cx, cy):
    if kind == "rgb":
        return "\n".join([rect(cx - 60, cy - 38, 120, 80, rx=12, fill="#1e293b"), rect(cx - 30, cy - 50, 44, 16, rx=4, fill="#1e293b"),
                          f'<circle cx="{cx}" cy="{cy + 2}" r="27" fill="#93c5fd" stroke="white" stroke-width="5"/>'])
    if kind == "lidar":
        out = [rect(cx - 34, cy - 44, 68, 88, rx=16, fill="#4c1d95")]
        for dy in (-30, -12, 6, 24):
            out.append(path(f"M{cx + 34},{cy} L{cx + 110},{cy + dy * 1.6}", stroke="#a78bfa", stroke_width=3))
        return "\n".join(out)
    ramp = ["#1e3a8a", "#2563eb", "#22c55e", "#facc15", "#f97316", "#dc2626"]
    return "\n".join(rect(cx - 66 + i * 22, cy - 44, 22, 88, fill=c) for i, c in enumerate(ramp))


def s_sensors(n, total):
    out = frame("① 센서별 촬영 설정", "같은 벽면을 세 가지 센서로 촬영해 서로 보완", n, total, stage=1,
                note="GSD: 이격 20 m 기준 계산값 (P1 4.4 µm·35 mm, H20T 12 µm·13.5 mm)")
    cards = [("rgb", "RGB 사진측량", "Zenmuse P1 (35 mm)", [("GSD ≈ 2.5 mm", 700), ("중복 80% / 70%", 400), ("RTK 촬영", 400)]),
             ("lidar", "라이다", "DJI L3 / Yellowscan Vx15", [("모든 반사 기록", 700), ("방지망 뒤 암면 복원", 400), ("벽면 정면 비행", 400)]),
             ("thermal", "열화상", "DJI H20T (640×512)", [("GSD ≈ 1.8 cm", 700), ("일출 전·흐린 날", 400), ("강우 후 재촬영", 400)])]
    for i, (kind, name, model, rows) in enumerate(cards):
        x, color = 56 + i * 512, SENSOR[kind]
        out.append(card(x, 180, 472, 640, name, color))
        out.append(text(x + 22, 290, model, SMALL, 700, MUTED))
        for k, (s, wgt) in enumerate(rows):
            out.append(f'<circle cx="{x + 30}" cy="{350 + k * 62}" r="7" fill="{color}"/>')
            out.append(text(x + 50, 361 + k * 62, s, BODY, wgt))
        out.append(sensor_icon(kind, x + 236, 700))
    return document(slide_defs(), "\n".join(out))


def s_thermal(n, total):
    red = SENSOR["thermal"]
    out = frame("① 열화상 촬영 시각 · 현장 보조자료", "열화상은 촬영 시각이 결과를 좌우 — 보조자료는 해석의 기준", n, total, stage=1)
    gx0, gx1, gy0, gy1 = 90, 880, 240, 610
    out.append(text(56, 210, "암면 vs 지하수 온도 (개념)", BODY, 700, red))
    out.append(rect(gx0 + (gx1 - gx0) * 3.5 / 24, gy0, (gx1 - gx0) * 2.5 / 24, gy1 - gy0, fill="#fee2e2"))
    out.append(rect(gx0 + (gx1 - gx0) * 11 / 24, gy0, (gx1 - gx0) * 5 / 24, gy1 - gy0, fill="#f3f4f6"))
    out.append(path(f"M{gx0},{gy1} L{gx1},{gy1} M{gx0},{gy0} L{gx0},{gy1}", stroke="#9ca3af", stroke_width=2))
    tt = np.linspace(0, 24, 97)
    mid = (gy0 + gy1) / 2 + 10
    pts = [(gx0 + (gx1 - gx0) * t / 24, mid - 130 * v) for t, v in zip(tt, np.sin(2 * np.pi * (tt - 8) / 24))]
    out.append(path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts), fill="none", stroke="#b45309", stroke_width=5))
    out.append(path(f"M{gx0},{mid + 24} L{gx1},{mid + 24}", stroke="#1d6fc2", stroke_width=5, stroke_dasharray="14 9"))
    for h in (0, 6, 12, 18, 24):
        out.append(text(gx0 + (gx1 - gx0) * h / 24, gy1 + 34, f"{h}시", NOTE, 400, MUTED, anchor="middle"))
    out.append(text(gx0 + (gx1 - gx0) * 4.75 / 24, gy0 + 32, "촬영 적기", SMALL, 700, "#b91c1c", anchor="middle"))
    out.append(text(gx0 + (gx1 - gx0) * 13.5 / 24, gy1 - 14, "일사 영향 큼", NOTE, 700, MUTED, anchor="middle"))
    out.append(path("M90,700 L140,700", stroke="#b45309", stroke_width=5))
    out.append(text(152, 708, "암면 표면온도", SMALL, 700, "#b45309"))
    out.append(path("M400,700 L450,700", stroke="#1d6fc2", stroke_width=5, stroke_dasharray="12 7"))
    out.append(text(462, 708, "지하수 수온 (거의 일정)", SMALL, 700, "#1d6fc2"))
    out.append(text(56, 780, "새벽: 일사 영향 없음 → 유출부 온도차 뚜렷", SMALL + 2, 700))
    out.append(text(56, 818, "(계절에 따라 유출부가 따뜻하게 또는 차갑게 보임)", NOTE, 400, MUTED))

    out.append(rect(940, 180, 612, 640, rx=14, fill="#eff6ff", stroke="#bfdbfe", stroke_width=2))
    out.append(text(972, 236, "현장 보조자료", BODY + 2, 700, SENSOR["rgb"]))
    items = ["RTK·GCP (RGB·열화상 겸용 표적)", "수온·기온·습도·방사율 기록", "현장 절리 실측 (스캔라인·시추공)", "비행 시각·기상·강우 이력",
             "DEM 습윤지수로 촬영 지점 선정"]
    y = 270
    for s in items:
        out.append(path(f"M972,{y + 22} l12,12 l22,-26", fill="none", stroke=DONE_TEXT, stroke_width=5, stroke_linecap="round"))
        rows = wrap(s, 510, SMALL + 4)
        for k, row in enumerate(rows):
            out.append(text(1022, y + 36 + k * 38, row, SMALL + 4))
        y += 38 * len(rows) + 44
    return document(slide_defs(), "\n".join(out))


def s_processing(n, total):
    teal = STAGES[2][1]
    out = frame("② 자료처리 흐름", "센서별 원자료를 상용 소프트웨어로 처리", n, total, stage=2,
                note="소프트웨어는 예시 — 보유 SW로 대체 가능")
    lanes = [("RGB 사진", "Zenmuse P1", SENSOR["rgb"], ["SfM 정렬 · GCP", "점군·메시·정사영상"], "Pix4D · Metashape", "LAS/PLY · OBJ · GeoTIFF"),
             ("라이다", "DJI L3 / Vx15", SENSOR["lidar"], ["궤적 처리 (RTK·IMU)", "점군 생성·노이즈 제거"], "DJI Terra 등", "LAS/LAZ 점군"),
             ("열화상", "DJI H20T", SENSOR["thermal"], ["R-JPEG → 온도(°C)", "방사율·기온·습도 보정"], "DJI Thermal SDK 등", "온도 영상 (°C)")]
    for i, (name, sensor, color, steps, sw, product) in enumerate(lanes):
        cy = 250 + i * 200
        out.append(rect(56, cy - 60, 240, 120, rx=12, fill=color))
        out.append(text(176, cy - 4, name, BODY + 2, 700, "white", anchor="middle"))
        out.append(text(176, cy + 34, sensor, NOTE, 400, "#e5e7eb", anchor="middle"))
        x = 340
        for j, step in enumerate(steps):
            out.append(rect(x, cy - 55, 340, 110, rx=12, fill="white", stroke=color, stroke_width=2.5))
            rows = wrap(step, 310, SMALL + 2, 700)
            for k, row in enumerate(rows):
                out.append(text(x + 170, round(cy + 10 + (k - (len(rows) - 1) / 2) * 34, 1), row, SMALL + 2, 700, anchor="middle"))
            out.append(arrow(f"M{x - 38},{cy} L{x - 8},{cy}", "#6b7280", 4, marker="gray"))
            x += 384
        out.append(pill(340, cy + 78, sw, "#f3f4f6", "#374151", size=20)[0])
        out.append(arrow(f"M{x - 38},{cy} L{x - 8},{cy}", "#6b7280", 4, marker="gray"))
        out.append(rect(x, cy - 55, 1552 - x, 110, rx=12, fill=STAGES[2][2], stroke=teal, stroke_width=2.5))
        out.append(text(x + 20, cy - 12, "산출물", NOTE, 700, teal))
        out.append(text(x + 20, cy + 30, product, SMALL, 700))
    return document(slide_defs(), "\n".join(out))


def s_registration(n, total):
    teal, orange = STAGES[2][1], STAGES[3][1]
    out = frame("② 좌표 정합 · 품질관리", "모든 자료를 하나의 좌표계로 맞춘 뒤 분석 프로그램에 입력", n, total, stage=2)
    out.append(card(56, 176, 728, 380, "좌표 정합", teal))
    svg, _ = bullets(80, 262, [("RTK·GCP 공통 좌표계", None), ("벽면 평행 정사영상·DSM ↔ 점군", "done"),
                               ("라이다 ↔ 사진 점군", "plan"), ("열화상 ↔ RGB", "plan")], 680, teal, gap=10)
    out.append(svg)
    out.append(card(824, 176, 728, 380, "품질관리 (QC)", teal))
    svg, _ = bullets(848, 270, [("점 간격이 충분히 조밀한가", None), ("정합 잔차 확인", None), ("방지망·식생 구간 기록", None)], 680, teal)
    out.append(svg)
    out.append(arrow("M800,566 L800,612", orange, 5, marker="gray"))
    out.append(rect(56, 624, 1496, 196, rx=14, fill="#fff7ed", stroke=orange, stroke_width=2.5))
    out.append(text(84, 674, "→ ③ 프로그램 입력", BODY + 2, 700, orange))
    x, y = 84, 740
    for s, planned in [("현장 폴더 (자동 인식)", False), ("점군 LAS · 메시 · 타일 모델", False), ("벽면 평행 영상·DSM", False),
                       ("현장 실측 CSV", False), ("라이다·열화상", True)]:
        svg, w = pill(x, y, s, PLAN_FILL if planned else "white", PLAN_TEXT if planned else TEXT, size=SMALL + 2,
                      stroke="#9ca3af" if planned else orange, dash="5 4" if planned else None)
        out.append(svg)
        x += w + 20
    return document(slide_defs(), "\n".join(out))


def s_pipeline(n, total):
    orange = STAGES[3][1]
    out = frame("③ 프로그램 분석 파이프라인", "점군에서 절리를 찾아 물길 후보와 그라우팅 설계까지 한 번에", n, total, stage=3,
                note="검증: 자동 테스트 58종 · 합성 벽면 방향 오차 < 1° · 반구대 실자료(6.5억 점) 전 과정 시험")
    steps = [("입력·전처리", "현장 폴더 자동 인식 · 식생 제외", "done"), ("법선·평면성 계산", "점 간격으로 크기 자동 설정", "done"),
             ("절리군 탐지", "스테레오넷 밀도 피크", "done"), ("절리면 분리·평면 적합", "경사·간격·연장성", "done"),
             ("벽면 좌표·입면 영상", "측점·표고로 전개 (곡선 벽)", "done"),
             ("입면 트레이스", "사진·DEM 6종 검출 · 교차검증", "done"),
             ("유출 가능성·안정성", "지표별 물길 지도 · 운동학 검토", "done"),
             ("그라우팅 설계", "절리망 기반 표적 천공 · 3D 시공", "done"),
             ("현장 실측 검증", "절리 방향 각도 오차", "done"), ("라이다·열화상 융합", "마스킹·열화상 이상 탐지", "plan")]
    for i, (title, desc, status) in enumerate(steps):
        col, row = divmod(i, 5)
        x, y = 56 + col * 760, 176 + row * 132
        planned = status == "plan"
        color = PLAN_TEXT if planned else orange
        if row < 4:
            out.append(path(f"M{x + 32},{y + 64} L{x + 32},{y + 132}", stroke="#d1d5db", stroke_width=4))
        dash = ' stroke-dasharray="6 4"' if planned else ""
        out.append(f'<circle cx="{x + 32}" cy="{y + 32}" r="30" fill="{"white" if planned else orange}" stroke="{color}" stroke-width="3"{dash}/>')
        out.append(text(x + 32, y + 43, str(i + 1), 30, 700, color if planned else "white", anchor="middle"))
        out.append(text(x + 84, y + 42, title, BODY + 2, 700, PLAN_TEXT if planned else TEXT))
        tag, fill, fg, stroke, dsh = (("개발 예정", PLAN_FILL, PLAN_TEXT, "#9ca3af", "4 3") if planned
                                     else ("구현", DONE_FILL, DONE_TEXT, None, None))
        out.append(pill(x + 84 + text_width(title, BODY + 2, 700) + 14, y + 32, tag, fill, fg, size=20, stroke=stroke, dash=dsh)[0])
        out.append(text(x + 84, y + 88, desc, SMALL + 2, 400, PLAN_TEXT if planned else MUTED))
    return document(slide_defs(), "\n".join(out))


BANG = OUT_DIR / "data" / "bangudae"
SET_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]        # Set 0, 1, 2 as in the program (groundwater.report.SET_COLORS)
ULSAN = REPO / "Filed_Data" / "Ulsan" / "분석결과"
VERIFY_BANNER = ("실제 현장 자료 — 울산 반구대 암각화 암벽 (익산 적용 전 사전 연습)", "#166534", "#dcfce7", "#86efac")


def picture(path, width=1400, crop=None, fmt="JPEG"):
    """An image file, optionally cropped (fractions l, t, r, b), resized for embedding."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
    im = im.resize((width, int(width * im.height / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, fmt, quality=88)
    return buf.getvalue(), ("image/jpeg" if fmt == "JPEG" else "image/png")


def embed(x, y, w, h, pic):
    import base64

    data, mime = pic
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" '
            f'href="{uri}" xlink:href="{uri}"/>')


def panel(x, y, w, h, pic, caption=None):
    out = [rect(x, y, w, h, rx=12, fill="white", stroke="#e5e7eb", stroke_width=2), embed(x + 8, y + 8, w - 16, h - 16, pic)]
    if caption:
        out.append(text(x + w / 2, y + h + 36, caption, SMALL + 2, 700, anchor="middle"))
    return "\n".join(out)


def verify_banner(x=56, y=166):
    s_, fg, bg, st = VERIFY_BANNER
    return banner(x, y, s_, fg=fg, bg=bg, stroke=st)


def s_bang_intro(n, total):
    orange = STAGES[3][1]
    out = frame("③ 사전 검증 — 반구대 암각화 암벽", "익산 적용 전, 실제 드론 자료로 프로그램 전 과정을 시험", n, total, stage=3,
                note="자료: 드론 사진측량 (점군 LAS · 벽면 평행 정사영상·DSM · 메시 OBJ · 타일 모델 3TZ)")
    out.append(verify_banner())
    out.append(panel(56, 224, 940, 440, picture(ULSAN / "facade_preview.jpg", 1600, crop=(0.0, 0.05, 1.0, 1.0))))
    out.append(text(526, 700, "벽면을 따라 펼친 입면 영상 (측점 0~78 m, 표고 76~106 m)", SMALL + 2, 700, anchor="middle"))
    out.append(panel(56, 724, 260, 116, picture(ULSAN / "암각화_실측도_정사영상.png", 600, crop=(0.12, 0.05, 0.95, 0.75))))
    out.append(text(336, 770, "현장 특화 기능: 암각화 실측도를 정사영상·", SMALL, 700, orange))
    out.append(text(336, 806, "DSM·타일 모델 제자리에 겹침 (수 cm 정합)", SMALL, 400, TEXT))
    out.append(side_items(1030, 228, 522, [
        ("왜 반구대인가", "수직 암벽 · 식생 · 오버행 · 넓은 틈 — 익산과 비슷한 어려운 조건"),
        ("시험 자료", "점군 6.5억 점 (점 간격 0.3 cm) · 벽면 평행 정사영상·DSM 2.83 mm"),
        ("시험 범위", "입력 → 절리·트레이스 → 유출 가능성 → 그라우팅 설계 → 3차원 가상 시공"),
        ("처리 성능", "6.5억 점 전체 분석 4.5분 · 화면 반응 0.1~0.3초"),
    ], orange))
    return document(slide_defs(), "\n".join(out))


def s_bang_joints(n, total):
    import json

    out = frame("③ 사전 검증 — 절리군 · 입면 트레이스", "3차원 절리면과 입면 트레이스가 서로 맞는지 확인", n, total, stage=3)
    out.append(verify_banner())
    # left: the poles and their density map (the dashboard's stereonet options), side by side
    peak = json.loads((BANG / "stereonet_density.json").read_text(encoding="utf-8"))["max_sigma"]
    out.append(panel(56, 224, 292, 292, picture(BANG / "stereonet.png", 600, fmt="PNG")))
    out.append(panel(364, 224, 292, 292, picture(BANG / "stereonet_density.png", 600, fmt="PNG")))
    out.append(text(202, 552, "극점 (절리면마다 한 점)", SMALL, 700, anchor="middle"))
    out.append(text(510, 552, "극점 밀도 지도", SMALL, 700, anchor="middle"))
    out.append(text(56, 590, f"밀도 지도: Kamb σ3 평활, 진할수록 극점이 모임 (최대 {peak}σ)", NOTE, 400, MUTED))
    rows = [("Set 0  15°/296°", "층리 (완경사)"), ("Set 1  78°/292°", "급경사 절리"), ("Set 2  89°/152°", "벽면과 평행")]
    for k, (a, b) in enumerate(rows):
        out.append(f'<circle cx="{78}" cy="{650 + k * 42}" r="10" fill="{SET_COLORS[k]}"/>')
        out.append(text(100, 660 + k * 42, a, SMALL + 2, 700))
        out.append(text(330, 660 + k * 42, b, SMALL + 2, 400, MUTED))
    out.append(text(56, 820, "밀도 지도: 세 절리군이 뚜렷한 세 집중으로 분리", SMALL, 700, DONE_TEXT))
    # right: the facade traces
    out.append(panel(676, 224, 876, 470, picture(BANG / "traces.png", 1700)))
    out.append(text(676, 744, "트레이스 577개 (414 m) → 방향군 2개, 연결된 절리군 색", SMALL + 2, 700))
    out.append(text(676, 786, "F1: Set 1·2가 벽면에서 같은 각도(115°) → 'Set 1 또는 2'", SMALL + 2, 400, MUTED))
    return document(slide_defs(), "\n".join(out))


def s_bang_structure(n, total):
    import json

    from PIL import Image

    orange = STAGES[3][1]
    info = json.loads((BANG / "major_planes.json").read_text(encoding="utf-8"))
    u0, u1 = info["window"]
    out = frame("③ 사전 검증 — 주요 절리면 구조 (암반 내부)", "절리군마다 나란히 놓인 여러 절리면을 암반 블록 끝까지 연장 — 거시 구조와 교선",
                n, total, stage=3,
                note=f"측점 {u0:.0f}~{u1:.0f} m · 암면 뒤 {info['depth']:.0f} m 블록 · 실제 평면 좌표 · 위치·방향 = 관측, "
                     "블록 끝까지 연장 = 해석 · 대시보드 '단면으로 보기'와 같은 그림")
    out.append(verify_banner())
    out.append(panel(56, 224, 508, 616, picture(BANG / "major_planes.png", 1000, fmt="PNG")))
    # sections A, B, C side by side, each image at its own size (sized to its shape)
    x, top = 576, 224
    for k in ("a", "b", "c"):
        f = BANG / f"major_sec_{k}.png"
        w, h = (v // 2 for v in Image.open(f).size)
        out.append(panel(x, top, w + 16, h + 16, picture(f, 2 * w, fmt="PNG")))
        x += w + 16 + 12
    from collections import Counter

    per = Counter(p["name"].split("-")[0] for p in info["planes"])
    ev = " · ".join(f"{k} {v}" for k, v in sorted(per.items())) + "개"
    items = [("나란한 주요 절리면", f"{ev} — 근거 {info['min_length_m']:g} m 이상, {info['min_sep_m']:g} m 이상 간격"),
             ("교선 (검은 선)", "층리 × 급경사 절리 → 선형 물길 후보 · 시추·ERT로 확인"),
             ("단면 A·B·C", "파선·글자표 = 다른 단면이 이 단면을 자르는 곳")]
    bx, by, bw = 576, 662, (1552 - 576 - 2 * 14) / 3
    for t_, d_ in items:
        out.append(rect(bx, by, bw, 178, rx=12, fill="white", stroke="#e5e7eb", stroke_width=2))
        out.append(rect(bx, by, 10, 178, fill=orange))
        out.append(text(bx + 28, by + 40, t_, BODY - 2, 700, orange))
        for r_, row in enumerate(wrap(d_, bw - 44, SMALL)[:4]):
            out.append(text(bx + 28, by + 78 + r_ * 31, row, SMALL, 400, TEXT))
        bx += bw + 14
    return document(slide_defs(), "\n".join(out))


def s_bang_seepage(n, total):
    import json

    orange = STAGES[3][1]
    kin = json.loads((BANG / "kinematic.json").read_text(encoding="utf-8"))
    ms = kin["map_share"]
    out = frame("③ 사전 검증 — 유출 가능성 · 안정성", "물길 후보 순위와 파괴 양상 (평사투영 · 벽면 셀 지도)", n, total, stage=3,
                note=f"'상' = 점수 상위 15% · 벽면 대표 {kin['face'][0]}°/{kin['face'][1]:03d}°, 마찰각 {kin['friction']:g}° · "
                     "운동학 검토는 '가능성' 선별 (점착력·수압 미고려)")
    out.append(verify_banner())
    # top: the two wall maps side by side (same frame)
    out.append(panel(56, 224, 740, 316, picture(BANG / "seepage.png", 1500)))
    out.append(panel(812, 224, 740, 316, picture(BANG / "kinematic_map.png", 1500)))
    out.append(text(426, 570, "유출 가능성 (번호 = 점수 순 구역)", SMALL, 700, anchor="middle"))
    out.append(text(1182, 570, "파괴 가능 구역 (1 m 셀: 그 자리 벽면 × 근처 절리군)", SMALL, 700, anchor="middle"))
    # bottom: the stereographic test, and the points
    out.append(panel(56, 592, 740, 248, picture(BANG / "kinematic_stereonet.png", 1500, fmt="PNG")))
    sp = json.loads((BANG / "seepage.json").read_text(encoding="utf-8"))
    (a0, a1), (e0, e1) = sp["top5_chainage"], sp["top5_elevation"]
    boxes = [(f"'상' 구역 {sp['zones']}곳", f"#1~#5 (점수 순): 측점 {a0:.0f}~{a1:.0f} m · 표고 {e0:.0f}~{e1:.0f} m 벽 하부"),
             (f"전도 {ms['전도파괴']}% · 쐐기 {ms['쐐기파괴']}% · 평면 {ms['평면파괴']}%", "암면 셀 중 운동학적으로 가능한 비율"),
             ("원인: Set 2 (89°, 벽면과 평행)", "판상 전도 · 쐐기 Set 1×Set 2 · 평면 Set 1")]
    for k, (t_, s_) in enumerate(boxes):
        y = 592 + k * 84
        out.append(rect(812, y, 740, 78, rx=10, fill="white", stroke="#e5e7eb", stroke_width=2))
        out.append(text(834, y + 33, t_, BODY - 4, 700, orange))
        out.append(text(834, y + 64, s_, SMALL, 400, MUTED))
    return document(slide_defs(), "\n".join(out))


def s_bang_indicators(n, total):
    """Sub-slide of the seepage slide (same page number): the four indicators one by one over the orthomosaic."""
    import json

    orange = STAGES[3][1]
    info = json.loads((BANG / "seepage_indicators.json").read_text(encoding="utf-8"))
    w_ = info["weights"]
    out = frame("③ 사전 검증 — 유출 가능성 네 지표", "점수를 이루는 지표를 하나씩, 고해상도 정사영상 위에서 — 어디가 왜 높은지 확인", n, total,
                stage=3, note="바탕: 벽면 평행 정사영상 (원본 2.83 mm) · 1 m 셀 · 낮은 값은 거의 투명 · "
                              "흰 상자 = 점수 상위 5개 '상' 구역 (번호 = 점수 순)")
    out.append(verify_banner())
    for k, name in enumerate(("p21", "openness", "connectivity", "wetness")):
        col, row = k % 2, k // 2
        out.append(panel(56 + col * 556, 224 + row * 316, 544, 300, picture(BANG / f"seepage_ind_{name}.png", 1200, fmt="PNG")))
    x, w = 1172, 380
    out.append(rect(x, 224, w, 616, rx=14, fill="#fff7ed", stroke="#fdba74", stroke_width=2))
    out.append(text(x + 24, 270, "종합 점수 (0~1)", BODY - 2, 700, orange))
    terms = [(f"{w_['p21']:.2f}", "절리 밀도 P21"), (f"{w_['openness']:.2f}", "열린 정도"),
             (f"{w_['connectivity']:.2f}", "연결성"), (f"{w_['wetness']:.2f}", "절리 옆 식생")]
    for i, (wt, nm) in enumerate(terms):
        y = 316 + i * 44
        out.append(text(x + 24, y, ("= " if i == 0 else "+ ") + wt, SMALL + 2, 700, TEXT))
        out.append(text(x + 120, y, nm, SMALL + 2, 400, TEXT))
    out.append(text(x + 24, 316 + 4 * 44 - 4, "지표마다 5~95 백분위로 0~1", NOTE, 400, MUTED))
    out.append(path(f"M{x + 24},{530} L{x + w - 24},{530}", stroke="#fdba74", stroke_width=1.5))
    out.append(text(x + 24, 572, "대시보드에서", BODY - 4, 700, orange))
    for i, row in enumerate(["바탕: 입면 영상·정사영상", "값: 지표 하나씩 선택", "새 창: 확대·셀 값 확인",
                             "타일 모델에도 같은 지도"]):
        out.append(f'<circle cx="{x + 32}" cy="{606 + i * 48}" r="5" fill="{orange}"/>')
        out.append(text(x + 46, 614 + i * 48, row, SMALL, 400, TEXT))
    out.append(text(x + 24, 812, "높은 이유를 구역마다 설명", SMALL, 700, DONE_TEXT))
    return document(slide_defs(), "\n".join(out))


def s_bang_grouting(n, total):
    purple = STAGES[4][1]
    out = frame("③ 사전 검증 — 절리망 기반 그라우팅 설계", "표면 절리를 암반 속으로 연장하고, 가장 많은 물길 절리를 지나도록 공을 배치", n, total,
                stage=3, note="공 길이 8 m · 절리 연장 = 트레이스 길이 ×2 · 메시로 암반 이탈 검사 · 3차원·실사 타일 모델로 확인")
    out.append(verify_banner())
    out.append(panel(56, 224, 1000, 410, picture(BANG / "grouting_plan.png", 1700)))
    out.append(panel(1076, 224, 476, 410, picture(BANG / "grouting_section.png", 900)))
    cols = [("배치 방식", 560), ("공 수 · 천공 길이", 400), ("유출 '상' 절리 관통", 536)]
    rows = [("절리망 기반 표적 천공", "30공 · 240 m", "104 / 247  (42%)"), ("얕은 균열 주입", "41공 · 85 m", "40 / 245"),
            ("균등 벽면 수평공", "59공 · 885 m", "1 / 245")]
    x0, y0, rh = 56, 650, 50
    out.append(rect(x0, y0, 1496, rh * 4, rx=10, fill="white", stroke="#d8b4fe", stroke_width=2))
    out.append(rect(x0, y0, 1496, rh, rx=10, fill="#f3e8ff"))
    x = x0
    for name, w in cols:
        out.append(text(x + 24, y0 + 32, name, SMALL + 2, 700, purple))
        x += w
    for i, row in enumerate(rows):
        yy = y0 + rh * (i + 1)
        out.append(path(f"M{x0},{yy} L{x0 + 1496},{yy}", stroke="#e9d5ff", stroke_width=1.5))
        x = x0
        for (name, w), v in zip(cols, row):
            out.append(text(x + 24, yy + 32, v, SMALL + 2, 700 if i == 0 else 400, "#7e22ce" if i == 0 else TEXT))
            x += w
    return document(slide_defs(), "\n".join(out))


def s_bang_lessons(n, total):
    orange, purple = STAGES[3][1], STAGES[4][1]
    out = frame("③ 사전 검증 — 교훈과 익산 적용 준비", "실자료 시험으로 고친 것 → 익산 촬영·처리에 반영", n, total, stage=3)
    out.append(card(56, 176, 728, 560, "시험으로 찾아 고친 것", orange))
    svg, _ = bullets(80, 266, [("가짜 긴 트레이스 → 검출 수정", "done"),
                               ("넓은 그늘 틈 → 다중 크기·넓은 틈 검출", "done"),
                               ("꺾인 벽면 기준선 입면 잘림 → 원호 처리", "done"),
                               ("대표 벽면 하나 → 1 m 셀별 운동학 판정", "done"),
                               ("하천 수면·마른 잎 면 → 자동 제외", "done"),
                               ("분석 12분 → 4.5분, 화면 반응 2초 → 0.2초", "done"),
                               ("큰 그림 새 창 확대 · 단면 위치 상호 표시", "done"),
                               ("절리군당 면 하나 → 나란한 여러 절리면", "done")], 680, orange, size=SMALL + 4, gap=10)
    out.append(svg)
    out.append(card(824, 176, 728, 560, "익산 적용 체크리스트", purple))
    svg, _ = bullets(848, 266, [("벽면 평행 정사영상·DSM + 평면 정의(JSON) 내보내기", None),
                                ("정사영상에서 벽면 기준선 클릭 (암사면 하단)", None),
                                ("절리군별 현장 실측 10~20개 (검증용)", None),
                                ("메시·타일 모델: 공 입구·암반 이탈·실사 확인", None),
                                ("열화상: 새벽·강우 후 촬영 · 유출 확인", "plan")], 680, purple, size=SMALL + 4, gap=14)
    out.append(text(80, 712, "자동 테스트 58종 · 실자료 전 과정 통과", SMALL + 2, 700, DONE_TEXT))
    out.append(text(848, 712, "자료 준비 순서: 촬영 → 처리 → 현장 폴더", SMALL + 2, 700, purple))
    out.append(svg)
    out.append(rect(56, 758, 1496, 72, rx=14, fill=purple))
    out.append(text(804, 805, "→ 익산 자료가 들어오면 같은 절차로 바로 분석", BODY, 700, "white", anchor="middle"))
    return document(slide_defs(), "\n".join(out))


def s_evidence(n, total):
    purple = STAGES[4][1]
    out = frame("④ 활성 물길 판정 근거", "여러 근거를 종합해 지하수 유출 통로 후보를 판정", n, total, stage=4)
    cards = [("A 기하 조건", ["유입: 피트 쪽으로 경사진 절리", "절리 교차부(교선) = 물길"], None, False),
             ("B 연속성·개구", ["입면 트레이스 밀도·연결성·개구", "→ 유출 가능성 지도 (구현)"], None, False),
             ("C 열화상 이상", ["새벽·강우 후 온도 이상", "습윤지수 높은 곳 우선 확인"], "개발 예정", True),
             ("D 현장·지형 증거", ["유출 흔적·지표 유역(DEM)", "관측정 수위·ERT 저비저항대"], None, False)]
    for i, (title, rows, tag, planned) in enumerate(cards):
        col, row = i % 2, i // 2
        x, y = 56 + col * 768, 178 + row * 272
        out.append(card(x, y, 728, 246, title, purple, fill="#faf5ff" if not planned else "#f9fafb", dashed=planned))
        if tag:
            out.append(pill(x + 32 + text_width(title, 32, 700), y + 33, tag, "white", PLAN_TEXT, size=20, stroke="white")[0])
        for k, s in enumerate(rows):
            out.append(f'<circle cx="{x + 34}" cy="{y + 118 + k * 58}" r="7" fill="{"#9ca3af" if planned else purple}"/>')
            out.append(text(x + 56, y + 129 + k * 58, s, BODY, 400, PLAN_TEXT if planned else TEXT))
    out.append(rect(56, 738, 1496, 76, rx=14, fill=purple))
    out.append(text(804, 788, "→ 지질기술자 검토로 활성 물길 후보 확정", BODY + 2, 700, "white", anchor="middle"))
    return document(slide_defs(), "\n".join(out))


def s_candidates(n, total):
    purple = STAGES[4][1]
    out = frame("④ 후보 목록 → 설계·조사 연계", "후보마다 측점·표고·방향·근거를 제공해 바로 활용", n, total, stage=4,
                note="표 값: 반구대 시험 결과 — 유출 가능성 '상' 구역 20곳 중 점수 상위 3곳 · 열화상 연계 시 우선순위 재산정")
    cols = [("측점(m)", 190), ("표고(m)", 190), ("면적(m²)", 190), ("벽면 절리군", 330), ("점수", 160), ("우선순위", 230)]
    rows = [("12–15", "77–78", "3", "Set 1·2·0", "0.62", "높음"), ("19–21", "82–85", "4", "Set 1·2·0", "0.58", "높음"),
            ("33–35", "84–85", "2", "Set 1·2", "0.55", "높음")]
    width = sum(w for _, w in cols)
    x0, y0, rh = 56 + (1496 - width) / 2, 180, 64
    out.append(rect(x0, y0, width, rh * (len(rows) + 1), rx=10, fill="white", stroke="#d8b4fe", stroke_width=2))
    out.append(rect(x0, y0, width, rh, rx=10, fill="#f3e8ff"))
    x = x0
    for name, w in cols:
        out.append(text(x + w / 2, y0 + 43, name, SMALL + 4, 700, purple, anchor="middle"))
        x += w
    for i, row in enumerate(rows):
        yy = y0 + rh * (i + 1)
        out.append(path(f"M{x0},{yy} L{x0 + width},{yy}", stroke="#e9d5ff", stroke_width=1.5))
        x = x0
        for (name, w), value in zip(cols, row):
            out.append(text(x + w / 2, yy + 43, value, SMALL + 4, 700 if name == "우선순위" else 400,
                            "#b91c1c" if value == "높음" else TEXT, anchor="middle"))
            x += w
    uses = [("2-4 그라우팅", "주입공 위치·방향"), ("2-3 시험시추", "시추 위치 선정"), ("2-2 시계열 ERT", "구조 경계 반영"), ("3-1 효과 검증", "시공 전후 비교")]
    bottom = y0 + rh * (len(rows) + 1)
    for i, (t, s) in enumerate(uses):
        x = 56 + i * 382
        out.append(arrow(f"M800,{bottom + 8} C800,{bottom + 50} {x + 176},{bottom + 40} {x + 176},{bottom + 92}", purple, 3, marker="gray"))
        out.append(rect(x, bottom + 100, 352, 170, rx=14, fill="white", stroke=purple, stroke_width=2.5))
        out.append(text(x + 176, bottom + 160, t, BODY, 700, purple, anchor="middle"))
        out.append(text(x + 176, bottom + 214, s, SMALL + 2, 400, TEXT, anchor="middle"))
    return document(slide_defs(), "\n".join(out))


def s_limits(n, total):
    out = frame("한계와 보완", "드론 분석이 할 수 있는 것과 없는 것을 구분", n, total)
    items = [("해상도 한계", "mm급 미세 절리(시추공 간격 0~20 mm)는 드론 해상도 밖", "→ 절리군 체계·주요 개구 절리 중심, 미세 절리는 시추공·물리검층으로 보완"),
             ("가려진 구간", "낙석방지망·식생 뒤 암면은 직접 관찰 불가", "→ 추정하지 않고 '평가 불가'로 명시"),
             ("지표수 ≠ 지하수", "DEM 해석은 지표 흐름만 · 5 m DEM은 평야부·도로 성토에서 오차", "→ 관측정 수위·ERT로 지하 흐름 확인, 하천 선형 각인으로 보정"),
             ("반자동 판정", "자동 트레이스는 놓침·오검출이 함께 있음 (반구대 시험에서 확인·보정)",
              "→ 입면 도구로 수동 보완, 최종 판정은 지질기술자 검토")]
    y = 174
    for i, (title, line1, line2) in enumerate(items):
        out.append(rect(56, y, 1496, 152, rx=14, fill="#fefce8", stroke="#facc15", stroke_width=2.5))
        out.append(f'<circle cx="112" cy="{y + 76}" r="34" fill="#ca8a04"/>')
        out.append(text(112, y + 88, str(i + 1), 32, 700, "white", anchor="middle"))
        out.append(text(176, y + 46, title, BODY + 2, 700, "#854d0e"))
        out.append(text(176, y + 88, line1, SMALL + 2, 400))
        out.append(text(176, y + 128, wrap(line2, 1340, SMALL + 2)[0], SMALL + 2, 700, "#374151"))
        y += 166
    return document(slide_defs(), "\n".join(out))


def figure_crop(name, box=(40, 168, 1030, 870), width=1400):
    """Map panel of another figure in this folder (box in its 1600x900 SVG units; PNGs are 2x)."""
    from PIL import Image

    im = Image.open(OUT_DIR / f"{name}.png")
    k = im.width / 1600
    im = im.crop(tuple(int(v * k) for v in box))
    im = im.resize((width, int(width * im.height / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def side_items(x, y, w, items, color):
    """Right-hand column: bold title + muted description per item."""
    out = []
    for title, desc in items:
        out.append(rect(x, y, 10, 40, fill=color))
        out.append(text(x + 26, y + 32, title, BODY, 700, color))
        cy = y + 32
        for row in wrap(desc, w - 26, SMALL + 2):
            cy += 38
            out.append(text(x + 26, cy, row, SMALL + 2, 400, TEXT))
        y = cy + 30
    return "\n".join(out)


def s_scope(n, total, relief=False):
    blue = STAGES[1][1]
    where = "0.5 m 드론 DTM 컬러 음영기복도" if relief else "정사영상"
    out = frame("① 조사 범위 — 매립지 전체 둘레", f"정면 암벽만이 아니라 유입부·벽 끝·하류까지 ({where} 위 실제 위치)", n, total,
                stage=1, note="상류 차수벽·HDPE: 드론 판독 · 하류 차수벽: DTM 추정 · 우물: 배치도 정합 · 지표 흐름: DEM 5 m + DTM 0.5 m")
    out.append(rect(56, 176, 940, 668, rx=12, fill="white", stroke="#e5e7eb", stroke_width=2))
    out.append(image(62, 182, 928, 656, figure_crop("leachate_plan_relief" if relief else "leachate_plan_ortho", box=(40, 168, 1062, 870))))
    out.append(side_items(1030, 186, 522, [
        ("유입부: 남·동·서 암벽", "드론으로 절리, 열화상으로 용출"),
        ("우회 유출: 차수벽 양 끝", "서북·동북측 사면 열화상·습윤 흔적"),
        ("하류: 서북쪽 골짜기", "유출 경로 따라 용출·수질 확인"),
        ("과거 영상 1973~2025", "굴착·매립 이력과 원지형 (뷰어)"),
    ], blue))
    return document(slide_defs(), "\n".join(out))


def s_auxiliary(n, total):
    teal = STAGES[2][1]
    out = frame("② 보조자료 — 과거 영상 · DEM 지표수", "현장 변천과 빗물의 흐름을 먼저 파악해 조사 지점을 고름", n, total, stage=2,
                note="정사영상 뷰어에 통합 (20개 시기, 지형·지표수 레이어, 물길 추적)")
    out.append(banner(56, 168, "지표수만 해석 — 지하 흐름은 관측정·ERT로 확인", fg="#0f766e", bg="#ccfbf1", stroke="#5eead4"))
    out.append(rect(56, 226, 940, 618, rx=12, fill="white", stroke="#e5e7eb", stroke_width=2))
    out.append(image(62, 232, 928, 606, figure_crop("leachate_surface_watershed")))
    out.append(side_items(1030, 186, 522, [
        ("과거 영상 20개 시기", "항공사진 9 + 정사영상 11, 자동 정합"),
        ("매립지 지표 유역 7.0 ha", "DEM 5 m 기준 (현재 지형 0.5 m: 4.9 ha)"),
        ("지표 유출 경로", "서북쪽 골짜기 → 큰 강 → 금강 방향"),
        ("습윤지수", "물이 모이는 곳 → 열화상 우선 지점"),
    ], teal))
    return document(slide_defs(), "\n".join(out))


def s_design_package(n, total):
    purple = STAGES[4][1]
    out = frame("④ 차수 설계 입력 자료", "깊은 벽으로 다 막는 대신 주요 물길을 찾아 막는 '정밀 차수'의 근거", n, total, stage=4,
                note="최종 설계(주입공 배치·주입재·압력)는 2-4 몫 — 드론 분석은 설계 입력과 근거")
    items = [("절리군 요약표", "방향·간격·연장성·실측 오차", True), ("유출 후보 절리 목록", "측점·표고·좌표 + 입면 지도", True),
             ("지표 유역·유출 경로", "DEM 해석 · 외수 차단 범위", True), ("절리면 암반 내 연장", "벽 안쪽 연장 · 단면도 · 차수선 교차", True),
             ("벽 끝 우회 후보", "차수벽 양 끝 바깥 절리", False), ("주입공 위치·방향·경사", "절리망 기반 표적 천공 · 3D 시공", True)]
    for i, (title, desc, done) in enumerate(items):
        col, row = i % 3, i // 3
        x, y = 56 + col * 508, 180 + row * 220
        out.append(rect(x, y, 480, 196, rx=14, fill="#faf5ff" if done else "#f9fafb",
                        stroke=purple if done else "#9ca3af", stroke_width=2.5, stroke_dasharray=None if done else "8 5"))
        out.append(text(x + 24, y + 58, title, BODY, 700, purple if done else PLAN_TEXT))
        tag = ("구현", DONE_FILL, DONE_TEXT, None, None) if done else ("개발 예정", PLAN_FILL, PLAN_TEXT, "#9ca3af", "4 3")
        out.append(pill(x + 24, y + 104, tag[0], tag[1], tag[2], size=20, stroke=tag[3], dash=tag[4])[0])
        out.append(text(x + 24, y + 164, desc, SMALL + 2, 400, TEXT if done else PLAN_TEXT))
    flow = [("2-4 타겟 그라우팅", purple), ("시공", "#6b7280"), ("3-1 시공 전후 검증", purple)]
    x = 56
    for i, (s, c) in enumerate(flow):
        w = 380 if i != 1 else 160
        out.append(rect(x, 650, w, 90, rx=14, fill=c))
        out.append(text(x + w / 2, 707, s, BODY, 700, "white", anchor="middle"))
        if i < 2:
            out.append(arrow(f"M{x + w + 10},695 L{x + w + 74},695", "#374151", 5, marker="gray"))
        x += w + 84
    out.append(text(x, 690, "← 드론 재촬영·ERT로", SMALL + 2, 700, purple))
    out.append(text(x, 726, "   남은 물길 추가 보완", SMALL + 2, 700, purple))
    return document(slide_defs(), "\n".join(out))


if __name__ == "__main__":
    missing = [f for f in ("stereonet", "traces", "seepage", "grouting_plan", "grouting_section", "major_planes",
                           "major_sec_a", "major_sec_b", "major_sec_c", "seepage_ind_p21", "seepage_ind_wetness")
               if not (BANG / f"{f}.png").exists()]
    if missing:
        sys.exit(f"Run make_bangudae_figures.py first (missing: {', '.join(missing)})")
    builders = [s_overview, s_flight, s_scope, s_site, s_sensors, s_thermal, s_processing, s_auxiliary,
                s_registration, s_pipeline, s_bang_intro, s_bang_joints, s_bang_structure, s_bang_seepage, s_bang_grouting, s_bang_lessons,
                s_evidence, s_candidates, s_design_package, s_limits]
    for old in OUT_DIR.glob("workflow_*.*"):
        try:
            old.unlink()
        except PermissionError:   # open in a viewer; it is overwritten below
            pass
    for i, build in enumerate(builders, start=1):
        save(build(i, len(builders)), OUT_DIR, f"workflow_{i:02d}")
        if build is s_scope:   # sub-slide: same slide on the 0.5 m DTM colour hillshade (same page number)
            save(s_scope(i, len(builders), relief=True), OUT_DIR, f"workflow_{i:02d}b")
        if build is s_bang_seepage:   # sub-slide: the four indicators of the seepage score (same page number)
            save(s_bang_indicators(i, len(builders)), OUT_DIR, f"workflow_{i:02d}b")
