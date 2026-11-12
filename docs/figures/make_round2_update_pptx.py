"""Update of the 2nd interim deck, on the author's edited version (라이다_2차발표자료_방은석 수정.pptx, kept as it is):
seven slides on the excavated surfaces (wall-parallel faces), the re-runs without them, the stability check of
both, the results on the tiled model and the site-wide underground joint model, put before the last slide
('한계와 다음 단계', which gets one more point in each column); every page number renumbered.
Run make_round2_update_figures.py first.

    C:\\dokdo\\.venv\\Scripts\\python.exe docs\\figures\\make_round2_update_pptx.py      (any Python with python-pptx)

Writes docs/라이다_2차발표자료_업데이트.pptx.
"""
import copy
import json
import re

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from make_round2_pptx import ACCENT, FONT, H_IN, MUTED, SEC, TEXT, W_IN, Deck, rgb  # noqa: F401

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "라이다_2차발표자료_방은석 수정.pptx"
OUT = HERE.parent / "라이다_2차발표자료_업데이트.pptx"
FIG = HERE / "round2" / "update"
SLATE = "475569"
CLASS = {"parallel": "9CA3AF", "flat": "D97706", "cross": "2563EB"}
FOOTER = "익산 폐석산 절벽 · 드론 모듈 (WBS 2-1) · 2차 발표"
PAGE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")


class Update(Deck):
    """Deck's slide helpers on an existing presentation."""

    def __init__(self, data, path):
        self.d = data
        self.prs = Presentation(str(path))
        self.blank = next(l for l in self.prs.slide_layouts if l.name.lower() == "blank")
        self.slides = []
        # the notes master of this file has no placeholders: a new notes page gets those of the first slide's
        self._notes_tpl = [copy.deepcopy(ph._element) for ph in self.prs.slides[0].notes_slide.placeholders]

    def slide(self, title, subtitle=None, colour=ACCENT, notes=""):
        s = self.prs.slides.add_slide(self.blank)
        self.rect(s, 0, 0, 0.12, H_IN, colour)
        self.text(s, 0.5, 0.32, 12.3, 0.7, [(title, 28, True, TEXT)], name="제목")
        if subtitle:
            self.text(s, 0.5, 0.98, 12.3, 0.45, [(subtitle, 16, False, MUTED)])
        ln = s.shapes.add_connector(1, Inches(0.5), Inches(1.45), Inches(W_IN - 0.5), Inches(1.45))
        ln.line.color.rgb = rgb("E5E7EB")
        ln.line.width = Inches(1.5 / 72)
        self.slides.append(s)
        ns = s.notes_slide
        if ns.notes_text_frame is None:
            tree = ns.shapes._spTree
            for el in self._notes_tpl:
                tree.append(copy.deepcopy(el))
        tf = ns.notes_text_frame
        for p in list(tf.paragraphs)[1:]:
            p._p.getparent().remove(p._p)
        tf.paragraphs[0].text = ""
        tf.text = notes
        return s


def fmt_sets(sets):
    return ", ".join(f"{s['dip']}/{s['dd']:03d} ({s['share']}%)" for s in sets)


# the site's joint sets: (dip range, dip directions); a near-vertical set also matches the opposite direction
J = {"J1": (75, 90, [35]), "J2": (70, 90, [262]), "J3": (80, 90, [171]), "J4": (0, 30, [318, 330, 348, 35])}


def in_j(x, j):
    lo, hi, dds = J[j]
    return lo <= x["dip"] <= hi and any(abs((x["dd"] - d + 180) % 360 - 180) <= 20 or
                                        (x["dip"] >= 80 and abs((x["dd"] - d) % 360 - 180) <= 20) for d in dds)


def j_of(sets, j):
    return next((x for x in sets if in_j(x, j)), None)


def out_set(sets):
    """A set inclined out of the west wall (30-60 deg toward about 046), as the candidate the trial photogrammetry
    and the LiDAR basic run found."""
    return next((x for x in sets if 30 <= x["dip"] <= 60 and abs((x["dd"] - 46 + 180) % 360 - 180) <= 25), None)


def o(x):
    """A set's orientation; a near-vertical one written with the dip direction of J1-J3 (90°/349° = 90°/169°)."""
    dd = x["dd"]
    if x["dip"] >= 85 and not any(abs((dd - d + 180) % 360 - 180) <= 30 for d in (35, 171, 262)):
        dd = (dd + 180) % 360
    return f"{x['dip']}°/{dd:03d}°"


def j_range(d, j):
    xs = [x for k in "ESW" for v in ("sets", "sets_x") for x in d[k][v] if in_j(x, j)]
    if not xs:
        return "—"
    dips = sorted(x["dip"] for x in xs)
    dds = sorted(int(o(x).split("/")[1][:3]) for x in xs)
    return f"{dips[0]}~{dips[-1]}°/{dds[0]:03d}~{dds[-1]:03d}°" if dips[0] != dips[-1] or dds[0] != dds[-1] else o(xs[0])


def pole_angle(a, b):
    import numpy as np

    def n(x):
        dp, dd = np.radians(x["dip"]), np.radians(x["dd"])
        return np.array([np.sin(dp) * np.sin(dd), np.sin(dp) * np.cos(dd), np.cos(dp)])
    return round(float(np.degrees(np.arccos(min(1.0, abs(n(a) @ n(b)))))))


def _ori(label):
    """{'dip', 'dd'} of a site-wide set label ('81°/265° · E·S'), None if it has none."""
    import re

    m = re.match(r"\s*(\d+)°/(\d+)°", label)
    return {"dip": int(m.group(1)), "dd": int(m.group(2))} if m else None


def legend_runs(items, size=14):
    """Coloured squares and labels of the site-wide sets ('81°/265° · E·S' -> '81°/265°', gentle -> '완경사')."""
    runs = []
    for lab, col in items:
        x = _ori(lab)
        ori, dip = (o(x), x["dip"]) if x else (lab.split(" · ")[0], 90)
        runs += [("■ ", size, True, col.lstrip("#").upper()), (("완경사" if dip <= 30 else ori) + "  ", size, False, TEXT)]
    return runs


# --- new slides -----------------------------------------------------------------------------------------------
def s_excavated(D):
    e, by = D.d["exposure"], D.d["exposure_by"]
    s = D.slide("굴착면과 절리면 — 석산 벽은 깎은 면", "찾은 '절리면' 중 벽과 나란한 면 = 굴착면 자체이거나 그 면을 따라 깨진 절리",
                colour=SLATE,
                notes="석산 벽은 절리를 따라 자연적으로 무너진 면이 아니라 발파·기계로 깎은 면입니다.\n"
                      "그래서 프로그램이 찾은 면에는 벤치면·발파면 같은 굴착면 자체, 소단·바닥, 그리고 벽을 가로지르는 절리가 섞여 있습니다.\n"
                      "면마다 그 자리의 벽면 방향과 비교해서 벽면 평행(경사 20° 초과, 벽과 20° 이내), 수평(경사 20° 이하), 교차로 나눴습니다.\n"
                      f"세 구간 전체 면적의 {e['parallel']}%가 벽면 평행이고, 동측 벽은 {by['E']['parallel']}%입니다. "
                      "그림은 동측 벽 입면으로, 넓은 회색 면이 매끈한 주 벽면에 몰려 있습니다. "
                      "벽과 나란한 면은 넓게 드러나고 가로지르는 절리는 좁은 띠·트레이스로만 보이므로, 면적으로 세면 벽면 평행이 과대평가됩니다.")
    D.picture(s, FIG / "exposure_elev_E.png", 0.3, 1.5, 12.7, 3.45)
    cards = [(f"{e['parallel']}%", "벽면 평행", "굴착면 또는 그 면을 따른 절리", CLASS["parallel"]),
             (f"{e['flat']}%", "수평 (경사 ≤ 20°)", "소단·바닥 가능", CLASS["flat"]),
             (f"{e['cross']}%", "교차", "벽을 가로지르는 절리", CLASS["cross"]),
             (f"{by['E']['parallel']}%", "동측 벽의 벽면 평행", f"서측 {by['W']['parallel']}% · 남측 {by['S']['parallel']}%", SLATE)]
    for i, (big, lab, sub, col) in enumerate(cards):
        D.stat(s, 0.5 + i * 3.1, 5.05, 2.95, big, lab, colour=col, sub=sub)
    D.text(s, 0.5, 6.62, 12.3, 0.35, [("면적 비율 (E·S·W 라이다 결과의 절리면 전체). 벽면 방향 = 벽선 + 5 m 규모의 벽 굽음", 12, False, MUTED)])


def s_exposure_stereo(D):
    s = D.slide("굴착면 구분 — 스테레오넷", "벽면 평행 면을 빼면 벽을 가로지르는 절리군이 드러남 (밀도: 면 개수 기준)", colour=SLATE,
                notes="왼쪽은 모든 면, 가운데는 벽면 평행 면, 오른쪽은 교차·수평 면입니다. 빨간 ×는 그 자리 벽면의 극점입니다.\n"
                      "벽면 평행 면의 극점은 빨간 × 바로 옆, 즉 벽면 자체의 방향에 몰려 있습니다.\n"
                      "등고선은 면적이 아니라 면 개수로 센 밀도입니다. 면적으로 세면 넓게 드러난 벽면 평행 면이 과대평가되기 때문입니다.\n"
                      "참고로 면 거칠기(평면 오차)는 벽면 평행 면과 교차 면이 약 36 mm로 같아서, 거칠기만으로 발파면을 가려낼 수는 없습니다.")
    D.picture(s, FIG / "exposure_stereo.png", 0.3, 1.55, 12.7, 4.75)
    D.bullets(s, 0.5, 6.25, 12.3, 0.8, ["거칠기(평면 오차)는 벽면 평행·교차 모두 약 36 mm → 발파면 판별은 현장 확인 (발파공 흔적·풍화·충전물)"],
              size=15, colour=SLATE)


def s_rerun(D):
    d = D.d
    E, S, W = d["E"], d["S"], d["W"]
    e2, e2x, e3 = j_of(E["sets"], "J2"), j_of(E["sets_x"], "J2"), j_of(E["sets"], "J3")
    s1, s1x = j_of(S["sets"], "J1"), j_of(S["sets_x"], "J1")
    w1, w1x = j_of(W["sets"], "J1"), j_of(W["sets_x"], "J1")
    wx = out_set(W["sets_x"]) if not out_set(W["sets"]) else None
    s = D.slide("벽면 평행 제외 재분석 — 남는 절리군", "벽과 나란한 점(법선)을 절리군 찾기에서 빼고 E·S·W 다시 분석", colour=SLATE,
                notes="분석 설정에 '절벽면과 평행한 면은 절리군 찾기에서 빼기'를 추가하고 세 구간을 다시 분석했습니다.\n"
                      f"동측의 {o(e2)}는 {o(e2x)}로 남습니다. 이 방향은 남측·서측에서는 벽을 가로지르므로 실제 절리군이고, "
                      "동측 벽은 이 절리를 따라 굴착된 것으로 봅니다."
                      + (f" 동측에서 벽을 가로지르는 {o(e3)}는 두 분석 모두에서 절리군으로 나오고, 남측의 J3와 같은 군입니다.\n" if e3 else "\n")
                      + "남측 끝은 벽이 크게 돌아가서 J2·J3가 모두 벽을 가로지르므로 두 분석에서 그대로 남습니다 (90°/349°는 90°/169°와 같은 수직 면)."
                      + (f" 기본 분석에만 나온 {o(s1)}는 남측 끝 북쪽 벽면과 나란한 J1이라 제외 분석에서는 빠집니다.\n" if s1 and not s1x else "\n")
                      + (f"서측의 {o(w1)}는 {pole_angle(w1, w1x)}° 돌아가 {o(w1x)}가 됩니다. 서측 벽 방향(약 47°)에 묶인 방향이라 실제 방향은 현장 확인이 필요합니다."
                         if w1 and w1x else "")
                      + (f"\n서측 제외 분석에는 {o(wx)} 절리군({wx['share']}%)이 새로 나옵니다. 기본 분석에서 후보로만 표시되던 방향으로, "
                         "벽 밖으로 기울어 있어 평면파괴가 가능합니다. 면은 대부분 망 뒤 뒤쪽 층 점이고 단면에서는 벽 중간의 경사 단으로 보여, "
                         "절리인지 발파로 생긴 경사면인지 현장 확인이 필요합니다." if wx else ""))
    rows = [["구간", "기본 분석", "벽면 평행 제외", "판단"],
            [("E 동측 벽", SEC["E"]), fmt_sets(E["sets"]), fmt_sets(E["sets_x"]),
             f"벽이 {o(e2)} 절리를 따라 굴착" + (f" · 가로지르는 {o(e3)}도 절리군" if e3 else "")],
            [("S 남측 끝", SEC["S"]), fmt_sets(S["sets"]), fmt_sets(S["sets_x"]),
             (("J2·J3 그대로 → 실제" + (f" · {o(s1)}(J1)는 벽과 나란 → 빠짐" if s1 and not s1x else "")), "15803D")],
            [("W 서측 벽", SEC["W"]), fmt_sets(W["sets"]), fmt_sets(W["sets_x"]),
             ((f"{o(w1)} → {o(w1x)} ({pole_angle(w1, w1x)}°) · 벽 방향에 묶임" if w1 and w1x else "")
              + (f" · {o(wx)} 새로 나옴 (벽 밖으로 기움)" if wx else ""), "C2410C")]]
    gt = D.table(s, 0.4, 1.7, 12.5, rows, [1.5, 3.3, 3.3, 4.4], size=13, row_h=0.78, first_bold=True)
    for i in range(1, 4):
        for j in (1, 2, 3):
            gt.cell(i, j).text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    D.bullets(s, 0.5, 5.05, 12.3, 1.9, [f"J2 ({j_range(d, 'J2')})·J3 ({j_range(d, 'J3')}): 벽 방향이 달라도 남음 → 실제 절리군",
                                        f"J1 (서측 {o(w1)}): 벽면과 겹쳐 방향이 불확실 → 현장 클리노미터 측정으로 확정",
                                        "J4 (완경사): 벽 위 나무 뒷면을 빼니 크게 줄고, 남은 면도 대부분 소단·바닥 → 현장 확인"]
                                       + ([f"서측 제외의 {o(wx)} ({wx['share']}%): 기본에선 후보 · 벽 밖으로 기울어 평면파괴 가능 → 현장 확인 우선"]
                                          if wx else []),
              size=16 if not wx else 15, colour=SLATE)


def plane_note(d, low=5, high=30):
    """'Plane failure: almost none', naming the runs where it is not -- first those where it is common."""
    def name(k, v):
        return f"{ {'E': '동측', 'S': '남측', 'W': '서측'}[k]} {'제외' if v == 'kin_x' else '기본'} {d[k][v]['평면파괴']}%"

    runs = [(k, v) for k in "ESW" for v in ("kin", "kin_x") if d[k][v]["평면파괴"] > low]
    big = [name(k, v) for k, v in runs if d[k][v]["평면파괴"] >= high]
    small = [name(k, v) for k, v in runs if d[k][v]["평면파괴"] < high]
    if not runs:
        return "평면파괴는 두 분석 모두 거의 없음"
    if not big:
        return f"평면파괴는 대부분 드묾 ({' · '.join(small)})"
    return f"평면파괴는 {' · '.join(big)}" + (f", 그 밖은 드묾 ({' · '.join(small)})" if small else ", 그 밖은 거의 없음")


def s_kin(D):
    d = D.d
    s = D.slide("안정성 검토 — 절리군 방향에 민감", "같은 벽, 같은 마찰각(30°)에서 기본 vs 벽면 평행 제외 결과", colour=SLATE,
                notes="절리군 방향이 몇 도만 바뀌어도 운동학 검토 결과가 달라집니다.\n"
                      f"벽면 평행을 빼면 동측 쐐기파괴 가능 구간이 {d['E']['kin']['쐐기파괴']}%에서 {d['E']['kin_x']['쐐기파괴']}%로, "
                      f"서측은 {d['W']['kin']['쐐기파괴']}%에서 {d['W']['kin_x']['쐐기파괴']}%로 늘어납니다. "
                      f"동측 주 절리군의 경사가 {j_of(d['E']['sets'], 'J2')['dip']}°에서 {j_of(d['E']['sets_x'], 'J2')['dip']}°로 "
                      "낮아지면서 교선이 벽면 밖으로 나오기 쉬워졌기 때문입니다.\n"
                      f"남측은 쐐기 {d['S']['kin']['쐐기파괴']}% → {d['S']['kin_x']['쐐기파괴']}%, "
                      f"전도 {d['S']['kin']['전도파괴']}% → {d['S']['kin_x']['전도파괴']}%입니다. "
                      "기본 분석의 쐐기는 남측에서 새로 나온 J1과 J2·J3의 교선에서 나옵니다."
                      + (f" 제외 분석에서는 J3를 이루는 면이 북쪽으로 기운 면({j_of(d['S']['sets_x'], 'J3')['dd']:03d}° 방향)이라 "
                         "남측 끝 북향 벽면 안쪽으로 기울지 않아 전도가 줄어듭니다. 기본 분석의 J3 면은 대부분 남쪽으로 기울어 있습니다. "
                         "거의 수직인 절리군은 어느 면이 남느냐에 따라 전도 판정이 갈리므로 현장에서 기운 방향을 확인합니다.\n"
                         if j_of(d['S']['sets_x'], 'J3') and o(j_of(d['S']['sets_x'], 'J3')) != f"{j_of(d['S']['sets_x'], 'J3')['dip']}°/{j_of(d['S']['sets_x'], 'J3')['dd']:03d}°"
                         else "\n")
                      + (f"서측 제외 분석의 평면파괴 {d['W']['kin_x']['평면파괴']}%는 새로 나온 {o(out_set(d['W']['sets_x']))} 절리군에서 나옵니다. "
                         "벽 밖으로 50° 안팎 기운 면이라, 실제 절리로 확인되면 서측의 가장 중요한 검토 대상입니다.\n"
                         if out_set(d['W']['sets_x']) and not out_set(d['W']['sets']) and d['W']['kin_x']['평면파괴'] >= 30 else "")
                      + "그래서 안정성 결과는 한 값이 아니라 두 분석의 범위로 보고, 현장에서 절리 방향을 확인한 뒤 확정하는 것이 맞습니다.")
    D.picture(s, FIG / "kin_compare.png", 0.3, 1.55, 8.3, 3.6)
    rows = [["구간", "쐐기 (기본→제외)", "전도 (기본→제외)"]]
    for k, name in (("E", "E 동측 벽"), ("S", "S 남측 끝"), ("W", "W 서측 벽")):
        a, b = d[k]["kin"], d[k]["kin_x"]
        rows.append([(name, SEC[k]), f"{a['쐐기파괴']}% → {b['쐐기파괴']}%", f"{a['전도파괴']}% → {b['전도파괴']}%"])
    D.table(s, 8.8, 1.75, 4.2, rows, [1.4, 1.4, 1.4], size=14, row_h=0.52, first_bold=True)
    D.bullets(s, 8.8, 4.0, 4.2, 1.6, [plane_note(d), "쐐기파괴는 절리 경사 몇 도에 크게 좌우"], size=15, colour=SLATE)
    D.rect(s, 0.5, 5.5, 12.3, 0.95, "F1F5F9", line="CBD5E1")
    D.text(s, 0.7, 5.58, 12.0, 0.85, [[("결과는 범위로: ", 16, True, SLATE),
                                       ("기본 분석 = 굴착면을 절리로 본 경우, 벽면 평행 제외 = 가로지르는 절리만 본 경우. "
                                        "현장에서 J1·J2 방향을 확인하면 그 사이에서 확정", 16, False, TEXT)]])


def s_tiles(D):
    s = D.slide("실사 타일 모델 위 중첩 — 절리면과 파괴 형태", "10/23 사진 타일 모델 (Metashape) 위에 E·S·W 결과를 한 장면으로 · 북쪽(개방부)에서 본 U자",
                colour=SLATE,
                notes="현장 전체 타일 모델 위에 세 구간의 절리면과 '안정성 검토'의 파괴 가능 구역 지도를 함께 올렸습니다.\n"
                      "왼쪽은 절리군별 면입니다. 색은 현장 전체 절리군이고, 두 구간 이상에서 나온 같은 방향은 같은 색입니다: "
                      + ", ".join(f"{lab}" for lab, _ in D.d.get("legend", [])) + ".\n"
                      "오른쪽은 파괴 형태별 셀입니다. 남측 끝은 전도(하늘색), 동측은 쐐기(주황)와 일부 평면(자홍)이 보입니다.\n"
                      "대시보드의 '현장 전체 실사 3D'에서 '파괴 가능 구역'을 켜면 같은 장면을 열 수 있고, 층마다 켜고 끌 수 있으며 화면 저장 단추로 그림을 저장합니다.")
    D.picture(s, FIG / "tiles_기본_U자_절리군.png", 0.3, 1.55, 6.3, 3.75)
    D.picture(s, FIG / "tiles_기본_U자_파괴형태.png", 6.75, 1.55, 6.3, 3.75)
    D.text(s, 0.3, 5.35, 6.3, 0.75, [legend_runs(D.d.get("legend", []), 13)], align=PP_ALIGN.CENTER)
    D.text(s, 6.75, 5.35, 6.3, 0.75, [[("■ ", 14, True, "C026D3"), ("평면파괴  ", 14, False, TEXT), ("■ ", 14, True, "F59E0B"),
                                       ("쐐기파괴  ", 14, False, TEXT), ("■ ", 14, True, "0EA5E9"), ("전도파괴", 14, False, TEXT)]],
           align=PP_ALIGN.CENTER)
    D.bullets(s, 0.5, 5.85, 12.3, 1.1, ["대시보드 '현장 전체 실사 3D' → 절리면·트레이스·유출 가능성·파괴 가능 구역을 층별로 켜고 끔, 화면 저장(PNG)",
                                        "파괴 가능 구역: 1 m 셀마다 그 자리 벽면 방향 × 근처(2 m)에서 관측된 절리군 (마찰각 30°, 측방 ±20°)"],
              size=14, colour=SLATE)


def s_tiles_w(D):
    d = D.d
    w1, w1x = j_of(d["W"]["sets"], "J1"), j_of(d["W"]["sets_x"], "J1")
    w = lambda key: [g for g in d.get(key, []) if "W" in g[0].split(" · ")[-1]]  # noqa: E731
    name = {"#1F77B4": "파랑", "#D62728": "빨강", "#2CA02C": "초록", "#FF7F0E": "주황", "#9467BD": "보라", "#8C564B": "갈색",
            "#E377C2": "분홍", "#17BECF": "청록"}
    def col(key, x):                                 # the colour of the site-wide set nearest to x
        best = min(((pole_angle(x, _ori(lab)), c) for lab, c in d.get(key, []) if _ori(lab)), default=(None, ""))
        return name.get(best[1].upper(), "")
    kw = sorted((d["W"]["kin"]["쐐기파괴"], d["W"]["kin_x"]["쐐기파괴"]))
    s = D.slide("서측 벽 근접 — 기본 vs 벽면 평행 제외", "같은 시점, 같은 타일 모델 · 색은 각 분석의 절리군", colour=SLATE,
                notes=f"서측 벽을 구덩이 쪽에서 본 모습입니다. 왼쪽 기본 분석은 벽면 대부분이 {o(w1)}({col('legend', w1)})인데, 이 방향은 서측 벽과 거의 나란합니다.\n"
                      f"오른쪽은 벽과 나란한 점을 빼고 다시 분석한 결과로, 같은 자리가 {o(w1x)}({col('legend_벽면평행제외', w1x)}) 등으로 나뉩니다.\n"
                      "매끈한 벽면 자체가 절리면인지, 발파로 생긴 면인지는 현장에서 확인해야 합니다.")
    D.picture(s, FIG / "tiles_기본_서측벽_절리군.png", 0.3, 1.55, 6.3, 3.6)
    D.picture(s, FIG / "tiles_벽면평행제외_서측벽_절리군.png", 6.75, 1.55, 6.3, 3.6)
    D.text(s, 0.3, 5.2, 6.3, 0.75, [("기본 분석", 16, True, TEXT), legend_runs(w("legend"), 13)], align=PP_ALIGN.CENTER)
    D.text(s, 6.75, 5.2, 6.3, 0.75, [("벽면 평행 제외", 16, True, SLATE), legend_runs(w("legend_벽면평행제외"), 13)],
           align=PP_ALIGN.CENTER)
    D.bullets(s, 0.5, 6.1, 12.3, 0.9, [f"서측 주 절리군의 실제 방향({w1['dd']:03d}° vs {w1x['dd']:03d}°)은 현장 측정으로 확정 → "
                                       f"쐐기파괴 비율 {kw[0]}% ↔ {kw[1]}%"], size=15, colour=SLATE)


def s_underground(D):
    sj = D.d["site_joints"]
    s = D.slide("지중 절리 연속성 — 현장 전체 모델", "구간 결과를 한 공간에 모아 절리면을 땅속으로 연장 (벽면 평행 제외 결과, 면 ≥ 5 m²)", colour=SLATE,
                notes="세 구간의 절리면을 원판으로 한 공간에 놓고, 같은 절리군이면서 한 평면 위에 놓인 면들을 '연속 가능면'으로 묶었습니다.\n"
                      "왼쪽 지도에서 기호는 주향·경사, 실선은 면을 3배로 연장했을 때 지표와 만나는 선, 점선은 연속 가능면을 끝까지 이었을 때의 지표 교선(가정)입니다.\n"
                      "오른쪽은 구덩이를 서→동으로 자른 수직 단면입니다.\n"
                      f"기본 결과로는 여러 구간에 걸친 연속 가능면이 {sj['기본']['multi']}개였지만, 벽면 평행 면을 빼면 {min(sj['기본_교차만']['multi'], sj['벽면평행제외']['multi'])}~{max(sj['기본_교차만']['multi'], sj['벽면평행제외']['multi'])}개로 줄어듭니다. "
                      "구간 사이 연속성의 상당 부분이 굴착 벽면에서 나온 것이므로, 땅속 연속성은 벽면 평행 제외 결과로 보는 것이 맞습니다.")
    D.picture(s, FIG / "sj_ortho.png", 0.3, 1.5, 5.4, 4.75)
    D.picture(s, FIG / "sj_section.png", 5.9, 1.5, 7.1, 3.0)
    D.text(s, 6.0, 4.5, 7.0, 0.35, [("수직 단면: 구덩이를 서 → 동으로 (회색 = 땅속)", 12, False, MUTED)], align=PP_ALIGN.CENTER)
    D.text(s, 0.2, 6.25, 5.8, 0.6, [legend_runs(D.d.get("legend_sj", []), 12),
                                     ("기호 = 주향·경사 · 실선 = ×3 연장 면의 지표 교선 · 점선 = 끝까지 (가정)", 11, False, MUTED)],
           align=PP_ALIGN.CENTER)
    rows = [["결과", "면", "연속 가능면", "여러 구간"],
            ["기본 (모든 면)", sj["기본"]["faces"], sj["기본"]["planes"], sj["기본"]["multi"]],
            ["기본 (교차 면만)", sj["기본_교차만"]["faces"], sj["기본_교차만"]["planes"], sj["기본_교차만"]["multi"]],
            [("벽면 평행 제외 분석", SLATE), sj["벽면평행제외"]["faces"], sj["벽면평행제외"]["planes"], sj["벽면평행제외"]["multi"]]]
    D.table(s, 6.1, 5.0, 6.9, rows, [2.6, 1.2, 1.6, 1.5], size=13, row_h=0.42, header_fill=SLATE, first_bold=True)
    D.text(s, 6.1, 6.72, 6.9, 0.3, [("연장·끝까지 이은 면은 가정 — 시추공·열화상으로 확인", 12, False, MUTED)])


NEW = [s_excavated, s_exposure_stereo, s_rerun, s_kin, s_tiles, s_tiles_w, s_underground]


# --- the author's slides --------------------------------------------------------------------------------------
def add_point(tb, text):
    """One more bullet at the end of a text box, as its last paragraph (formatting copied)."""
    last = tb.text_frame.paragraphs[-1]._p
    new = copy.deepcopy(last)
    last.addnext(new)
    runs = new.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r")
    for r in runs[1:]:
        r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = ""
    if len(runs) > 1:
        runs[1].find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = text


def update_last(slide):
    """'한계와 다음 단계': the excavated-surface point in each column."""
    boxes = [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip().startswith("•")]
    boxes.sort(key=lambda sh: sh.left)
    if len(boxes) >= 2:
        add_point(boxes[0], "벽면 평행 면은 굴착면·절리 구분 불가 → 결과를 범위(기본 ~ 제외)로 제시")
        add_point(boxes[1], "벽면 평행 면 현장 판별 · J1 방향 측정 (발파공 흔적·풍화·충전물)")


def renumber(prs, footer_slides):
    n = len(prs.slides)
    for i, s in enumerate(prs.slides, start=1):
        for sh in s.shapes:
            if sh.has_text_frame and PAGE.match(sh.text_frame.text):
                runs = sh.text_frame.paragraphs[0].runs
                runs[0].text = f"{i} / {n}"
                for r in runs[1:]:
                    r.text = ""
    return n


def main():
    data = json.loads((FIG / "update.json").read_text(encoding="utf-8"))
    D = Update(data, SRC)
    n0 = len(D.prs.slides)
    user = list(D.prs.slides)                           # the author's slides, before the new ones go in
    for f in NEW:
        f(D)
    for s in D.slides:                                  # page number and footer as the other slides
        D.text(s, W_IN - 1.6, H_IN - 0.45, 1.2, 0.3, [("0 / 0", 11, False, MUTED)], align=PP_ALIGN.RIGHT)
        D.text(s, 0.5, H_IN - 0.45, 8, 0.3, [(FOOTER, 11, False, MUTED)])
    # the new slides before the author's last slide
    ids = D.prs.slides._sldIdLst
    items = list(ids)
    last, new = items[n0 - 1], items[n0:]
    for el in new:
        ids.remove(el)
    for el in new:                                      # each lands just before the last slide, in order
        last.addprevious(el)
    import make_round2_full_pptx as FP                  # the author's result slides with the current results

    D.d = {**data, "U": data, "F": json.loads((FP.FULL / "full.json").read_text(encoding="utf-8"))}
    FP.refresh_user(user, D, json.loads((FP.R2 / "summary.json").read_text(encoding="utf-8")))
    update_last(D.prs.slides[len(D.prs.slides) - 1])
    n = renumber(D.prs, D.slides)
    D.prs.save(OUT)
    print(f"wrote {OUT} ({n} slides; {len(NEW)} new before the last of {n0})")


if __name__ == "__main__":
    main()
