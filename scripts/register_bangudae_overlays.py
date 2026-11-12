"""Bangudae (Ulsan) petroglyph panel: register reference images to the wall-parallel planar orthomosaic and
write reference_overlays.json next to it (site-specific, run once; the dashboard and the tiled-model viewer
read the JSON).

* photos of the panel: SIFT features matched to the planar orthomosaic (panel crop, ~5.7 mm), RANSAC
  homography photo px -> plane coordinates;
* the line tracing (실측도) has no texture to match: figure correspondences picked between the tracing and
  the highlighted panel photo (recognisable figures, below), carried to the plane through that photo's
  homography, and a thin-plate spline through them (the panel is not one plane: the left block stands
  forward of the right one, so a single homography leaves 10-20 cm misfits).

    .venv\\Scripts\\python.exe scripts\\register_bangudae_overlays.py [planar folder]
"""
import json
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

FOLDER = sys.argv[1] if len(sys.argv) > 1 else r"C:\Groundwater\Filed_Data\Ulsan_rock_carving"
ORTHO = "Ortho(2.83mm)_wall.tif"
PANEL_1CM = (2450, 2050, 3500, 2750)            # panel search window at 1 cm (x0, y0, x1, y1), from a whole-wall match
READ_FACTOR = 2                                 # panel crop read at 2 x the native 2.83 mm
TRACING = "20211011142537200_BSKS8EJM.jpg"
BASE_PHOTO = "202507121951347897_0.jpg"
PHOTOS = {                                      # file -> (name, note)
    BASE_PHOTO: ("암각화 강조 사진 (전체 암면)", "암각화 형상을 어둡게 강조한 정면 사진"),
    "20211011142838278_QKC538SM.jpg": ("근접 사진: 오른쪽 암면 중앙", ""),
    "20211011142801870_ADQ85BTC.jpg": ("근접 사진: 왼쪽 암면 중앙 (인물·그물)", ""),
    "20211011142434418_DAD9X6OM.jpg": ("근접 사진: 왼쪽 암면 전체 (비스듬히)", "비스듬히 찍은 사진이라 위치 정확도 낮음"),
}
# tracing px -> base photo px: recognisable figures
TRACING_PAIRS = {
    "왼쪽 암면 그물무늬 타원": ((465, 370), (235, 240)),
    "울타리 모양": ((300, 215), (148, 152)),
    "줄무늬 호랑이": ((585, 272), (290, 183)),
    "왼쪽 암면 오른쪽 끝 긴 타원": ((640, 245), (325, 163)),
    "오른쪽 암면 큰 검은 고래": ((761, 303), (393, 200)),
    "그 왼쪽 아래 검은 고래": ((714, 373), (365, 240)),
    "보라색 호 (위쪽)": ((934, 133), (522, 88)),
    "갈색 그물무늬 형상": ((740, 210), (395, 140)),
    "격자무늬 동물": ((1072, 264), (603, 170)),
    "위쪽 사슴": ((1093, 381), (612, 262)),
    "큰 네발짐승": ((1085, 440), (612, 300)),
}


def main():
    with rasterio.open(os.path.join(FOLDER, ORTHO)) as ds:
        k = 0.01 / abs(ds.transform.a)
        x0, y0, x1, y1 = PANEL_1CM
        c0, r0, c1, r1 = int(x0 * k), int(y0 * k), int(x1 * k), int(y1 * k)
        img = ds.read([1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0),
                      out_shape=(3, (r1 - r0) // READ_FACTOR, (c1 - c0) // READ_FACTOR), resampling=Resampling.average)
        A = ds.transform
    crop = np.moveaxis(img, 0, -1)[:, :, ::-1].copy()
    # crop px -> plane: a crop pixel coordinate (cx, cy) (OpenCV: pixel centres at integers) is full-res
    # pixel position c0 + (cx + 0.5) f, r0 + (cy + 0.5) f, then the raster transform
    f = READ_FACTOR
    C2P = np.array([[A.a * f, A.b * f, A.a * (c0 + 0.5 * f) + A.b * (r0 + 0.5 * f) + A.c],
                    [A.d * f, A.e * f, A.d * (c0 + 0.5 * f) + A.e * (r0 + 0.5 * f) + A.f],
                    [0, 0, 1]])
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
    sift = cv2.SIFT_create(nfeatures=150000)
    ko, do = sift.detectAndCompute(clahe.apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
    overlays, H_base = [], None
    for f, (name, note) in PHOTOS.items():
        ref = cv2.imread(os.path.join(FOLDER, f))
        best = None
        for s in (1.0, 0.6, 1.6):
            r = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            kr, dr = sift.detectAndCompute(clahe.apply(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)), None)
            good = [a for a, b in flann.knnMatch(dr, do, k=2) if a.distance < 0.8 * b.distance]
            if len(good) < 10:
                continue
            src = np.float32([kr[g.queryIdx].pt for g in good]) / s
            dst = np.float32([ko[g.trainIdx].pt for g in good])
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
            if H is not None and (best is None or inl.sum() > best[0]):
                best = (int(inl.sum()), H)
        if best is None or best[0] < 20:
            print(f"{f}: not registered ({0 if best is None else best[0]} inliers)")
            continue
        Hp = C2P @ best[1]
        if f == BASE_PHOTO:
            H_base = Hp
        overlays.append({"name": name, "file": f, "kind": "photo", "method": "homography",
                         "H": (Hp / Hp[2, 2]).tolist(), "inliers": best[0], "note": note})
        print(f"{f}: {best[0]} inliers")
    if H_base is not None:
        pts = []
        for label, (t, p) in TRACING_PAIRS.items():
            q = H_base @ np.array([p[0], p[1], 1.0])
            pts.append([t[0], t[1], q[0] / q[2], q[1] / q[2], label])
        overlays.insert(0, {"name": "암각화 실측도 (전체 형상)", "file": TRACING, "kind": "tracing", "method": "tps",
                            "points": pts, "note": "형상 대응점 11개로 맞춘 박판 스플라인, 위치 오차 수 cm~10 cm"})
    out = os.path.join(FOLDER, "reference_overlays.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"ortho": ORTHO, "overlays": overlays}, fh, ensure_ascii=False, indent=1)
    print("wrote", out, len(overlays), "overlays")


if __name__ == "__main__":
    main()
