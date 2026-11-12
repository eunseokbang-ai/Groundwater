"""Reproducible per-capture image edits for the viewer tiles (the source photos are never changed).

edits.json maps a capture id (e.g. "2002_2") to a list of operations. Supported:

  shadow_relight -- brighten a cast shadow (e.g. inside a quarry pit) to match the sunlit surface:
      roi             polygon (capture pixels) that bounds where the shadow is searched for
      keep_side_of    optional line [[x1, y1], [x2, y2]]; only the part to the right of the
                      direction p1 -> p2 (in image coordinates) is kept, e.g. a pit wall line
      max_mean, max_std   shadow = dark (7x7 mean) and smooth (7x7 std) pixels inside roi
      target          mean grey level the shadow is brought to
      matte_radius, matte_eps, seed_blur   edge-aware (guided filter) soft matte, so the
                      correction fades out along the photo's own light/shade boundary
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

EDITS_FILE = Path(__file__).resolve().parent / "edits.json"


def load_edits() -> dict:
    return json.loads(EDITS_FILE.read_text(encoding="utf-8")) if EDITS_FILE.exists() else {}


def edit_stamp(cid: str, edits: dict) -> str:
    """Short hash of a capture's edit recipe ('' when it has none), so a changed recipe rebuilds."""
    ops = edits.get(cid)
    return hashlib.sha1(json.dumps(ops, sort_keys=True).encode()).hexdigest()[:10] if ops else ""


def _guided(I, p, r, eps):
    """He et al. guided filter with a grey guide image."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    return box(a) * I + box(mp - a * mI)


def shadow_mask(gray: np.ndarray, op: dict) -> np.ndarray:
    roi = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(roi, [np.array(op["roi"], np.int32)], 1)
    mean = cv2.blur(gray, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mean ** 2, 0))
    m = ((roi > 0) & (mean < op.get("max_mean", 95)) & (std < op.get("max_std", 12))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros(gray.shape, bool)
    m = (lab == 1 + np.argmax(stats[1:, 4])).astype(np.uint8)
    # fill small interior holes
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - m)
    h, w = gray.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h and area < 3000:
            m[lab == i] = 1
    if "keep_side_of" in op:
        (x1, y1), (x2, y2) = op["keep_side_of"]
        yy, xx = np.mgrid[:h, :w]
        m &= ((xx - x1) * (y2 - y1) - (yy - y1) * (x2 - x1) > 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m > 0


def shadow_relight(rgb: np.ndarray, icons: np.ndarray, op: dict) -> np.ndarray:
    img = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shadow = shadow_mask(gray, op)
    if not shadow.any():
        return rgb
    k = op["target"] / max(float(img[shadow & ~icons].mean()), 1.0)
    seed = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), op.get("seed_blur", 1.5))
    matte = np.clip(_guided(gray / 255, seed, op.get("matte_radius", 4), op.get("matte_eps", 1e-4)), 0, 1)
    return np.clip(img * (1 + (k - 1) * matte)[..., None], 0, 255).astype(np.uint8)


OPERATIONS = {"shadow_relight": shadow_relight}


def apply_edits(cid: str, rgb: np.ndarray, icons: np.ndarray, edits: dict, inpaint) -> np.ndarray:
    """rgb: capture with icons already inpainted; icons are inpainted again after editing so the
    filled spots match the edited surroundings."""
    ops = edits.get(cid)
    if not ops:
        return rgb
    for op in ops:
        rgb = OPERATIONS[op["op"]](rgb, icons, op)
    return inpaint(rgb, icons)
