"""Georeference screen captures of historical aerial photos (e.g. `1973.png`) against the orthophotos.

The captures come from the NGII aerial-photo viewer (국토정보플랫폼): single, unrectified photos that
the viewer places only roughly, so every capture has its own scale, rotation and offset. They also
carry the viewer's photo-centre icons and their year labels.

Registration (one 6-parameter affine per capture, capture pixel -> EPSG:5186):
  1. mask the icons (found by their flat blue/red colour) and the labels under them;
  2. match directly against the orthophotos with LoFTR (a learned dense matcher, robust to the
     tonal and land-use change between decades), first coarse over the site, then at 1 m around
     the coarse footprint;
  3. captures that do not match the orthophotos are chained through captures already registered
     (e.g. 1973 via 1995), pooling every link;
  4. accept only with >= MIN_INLIERS RANSAC inliers spread over the image; a capture that overlaps
     the others only in a corner gets a rough guess from its best partner (confidence "낮음", with
     the disagreement between partners as its uncertainty); the rest are reported as skipped
     rather than placed wrongly.

A world file next to the capture (`1947.pgw`, EPSG:5186) overrides the automatic result, so a
capture can be georeferenced by hand (e.g. in the QGIS Georeferencer) when matching fails.
"""
from pathlib import Path

import cv2
import numpy as np
from affine import Affine
from PIL import Image

MIN_INLIERS = 20
MIN_SPREAD = 0.15        # inlier convex hull / image area, for a full affine
MIN_SPREAD_SIMILARITY = 0.03
WEAK_MIN_INLIERS = 8     # rough first guess, to be confirmed against the orthophotos
ROUGH_MIN_INLIERS = 30   # unconfirmed guess still shown as an approximate position
RANSAC_M = 4.0           # RANSAC threshold on the ground, metres
REF_RES = 1.0            # orthophoto reference resolution, metres
COARSE_MAX_PX = 1100
CHAIN_PARTNERS = 6       # registered captures (nearest in time) each chained capture is matched to


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


ICON_COLORS = [(45, 96, 157), (237, 47, 19)]   # viewer photo-centre icon: normal blue, selected red


def icon_mask(rgb: np.ndarray) -> np.ndarray:
    """The viewer's photo-centre icons (flat blue or red squares) plus the year label under each.

    Found by colour in each capture on its own, so captures of any window size work."""
    px = rgb.astype(np.int16)
    hit = np.zeros(rgb.shape[:2], bool)
    for colour in ICON_COLORS:
        hit |= (np.abs(px - colour) <= 28).all(axis=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(hit.astype(np.uint8))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    H, W = rgb.shape[:2]
    for x, y, w, h, area in stats[1:]:
        on_edge = x == 0 or y == 0 or x + w == W or y + h == H     # icon cut off by the window edge
        if not ((100 if not on_edge else 20) <= area <= 2000 and w <= 45 and h <= 50
                and (on_edge or (w >= 10 and h >= 10))):
            continue
        cx = x + w // 2                                  # label: ~46 px wide, centred, below the icon
        mask[max(0, y - 8):y + h + 22, max(0, cx - 24):cx + 24] = 1
    return mask > 0


def clean_capture(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(rgb, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA) if mask.any() else rgb


def read_world_file(path: Path) -> Affine | None:
    for ext in (".pgw", ".pngw", ".wld"):
        wf = path.with_suffix(ext)
        if wf.exists():
            a, d, b, e, c, f = [float(v) for v in wf.read_text().split()[:6]]
            # world files reference pixel centres; Affine references the corner
            return Affine(a, b, c, d, e, f) * Affine.translation(-0.5, -0.5)
    return None


class Matcher:
    """Lazy LoFTR (kornia); the outdoor weights are downloaded once to the torch hub cache."""

    def __init__(self):
        try:
            import truststore  # use the Windows certificate store (institutional TLS inspection)
            truststore.inject_into_ssl()
        except ImportError:
            pass
        import torch
        import kornia.feature as KF

        self.torch = torch
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) - 1))
        self.model = KF.LoFTR(pretrained="outdoor").eval()
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def _tensor(self, gray, scale):
        h, w = gray.shape
        size = (max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8))
        g = cv2.resize(self.clahe.apply(gray), size, interpolation=cv2.INTER_AREA)
        return self.torch.from_numpy(g).float()[None, None] / 255, (w / size[0], h / size[1])

    def __call__(self, g0, g1, s0=1.0, s1=1.0, conf=0.5):
        t0, f0 = self._tensor(g0, s0)
        t1, f1 = self._tensor(g1, s1)
        with self.torch.inference_mode():
            out = self.model({"image0": t0, "image1": t1})
        keep = out["confidence"].numpy() > conf
        return out["keypoints0"].numpy()[keep] * f0, out["keypoints1"].numpy()[keep] * f1


class Reference:
    """An orthophoto resampled to REF_RES on a north-up grid: gray array + grid transform."""

    def __init__(self, year, gray, transform: Affine):
        self.year, self.gray, self.transform = year, gray, transform

    def window(self, bounds):
        """Sub-array covering bounds (x0, y0, x1, y1) in map units, and its transform."""
        inv = ~self.transform
        c0, r0 = inv * (bounds[0], bounds[3])
        c1, r1 = inv * (bounds[2], bounds[1])
        h, w = self.gray.shape
        c0, r0 = max(0, int(c0)), max(0, int(r0))
        c1, r1 = min(w, int(c1)), min(h, int(r1))
        return self.gray[r0:r1, c0:c1], self.transform * Affine.translation(c0, r0)


def _to_map(transform: Affine, pts: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = transform[:6]
    return np.column_stack([a * pts[:, 0] + b * pts[:, 1] + c, d * pts[:, 0] + e * pts[:, 1] + f])


def fit_model(src_px: np.ndarray, dst_map: np.ndarray, shape, similarity_ok=False,
              min_inliers=None, min_spread_similarity=None):
    """RANSAC capture-pixel -> map fit: a full affine when the inliers cover the image, otherwise
    (similarity_ok) a 4-parameter similarity, which extrapolates safely from a clustered overlap.

    Returns {"transform", "inliers", "rms_m", "spread", "model"} or None.
    """
    min_inliers = min_inliers or MIN_INLIERS
    if min_spread_similarity is None:
        min_spread_similarity = MIN_SPREAD_SIMILARITY
    if len(src_px) < min_inliers:
        return None
    cv2.setRNGSeed(0)  # reproducible RANSAC
    # Pixel rows grow downwards and northings upwards; a similarity cannot mirror, so fit in (x, -y).
    src32 = (src_px * [1, -1]).astype(np.float32)
    origin = dst_map.mean(axis=0)
    dst32 = (dst_map - origin).astype(np.float32)
    models = [("affine", cv2.estimateAffine2D, MIN_SPREAD)]
    if similarity_ok:
        models.append(("similarity", cv2.estimateAffinePartial2D, min_spread_similarity))
    for name, estimate, min_spread in models:
        A, inl = estimate(src32, dst32, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_M,
                          maxIters=20000, confidence=0.999)
        if A is None:
            continue
        inl = inl.ravel().astype(bool)
        if inl.sum() < min_inliers:
            continue
        A, _ = estimate(src32[inl], dst32[inl], method=cv2.LMEDS)  # least-squares refit on inliers
        t = Affine(A[0, 0], -A[0, 1], A[0, 2] + origin[0], A[1, 0], -A[1, 1], A[1, 2] + origin[1])
        spread = cv2.contourArea(cv2.convexHull(src32[inl])) / (shape[0] * shape[1])
        sx, sy = np.hypot(t.a, t.d), np.hypot(t.b, t.e)
        if spread < min_spread or not (0.2 < sx < 5 and 0.7 < sx / sy < 1.4):
            continue
        rms = float(np.sqrt(((_to_map(t, src_px[inl]) - dst_map[inl]) ** 2).sum(axis=1).mean()))
        return {"transform": t, "inliers": int(inl.sum()), "rms_m": rms, "spread": spread, "model": name}
    return None


def _in_mask(mask, pts):
    h, w = mask.shape
    return mask[pts[:, 1].astype(int).clip(0, h - 1), pts[:, 0].astype(int).clip(0, w - 1)]


def capture_year(cid: str) -> int:
    return int(cid[:4])


def register(captures: dict, masks: dict, refs: list, site_bounds, fixed=None, log=print) -> dict:
    """captures: {id: gray array (icons inpainted)}, ids like "1984" or "1984_2"; masks: {id: icon
    mask}; refs: [Reference]; fixed: {id: Affine} captures registered earlier, kept as they are and
    used only as chaining partners.

    Returns {id: {"transform": Affine, "inliers", "rms_m", "model", "method", "confidence"}} for the
    captures that registered; `method` says whether the final fit is tied to the orthophotos directly.
    """
    match = Matcher()
    done = {cid: {"transform": t} for cid, t in (fixed or {}).items()}
    pair_cache = {}

    def footprint(t, shape, pad):
        h, w = shape
        fx, fy = _to_map(t, np.array([[0, 0], [w, 0], [w, h], [0, h]], float)).T
        return fx.min() - pad, fy.min() - pad, fx.max() + pad, fy.max() + pad

    def direct(cid):
        """Coarse match of the raw capture against each orthophoto over the whole site."""
        gray, src, dst = captures[cid], [], []
        for ref in refs:
            win, wt = ref.window(site_bounds)
            p0, p1 = match(gray, win, 1.0, min(1.0, COARSE_MAX_PX / max(win.shape)))
            keep = ~_in_mask(masks[cid], p0)
            src.append(p0[keep])
            dst.append(_to_map(wt, p1[keep]))
        return fit_model(np.vstack(src), np.vstack(dst), gray.shape)

    def refine(cid, t):
        """Resample the capture to the map grid with the current estimate and match it against the
        orthophotos at 1 m: same scale and orientation, so this ties it to the orthophotos directly
        even where the raw capture would not match."""
        gray = captures[cid]
        bounds = footprint(t, gray.shape, 60)
        grid = Affine(REF_RES, 0, bounds[0], 0, -REF_RES, bounds[3])
        size = (int((bounds[2] - bounds[0]) / REF_RES), int((bounds[3] - bounds[1]) / REF_RES))
        to_cap = ~t * grid                                # warped pixel -> capture pixel
        M = np.array(to_cap[:6], np.float64).reshape(2, 3)
        inverse = cv2.WARP_INVERSE_MAP
        warped = cv2.warpAffine(gray, M, size, flags=cv2.INTER_LINEAR | inverse)
        valid = cv2.warpAffine((~masks[cid]).astype(np.uint8), M, size, flags=cv2.INTER_NEAREST | inverse) > 0
        src, dst = [], []
        for ref in refs:
            win, wt = ref.window(bounds)
            s = min(1.0, 1400 / max(max(win.shape), max(warped.shape)))
            p0, p1 = match(warped, win, s, s)
            keep = _in_mask(valid, p0)
            src.append(_to_map(to_cap, p0[keep]))
            dst.append(_to_map(wt, p1[keep]))
        return fit_model(np.vstack(src), np.vstack(dst), gray.shape, similarity_ok=True)

    def accept(cid, fit, method, confidence):
        done[cid] = {**fit, "method": method, "confidence": confidence}
        log(f"    capture {cid}: {method} ({confidence}), {fit['model']}, {fit['inliers']} inliers, "
            f"RMS {fit['rms_m']:.1f} m")

    def partners(cid):
        """Registered captures nearest in time (the landscape changes least between them)."""
        return sorted(done, key=lambda o: (abs(capture_year(o) - capture_year(cid)), o))[:CHAIN_PARTNERS]

    def pair(cid, other):
        """Capture-to-capture matches (cid px, other px), icons dropped on both sides."""
        if (cid, other) not in pair_cache:
            p0, p1 = match(captures[cid], captures[other])
            keep = ~(_in_mask(masks[cid], p0) | _in_mask(masks[other], p1))
            pair_cache[(cid, other)] = (p0[keep], p1[keep])
        return pair_cache[(cid, other)]

    def chain_links(cid):
        src, dst = [np.zeros((0, 2))], [np.zeros((0, 2))]
        for other in partners(cid):
            p0, p1 = pair(cid, other)
            src.append(p0)
            dst.append(_to_map(done[other]["transform"], p1))
        return np.vstack(src), np.vstack(dst)

    pending = sorted(cid for cid in captures if cid not in done)
    for cid in list(pending):
        fit = direct(cid)
        if fit:
            fine = refine(cid, fit["transform"])
            accept(cid, fine or fit, "정사영상 직접 정합", "높음")
            pending.remove(cid)

    # Chain the rest through registered captures until nothing changes, then tie each chained
    # estimate back to the orthophotos so errors do not accumulate along the chain.
    progress = True
    while pending and progress:
        progress = False
        for cid in list(pending):
            fit = fit_model(*chain_links(cid), captures[cid].shape, similarity_ok=True)
            if not fit:
                continue
            fine = refine(cid, fit["transform"])
            if fine:
                accept(cid, fine, "항공사진 연계 후 정사영상 정합", "보통")
            else:
                accept(cid, fit, "항공사진 연계", "보통")
            pending.remove(cid)
            progress = True

    # Last resort for captures that overlap the others only in a corner: a similarity per partner
    # from the clustered overlap. The best one is kept if the orthophotos confirm it; otherwise it is
    # kept only as an approximate position (ROUGH_MIN_INLIERS), with the spread between the partners'
    # estimates reported as its uncertainty.
    for cid in list(pending):
        h, w = captures[cid].shape
        centre = np.array([[w / 2, h / 2]])
        guesses = []
        for other in partners(cid):
            p0, p1 = pair(cid, other)
            fit = fit_model(p0, _to_map(done[other]["transform"], p1), (h, w), similarity_ok=True,
                            min_inliers=WEAK_MIN_INLIERS, min_spread_similarity=0.0)
            if fit:
                guesses.append((fit["inliers"], other, fit))
        if not guesses:
            continue
        guesses.sort(key=lambda g: g[0], reverse=True)
        n, other, rough = guesses[0]
        centres = np.vstack([_to_map(g[2]["transform"], centre) for g in guesses])
        spread_m = float(np.linalg.norm(centres - centres[0], axis=1).max()) if len(guesses) > 1 else None
        fine = refine(cid, rough["transform"])
        if fine:
            accept(cid, fine, "부분 중첩 연계 후 정사영상 정합", "낮음")
        elif n >= ROUGH_MIN_INLIERS:
            accept(cid, {**rough, "uncertainty_m": spread_m}, f"항공사진 {other} 부분 중첩 연계 (위치 대략)", "낮음")
        else:
            continue
        pending.remove(cid)
    for cid in pending:
        log(f"    capture {cid}: not enough reliable matches, skipped")
    return {cid: r for cid, r in done.items() if cid not in (fixed or {})}

