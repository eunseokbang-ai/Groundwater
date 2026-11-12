"""Trace families on the facade: the joints a 3D facet analysis cannot see.

A joint set that strikes across the wall exposes almost no surface -- it shows up as lines (traces), not
faces (Terzaghi sampling bias). Grouping the facade traces by apparent angle and comparing each group with
the apparent angle the 3D sets would produce separates "already known" from "trace-only" families.

Orientation of a trace-only family: each trace has a 3D direction (its endpoints lifted onto the rock
surface with the facade depth). All traces of one joint set lie in planes of the same orientation, so their
directions are perpendicular to the common pole; with traces on faces of different orientation (a stepped
or curved wall) the pole is the least-variance axis of the direction tensor. The ratio of its two larger
eigenvalues tells whether the directions spread enough to fix the pole; on a single flat face they do not
and only the apparent angle is reported.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import normal_to_dip_dipdir
from .trace_mapping import Trace2D

PEAK_SMOOTH_DEG = 5.0
FAMILY_HALFWIDTH_DEG = 15.0
MIN_PEAK_SEPARATION_DEG = 30.0   # two families closer than this are one; traces go to the nearest peak
MIN_FAMILY_SHARE = 0.05
LINK_TO_SET_DEG = 15.0
LINK_TIE_DEG = 5.0         # another set expected this close to the linked one's angle fits the family as well
MIN_SPREAD = 0.08          # second eigenvalue / largest: below this the 3D directions are ~parallel


@dataclass
class TraceFamily:
    family_id: int
    apparent_angle: float            # deg on the facade, 0 = horizontal, 90 = vertical (u right, v up)
    n_traces: int
    total_length_m: float
    mean_length_m: float
    linked_set: int | None           # 3D set whose expected trace angle matches, None = trace-only
    dip: float | None = None         # estimated 3D orientation (trace-only families), if determinable
    dip_direction: float | None = None
    orientation_quality: float | None = None
    apparent_spacing_m: float | None = None
    members: list = field(default_factory=list, repr=False)
    # every set the family may belong to, linked_set first: sets whose expected angle on this wall is as close
    # (within LINK_TIE_DEG) cannot be told apart by the angle -- e.g. two steep sets crossing the wall alike
    candidate_sets: list = field(default_factory=list)


def apparent_angle(trace: Trace2D) -> float:
    d = trace.coords[1] - trace.coords[0]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)


def _axial_peaks(angles: np.ndarray, weights: np.ndarray) -> list[float]:
    """Peaks of a length-weighted, smoothed circular (180 deg) histogram."""
    grid = np.arange(0.0, 180.0, 1.0)
    diff = np.abs(grid[:, None] - angles[None, :])
    diff = np.minimum(diff, 180.0 - diff)
    dens = (np.exp(-0.5 * (diff / PEAK_SMOOTH_DEG) ** 2) * weights[None, :]).sum(axis=1)
    peaks = [i for i in range(180) if dens[i] >= dens[i - 1] and dens[i] > dens[(i + 1) % 180]]
    peaks.sort(key=lambda i: -dens[i])
    chosen = []
    for i in peaks:
        if all(min(abs(i - j), 180 - abs(i - j)) >= MIN_PEAK_SEPARATION_DEG for j in chosen):
            chosen.append(i)
    return [float(grid[i]) for i in chosen]


def _pole_from_directions(dirs: np.ndarray, weights: np.ndarray):
    """(unit pole, quality) from 3D trace directions; quality = lambda2 / lambda1."""
    t = (dirs * weights[:, None]).T @ dirs
    vals, vecs = np.linalg.eigh(t)
    return vecs[:, 0], float(vals[1] / max(vals[2], 1e-12))


def apparent_spacing(traces: list[Trace2D], angle: float, n_lines: int = 200, seed: int = 0) -> float | None:
    """Mean gap between traces along facade scanlines perpendicular to `angle` (2D, apparent)."""
    if len(traces) < 2:
        return None
    a = np.radians(angle)
    d = np.array([np.cos(a), np.sin(a)])
    nrm = np.array([-d[1], d[0]])
    p0 = np.array([t.coords[0] for t in traces])
    p1 = np.array([t.coords[1] for t in traces])
    s0, s1 = p0 @ d, p1 @ d
    lo, hi = np.minimum(s0, s1), np.maximum(s0, s1)
    off = (p0 @ nrm + p1 @ nrm) / 2
    rng = np.random.default_rng(seed)
    lengths = hi - lo
    pick = rng.choice(len(traces), n_lines, p=lengths / lengths.sum())
    pos = lo[pick] + rng.uniform(0, 1, n_lines) * lengths[pick]
    gaps = []
    for s in pos:
        hit = np.sort(off[(lo <= s) & (hi >= s)])
        if len(hit) >= 2:
            g = np.diff(hit)
            gaps.append(g[g > 0.05])
    gaps = np.concatenate(gaps) if gaps else np.empty(0)
    return float(gaps.mean()) if len(gaps) else None


def _axial_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def expected_angles(frame, sets) -> dict:
    """set id -> the apparent angle its joints make on the wall (area-weighted axial mean over its faces, each
    cut with the wall where it lies); sets parallel to the wall (no trace) are left out."""
    expected = {}
    if frame is None:
        return expected
    for s in sets:
        vals = []
        for p in s.patches:
            u = frame.to_wall(p.centroid[None])[0, 0]
            v = frame.apparent_trace_direction(p.normal, u)
            if v is not None:
                vals.append((np.degrees(np.arctan2(v[1], v[0])) % 180.0, p.area))
        if vals:
            a = np.radians(2 * np.array([x for x, _ in vals]))
            wt = np.array([y for _, y in vals])
            expected[s.set_id] = float(np.degrees(np.arctan2((np.sin(a) * wt).sum(), (np.cos(a) * wt).sum())) / 2 % 180)
    return expected


def _candidates(angle: float, expected: dict, linked) -> list:
    """linked first, then the other sets within LINK_TO_SET_DEG of `angle` whose difference is within
    LINK_TIE_DEG of the linked one's."""
    if linked is None or linked not in expected:
        return [] if linked is None else [linked]
    best = _axial_diff(expected[linked], angle)
    others = sorted((_axial_diff(e, angle), sid) for sid, e in expected.items()
                    if sid != linked and _axial_diff(e, angle) <= LINK_TO_SET_DEG)
    return [linked] + [sid for d, sid in others if d - best <= LINK_TIE_DEG]


def family_sets(family, frame=None, sets=()) -> list:
    """The sets a trace family may belong to, linked set first ([] = trace-only). Results saved before the
    candidates were kept get them worked out here (with the frame and sets)."""
    stored = getattr(family, "candidate_sets", None)
    if stored:
        return list(stored)
    if family.linked_set is None:
        return []
    return _candidates(family.apparent_angle, expected_angles(frame, sets), family.linked_set) if frame is not None \
        else [family.linked_set]


def family_tag(candidates: list) -> str:
    """"Set 1", "Set 1 또는 2" or "트레이스 전용" (trace-only)."""
    if not candidates:
        return "트레이스 전용"
    return "Set " + " 또는 ".join(str(c) for c in candidates)


def trace_families(traces: list[Trace2D], frame=None, sets=()) -> list[TraceFamily]:
    """Group traces by apparent angle; link each family to a 3D set or estimate its own orientation."""
    traces = [t for t in traces if t.length_m > 0]
    if not traces:
        return []
    ang = np.array([apparent_angle(t) for t in traces])
    w = np.array([t.length_m for t in traces])

    expected = expected_angles(frame, sets)

    families = []
    peaks = _axial_peaks(ang, w)
    if not peaks:
        return []
    dist = np.abs(ang[:, None] - np.array(peaks)[None, :])
    dist = np.minimum(dist, 180 - dist)
    nearest = np.argmin(dist, axis=1)
    for pi, peak in enumerate(peaks):
        member = (nearest == pi) & (dist[:, pi] <= FAMILY_HALFWIDTH_DEG)
        if w[member].sum() < MIN_FAMILY_SHARE * w.sum():
            continue
        idx = np.flatnonzero(member)
        linked = None
        best = LINK_TO_SET_DEG
        for sid, e in expected.items():
            dd = min(abs(e - peak), 180 - abs(e - peak))
            if dd <= best:
                best, linked = dd, sid
        fam = TraceFamily(family_id=len(families), apparent_angle=peak, n_traces=len(idx),
                          total_length_m=float(w[idx].sum()), mean_length_m=float(w[idx].mean()),
                          linked_set=linked, members=[traces[i] for i in idx],
                          candidate_sets=_candidates(peak, expected, linked))
        xyz = [traces[i].xyz for i in idx if traces[i].xyz is not None]
        if len(xyz) >= 5:
            dirs = np.array([x[1] - x[0] for x in xyz])
            ln = np.linalg.norm(dirs, axis=1)
            ok = ln > 0
            pole, q = _pole_from_directions(dirs[ok] / ln[ok, None], ln[ok])
            fam.orientation_quality = round(q, 3)
            if q >= MIN_SPREAD:
                fam.dip, fam.dip_direction = normal_to_dip_dipdir(pole)
        fam.apparent_spacing_m = apparent_spacing(fam.members, peak)
        families.append(fam)
    return families


def families_table(families: list[TraceFamily], frame=None, sets=()):
    import pandas as pd

    rows = []
    for f in families:
        cand = family_sets(f, frame, sets)
        rows.append({
            "family": f.family_id, "apparent_angle_deg": round(f.apparent_angle, 0), "n_traces": f.n_traces,
            "total_length_m": round(f.total_length_m, 1), "mean_length_m": round(f.mean_length_m, 2),
            "linked_3d_set": " 또는 ".join(str(c) for c in cand) if cand else None, "trace_only": f.linked_set is None,
            "est_dip": None if f.dip is None else round(f.dip, 0),
            "est_dip_direction": None if f.dip_direction is None else round(f.dip_direction, 0),
            "orientation_quality": f.orientation_quality,
            "apparent_spacing_m": None if f.apparent_spacing_m is None else round(f.apparent_spacing_m, 2),
        })
    return pd.DataFrame(rows)
