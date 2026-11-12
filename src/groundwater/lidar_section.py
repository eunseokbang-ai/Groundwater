"""Vertical sections through a LiDAR scan: the returns in a thin slice (single / first / intermediate / last) and the
rock layer the program keeps (lidar.rear_layer, then lidar.rock_surface and not above the ground), to check the scan and the
classification at any place of the site -- and to show it.

A 62 GB scan cannot be read for every section. index() makes, once, a small index next to the cache: the bounding
box of every CHUNK consecutive points (a LAS is in acquisition order, so a chunk is a short stretch of one flight
line) and a plan image (the top-most point's colour and height per PLAN_RES cell). A section then reads only the
chunks whose box meets it (LAS seek; ~1 s instead of minutes).

Section geometry: the vertical plane through the plan line A -> B; s = distance along A -> B, d = signed offset from
the plane (+ to the left of A -> B), z = height (as in the file; DJI L2 LAS: orthometric).
"""
import hashlib
from pathlib import Path

import numpy as np

CHUNK = 100_000
PLAN_RES = 0.25          # m, plan image cell
CONTEXT_M = 1.5          # half-width of the slab read around the plane (the classification needs neighbours)
CACHE_DIR = Path.home() / ".groundwater" / "lidar_index"
RETURN_KINDS = ("단일 반사", "첫 반사", "중간 반사", "마지막 반사")


def index_path(las) -> Path:
    p = Path(las)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{p.stem}_{key}.npz"


def load_index(las) -> dict | None:
    f = index_path(las)
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def build_index(las, progress=None) -> dict:
    """One pass over the file: chunk boxes and the plan image. Saved as npz in CACHE_DIR."""
    import laspy

    say = progress or (lambda frac, s: None)
    with laspy.open(str(las)) as f:
        h = f.header
        n = int(h.point_count)
        lo, hi = np.asarray(h.mins[:2], float), np.asarray(h.maxs[:2], float)
        W = int(np.ceil((hi[0] - lo[0]) / PLAN_RES)) + 1
        H = int(np.ceil((hi[1] - lo[1]) / PLAN_RES)) + 1
        ztop = np.full(H * W, -np.inf, np.float32)
        rgb = np.zeros((H * W, 3), np.uint8)
        has_rgb = {"red", "green", "blue"} <= set(h.point_format.dimension_names)
        boxes, read, scale = [], 0, None
        for ch in f.chunk_iterator(CHUNK):
            x, y, z = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z, np.float32)
            boxes.append((x.min(), y.min(), x.max(), y.max()))
            c = ((x - lo[0]) / PLAN_RES).astype(np.int64)
            r = ((hi[1] - y) / PLAN_RES).astype(np.int64)            # row 0 = north
            idx = r * W + c
            o = np.lexsort((z, idx))
            last = np.r_[idx[o][1:] != idx[o][:-1], True]             # the top-most point of each cell
            sel, cell = o[last], idx[o[last]]
            up = z[sel] > ztop[cell]
            ztop[cell[up]] = z[sel][up]
            if has_rgb and up.any():
                col = np.column_stack([ch.red, ch.green, ch.blue])[sel[up]].astype(np.float32)
                if scale is None:                          # 16-bit colours, or 8-bit ones in the 16-bit fields
                    scale = 256.0 if col.max() > 255 else 1.0
                rgb[cell[up]] = np.clip(col / scale, 0, 255).astype(np.uint8)
            read += len(ch)
            if len(boxes) % 200 == 0:
                say(read / n, f"라이다 색인 만드는 중: {read / 1e6:,.0f} / {n / 1e6:,.0f}백만 점")
    idx = {"boxes": np.array(boxes), "chunk": np.array(CHUNK), "n": np.array(n), "plan_origin": np.array([lo[0], hi[1]]),
           "plan_res": np.array(PLAN_RES), "plan_z": ztop.reshape(H, W), "plan_rgb": rgb.reshape(H, W, 3)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_path(las), **idx)
    say(1.0, "색인 완료")
    return idx


def _chunks_near(index, a, b, pad) -> np.ndarray:
    """Indices of the chunks whose box comes within `pad` of the segment a-b (box vs the segment's padded box,
    then the distance from the box centre to the segment against the box's half diagonal)."""
    bx = index["boxes"]
    lo, hi = np.minimum(a, b) - pad, np.maximum(a, b) + pad
    m = (bx[:, 2] >= lo[0]) & (bx[:, 0] <= hi[0]) & (bx[:, 3] >= lo[1]) & (bx[:, 1] <= hi[1])
    i = np.flatnonzero(m)
    c = np.column_stack([(bx[i, 0] + bx[i, 2]) / 2, (bx[i, 1] + bx[i, 3]) / 2])
    half = 0.5 * np.hypot(bx[i, 2] - bx[i, 0], bx[i, 3] - bx[i, 1])
    ab = b - a
    t = np.clip(((c - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    dist = np.hypot(*(c - (a + t[:, None] * ab)).T)
    return i[dist <= half + pad]


def read_slab(las, index, a, b, half_width: float = CONTEXT_M, margin: float = 1.0) -> dict:
    """The points within `half_width` of the vertical plane through a -> b (and up to `margin` beyond its ends)."""
    import laspy

    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    t = (b - a) / max(L, 1e-9)
    nrm = np.array([-t[1], t[0]])
    out = {k: [] for k in ("xyz", "ret", "nret", "rgb")}
    chunks = _chunks_near(index, a, b, half_width + margin)
    step = int(index["chunk"])
    with laspy.open(str(las)) as f:
        has_rgb = {"red", "green", "blue"} <= set(f.header.point_format.dimension_names)
        for i in chunks:
            f.seek(int(i) * step)
            p = f.read_points(step)
            x, y = np.asarray(p.x), np.asarray(p.y)
            rel = np.column_stack([x - a[0], y - a[1]])
            s, d = rel @ t, rel @ nrm
            m = (np.abs(d) <= half_width) & (s >= -margin) & (s <= L + margin)
            if not m.any():
                continue
            out["xyz"].append(np.column_stack([x[m], y[m], np.asarray(p.z)[m]]))
            out["ret"].append(np.asarray(p.return_number)[m].astype(np.int8))
            out["nret"].append(np.asarray(p.number_of_returns)[m].astype(np.int8))
            if has_rgb:
                c = np.column_stack([p.red[m], p.green[m], p.blue[m]]).astype(np.float32)
                out["rgb"].append(np.clip(c / (256.0 if c.max() > 255 else 1.0), 0, 255).astype(np.uint8))
    res = {k: (np.concatenate(v) if v else np.zeros((0, 3) if k in ("xyz", "rgb") else 0)) for k, v in out.items()}
    rel = res["xyz"][:, :2] - a if len(res["xyz"]) else np.zeros((0, 2))
    res.update({"s": rel @ t, "d": rel @ nrm, "length": L, "a": a, "b": b, "chunks_read": int(len(chunks))})
    return res


def return_kind(ret: np.ndarray, nret: np.ndarray) -> np.ndarray:
    """0 single, 1 first of several, 2 intermediate, 3 last of several."""
    k = np.full(len(ret), 2, np.int8)
    k[nret <= 1] = 0
    k[(nret > 1) & (ret == 1)] = 1
    k[(nret > 1) & (ret >= nret)] = 3
    return k


def classify(xyz: np.ndarray, frame, dtm=None, context: np.ndarray | None = None) -> dict:
    """The program's rock layer in the slab: rear_layer per wall cell (the rearmost layer seen from the pit), and of
    it the rock face with its gentle ledges (rock_surface; `ledge` = the gentle part kept) -- the rear layer above
    the face (original ground, tree trunks, the scan's far edge) or on the floor below it is left out (`outside`) --
    and, with a bare-earth `dtm`, not above the ground ("air": the back of a tree crown, cleaning.above_ground).
    `context`: more points of the wall around the slab (e.g. the section's analysis points) -- a slab oblique to the
    wall holds only part of each wall column, and the face's span is found on the whole column as in the analysis."""
    from .lidar import rear_layer, rock_surface

    rear = np.zeros(len(xyz), bool)
    rocky = np.zeros(len(xyz), bool)
    ledge = np.zeros(len(xyz), bool)
    air = np.zeros(len(xyz), bool)
    if len(xyz) < 10:
        return {"rear": rear, "rock": rocky, "ledge": ledge, "outside": rocky.copy(), "air": air}
    uvw = frame.to_wall(xyz)
    rear = rear_layer(uvw)
    if rear.any():
        ctx = None
        if context is not None and len(context):
            cu = frame.to_wall(np.asarray(context, float))
            lo, hi = uvw[rear, 0].min() - 2.0, uvw[rear, 0].max() + 2.0
            cu = cu[(cu[:, 0] >= lo) & (cu[:, 0] <= hi)]
            ctx = cu[:: max(1, len(cu) // 2_000_000)] if len(cu) else None
        idx = np.flatnonzero(rear)
        rocky[idx], ledge[idx] = rock_surface(uvw[rear], ctx)
    if dtm and rear.any():
        from .discontinuity.cleaning import above_ground

        idx = np.flatnonzero(rear)
        air[idx] = above_ground(xyz[idx], dtm, check_datum=False)[0]
    return {"rear": rear, "rock": rear & rocky & ~air, "ledge": rear & ledge & ~air, "outside": rear & ~rocky & ~air,
            "air": air}


def owning_section(site, xy):
    """(section name or None for the site folder itself, its folder) whose section_area.csv holds xy, else the
    first section with a wall line; (None, site) when the site has no sections."""
    from . import site_folder as S
    from .facade.region import SECTION_FILE, in_polygon, load_section

    site = Path(site)
    names = S.sections(site)
    for name in names:
        poly = load_section(S.section_dir(site, name) / SECTION_FILE)
        if poly and in_polygon(poly, np.asarray(xy, float)[None])[0]:
            return name, S.section_dir(site, name)
    poly = load_section(site / SECTION_FILE)
    if poly is None or in_polygon(poly, np.asarray(xy, float)[None])[0]:
        return None, site
    return (names[0], S.section_dir(site, names[0])) if names else (None, site)


def section_frame(folder, sample_xyz):
    """The wall frame of a section folder (its wall_line.csv), or a straight wall fitted to the sample."""
    from .facade.frame import build_wall_frame
    from .site_folder import WALL_LINE_NAME, read_wall_line

    wl = Path(folder) / WALL_LINE_NAME
    poly, pit = read_wall_line(wl) if wl.exists() else (None, None)
    return build_wall_frame(sample_xyz, poly, pit)


def perpendicular_line(frame, u: float, front_m: float = 15.0, back_m: float = 40.0):
    """Plan line across the wall at chainage u: from `front_m` in front of the wall line (pit) to `back_m` behind."""
    a = frame.to_world(np.array([[u, 0.0, front_m]]))[0, :2]
    b = frame.to_world(np.array([[u, 0.0, -back_m]]))[0, :2]
    return a, b
