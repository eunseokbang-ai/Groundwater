"""Point cloud loading for LAS/LAZ (laspy) and PLY/PCD (Open3D)."""
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: str | Path, max_points: int | None = None, bbox=None) -> o3d.geometry.PointCloud:
    """Load a point cloud into an Open3D PointCloud, normalizing across formats.

    Supports .las/.laz (via laspy) and anything Open3D reads natively
    (.ply, .pcd, .xyz). Color is carried over when present in the source file.
    A LAS/LAZ with more than `max_points` points is read in chunks keeping every k-th point, so a
    multi-GB photogrammetry cloud never has to fit in memory at full density. With `bbox` ((xmin, ymin), (xmax,
    ymax)) only the points in it are read -- one section of a scan of the whole site -- and `max_points` applies to
    those (load_stats has the counts).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".las", ".laz"):
        return _load_las_area(path, max_points, bbox) if bbox is not None else _load_las(path, max_points)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"No points loaded from {path}")
    return pcd


load_stats: dict = {}          # of the last area read (_load_las_area): points in the area, final step


def _load_las_area(path: Path, max_points: int | None, bbox) -> o3d.geometry.PointCloud:
    """The points of a LAS/LAZ inside `bbox`, evenly thinned to at most `max_points`: kept at random with
    probability 1/step, the step doubling (and the points kept so far halved) whenever they exceed the limit --
    one pass, without knowing beforehand how many points the area holds."""
    import laspy

    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    rng = np.random.default_rng(0)
    step, n_area, n_total = 1, 0, 0
    xyz_parts, rgb_parts, max_val, has_rgb = [], [], 1.0, False
    with laspy.open(str(path)) as f:
        has_rgb = {"red", "green", "blue"} <= set(f.header.point_format.dimension_names)
        for chunk in f.chunk_iterator(10_000_000):
            n_total += len(chunk)
            x, y = np.asarray(chunk.x), np.asarray(chunk.y)
            m = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            n_area += int(m.sum())
            m &= rng.random(len(m)) < 1.0 / step
            xyz_parts.append(np.column_stack([x[m], y[m], np.asarray(chunk.z)[m]]))
            if has_rgb:
                rgb_parts.append(np.column_stack([chunk.red[m], chunk.green[m], chunk.blue[m]]))
                if np.issubdtype(chunk.red.dtype, np.integer):
                    max_val = np.iinfo(chunk.red.dtype).max
            while max_points and sum(len(p) for p in xyz_parts) > max_points:
                step *= 2
                keep = [rng.random(len(p)) < 0.5 for p in xyz_parts]
                xyz_parts = [p[k] for p, k in zip(xyz_parts, keep)]
                rgb_parts = [p[k] for p, k in zip(rgb_parts, keep)] if has_rgb else rgb_parts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(np.concatenate(xyz_parts), dtype=np.float64))
    if has_rgb:
        rgb = np.concatenate(rgb_parts).astype(np.float64)
        if rgb.max() <= 255:
            max_val = 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(np.clip(rgb / max_val, 0.0, 1.0)))
    load_stats.update({"points_total": n_total, "points_in_area": n_area, "step": step})
    print(f"{path.name}: {n_area:,} of {n_total:,} points in the area, 1 in {step} kept ({len(pcd.points):,})")
    return pcd


def _load_las(path: Path, max_points: int | None = None) -> o3d.geometry.PointCloud:
    import laspy

    with laspy.open(str(path)) as f:
        n = int(f.header.point_count)
        step = max(1, -(-n // max_points)) if max_points else 1
        has_rgb = {"red", "green", "blue"} <= set(f.header.point_format.dimension_names)
        xyz_parts, rgb_parts, max_val = [], [], 1.0
        for chunk in f.chunk_iterator(10_000_000):
            xyz_parts.append(np.vstack([chunk.x[::step], chunk.y[::step], chunk.z[::step]]).T.astype(np.float64))
            if has_rgb:
                rgb_parts.append(np.vstack([chunk.red[::step], chunk.green[::step], chunk.blue[::step]]).T)
                if np.issubdtype(chunk.red.dtype, np.integer):
                    max_val = np.iinfo(chunk.red.dtype).max
    pcd = o3d.geometry.PointCloud()
    # C-contiguous float64 rows: Open3D copies those in one pass; the transposed (column-major) parts would go
    # through its element-by-element path, ~80x slower (minutes for a 10^8-point cloud)
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(np.concatenate(xyz_parts), dtype=np.float64))
    del xyz_parts
    if has_rgb:
        rgb = np.concatenate(rgb_parts).astype(np.float64)
        if rgb.max() <= 255:            # 8-bit colours stored in the 16-bit fields (some exporters)
            max_val = 255.0
        rgb /= max_val
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(np.clip(rgb, 0.0, 1.0)))
    if step > 1:
        print(f"{path.name}: {n:,} points, every {step}th kept ({len(pcd.points):,})")
    return pcd


def las_point_count(path: str | Path) -> int | None:
    if Path(path).suffix.lower() not in (".las", ".laz"):
        return None
    import laspy

    with laspy.open(str(path)) as f:
        return int(f.header.point_count)


def iter_las_chunks(path: str | Path, chunk_size: int = 10_000_000, prefetch: bool = True):
    """(xyz float64, rgb float32 0..1 or None) for consecutive chunks of a LAS/LAZ at full density.

    With `prefetch` the next chunk is read in a background thread while the caller works on this one (reading and
    scaling a chunk takes about as long as projecting and rasterising it)."""
    if not prefetch:
        yield from _las_chunks(path, chunk_size)
        return
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=1)
    done = object()

    def reader():
        try:
            for item in _las_chunks(path, chunk_size):
                q.put(item)
        except BaseException as e:  # noqa: BLE001 -- handed to the consumer
            q.put(e)
        q.put(done)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _las_chunks(path: str | Path, chunk_size: int):
    import laspy

    with laspy.open(str(path)) as f:
        has_rgb = {"red", "green", "blue"} <= set(f.header.point_format.dimension_names)
        scale = None
        for chunk in f.chunk_iterator(chunk_size):
            xyz = np.vstack([chunk.x, chunk.y, chunk.z]).T.astype(np.float64)
            rgb = None
            if has_rgb:
                rgb = np.vstack([chunk.red, chunk.green, chunk.blue]).T.astype(np.float32)
                if scale is None:
                    scale = 255.0 if rgb.max() <= 255 else float(np.iinfo(chunk.red.dtype).max)
                rgb /= scale
            yield xyz, rgb


def crop_bbox(pcd: o3d.geometry.PointCloud, bbox: list[float] | None) -> o3d.geometry.PointCloud:
    """Crop to an axis-aligned box [xmin, ymin, zmin, xmax, ymax, zmax]. No-op if bbox is None."""
    if bbox is None:
        return pcd
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, zmin), max_bound=(xmax, ymax, zmax)
    )
    return pcd.crop(aabb)


def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if not voxel_size:
        return pcd
    return pcd.voxel_down_sample(voxel_size)
