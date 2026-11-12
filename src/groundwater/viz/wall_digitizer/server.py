"""Wall plan line digitizer: click the wall line and a pit point on the drone orthomosaic, save wall_line.csv.

A local web page (Leaflet) over the orthomosaic tiles, with a DTM (or DSM) slope overlay to find the rock
face. The CSV is written in the orthomosaic's own projected CRS (e.g. EPSG:5186 central, 5187 east belt).
The page is stateless on the server side: every request names the orthomosaic (`ortho`, a GeoTIFF or a
folder to search) and the site folder the CSV goes to (`out`), so one running server serves any site.

Tiles: a GeoTIFF already tiled by the orthophoto history viewer (same file name and size) is reused as is;
any other one is tiled once in the background (large_ortho.build_pyramid) into data/cache/ortho_tiles/.

    .venv\\Scripts\\python.exe -m groundwater.viz.wall_digitizer --ortho <tif or folder> --out <site folder>
"""
import argparse
import hashlib
import json
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
VIEWER = ROOT / "webapp" / "ortho_history"
CACHE = ROOT / "data" / "cache" / "ortho_tiles"
DEFAULT_PORT = 8612
SLOPE_MIN_RES = 0.5                 # m; a cm-scale DSM is averaged to this first (canopy/grass texture is not slope)
SLOPE_MAX_PX = 5000                 # overlay never wider / taller than this
SLOPE_FROM = 40.0                   # deg, gentler ground stays transparent
PIT_PROBE = 15.0                    # m, auto pit point: this far off the wall's middle on the lower side

sys.path.insert(0, str(ROOT / "src"))
from groundwater.facade.region import REGION_FILE, load_polygon, save_polygon  # noqa: E402
from groundwater.site_folder import WALL_LINE_NAME, read_wall_line  # noqa: E402

MANUAL_TRACES = "manual_traces.csv"           # same name as groundwater.pipeline.MANUAL_TRACES

_builds = {}                        # tile dir -> {"status", "log"}
_tilesets = {}                      # id -> 3D Tiles archive / folder served under /3dtiles/<id>/
_lock = threading.Lock()


# --- finding the inputs ------------------------------------------------------------------------------

def _raster_info(p: Path):
    import rasterio
    try:
        with rasterio.open(p) as ds:
            if ds.crs is None or ds.transform.is_identity:
                return None
            return ds.count
    except Exception:  # noqa: BLE001
        return None


def find_ortho(path: str | Path) -> Path | None:
    """The GeoTIFF itself, or the largest georeferenced 3-4 band GeoTIFF under a folder (DTM/DSM skipped)."""
    p = Path(path)
    if p.is_file():
        return p if p.suffix.lower() in (".tif", ".tiff") and (_raster_info(p) or 0) >= 3 else None
    if not p.is_dir():
        return None
    cands = [f for f in p.rglob("*.tif") if not any(k in str(f.relative_to(p)).lower() for k in ("dtm", "dsm", "dem"))]
    cands = [f for f in sorted(cands, key=lambda f: f.stat().st_size, reverse=True) if (_raster_info(f) or 0) >= 3]
    return cands[0] if cands else None


def find_dtm(ortho: Path, out: Path | None) -> Path | None:
    """A 1-band georeferenced GeoTIFF with dtm/dem in its path near the orthomosaic or in the site folder;
    a DSM when there is none (on a bare rock face the two hardly differ)."""
    roots = [r for r in (ortho.parent, ortho.parent.parent, out) if r and r.is_dir()]
    for words in (("dtm", "dem"), ("dsm",)):
        for r in roots:
            for f in sorted(r.rglob("*.tif"), key=lambda f: ("dtm" not in str(f).lower(), str(f))):
                if any(k in str(f.relative_to(r)).lower() for k in words) and _raster_info(f) == 1:
                    return f
    return None


def tile_source(ortho: Path) -> dict:
    """{'dir', 'max_zoom', 'bounds', 'status'} for the orthomosaic's tiles (viewer tiles reused when identical)."""
    layers = VIEWER / "layers.json"
    if layers.exists():
        for d in json.loads(layers.read_text(encoding="utf-8"))["layers"]:
            src = ROOT / "과거정사영상" / d.get("file", "")
            if (d.get("kind") == "drone" and d.get("file") == ortho.name and src.exists()
                    and src.stat().st_size == ortho.stat().st_size):
                return {"dir": VIEWER / "tiles" / d["id"], "max_zoom": d["max_zoom"], "min_zoom": d["min_zoom"],
                        "bounds": d["bounds"], "status": "ready", "reused": "정사영상 변천 뷰어 타일"}
    key = hashlib.md5(f"{ortho.resolve()}|{ortho.stat().st_size}|{int(ortho.stat().st_mtime)}".encode()).hexdigest()[:10]
    out = CACHE / f"{ortho.stem}-{key}"
    meta = out / "meta.json"
    if meta.exists():
        return {**json.loads(meta.read_text(encoding="utf-8")), "dir": out, "status": "ready", "reused": "캐시"}
    with _lock:
        job = _builds.get(str(out))
        if job is None:
            job = _builds[str(out)] = {"status": "building", "log": "타일 만드는 중..."}
            threading.Thread(target=_build_tiles, args=(ortho, out, job), daemon=True).start()
    return {"dir": out, "status": job["status"], "log": job["log"]}


def _build_tiles(ortho: Path, out: Path, job: dict):
    sys.path.insert(0, str(VIEWER))
    from large_ortho import build_pyramid

    def log(msg, **_):
        job["log"] = str(msg).strip()
    try:
        out.mkdir(parents=True, exist_ok=True)
        max_zoom, bounds, gsd = build_pyramid(ortho, out, min_zoom=13, log=log)
        (out / "meta.json").write_text(json.dumps({"max_zoom": max_zoom, "min_zoom": 13, "bounds": bounds,
                                                   "gsd": gsd}), encoding="utf-8")
        job["status"] = "ready"
    except Exception as e:  # noqa: BLE001
        job["status"], job["log"] = "error", f"타일 생성 실패: {e}"


def slope_overlay(dtm: Path) -> tuple[Path, list]:
    """Slope of the DTM as a transparent PNG on a Web Mercator grid (steep = orange-red); cached."""
    import rasterio
    from PIL import Image
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

    key = hashlib.md5(f"{dtm.resolve()}|{dtm.stat().st_size}|{int(dtm.stat().st_mtime)}".encode()).hexdigest()[:10]
    png, meta = CACHE / f"slope2-{key}.png", CACHE / f"slope2-{key}.json"
    if png.exists() and meta.exists():
        return png, json.loads(meta.read_text())
    with rasterio.open(dtm) as ds:
        res = abs(ds.transform.a)
        k = max(1.0, SLOPE_MIN_RES / res, max(ds.width, ds.height) / SLOPE_MAX_PX)
        h0, w0 = max(1, int(ds.height / k)), max(1, int(ds.width / k))
        z = ds.read(1, out_shape=(h0, w0), resampling=Resampling.average, masked=True).astype(np.float32)
        z = z.filled(np.nan)
        z[z < -1000] = np.nan
        st = ds.transform * ds.transform.scale(ds.width / w0, ds.height / h0)
        gy, gx = np.gradient(z, abs(st.e), st.a)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        t, w, h = calculate_default_transform(ds.crs, "EPSG:3857", w0, h0, *ds.bounds)
        dst = np.full((h, w), np.nan, np.float32)
        reproject(slope, dst, src_transform=st, src_crs=ds.crs, dst_transform=t, dst_crs="EPSG:3857",
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        w_, s_, e_, n_ = transform_bounds("EPSG:3857", "EPSG:4326", t.c, t.f + h * t.e, t.c + w * t.a, t.f)
    v = np.clip((np.nan_to_num(dst) - SLOPE_FROM) / 35, 0, 1)    # transparent below SLOPE_FROM, solid from +35 deg
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (200 * (1 - v) + 30).astype(np.uint8)
    rgba[..., 2] = 20
    rgba[..., 3] = (v ** 0.8 * 220).astype(np.uint8)
    CACHE.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(png, optimize=True)
    bounds = [[s_, w_], [n_, e_]]
    meta.write_text(json.dumps(bounds))
    return png, bounds


# --- coordinates and saving --------------------------------------------------------------------------

def ortho_crs(ortho: Path):
    """The orthomosaic's own projected CRS: wall_line.csv is written in it (same as the point cloud's)."""
    import rasterio
    with rasterio.open(ortho) as ds:
        return ds.crs


def to_map(latlngs, crs):
    from rasterio.warp import transform
    if not latlngs:
        return []
    xs, ys = transform("EPSG:4326", crs, [p[1] for p in latlngs], [p[0] for p in latlngs])
    return [[round(x, 3), round(y, 3)] for x, y in zip(xs, ys)]


def to_latlng(xy, crs):
    from rasterio.warp import transform
    if not xy:
        return []
    lon, lat = transform(crs, "EPSG:4326", [p[0] for p in xy], [p[1] for p in xy])
    return [[b, a] for a, b in zip(lon, lat)]


def auto_pit(wall, dtm: Path | None):
    """A point PIT_PROBE m off the middle of the wall line on its lower (open) side, from the DTM."""
    if dtm is None:
        return None
    import rasterio
    w = np.asarray(wall, float)
    seg = np.diff(w, axis=0)
    i = int(np.argmax(np.hypot(*seg.T)))
    mid = (w[i] + w[i + 1]) / 2
    d = seg[i] / np.hypot(*seg[i])
    nrm = np.array([-d[1], d[0]])
    probes = [mid + nrm * PIT_PROBE, mid - nrm * PIT_PROBE]
    with rasterio.open(dtm) as ds:
        zs = [v[0] for v in ds.sample([tuple(p) for p in probes])]
    if any(z is None or z <= -1000 for z in zs):
        return None
    return [round(float(c), 3) for c in probes[int(np.argmin(zs))]]


def save_wall_line(out: Path, wall, pit) -> Path:
    path = out / WALL_LINE_NAME
    lines = ["kind,x,y"] + [f"wall,{x:.3f},{y:.3f}" for x, y in wall] + ([f"pit,{pit[0]:.3f},{pit[1]:.3f}"] if pit else [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- HTTP --------------------------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".webp": "image/webp"}

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str, cache="no-cache"):
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _params(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        ortho = find_ortho(q.get("ortho", [""])[0]) if q.get("ortho") else None
        out = Path(q["out"][0]) if q.get("out") else None
        return ortho, out

    def _host_ok(self) -> bool:
        """Only requests addressed to this machine (a DNS-rebinding page sends its own host name)."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _post_ok(self) -> bool:
        """POSTs only from the tool's own pages: JSON body (a cross-site page cannot send that without a CORS
        preflight this server never grants) and, when the browser says, the same origin."""
        if not self._host_ok() or not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            host = urllib.parse.urlparse(origin).hostname or ""
            return host in ("127.0.0.1", "localhost", "::1")
        return True

    def do_GET(self):
        if not self._host_ok():
            return self.send_error(403)
        route = urllib.parse.urlparse(self.path).path
        if route in ("/", "/index.html"):
            return self._file(HERE / "index.html", "text/html; charset=utf-8", "no-store")
        if route == "/facade":
            return self._file(HERE / "facade.html", "text/html; charset=utf-8", "no-store")
        if route == "/tiles3d":
            return self._file(HERE / "tiles3d.html", "text/html; charset=utf-8", "no-store")
        if route == "/api/tiles3d/scene":
            return self._tiles_scene()
        if route.startswith("/3dtiles/"):
            return self._tile_file(route)
        if route in ("/facade/preview.jpg", "/facade/setmap.png"):
            res_dir = self._result_dir()
            name = "facade_preview.jpg" if route.endswith(".jpg") else "facade_setmap.png"
            return self._file(res_dir / name if res_dir else Path(), "image/jpeg" if name.endswith("jpg") else "image/png")
        if route == "/api/facade/info":
            return self._facade_info()
        if route == "/ping":
            return self._json({"app": "wall_digitizer"})
        if route == "/api/info":
            ortho, out = self._params()
            if ortho is None:
                return self._json({"error": "정사영상(GeoTIFF)을 찾지 못했습니다."}, 404)
            src = tile_source(ortho)
            dtm = find_dtm(ortho, out)
            existing = None
            if out and (out / WALL_LINE_NAME).exists():
                wall, pit = read_wall_line(out / WALL_LINE_NAME)
                crs = ortho_crs(ortho)
                existing = {"wall": to_latlng(wall or [], crs), "pit": to_latlng([pit], crs)[0] if pit else None}
            return self._json({
                "ortho": str(ortho), "status": src["status"], "log": src.get("log"), "reused": src.get("reused"),
                "max_zoom": src.get("max_zoom"), "min_zoom": src.get("min_zoom"), "bounds": src.get("bounds"),
                "dtm": str(dtm) if dtm else None, "out": str(out) if out else None,
                "out_exists": bool(out and out.is_dir()), "existing": existing})
        if route.startswith("/tiles/"):
            ortho, _ = self._params()
            if ortho is None:
                return self.send_error(404)
            src = tile_source(ortho)
            base = Path(src["dir"]).resolve()
            f = (base / urllib.parse.unquote(route[len("/tiles/"):])).resolve()
            if base not in f.parents:                      # no absolute paths or ../ out of the tile folder
                return self.send_error(404)
            return self._file(f, "image/webp")
        if route == "/slope.png":
            ortho, out = self._params()
            dtm = find_dtm(ortho, out) if ortho else None
            if dtm is None:
                return self.send_error(404)
            return self._file(slope_overlay(dtm)[0], "image/png")
        if route == "/api/slope":
            ortho, out = self._params()
            dtm = find_dtm(ortho, out) if ortho else None
            return self._json({"bounds": slope_overlay(dtm)[1] if dtm else None})
        self.send_error(404)

    def _tiles_scene(self):
        """The grouting scene JSON (?scene=path) plus the id its tileset is served under."""
        from groundwater.viz import tiles3d

        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        p = Path(q.get("scene", [""])[0])
        if not p.is_file():
            return self._json({"error": f"장면 파일이 없습니다: {p}"}, 404)
        scene = json.loads(p.read_text(encoding="utf-8"))
        tiles = tiles3d.find_tileset(scene.get("tiles", ""))
        if tiles is None:
            return self._json({"error": f"타일 모델을 찾을 수 없습니다: {scene.get('tiles')}"}, 404)
        tid = hashlib.sha1(str(tiles.resolve()).encode()).hexdigest()[:12]
        _tilesets[tid] = tiles
        scene["tileset_url"] = f"/3dtiles/{tid}/tileset.json"
        return self._json(scene)

    def _tiles_snapshot(self):
        """A view of the 3D page (PNG data URI of its canvas) saved in <scene folder>/3D화면저장/."""
        import base64
        import time

        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        scene = Path(str(body.get("scene") or ""))
        data = str(body.get("png") or "")
        if not (scene.is_file() and scene.suffix.lower() == ".json"):
            return self._json({"error": f"장면 파일이 없습니다: {scene}"}, 400)
        if not data.startswith("data:image/png;base64,"):
            return self._json({"error": "PNG 자료가 아닙니다."}, 400)
        raw = base64.b64decode(data.split(",", 1)[1])
        out = scene.parent / "3D화면저장"
        out.mkdir(exist_ok=True)
        name = "".join(c for c in str(body.get("name") or "") if c.isalnum() or c in "-_ ").strip()
        path = out / f"{scene.stem}_{name or time.strftime('%Y%m%d_%H%M%S')}.png"
        path.write_bytes(raw)
        return self._json({"path": str(path), "bytes": len(raw)})

    def _tile_file(self, route: str):
        """/3dtiles/<id>/<path inside the tileset>: straight out of the .3tz (or folder), no extraction."""
        from groundwater.viz import tiles3d

        parts = route.split("/", 3)
        if len(parts) < 4 or parts[2] not in _tilesets:
            return self.send_error(404)
        data = tiles3d.archive(_tilesets[parts[2]]).read(urllib.parse.unquote(parts[3]))
        if data is None:
            return self.send_error(404)
        ctype = {"json": "application/json", "glb": "model/gltf-binary", "b3dm": "application/octet-stream",
                 "jpg": "image/jpeg", "png": "image/png", "ktx2": "image/ktx2"}.get(parts[3].rsplit(".", 1)[-1].lower(),
                                                                                  "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "max-age=86400")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _result_dir(self) -> Path | None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        d = Path(q["result"][0]) if q.get("result") else None
        return d if d and (d / "facade_preview.json").exists() else None

    def _site_dir(self) -> Path | None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        d = Path(q["site"][0]) if q.get("site") and q["site"][0] else None
        return d if d and d.is_dir() else None

    def _facade_info(self):
        import pandas as pd

        res_dir = self._result_dir()
        if res_dir is None:
            return self._json({"error": "분석 결과(입면 영상)가 없습니다. 분석을 한 번 실행하세요."}, 404)
        meta = json.loads((res_dir / "facade_preview.json").read_text(encoding="utf-8"))
        auto = []
        if (res_dir / "traces.csv").exists():
            t = pd.read_csv(res_dir / "traces.csv")
            t = t[t["source"] == "auto"]
            auto = [{"u1": r.u1, "v1": r.v1, "u2": r.u2, "v2": r.v2,
                     "family": None if pd.isna(r.family) else int(r.family)} for r in t.itertuples()]
        manual = []
        if (res_dir / MANUAL_TRACES).exists():
            m = pd.read_csv(res_dir / MANUAL_TRACES)
            for tid, g in m.groupby("trace_id", sort=False):
                note = g["note"].iloc[0] if "note" in g and not pd.isna(g["note"].iloc[0]) else ""
                manual.append({"id": int(tid), "note": str(note), "points": g[["u", "v"]].to_numpy().tolist()})
        site = self._site_dir()
        region = load_polygon(site / REGION_FILE) if site else None
        return self._json({"result": str(res_dir), "meta": meta, "auto": auto, "manual": manual, "region": region})

    def _facade_save(self):
        res_dir = self._result_dir()
        if res_dir is None:
            return self._json({"error": "분석 결과 폴더를 찾을 수 없습니다."}, 400)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        rows = ["trace_id,u,v,note"]
        n = 0
        for t in body.get("traces") or []:
            pts = t.get("points") or []
            if len(pts) < 2:
                continue
            n += 1
            note = str(t.get("note", "")).replace(",", " ").replace("\n", " ")
            rows += [f"{int(t['id'])},{u:.3f},{v:.3f},{note}" for u, v in pts]
        (res_dir / MANUAL_TRACES).write_text("\n".join(rows) + "\n", encoding="utf-8")
        site = self._site_dir()
        region_path = None
        if site:
            rp = site / REGION_FILE
            if body.get("region") and len(body["region"]) >= 3:
                save_polygon(rp, body["region"])
                region_path = str(rp)
            elif rp.exists():
                rp.unlink()
        return self._json({"n_traces": n, "traces_path": str(res_dir / MANUAL_TRACES), "region_path": region_path})

    def do_POST(self):
        if not self._post_ok():
            return self.send_error(403)
        try:
            return self._post()
        except (ValueError, KeyError, TypeError) as e:          # bad body / parameters: an answer, not a dropped socket
            return self._json({"error": f"잘못된 요청: {e}"}, 400)

    def _post(self):
        route = urllib.parse.urlparse(self.path).path
        if route == "/api/facade/save":
            return self._facade_save()
        if route == "/api/tiles3d/snapshot":
            return self._tiles_snapshot()
        if route != "/api/save":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        ortho, out = self._params()
        out_s = str(body.get("out") or out or "").strip()
        out = Path(out_s) if out_s else None
        if out is None or not out.is_dir():
            return self._json({"error": f"저장 폴더가 없습니다: {out}"}, 400)
        if ortho is None:
            return self._json({"error": "정사영상을 찾을 수 없습니다."}, 400)
        crs = ortho_crs(ortho)
        wall = to_map(body.get("wall") or [], crs)
        if len(wall) < 2:
            return self._json({"error": "벽면선은 점이 2개 이상 필요합니다."}, 400)
        pit = to_map([body["pit"]], crs)[0] if body.get("pit") else None
        pit_auto = False
        if pit is None:
            pit = auto_pit(wall, find_dtm(ortho, out) if ortho else None)
            pit_auto = pit is not None
        path = save_wall_line(out, wall, pit)
        length = float(np.sum(np.hypot(*np.diff(np.asarray(wall), axis=0).T)))
        return self._json({"path": str(path), "length_m": round(length, 1), "n": len(wall),
                           "pit": to_latlng([pit], crs)[0] if pit else None, "pit_auto": pit_auto,
                           "crs": crs.to_string()})


def is_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=1.5) as r:
            return b"wall_digitizer" in r.read()
    except OSError:
        return False


def facade_url(port: int, result_dir, site_dir) -> str:
    q = urllib.parse.urlencode({"result": str(result_dir), "site": str(site_dir) if site_dir else ""})
    return f"http://localhost:{port}/facade?{q}"


def tiles3d_url(port: int, scene_path) -> str:
    return f"http://localhost:{port}/tiles3d?" + urllib.parse.urlencode({"scene": str(scene_path)})


def page_url(port: int, ortho, out) -> str:
    q = urllib.parse.urlencode({"ortho": str(ortho), "out": str(out) if out else ""})
    return f"http://localhost:{port}/?{q}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ortho", help="orthomosaic GeoTIFF or a folder to search for one")
    ap.add_argument("--out", help="site folder that receives wall_line.csv")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = page_url(args.port, args.ortho or "", args.out)
    if is_running(args.port):
        print(f"이미 실행 중입니다: {url}")
    else:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(f"벽면 기준선 그리기: {url}")
        if not args.no_browser and args.ortho:
            webbrowser.open(url)
        server.serve_forever()
        return
    if not args.no_browser and args.ortho:
        webbrowser.open(url)


if __name__ == "__main__":
    main()
