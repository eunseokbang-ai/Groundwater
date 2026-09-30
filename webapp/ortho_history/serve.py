"""Serve the orthophoto history viewer on localhost and open it in the default browser.

The page and layers.json are sent with `Cache-Control: no-store` and tiles with `no-cache`
(revalidated via Last-Modified), so the browser always shows the current code and imagery
instead of a cached copy.

Usage:
    .venv\\Scripts\\python.exe webapp\\ortho_history\\serve.py [--port 8610] [--no-browser]
"""
import argparse
import http.server
import sys
import urllib.request
import webbrowser
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".webp": "image/webp",
                      ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache" if self.path.startswith("/tiles/") else "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # tiles outside the imagery 404 by design; keep the console quiet


class Server(http.server.ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already taken.
    allow_reuse_address = False


def is_viewer(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/layers.json", timeout=2) as r:
            return r.status == 200 and b'"layers"' in r.read(4096)
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8610)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not (HERE / "layers.json").exists():
        sys.exit("layers.json이 없습니다. 먼저 build_tiles.py를 실행하세요.")

    for port in range(args.port, args.port + 10):
        try:
            server = Server(("127.0.0.1", port), partial(Handler, directory=str(HERE)))
            break
        except OSError:
            if is_viewer(port):
                url = f"http://localhost:{port}/"
                print(f"뷰어가 이미 실행 중입니다: {url}")
                if not args.no_browser:
                    webbrowser.open(url)
                return
    else:
        sys.exit(f"포트 {args.port}-{args.port + 9}를 모두 사용할 수 없습니다.")

    url = f"http://localhost:{port}/"
    print(f"정사영상 변천 뷰어: {url}")
    print("이 창을 닫거나 Ctrl+C를 누르면 종료됩니다.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
