"""Shared SVG helpers for the presentation figures in docs/figures (16:9, 1600x900).

Text widths are measured with the real Malgun Gothic font (via Pillow) so label
boxes fit; PNGs are rendered with headless Microsoft Edge at 2x.
"""
import html
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

W, H = 1600, 900
FONT = "'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
TEXT, MUTED = "#1f2937", "#6b7280"
MARKER_COLORS = {"blue": "#1d6fc2", "red": "#d9480f", "gray": "#374151", "green": "#15803d"}


def fmt(p):
    return f"{p[0]:.1f},{p[1]:.1f}"


def curve(points):
    """Catmull-Rom spline through `points` as cubic Bezier segments (current point = points[0])."""
    segs = []
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1, p2 = points[i], points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        segs.append(f"C{fmt(c1)} {fmt(c2)} {fmt(p2)}")
    return " ".join(segs)


def lines(points):
    return " ".join(f"L{fmt(p)}" for p in points)


def attrs(**kw):
    return " ".join(f'{k.rstrip("_").replace("_", "-")}="{v}"' for k, v in kw.items() if v is not None)


def path(d, **kw):
    return f'<path d="{d}" {attrs(**kw)}/>'


def rect(x, y, w, h, **kw):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" {attrs(**kw)}/>'


_FONT_FILES = {False: Path(r"C:\Windows\Fonts\malgun.ttf"), True: Path(r"C:\Windows\Fonts\malgunbd.ttf")}
_font_cache = {}


def text_width(s, size, weight=400):
    """Rendered width in px, measured with the actual Malgun Gothic font when available."""
    bold = weight >= 600
    try:
        from PIL import ImageFont

        key = (bold, size)
        if key not in _font_cache:
            _font_cache[key] = ImageFont.truetype(str(_FONT_FILES[bold]), size)
        return _font_cache[key].getlength(s)
    except (ImportError, OSError):
        per_char = [1.0 if "가" <= ch <= "힣" or ch in "①②③④→" else 0.32 if ch == " " else 0.62 for ch in s]
        return sum(per_char) * size * (1.06 if bold else 1.0)


def wrap(s, max_width, size, weight=400):
    """Greedy word wrap on spaces, using measured widths."""
    out, current = [], ""
    for word in s.split(" "):
        trial = f"{current} {word}".strip()
        if current and text_width(trial, size, weight) > max_width:
            out.append(current)
            current = word
        else:
            current = trial
    return out + ([current] if current else [])


def text(x, y, s, size=18, weight=400, fill=None, anchor="start", **kw):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill or TEXT}" '
            f'text-anchor="{anchor}" {attrs(**kw)}>{html.escape(s)}</text>')


def label(x, y, rows, anchor="start", pad=8, fill="white", opacity=0.92, stroke="#cbd5e1"):
    """Box with one or more text rows; rows = [(text, size, weight, color)]. (x, y) = top anchor point."""
    width = max(text_width(r[0], r[1], r[2]) for r in rows) + 2 * pad
    height = sum(r[1] * 1.3 for r in rows) + 2 * pad - rows[-1][1] * 0.3
    left = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
    out = [f'<rect x="{left:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="6" '
           f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}"/>']
    cy = y + pad
    for s, size, weight, color in rows:
        cy += size
        out.append(text(left + pad, round(cy - size * 0.12, 1), s, size, weight, color))
        cy += size * 0.3
    return "\n".join(out)


def pill(x, y, s, fill, color, size=13, stroke=None, dash=None):
    """Small rounded tag; (x, y) = left, vertical centre. Returns (svg, width)."""
    w = text_width(s, size, 700) + 14
    h = size + 8
    svg = (rect(x, y - h / 2, w, h, rx=h / 2, fill=fill, stroke=stroke, stroke_dasharray=dash)
           + text(round(x + 7, 1), round(y + size * 0.36, 1), s, size, 700, color))
    return svg, w


def image(x, y, w, h, png_bytes):
    """Embedded PNG, scaled to fit the box without distortion."""
    import base64

    uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" '
            f'href="{uri}" xlink:href="{uri}"/>')


def arrow(d, color, width=3, dash=None, marker="blue"):
    return path(d, fill="none", stroke=color, stroke_width=width, stroke_linecap="round",
                stroke_dasharray=dash, marker_end=f"url(#ah-{marker})")


def marker_defs():
    size = {"gray": 4}
    return "\n".join(
        f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="{size.get(name, 4.5)}" '
        f'markerHeight="{size.get(name, 4.5)}" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
        for name, color in MARKER_COLORS.items()
    )


def document(defs, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">\n'
            f'<defs>\n{marker_defs()}\n{defs}\n</defs>\n{body}\n</svg>\n')


def render_png(svg_path: Path, png_path: Path) -> str:
    edge = next((p for p in [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                             Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")] if p.exists()), None)
    if edge is None:
        return "Edge not found: PNG skipped"
    png_path.unlink(missing_ok=True)
    profile = tempfile.mkdtemp(prefix="edge-render-")  # never touch the user's own Edge profile
    try:
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        "--force-device-scale-factor=2", f"--window-size={W},{H}", f"--screenshot={png_path}",
                        svg_path.as_uri()], capture_output=True, timeout=120)
        # msedge.exe can return before its renderer child has written the file.
        deadline = time.time() + 60
        while time.time() < deadline and not (png_path.exists() and png_path.stat().st_size > 0):
            time.sleep(0.5)
        time.sleep(1.0)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    return f"Wrote {png_path}" if png_path.exists() else "Edge ran but wrote no PNG: open the SVG in a browser instead"


def save(svg: str, out_dir: Path, name: str) -> None:
    svg_path = out_dir / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {svg_path}")
    print(render_png(svg_path, out_dir / f"{name}.png"))
