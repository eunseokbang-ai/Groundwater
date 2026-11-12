"""Render docs/workflow_manual.md to docs/workflow_manual.pdf (A4) with headless Microsoft Edge.

    .venv\\Scripts\\python.exe docs\\make_manual_pdf.py
"""
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
SRC, OUT = DOCS / "workflow_manual.md", DOCS / "workflow_manual.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; font-size: 10pt; line-height: 1.6;
       color: #1f2937; }
h1 { font-size: 20pt; margin: 0 0 6pt; color: #111827; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; padding-bottom: 3pt; border-bottom: 2px solid #c2410c; color: #111827;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; color: #9a3412; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 18pt; }
hr { border: 0; border-top: 1px solid #e5e7eb; margin: 14pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 4pt 6pt; vertical-align: top; text-align: left; }
th { background: #f3f4f6; font-weight: 700; }
code { font-family: Consolas, 'D2Coding', monospace; font-size: 8.8pt; background: #f3f4f6; padding: 0 3pt;
       border-radius: 3px; }
pre { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8pt 10pt; font-size: 8.6pt;
      line-height: 1.45; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
strong { color: #111827; }
"""


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>드론 기반 절리군 분석 작업 설명서</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    edge = next((p for p in [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                             Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")] if p.exists()), None)
    if edge is None:
        raise SystemExit("Microsoft Edge not found")
    tmp = Path(tempfile.mkdtemp(prefix="manual-pdf-"))
    try:
        page = tmp / "manual.html"
        page.write_text(html, encoding="utf-8")
        OUT.unlink(missing_ok=True)
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", f"--user-data-dir={tmp / 'profile'}",
                        "--no-pdf-header-footer", f"--print-to-pdf={OUT}", page.as_uri()],
                       capture_output=True, timeout=120)
        deadline = time.time() + 60  # msedge.exe can return before its renderer has written the file
        while time.time() < deadline and not (OUT.exists() and OUT.stat().st_size > 0):
            time.sleep(0.5)
        time.sleep(1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {OUT}" if OUT.exists() else "Edge ran but wrote no PDF")


if __name__ == "__main__":
    main()
