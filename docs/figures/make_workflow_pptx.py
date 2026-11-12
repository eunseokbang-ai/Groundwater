"""The workflow deck as a PowerPoint file: every slide as its SVG drawing (sharp at any size; in PowerPoint,
right-click > "Convert to Shape" makes its text and shapes editable) with a JPEG copy for viewers without SVG
support, and the speaker notes from 워크플로_발표대본.md. Run after make_workflow_slides.py.

    .venv\\Scripts\\python.exe docs\\figures\\make_workflow_pptx.py

Needs python-pptx (pip install python-pptx).
"""
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
OUT = DOCS / "워크플로_발표자료.pptx"
NOTES = DOCS / "워크플로_발표대본.md"
# slide order of the deck (the sub-slides 3b and 14b share pages 3 and 14), as in the PDF
ORDER = ["01", "02", "03", "03b"] + [f"{i:02d}" for i in range(4, 15)] + ["14b"] + [f"{i:02d}" for i in range(15, 21)]
SVG_EXT = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"          # PowerPoint's SVG image extension
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_SVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


def notes_by_slide() -> dict:
    """'03', '03b', ... -> (title, notes text) from the script's '### N쪽 — title' sections (```-fenced text)."""
    text = NOTES.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"^### (\d+)쪽( \(서브\))? — (.+?)\n```\n(.*?)```", text, re.S | re.M):
        key = f"{int(m.group(1)):02d}" + ("b" if m.group(2) else "")
        out[key] = (m.group(3).strip(), m.group(4).strip())
    return out


def fallback_jpeg(png: Path, width: int = 1920) -> bytes:
    from PIL import Image

    im = Image.open(png).convert("RGB")
    im = im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    from lxml import etree
    from pptx import Presentation
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI

    notes = notes_by_slide()
    prs = Presentation()
    prs.slide_width, prs.slide_height = 12192000, 6858000             # 16:9, 13.33 x 7.5 in
    blank = prs.slide_layouts[6]
    missing = []
    for i, key in enumerate(ORDER, start=1):
        svg, png = HERE / f"workflow_{key}.svg", HERE / f"workflow_{key}.png"
        if not svg.exists() or not png.exists():
            sys.exit(f"Run make_workflow_slides.py first (missing workflow_{key})")
        slide = prs.slides.add_slide(blank)
        pic = slide.shapes.add_picture(io.BytesIO(fallback_jpeg(png)), 0, 0, prs.slide_width, prs.slide_height)
        title, text = notes.get(key, (f"{key}", ""))
        if not text:
            missing.append(key)
        pic.name = f"슬라이드 {key}"
        pic._element.nvPicPr.cNvPr.set("descr", title)                 # alt text: the slide title
        # the SVG drawing itself, shown by PowerPoint 2016+ (the JPEG above is its fallback)
        part = Part(PackURI(f"/ppt/media/workflow_{key}.svg"), "image/svg+xml", prs.part.package, svg.read_bytes())
        rid = slide.part.relate_to(part, RT.IMAGE)
        blip = pic._element.blipFill.find(f"{{{NS_A}}}blip")
        ext = etree.fromstring(f'<a:extLst xmlns:a="{NS_A}"><a:ext uri="{SVG_EXT}">'
                               f'<asvg:svgBlip xmlns:asvg="{NS_SVG}" xmlns:r="{NS_R}" r:embed="{rid}"/></a:ext></a:extLst>')
        blip.append(ext)
        slide.notes_slide.notes_text_frame.text = text
    prs.save(OUT)
    print(f"wrote {OUT} ({len(ORDER)} slides, {OUT.stat().st_size / 1e6:.1f} MB)"
          + (f"; no notes for {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
