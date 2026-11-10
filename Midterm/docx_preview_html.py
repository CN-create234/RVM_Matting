"""Fallback visual QA renderer used when LibreOffice is unavailable on Windows."""

import base64
import html
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Midterm" / "RVM深度信息融合调研与实验报告.docx"
OUTPUT = ROOT / "Midterm" / "_report_render" / "preview.html"


def iter_blocks(document):
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def paragraph_html(paragraph):
    xml = paragraph._p.xml
    if "w:type=\"page\"" in xml or "w:type='page'" in xml:
        return '<div class="page-break"></div>'
    if "<a:blip" in xml:
        for rel_id in paragraph._p.xpath(".//a:blip/@r:embed"):
            part = paragraph.part.related_parts[rel_id]
            encoded = base64.b64encode(part.blob).decode("ascii")
            content_type = part.content_type
            return f'<figure><img src="data:{content_type};base64,{encoded}"></figure>'
    text = html.escape(paragraph.text).replace("\n", "<br>")
    style = paragraph.style.name if paragraph.style else "Normal"
    if not text:
        return '<p class="empty">&nbsp;</p>'
    if style == "Title":
        return f"<h1 class=\"title\">{text}</h1>"
    if style.startswith("Heading 1"):
        return f"<h1>{text}</h1>"
    if style.startswith("Heading 2"):
        return f"<h2>{text}</h2>"
    if style.startswith("Heading 3"):
        return f"<h3>{text}</h3>"
    if style.startswith("List Bullet"):
        return f"<div class=\"bullet\">• {text}</div>"
    if style.startswith("List Number"):
        return f"<div class=\"number\">{text}</div>"
    css = " centered" if paragraph.alignment == 1 else ""
    return f'<p class="body{css}">{text}</p>'


def table_html(table):
    rows = []
    for row_index, row in enumerate(table.rows):
        cells = []
        tag = "th" if row_index == 0 else "td"
        for cell in row.cells:
            text = "<br>".join(html.escape(p.text) for p in cell.paragraphs)
            cells.append(f"<{tag}>{text}</{tag}>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def main():
    document = Document(DOCX)
    body = []
    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            rendered = paragraph_html(block)
            body.append(rendered)
        else:
            body.append(table_html(block))
    css = """
    @page { size: Letter; margin: 0.72in 0.8in; }
    * { box-sizing: border-box; }
    body { font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; color: #000; font-size: 10.5pt; line-height: 1.38; }
    h1.title { text-align:center; font-size:24pt; margin:1.05in 0 0.25in; page-break-after:avoid; }
    h1 { font-size:16pt; margin:16pt 0 7pt; page-break-after:avoid; }
    h2 { font-size:13pt; margin:13pt 0 6pt; page-break-after:avoid; }
    h3 { font-size:11.5pt; margin:10pt 0 5pt; page-break-after:avoid; }
    p.body { margin:0 0 6pt; text-align:justify; }
    p.centered { text-align:center; }
    p.empty { margin:0 0 12pt; }
    .bullet, .number { margin:0 0 5pt 0.24in; text-indent:-0.18in; }
    .page-break { break-before:page; }
    table { width:100%; border-collapse:collapse; margin:8pt 0 12pt; font-size:9.3pt; break-inside:auto; }
    tr { break-inside:avoid; }
    th, td { border:1px solid #d9d9d9; padding:7px 8px; vertical-align:middle; }
    th { color:white; background:#1f4e78; text-align:center; }
    tr:nth-child(odd) td { background:#eaf2f8; }
    figure { margin:10pt auto 4pt; text-align:center; break-inside:avoid; }
    figure img { max-width:100%; height:auto; }
    a { color:#0563c1; }
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>" + css + "</style></head><body>" + "".join(body) + "</body></html>",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
