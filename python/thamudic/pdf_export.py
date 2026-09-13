"""PDF exports for translation history and ancient-script reports.

Uses ReportLab Platypus so reports can flow across pages and remain readable for
large translation histories. PDF generation is optional at import time; callers
receive a clear RuntimeError when ReportLab is not installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _escape(value: Any) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _flatten(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return str(value)


def _reportlab():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
    except ImportError as exc:
        raise RuntimeError("PDF export requires ReportLab; install the project PDF dependencies") from exc
    return A4, getSampleStyleSheet, ParagraphStyle, TA_CENTER, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, colors


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(40, 24, "Ancient Language NLP — evidence-aware export")
    canvas.drawRightString(555, 24, f"Page {doc.page}")
    canvas.restoreState()


def _document(title: str, output: str | Path):
    A4, getSampleStyleSheet, ParagraphStyle, TA_CENTER, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, colors = _reportlab()
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=44, bottomMargin=38, title=title, author="amerhwitat/nlp")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="Key", parent=styles["Heading3"], fontSize=10, leading=12, spaceBefore=8, spaceAfter=4))
    return doc, styles, Paragraph, Spacer, Table, TableStyle, colors


def write_records_pdf(records: Iterable[dict[str, Any]], output: str | Path, title: str = "Translation History") -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    doc, styles, Paragraph, Spacer, Table, TableStyle, colors = _document(title, output)
    story = [Paragraph(_escape(title), styles["ReportTitle"])]
    rows = list(records)
    story.append(Paragraph(f"Records: {len(rows)}", styles["Small"]))
    for index, record in enumerate(rows, 1):
        story.append(Paragraph(f"Record {index}", styles["Heading2"]))
        data = [[Paragraph("Field", styles["Small"]), Paragraph("Value", styles["Small"])]]
        for key, value in record.items():
            rendered = _escape(_flatten(value)).replace("\n", "<br/>")
            data.append([Paragraph(_escape(key), styles["Small"]), Paragraph(rendered, styles["Small"])])
        table = Table(data, colWidths=[125, 390], repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([table, Spacer(1, 12)])
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output


def write_report_pdf(report: dict[str, Any], output: str | Path, title: str | None = None) -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    title = title or f"{report.get('script_information', {}).get('name', 'Ancient Script')} — Script Report"
    doc, styles, Paragraph, Spacer, Table, TableStyle, colors = _document(title, output)
    story = [Paragraph(_escape(title), styles["ReportTitle"])]
    for key, value in report.items():
        story.append(Paragraph(_escape(key.replace("_", " ").title()), styles["Key"]))
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                rendered = _escape(_flatten(subvalue)).replace("\n", "<br/>")
                story.append(Paragraph(f"<b>{_escape(subkey.replace('_', ' ').title())}:</b> {rendered}", styles["Small"]))
        elif isinstance(value, list):
            story.append(Paragraph(_escape(_flatten(value)).replace("\n", "<br/>"), styles["Small"]))
        else:
            story.append(Paragraph(_escape(value), styles["Small"]))
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output


def records_pdf_bytes(records: Iterable[dict[str, Any]], title: str = "Translation History") -> bytes:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        temp = Path(handle.name)
    try:
        write_records_pdf(records, temp, title)
        return temp.read_bytes()
    finally:
        temp.unlink(missing_ok=True)


def report_pdf_bytes(report: dict[str, Any], title: str | None = None) -> bytes:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        temp = Path(handle.name)
    try:
        write_report_pdf(report, temp, title)
        return temp.read_bytes()
    finally:
        temp.unlink(missing_ok=True)
