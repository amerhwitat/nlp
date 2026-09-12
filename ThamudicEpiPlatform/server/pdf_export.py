from __future__ import annotations

import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def export_research_pdf(report: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Generate a provenance-aware research PDF and a machine-readable manifest.

    The caller supplies scholarly/transliteration/translation data; this function never
    invents historical readings. Confidence and provenance are displayed verbatim.
    """
    report_id = str(report.get("report_id") or uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=str(report.get("title", "Historical Object Research Report")))
    styles = getSampleStyleSheet()
    story: list[Any] = [Paragraph(str(report.get("title", "Historical Object Research Report")), styles["Title"])]
    story.append(Paragraph(f"Report ID: {report_id}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {generated_at}", styles["Normal"]))
    story.append(Spacer(1, 12))

    for section in report.get("sections", []):
        story.append(Paragraph(str(section.get("title", "Section")), styles["Heading2"]))
        body = section.get("body")
        if body:
            story.append(Paragraph(str(body).replace("\n", "<br/>"), styles["BodyText"]))
        rows = section.get("rows")
        if rows:
            table = Table([[str(c) for c in row] for row in rows], repeatRows=1)
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "black"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(table)
        story.append(Spacer(1, 10))

    citations = report.get("citations", [])
    if citations:
        story.append(Paragraph("Citations and provenance", styles["Heading2"]))
        for citation in citations:
            story.append(Paragraph(str(citation), styles["BodyText"]))

    doc.build(story)
    pdf = buffer.getvalue()
    manifest = {
        "schema_version": "1.0",
        "report_id": report_id,
        "generated_at": generated_at,
        "source": {
            "sha256": str(report.get("source_sha256", hashlib.sha256(pdf).hexdigest())),
            "source_id": report.get("source_id"),
            "rights": report.get("rights"),
        },
        "sections": report.get("sections", []),
        "provenance": report.get("provenance", []),
        "citations": citations,
        "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
    }
    # Validate serializability before returning the sidecar.
    json.dumps(manifest, ensure_ascii=False)
    return pdf, manifest
