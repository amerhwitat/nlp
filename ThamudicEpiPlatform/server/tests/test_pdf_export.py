from __future__ import annotations

from server.pdf_export import export_research_pdf


def test_export_contains_pdf_and_provenance_manifest():
    pdf, manifest = export_research_pdf({
        "title": "Historical Object",
        "source_sha256": "0" * 64,
        "source_id": "demo-source",
        "sections": [{"title": "Reading", "body": "𐪑 transliteration and translation"}],
        "provenance": [{"reviewer": "researcher", "confidence": 0.8}],
        "citations": ["Unicode / CLDR transliteration guidance"],
    })
    assert pdf.startswith(b"%PDF")
    assert manifest["schema_version"] == "1.0"
    assert manifest["source"]["sha256"] == "0" * 64
    assert len(manifest["pdf_sha256"]) == 64
