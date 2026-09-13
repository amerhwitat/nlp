from pathlib import Path

from python.thamudic.pdf_export import write_records_pdf, write_report_pdf
from python.thamudic.translation_log import export_records


def test_records_pdf(tmp_path: Path):
    output = tmp_path / "history.pdf"
    write_records_pdf([{"source": "abc", "translation": "test", "record_hash": "x"}], output)
    assert output.exists()
    assert output.read_bytes().startswith(b"%PDF")


def test_report_pdf(tmp_path: Path):
    output = tmp_path / "report.pdf"
    write_report_pdf({"source_language": "Dadanitic", "translation": "test"}, output)
    assert output.read_bytes().startswith(b"%PDF")
