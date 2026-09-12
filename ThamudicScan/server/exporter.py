from __future__ import annotations

import csv
import io
import json

COLUMNS = [
    "id", "session_id", "source", "text", "transliteration", "confidence",
    "language", "script_variant", "codepoints",
]


def _row(result: dict) -> dict:
    value = {column: result.get(column, "") for column in COLUMNS}
    if isinstance(value["codepoints"], list):
        value["codepoints"] = " ".join(f"U+{cp:04X}" for cp in value["codepoints"])
    return value


def export_results_csv(results: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_row(result) for result in results)
    return buffer.getvalue()


def export_results_json(results: list[dict]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)
