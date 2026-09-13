"""Exportable, evidence-aware summaries for a selected ancient script/language."""
from __future__ import annotations

import json
from typing import Any

from .ancient_alphabet_registry import language_profile, variations, translation_capabilities, translation_directions


def build_script_summary(language: str) -> dict[str, Any]:
    p = language_profile(language)
    return {
        "language_id": p["id"],
        "name": p.get("name", p["id"]),
        "iso639": p.get("iso639", []),
        "original_script": p.get("scripts", p.get("script", [])),
        "script_family": p.get("script_family"),
        "script_type": p.get("script_type"),
        "writing_direction": p.get("direction"),
        "writing_direction_description": p.get("writing_direction_description"),
        "unicode_blocks": p.get("unicode_blocks", []),
        "variations": list(variations(language)),
        "dating": p.get("dating"),
        "dating_status": p.get("dating_status"),
        "geographic_scope": p.get("geographic_scope"),
        "materials": p.get("materials", []),
        "related_scripts": p.get("related_scripts", []),
        "translation_modes": list(translation_capabilities(language)),
        "translation_directions": translation_directions(language),
        "transliteration_systems": p.get("transliteration_systems", []),
        "notes": p.get("notes"),
        "evidence_policy": "Separate attested original text, scholarly transliteration, corpus-backed translation, model output, and reconstructed/uncertain readings.",
        "source_metadata": {
            "unicode_reference": "https://www.unicode.org/charts/",
            "dating_note": "Historical dates are approximate and should be checked against corpus-specific scholarship.",
        },
    }


def export_script_summary(language: str, format: str = "json") -> tuple[str, str, str]:
    summary = build_script_summary(language)
    fmt = format.casefold()
    if fmt == "json":
        return json.dumps(summary, ensure_ascii=False, indent=2), "application/json; charset=utf-8", f"{summary['language_id']}-script-summary.json"
    if fmt == "txt":
        lines = [f"{summary['name']} — Script Summary", ""]
        for key, value in summary.items(): lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n", "text/plain; charset=utf-8", f"{summary['language_id']}-script-summary.txt"
    if fmt == "md":
        lines = [f"# {summary['name']} — Script Summary", ""]
        for key, value in summary.items(): lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        return "\n".join(lines) + "\n", "text/markdown; charset=utf-8", f"{summary['language_id']}-script-summary.md"
    raise ValueError("format must be json, md, or txt")
