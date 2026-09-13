"""Exportable, evidence-aware summaries and complete script reports."""
from __future__ import annotations

import json
from typing import Any

from .ancient_alphabet_registry import language_profile, variations, translation_capabilities, translation_directions
from .universal_translation import translate_ancient


def build_script_summary(language: str) -> dict[str, Any]:
    p = language_profile(language)
    return {
        "language_id": p["id"], "name": p.get("name", p["id"]), "iso639": p.get("iso639", []),
        "original_script": p.get("scripts", p.get("script", [])), "script_family": p.get("script_family"),
        "script_type": p.get("script_type"), "writing_direction": p.get("direction"),
        "writing_direction_description": p.get("writing_direction_description"), "unicode_blocks": p.get("unicode_blocks", []),
        "variations": list(variations(language)), "dating": p.get("dating"), "dating_status": p.get("dating_status"),
        "geographic_scope": p.get("geographic_scope"), "materials": p.get("materials", []),
        "related_scripts": p.get("related_scripts", []), "translation_modes": list(translation_capabilities(language)),
        "translation_directions": translation_directions(language), "transliteration_systems": p.get("transliteration_systems", []),
        "notes": p.get("notes"),
        "evidence_policy": "Separate attested original text, scholarly transliteration, corpus-backed translation, model output, and reconstructed/uncertain readings.",
        "source_metadata": {"unicode_reference": "https://www.unicode.org/charts/", "dating_note": "Historical dates are approximate and should be checked against corpus-specific scholarship."},
    }


def build_script_report(language: str, original_text: str, target_language: str = "en") -> dict[str, Any]:
    if not original_text.strip(): raise ValueError("original_text is required")
    if not target_language.strip(): raise ValueError("target_language is required")
    result = translate_ancient(original_text, language, target_language, source_form="script").as_dict()
    return {
        "report_version": "1.1", "script_information": build_script_summary(language),
        "original_text": original_text, "source_language": language, "target_language": target_language,
        "transliteration": result.get("transliteration"), "translation": result.get("translation"),
        "translation_status": result.get("status", result.get("translation_status")),
        "confidence": result.get("confidence"), "provider": result.get("provider"), "provenance": result.get("provenance"),
        "translation_result": result,
    }


def _export(payload: dict[str, Any], fmt: str, filename_base: str) -> tuple[str | bytes, str, str]:
    if fmt == "json": return json.dumps(payload, ensure_ascii=False, indent=2), "application/json; charset=utf-8", f"{filename_base}.json"
    if fmt == "txt":
        lines = [f"{payload.get('script_information', {}).get('name', filename_base)} — Ancient Script Report", ""]
        for key, value in payload.items(): lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n", "text/plain; charset=utf-8", f"{filename_base}.txt"
    if fmt == "md":
        lines = [f"# {payload.get('script_information', {}).get('name', filename_base)} — Ancient Script Report", ""]
        for key, value in payload.items(): lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        return "\n".join(lines) + "\n", "text/markdown; charset=utf-8", f"{filename_base}.md"
    if fmt == "pdf":
        from .pdf_export import report_pdf_bytes
        return report_pdf_bytes(payload), "application/pdf", f"{filename_base}.pdf"
    raise ValueError("format must be json, md, txt, or pdf")


def export_script_summary(language: str, format: str = "json") -> tuple[str | bytes, str, str]:
    return _export(build_script_summary(language), format.casefold(), f"{language_profile(language)['id']}-script-summary")


def export_script_report(language: str, original_text: str, target_language: str = "en", format: str = "json") -> tuple[str | bytes, str, str]:
    return _export(build_script_report(language, original_text, target_language), format.casefold(), f"{language_profile(language)['id']}-script-report")
