"""Verify the JSON export script keeps the static app in sync with Python sources."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exported_data() -> dict[str, dict]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "nlp-studio" / "scripts" / "export_static_data.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr

    data_dir = repo_root / "nlp-studio" / "web" / "data"
    return {
        "ona": json.loads((data_dir / "ona_chars.json").read_text(encoding="utf-8")),
        "registry": json.loads((data_dir / "alphabet_registry.json").read_text(encoding="utf-8")),
        "corpus": json.loads((data_dir / "translation_corpus.json").read_text(encoding="utf-8")),
        "profiles": json.loads((data_dir / "source_language_profiles.json").read_text(encoding="utf-8")),
    }


def test_ona_lookup_table(exported_data):
    ona = exported_data["ona"]
    assert ona["first"] == 0x10A80
    assert ona["last"] == 0x10A9F
    assert ona["by_character"]["𐪀"] == "h"
    assert ona["by_character"]["𐪁"] == "l"
    assert len(ona["characters"]) == 32
    assert ona["variant_forms"]


def test_alphabet_registry_exported(exported_data):
    registry = exported_data["registry"]
    assert "ancient-north-arabian" in registry
    profile = registry["ancient-north-arabian"]
    assert profile["name"] == "Ancient North Arabian"
    assert profile.get("direction") == "rtl"


def test_translation_corpus_exported(exported_data):
    corpus = exported_data["corpus"]
    assert len(corpus) == 4
    first = corpus[0]
    assert first["identifier"] == "TIJ 503"
    assert "en" in first["translations"]
    assert "ar" in first["translations"]
    assert first["source_url"].startswith("http")


def test_source_language_profiles_exported(exported_data):
    profiles = exported_data["profiles"]
    assert "ancient-north-arabian" in profiles
    assert profiles["ancient-north-arabian"]["ranges"]
