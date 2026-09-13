from pathlib import Path

from thamudic.artifacts_db import ArtifactDatabase


def test_artifact_round_trip(tmp_path: Path):
    db = ArtifactDatabase(tmp_path / "artifacts.sqlite")
    try:
        artifact_id = db.create_artifact({
            "title": "test inscription", "source": "pytest", "media_type": "text",
            "original_text": "test", "source_language": "Dadanitic", "script_variant": "Dadanitic",
            "target_language": "en", "tags": ["test"],
        })
        db.add_scan(artifact_id, {"matched": True, "language": "Old North Arabian", "script_variant": "Dadanitic", "confidence": 0.9, "codepoints": ["U+0061"]})
        db.add_translation(artifact_id, {"target_language": "en", "transliteration": "test", "translation": None, "translation_status": "not_available", "confidence": "unknown"})
        result = db.get(artifact_id)
        assert result is not None
        assert result["id"] == artifact_id
        assert result["tags"] == ["test"]
        assert len(result["scans"]) == 1
        assert result["translations"][0]["status"] == "not_available"
        assert db.statistics()["artifacts"] == 1
    finally:
        db.close()
