from pathlib import Path

from ancient_languages.pronunciation import resolve_pronunciation


ROOT = Path(__file__).parents[2]
DATA = str(ROOT / "data" / "ancient_languages" / "pronunciation.json")


def test_modern_hebrew_profile():
    result = resolve_pronunciation("shalom", "hebrew.modern", "hebrew.modern.he", DATA)
    assert result.locale == "he-IL"
    assert result.pronunciation_type == "modern"
    assert result.backend == "web-speech"


def test_ancient_hebrew_is_explicitly_reconstructed():
    result = resolve_pronunciation("šālôm", "hebrew.ancient", "hebrew.ancient.reference", DATA)
    assert result.pronunciation_type == "reconstructed"
    assert "reference" in result.label.lower()


def test_profile_stage_mismatch_is_rejected():
    try:
        resolve_pronunciation("λόγος", "hebrew.modern", "greek.ancient.reference", DATA)
    except ValueError:
        return
    raise AssertionError("mismatched pronunciation stage must fail")
