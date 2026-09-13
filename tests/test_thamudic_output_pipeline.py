import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thamudic_all_in_one import build_text_outputs, scan_image_safe, transliterate_ona
from nlp_thamudic_all_in_one import nlp_analyze


def test_ona_unicode_is_transliterated():
    assert transliterate_ona("𐪀𐪁𐪂") == "hlḥ"


def test_known_transliteration_produces_arabic_and_english_output():
    result = build_text_outputs("mlk")
    assert result["transliteration"] == "mlk"
    assert result["translation_ar"] == "ملك"
    assert result["translation_en"] == "king"


def test_nlp_result_preserves_visible_output_fields():
    result = nlp_analyze("mlk bn")
    assert result["transliteration"] == "mlk bn"
    assert result["translation_ar"] == "ملك بن"
    assert result["translation_en"] == "king son"
    assert result["token_count"] == 2


def test_safe_worker_returns_structured_error_instead_of_escaping_worker():
    result = scan_image_safe(Path("does-not-exist.png"))
    assert result["ok"] is False
    assert result["error_type"] in {"FileNotFoundError", "OSError"}
    assert "does-not-exist.png" in result["error"]
