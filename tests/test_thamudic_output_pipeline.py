import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thamudic_all_in_one import build_text_outputs, translate_transliteration, transliterate_ona
from nlp_thamudic_all_in_one import nlp_analyze


def test_ona_unicode_is_transliterated():
    assert transliterate_ona("𐪀𐪁𐪂") == "hlḥ"


def test_known_transliteration_produces_arabic_and_english_output():
    assert translate_transliteration("mlk") == ("ملك", "king")
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
