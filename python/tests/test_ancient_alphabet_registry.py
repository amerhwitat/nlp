from python.thamudic import (
    supported_alphabet_languages,
    language_profile,
    variations,
    translation_capabilities,
    translation_directions,
)


def test_registry_contains_major_ancient_language_families():
    supported = set(supported_alphabet_languages())
    assert {"ancient-egyptian", "akkadian", "sumerian", "ugaritic", "phoenician", "ancient-hebrew", "aramaic", "ancient-north-arabian", "greek", "latin", "chinese", "japanese"}.issubset(supported)


def test_variations_are_exposed():
    assert "Classical Greek" in variations("greek")
    assert "Paleo-Hebrew" in variations("ancient-hebrew")
    assert "Old Babylonian" in variations("akkadian")
    assert "Oracle Bone Script" in variations("chinese")
    assert "Hentaigana" in variations("japanese")


def test_profile_has_unicode_and_translation_metadata():
    profile = language_profile("ancient-egyptian")
    assert "Egyptian Hieroglyphs" in profile["scripts"]
    assert "Egyptian Hieroglyphs" in profile["unicode_blocks"]
    assert "hieroglyph-to-transliteration" in translation_capabilities("ancient-egyptian")


def test_bidirectional_metadata_does_not_claim_unsupported_translation_models():
    directions = translation_directions("ancient-egyptian")
    assert directions["source_to_transliteration"] is True
    assert directions["transliteration_to_translation"] is True
    assert directions["translation_to_source_retrieval"] is True
