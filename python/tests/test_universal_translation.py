from python.thamudic import translate_ancient, translation_matrix


def test_registered_language_returns_explicit_provider_required_status():
    result = translate_ancient("ἄνθρωπος", "ancient-greek", "en")
    assert result.status == "provider_required"
    assert result.source_language == "greek"
    assert result.confidence == 0.0


def test_translation_input_can_request_source_retrieval_direction():
    result = translate_ancient("man", "ancient-greek", "grc", source_form="translation")
    assert result.status == "provider_required"
    assert result.target_form is None


def test_translation_matrix_contains_major_requested_languages():
    matrix = translation_matrix()
    for language in ("ancient-egyptian", "chinese", "japanese", "greek", "latin", "ancient-north-arabian"):
        assert language in matrix
