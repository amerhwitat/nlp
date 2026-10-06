def test_translate_corpus_match(client):
    response = client.post(
        "/api/translate",
        json={"text": "ytm bn ʿbny w wgm ʿl- ḫll -h", "script": "Safaitic", "target_language": "en"},
    )
    assert response.status_code == 200
    translation = response.json()["translation"]
    assert translation["translation_status"] == "corpus_match"
    assert translation["corpus_id"] == "TIJ 503"
    assert translation["translation"]
    assert translation["provider"] == "corpus"


def test_translate_arabic_target(client):
    response = client.post(
        "/api/translate",
        json={"text": "bḏkrh wdd ḏ{h}k", "script": "Dadanitic", "target_language": "ar"},
    )
    assert response.status_code == 200
    translation = response.json()["translation"]
    assert translation["target_language"] == "ar"
    assert translation["translation"]


def test_translate_unknown_is_not_fabricated(client):
    response = client.post(
        "/api/translate",
        json={"text": "totally unknown fragment", "script": "Thamudic B", "target_language": "en"},
    )
    assert response.status_code == 200
    translation = response.json()["translation"]
    assert translation["translation"] is None
    assert translation["translation_status"] == "not_available"


def test_translate_empty_rejected(client):
    response = client.post("/api/translate", json={"text": ""})
    assert response.status_code == 400


def test_translate_ona_script_transliteration(client):
    response = client.post(
        "/api/translate",
        json={"text": "𐪀𐪁𐪂", "script": "Dadanitic", "target_language": "en"},
    )
    assert response.status_code == 200
    translation = response.json()["translation"]
    assert translation["transliteration"] == "hlḥ"
