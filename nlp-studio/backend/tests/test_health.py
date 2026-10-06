def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "NLP / Ancient Script Studio"
    assert "ocr" in body
    assert "translation_provider_enabled" in body
    assert "target_languages" in body


def test_languages_lists_ona_scripts(client):
    response = client.get("/api/languages")
    assert response.status_code == 200
    body = response.json()
    assert "Dadanitic" in body["ona_scripts"]
    assert any(lang["id"] == "ancient-north-arabian" for lang in body["source_languages"])
    assert "en" in body["target_languages"]


def test_script_summary(client):
    response = client.get("/api/script-summary/ancient-north-arabian")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Ancient North Arabian"
    assert body["writing_direction"]
