import io


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validate_returns_old_north_arabian_codepoints(client):
    response = client.post("/validate", json={"text": "𐪀𐪁𐪂"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["codepoints"] == [0x10A80, 0x10A81, 0x10A82]


def test_scan_creates_session_and_result(client):
    response = client.post("/scan", json={"text": "𐪀𐪁", "keywords": []})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["results"][0]["transliteration"] == "hl"
    assert body["summary"]["match_count"] == 1


def test_session_can_be_reopened(client):
    created = client.post("/scan", json={"text": "𐪀", "keywords": []}).json()
    reopened = client.get(f"/sessions/{created['session_id']}")
    assert reopened.status_code == 200
    assert reopened.json()["results"][0]["text"] == "𐪀"


def test_export_endpoints(client):
    created = client.post("/scan", json={"text": "𐪀", "keywords": []}).json()
    session_id = created["session_id"]
    csv_response = client.get(f"/export/{session_id}?format=csv")
    json_response = client.get(f"/export/{session_id}?format=json")
    assert csv_response.status_code == 200
    assert "𐪀" in csv_response.text
    assert json_response.status_code == 200
    assert "𐪀" in json_response.text


def test_scan_file_accepts_utf8_text(client):
    response = client.post(
        "/scan_file",
        files={"file": ("sample.txt", io.BytesIO("𐪀𐪁".encode("utf-8")), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "𐪀𐪁"


def test_scan_file_rejects_unsafe_extension(client):
    response = client.post(
        "/scan_file",
        files={"file": ("run.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert response.status_code == 415
