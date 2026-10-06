def test_history_and_export(client):
    client.post(
        "/api/translate",
        json={"text": "ytm bn ʿbny w wgm ʿl- ḫll -h", "script": "Safaitic", "target_language": "en"},
    )

    history = client.get("/api/history")
    assert history.status_code == 200
    body = history.json()
    assert body["count"] >= 1
    assert body["integrity"]["invalid"] == 0

    for fmt, expected_type in (("json", "application/json"), ("jsonl", "application/x-ndjson"), ("txt", "text/plain")):
        response = client.get(f"/api/export/history?format={fmt}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(expected_type)
        assert response.headers["content-disposition"].startswith("attachment")

    pdf = client.get("/api/export/history?format=pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")


def test_export_result_pdf_and_md(client):
    response = client.post(
        "/api/translate",
        json={"text": "bḏkrh wdd ḏ{h}k", "script": "Dadanitic", "target_language": "en"},
    )
    result_id = response.json()["result_id"]

    md = client.get(f"/api/export/result/{result_id}?format=md")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert "Dadanitic" in md.text

    pdf = client.get(f"/api/export/result/{result_id}?format=pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")


def test_export_missing_result_404(client):
    response = client.get("/api/export/result/does-not-exist?format=json")
    assert response.status_code == 404
