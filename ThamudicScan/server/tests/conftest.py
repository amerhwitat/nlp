import pytest
from fastapi.testclient import TestClient

from ThamudicScan.server.main import create_app
from ThamudicScan.server.db import Database


@pytest.fixture()
def database(tmp_path):
    return Database(tmp_path / "test.sqlite3")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("THAMUDIC_DB_PATH", str(tmp_path / "test.sqlite3"))
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
