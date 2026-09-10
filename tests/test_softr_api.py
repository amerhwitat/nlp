import pytest
from softr_api import SoftrDatabaseClient

def test_softr_client_requires_key(monkeypatch):
    monkeypatch.delenv('SOFTR_API_KEY',raising=False)
    with pytest.raises(ValueError): SoftrDatabaseClient()
