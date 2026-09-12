from fastapi.testclient import TestClient
from server.app import app

def test_health():
    c=TestClient(app); r=c.get('/api/health'); assert r.status_code==200; assert r.json()['ok'] is True

def test_create_and_list():
    c=TestClient(app); r=c.post('/api/objects',json={'title':'Test 𐪀','site':'Test'}); assert r.status_code==200
    r=c.get('/api/objects',params={'q':'Test'}); assert r.status_code==200; assert any(x['title']=='Test 𐪀' for x in r.json())
