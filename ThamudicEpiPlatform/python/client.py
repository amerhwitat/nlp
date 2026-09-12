import requests

BASE='http://127.0.0.1:8010/api'

def objects(query=''):
    r=requests.get(f'{BASE}/objects',params={'q':query},timeout=10); r.raise_for_status(); return r.json()

def create(title,site=None):
    r=requests.post(f'{BASE}/objects',json={'title':title,'site':site},timeout=10); r.raise_for_status(); return r.json()

if __name__=='__main__':
    print(objects())
