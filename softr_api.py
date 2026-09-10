"""Optional Softr Database API client. Never commit SOFTR_API_KEY."""
from __future__ import annotations
import os
import requests

class SoftrAPIError(RuntimeError): pass

class SoftrDatabaseClient:
    def __init__(self,api_key=None,base_url='https://tables-api.softr.io/api/v1',timeout=30):
        self.api_key=api_key or os.getenv('SOFTR_API_KEY')
        if not self.api_key: raise ValueError('SOFTR_API_KEY is required')
        self.base_url=base_url.rstrip('/'); self.timeout=timeout
    def _request(self,method,path,**kwargs):
        headers=kwargs.pop('headers',{}); headers['Softr-Api-Key']=self.api_key; headers.setdefault('Content-Type','application/json')
        r=requests.request(method,self.base_url+path,headers=headers,timeout=self.timeout,**kwargs)
        if not r.ok: raise SoftrAPIError(f'Softr API {r.status_code}: {r.text[:500]}')
        return r.json()
    def list_databases(self): return self._request('GET','/databases').get('data',[])
    def list_tables(self,database_id): return self._request('GET',f'/databases/{database_id}/tables').get('data',[])
    def list_records(self,database_id,table_id,limit=100,offset=0,field_names=True):
        q=f'?limit={int(limit)}&offset={int(offset)}&fieldNames={str(bool(field_names)).lower()}'
        return self._request('GET',f'/databases/{database_id}/tables/{table_id}/records{q}').get('data',[])
    def create_record(self,database_id,table_id,fields): return self._request('POST',f'/databases/{database_id}/tables/{table_id}/records',json={'fields':fields}).get('data')
    def update_record(self,database_id,table_id,record_id,fields): return self._request('PUT',f'/databases/{database_id}/tables/{table_id}/records/{record_id}',json={'fields':fields}).get('data')
