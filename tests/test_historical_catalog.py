from ancient_objects_db import ObjectDatabase
from historical_periods import PERIODS, period_tree
from object_sources import build_iiif_url, normalize_source_record
from softr_export import object_to_softr_row

def test_period_registry_covers_historical_span():
    keys={p['key'] for p in PERIODS}
    assert {'paleolithic','epipaleolithic','neolithic','chalcolithic','bronze_age','iron_age','hellenistic','roman','byzantine','early_islamic','medieval','modern'} <= keys

def test_period_tree_has_children():
    tree=period_tree(); assert any(x['key']=='bronze_age' for x in tree['ancient']['children'])

def test_object_database_round_trip(tmp_path):
    db=ObjectDatabase(tmp_path/'objects.sqlite'); oid=db.add_object({'title':'Test stele','period_key':'bronze_age','object_type':'inscription','license':'CC0'}); row=db.get_object(oid)
    assert row['title']=='Test stele' and row['period_name']=='Bronze Age' and row['license']=='CC0'

def test_iiif_and_source_normalization():
    assert build_iiif_url('https://example.org/iiif/abc',width=800).endswith('/full/800,/0/default.jpg')
    row=normalize_source_record({'id':'1','title':'Object','image':'https://x/y.jpg','license':'CC0'},source='test'); assert row['source']=='test' and row['image_url'].endswith('.jpg')

def test_softr_export_columns():
    row=object_to_softr_row({'id':'x','title':'Stele','period_name':'Iron Age','image_url':'https://x/i.jpg','license':'CC0'}); assert row['Record ID']=='x' and row['Image URL'].endswith('.jpg') and row['Rights / License']=='CC0'
