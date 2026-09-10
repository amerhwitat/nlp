import json
from pathlib import Path

from ancient_objects_db import ObjectDatabase


def test_sqlite_round_trip_and_upsert(tmp_path: Path):
    db = ObjectDatabase(tmp_path / "objects.sqlite")
    record = {
        "id": "TEST-001",
        "title": "Test inscription",
        "period_key": "iron_age",
        "object_type": "inscription",
        "script_key": "safaitic",
        "tags": ["Jordan", "Safaitic"],
        "subjects": ["epigraphy"],
    }
    assert db.add_object(record) == "TEST-001"
    assert db.get_object("TEST-001")["title"] == "Test inscription"
    assert db.list_objects(query="Safaitic")[0]["id"] == "TEST-001"
    record["title"] = "Updated inscription"
    db.add_object(record)
    assert db.get_object("TEST-001")["title"] == "Updated inscription"
    db.close()


def test_seed_import_and_json_backup(tmp_path: Path):
    db = ObjectDatabase(tmp_path / "objects.sqlite")
    seed = [{"id": "A", "title": "A", "object_type": "artifact"}, {"id": "B", "title": "B", "object_type": "inscription"}]
    assert db.import_records(seed) == 2
    out = tmp_path / "backup.json"
    assert db.export_json(out) == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {x["id"] for x in payload} == {"A", "B"}
    db.close()


def test_sql_dump_and_statistics(tmp_path: Path):
    db = ObjectDatabase(tmp_path / "objects.sqlite")
    db.add_object({"id": "A", "title": "A", "country": "Jordan"})
    dump = tmp_path / "backup.sql"
    assert db.export_sql(dump) > 0
    stats = db.statistics()
    assert stats["objects"] == 1
    assert stats["database"] == str(tmp_path / "objects.sqlite")
    db.close()
