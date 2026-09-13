"""Bridge the legacy Ancient Object Research database into ArtifactDatabase."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ancient_objects_db import ObjectDatabase
from .artifacts_db import ArtifactDatabase


def migrate_legacy_objects(source_db: str | Path, target_db: str | Path = "data/artifacts.sqlite") -> int:
    """Copy legacy object records into the unified artifact database.

    The original database is never modified. Every imported record retains its
    complete legacy row under metadata. This makes migration reversible and
    preserves source/provenance fields while the new API evolves.
    """
    count = 0
    with ObjectDatabase(source_db) as source, ArtifactDatabase(target_db) as target:
        for row in source.list_objects():
            target.import_legacy_object(dict(row))
            count += 1
    return count


def convert_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready unified artifact record without writing it."""
    return {
        **record,
        "source": record.get("source_name", "legacy-ancient-object-db"),
        "source_language": record.get("language", ""),
        "script_variant": record.get("script_key", ""),
        "translation": record.get("translation_en") or record.get("translation_ar"),
        "metadata": {"legacy_schema": "ancient_objects_db.v2", "legacy_record": record},
    }


__all__ = ["migrate_legacy_objects", "convert_legacy_record"]
