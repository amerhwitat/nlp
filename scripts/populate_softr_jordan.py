#!/usr/bin/env python3
"""Populate a Softr Database table from reviewed Jordan heritage seeds.

Credentials are read from SOFTR_API_KEY and are never written to the repository.
Use --dry-run to inspect the exact Softr payload before writing records.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from softr_api import SoftrDatabaseClient  # noqa: E402
from softr_export import object_to_softr_row  # noqa: E402

DEFAULT_SEEDS = [
    ROOT / "data" / "jordan_heritage_seed.json",
    ROOT / "data" / "jordan_high_value_artifacts_seed.json",
    ROOT / "data" / "jordan_epigraphy_seed.json",
]


def load_rows(paths: list[Path]):
    rows = []
    seen = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Seed file must contain a JSON list: {path}")
        for record in data:
            record_id = record.get("id")
            if record_id in seen:
                continue
            seen.add(record_id)
            rows.append(object_to_softr_row(record))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-id", required=True)
    ap.add_argument("--table-id", required=True)
    ap.add_argument("--seed", type=Path, action="append", help="Additional JSON seed; repeatable")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = DEFAULT_SEEDS + (args.seed or [])
    rows = load_rows(paths)
    if args.limit > 0:
        rows = rows[:args.limit]

    if args.dry_run:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    client = SoftrDatabaseClient()
    created = 0
    for row in rows:
        client.create_record(args.database_id, args.table_id, row)
        created += 1
        print(f"created {created}/{len(rows)}: {row.get('Record ID', '')}")
    print(f"created={created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
