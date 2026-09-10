#!/usr/bin/env python3
"""Command-line database utility for the Ancient Object Research database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ancient_objects_db import ObjectDatabase


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage the local SQLite research database")
    ap.add_argument("--db", default="ancient_objects.sqlite", help="SQLite database path")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create/upgrade the database")
    p = sub.add_parser("import-json", help="Import a JSON list of records")
    p.add_argument("path", type=Path)
    p = sub.add_parser("export-json", help="Export all objects to JSON")
    p.add_argument("path", type=Path)
    p = sub.add_parser("export-csv", help="Export all objects to CSV")
    p.add_argument("path", type=Path)
    p = sub.add_parser("export-sql", help="Export a portable SQL dump")
    p.add_argument("path", type=Path)
    p = sub.add_parser("backup", help="Create a consistent SQLite backup")
    p.add_argument("path", type=Path)
    sub.add_parser("stats", help="Show database statistics")
    p = sub.add_parser("search", help="Search the object catalog")
    p.add_argument("query")
    p.add_argument("--country")
    p.add_argument("--script")
    p.add_argument("--period")
    p.add_argument("--type")
    p.add_argument("--limit", type=int, default=50)

    args = ap.parse_args()
    with ObjectDatabase(args.db) as db:
        if args.command == "init":
            print(json.dumps(db.statistics(), ensure_ascii=False, indent=2))
        elif args.command == "import-json":
            print(f"imported={db.import_json(args.path)}")
        elif args.command == "export-json":
            print(f"exported={db.export_json(args.path)}")
        elif args.command == "export-csv":
            print(f"exported={db.export_csv(args.path)}")
        elif args.command == "export-sql":
            print(f"bytes={db.export_sql(args.path)}")
        elif args.command == "backup":
            print(db.backup(args.path))
        elif args.command == "stats":
            print(json.dumps(db.statistics(), ensure_ascii=False, indent=2))
        elif args.command == "search":
            print(json.dumps(db.list_objects(query=args.query, country=args.country,
                                              script_key=args.script, period_key=args.period,
                                              object_type=args.type, limit=args.limit),
                              ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
