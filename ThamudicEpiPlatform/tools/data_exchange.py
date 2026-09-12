from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Any


def write_json(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sqlite_export(db: str | Path, query: str, path: str | Path) -> None:
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(query).fetchall()]; con.close(); write_json(rows, path)


def access_export(connection_string: str, query: str, path: str | Path) -> None:
    """Optional Microsoft Access export through pyodbc; dependency is not mandatory."""
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError('Install optional dependency pyodbc and an Access ODBC driver') from exc
    con = pyodbc.connect(connection_string); cur = con.cursor(); cur.execute(query)
    columns = [d[0] for d in cur.description]; rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    con.close(); write_json(rows, path)
