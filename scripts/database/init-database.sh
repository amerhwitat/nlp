#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-check}"
cd "$ROOT"
case "$MODE" in
  check)
    command -v sqlite3 >/dev/null 2>&1 && sqlite3 --version || true
    command -v psql >/dev/null 2>&1 && psql --version || true
    echo 'Database prerequisite check completed.'
    ;;
  init)
    mkdir -p artifacts/database
    if [[ -n "${NLP_DATABASE_URL:-${DATABASE_URL:-}}" ]] && command -v psql >/dev/null 2>&1; then
      for f in db/sql/*.sql; do psql "${NLP_DATABASE_URL:-$DATABASE_URL}" -v ON_ERROR_STOP=1 -f "$f"; done
    elif command -v sqlite3 >/dev/null 2>&1; then
      db=artifacts/database/nlp.sqlite
      for f in db/sql/*.sql; do sqlite3 "$db" < "$f"; done
    else
      echo 'Install sqlite3 or PostgreSQL client.' >&2; exit 2
    fi
    ;;
  *) echo "Usage: $0 check|init" >&2; exit 2;;
esac
