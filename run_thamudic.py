#!/usr/bin/env python3
"""Unified launcher for the single-file Thamudic + NLP desktop workbench."""
from __future__ import annotations
import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the unified Thamudic/NLP scanner")
    parser.add_argument("mode", choices=("desktop", "nlp", "web"), nargs="?", default="desktop")
    parser.add_argument("--db", default="ancient_objects.sqlite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if args.mode in ("desktop", "nlp"):
        from thamudic_all_in_one import App
        App(mode="nlp" if args.mode == "nlp" else "thamudic", db_path=args.db).mainloop()
        return 0
    from thamudic_web_app import create_app
    create_app(args.db).run(host=args.host, port=args.port, debug=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
