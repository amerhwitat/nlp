#!/usr/bin/env python3
"""Unified launcher for the fixed all-in-one Thamudic Python applications."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Thamudic scanner or NLP Thamudic scanner")
    parser.add_argument("mode", choices=("desktop", "nlp", "web"), nargs="?", default="desktop")
    parser.add_argument("--db", default="ancient_objects.sqlite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "desktop":
        from thamudic_all_in_one import App
        App().mainloop()
        return 0

    if args.mode == "nlp":
        from nlp_thamudic_all_in_one import NLPApp
        NLPApp().mainloop()
        return 0

    from thamudic_web_app import create_app
    create_app(args.db).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
