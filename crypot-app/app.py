"""Unified entry point for the cooperative crypto research application."""
from __future__ import annotations
import argparse
from crypto_research_lab import digest_text
from rnn_llm_module import CryptoResearchAssistant


def main():
    parser = argparse.ArgumentParser(description="Chimera Crypto Research App")
    parser.add_argument("text", nargs="?", default="crypto research")
    parser.add_argument("--hash", action="store_true", help="print standard digest set")
    parser.add_argument("--analyze", action="store_true", help="run local RNN/LLM facade")
    parser.add_argument("--gui", action="store_true", help="launch the unified Tkinter GUI")
    args = parser.parse_args()
    if args.gui or not (args.hash or args.analyze):
        from crypot_gui import main as gui_main
        gui_main()
        return
    if args.hash:
        print(digest_text(args.text))
    if args.analyze:
        print(CryptoResearchAssistant().analyze(args.text))


if __name__ == "__main__":
    main()
