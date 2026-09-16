"""Unified entry point for the cooperative crypto research application."""
from __future__ import annotations
import argparse
import hashlib
from crypto_research_lab import digest_text
from rnn_llm_module import CryptoResearchAssistant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="crypto research")
    parser.add_argument("--hash", action="store_true", help="print standard digest set")
    parser.add_argument("--analyze", action="store_true", help="run local RNN/LLM facade")
    args = parser.parse_args()
    if args.hash:
        print(digest_text(args.text))
    if args.analyze:
        print(CryptoResearchAssistant().analyze(args.text))
    if not args.hash and not args.analyze:
        print("Crypto Research App ready. Use --hash or --analyze.")


if __name__ == "__main__":
    main()
