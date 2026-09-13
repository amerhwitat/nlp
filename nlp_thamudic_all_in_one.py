#!/usr/bin/env python3
"""All-in-one NLP Thamudic scanner entry point.

Uses the same Tkinter GUI and scanner engine as thamudic_all_in_one.py and adds
NLP token analysis. Every analysis result is written into the visible
transliteration, Arabic translation, English translation and evidence boxes.
"""
from __future__ import annotations
import json
import re
from thamudic_all_in_one import App, LEXICON, build_text_outputs, _normalize_token


def nlp_analyze(text: str) -> dict:
    result = build_text_outputs(text)
    tokens = [t for t in re.split(r"[\s,;|]+", result["transliteration"].strip()) if t]
    known = [t for t in tokens if _normalize_token(t) in LEXICON]
    unknown = [t for t in tokens if _normalize_token(t) not in LEXICON]
    result.update({
        "tokens": tokens,
        "known_tokens": known,
        "unknown_tokens": unknown,
        "token_count": len(tokens),
        "confidence": "candidate / scholarly review required",
    })
    return result


class NLPApp(App):
    def __init__(self):
        super().__init__()
        self.title("NLP Thamudic Scanner — Ancient Languages Research Workbench")

    def _scanner(self):
        super()._scanner()
        right = self.workspace.winfo_children()[0].winfo_children()[1]
        box = right.winfo_children()[0]
        ttk_button = __import__("tkinter").ttk.Button
        ttk_button(box, text="NLP Analyze", style="Primary.TButton", command=self.nlp_run).pack(fill="x", pady=3)

    def nlp_run(self):
        result = nlp_analyze(self.source_text.get("1.0", "end-1c"))
        for widget, key in ((self.translit, "transliteration"), (self.arabic, "translation_ar"), (self.english, "translation_en")):
            widget.delete("1.0", "end")
            widget.insert("1.0", result[key])
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", json.dumps({
            "tokens": result["tokens"],
            "known_tokens": result["known_tokens"],
            "unknown_tokens": result["unknown_tokens"],
            "confidence": result["confidence"],
        }, ensure_ascii=False, indent=2))
        self.status_var.set(f"NLP analysis complete: {result['token_count']} token(s)")


if __name__ == "__main__":
    NLPApp().mainloop()
