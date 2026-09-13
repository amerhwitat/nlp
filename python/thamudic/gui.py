"""Tkinter desktop UI for scanning, importing media, transliteration, translation and voice."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from .ancient_translation import supported_targets, translate
from .media_pipeline import extract_media_text
from .voice import VoiceRequest, speak as voice_capability
from .desktop_voice import DesktopVoice


class ThamudicTranslatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ancient Script / Thamudic Scanner + Translator")
        self.geometry("1120x900")
        self._voice = DesktopVoice()
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self); top.pack(fill="x", padx=12, pady=10)
        ttk.Button(top, text="Import image / PDF", command=self.import_media).pack(side="left")
        ttk.Label(top, text="Script").pack(side="left", padx=(14, 4))
        self.script = ttk.Combobox(top, values=["Dadanitic", "Safaitic", "Hismaic", "Taymanitic", "Thamudic B"], state="readonly", width=18)
        self.script.set("Dadanitic"); self.script.pack(side="left")
        ttk.Label(top, text="Target").pack(side="left", padx=(12, 4))
        self.target = ttk.Combobox(top, values=list(supported_targets()), state="readonly", width=8)
        self.target.set("en"); self.target.pack(side="left")
        ttk.Button(top, text="Translate + transliterate", command=self.translate).pack(side="left", padx=12)

        ttk.Label(self, text="Imported/OCR source or scholarly transliteration").pack(anchor="w", padx=12)
        self.source = tk.Text(self, height=10, wrap="word"); self.source.pack(fill="x", padx=12)
        ttk.Label(self, text="Transliteration").pack(anchor="w", padx=12, pady=(10, 4))
        self.transliteration = tk.Text(self, height=5, wrap="word"); self.transliteration.pack(fill="x", padx=12)
        ttk.Label(self, text="Translation + evidence").pack(anchor="w", padx=12, pady=(10, 4))
        self.translation = tk.Text(self, height=9, wrap="word"); self.translation.pack(fill="both", expand=True, padx=12)

        vf = ttk.LabelFrame(self, text="Voice controls")
        vf.pack(fill="x", padx=12, pady=10)
        ttk.Button(vf, text="▶ Transliteration", command=self.speak_transliteration).pack(side="left", padx=4, pady=6)
        ttk.Button(vf, text="▶ Translation", command=self.speak_translation).pack(side="left", padx=4, pady=6)
        ttk.Button(vf, text="■ Stop", command=self.stop_voice).pack(side="left", padx=4, pady=6)
        ttk.Label(vf, text="Rate").pack(side="left", padx=(18, 4))
        self.rate = tk.IntVar(value=160)
        ttk.Scale(vf, from_=60, to=300, variable=self.rate, orient="horizontal", length=160).pack(side="left")
        self.voice_status = ttk.Label(vf, text="TTS: checking…"); self.voice_status.pack(side="left", padx=10)

        self.status = ttk.Label(self, text="Ready"); self.status.pack(anchor="w", padx=12, pady=8)
        self._update_voice_status()

    def _update_voice_status(self) -> None:
        self.voice_status.config(text="TTS: available" if self._voice.available() else "TTS: install pyttsx3")

    def import_media(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Images/PDF", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf"), ("Text", "*.txt *.md *.csv"), ("All files", "*.*")])
        if not p: return
        try:
            text, meta = extract_media_text(Path(p))
            self.source.delete("1.0", "end"); self.source.insert("1.0", text)
            self.status.config(text=f"Imported {Path(p).name} via {meta.get('provider', 'unknown')}")
            self.translate()
        except Exception as exc:
            messagebox.showerror("Import/OCR error", str(exc))

    def translate(self) -> None:
        text = self.source.get("1.0", "end-1c").strip()
        if not text:
            self.status.config(text="Enter or import source text first"); return
        result = translate(text, script=self.script.get(), target_language=self.target.get())
        self.transliteration.delete("1.0", "end"); self.transliteration.insert("1.0", result["transliteration"])
        self.translation.delete("1.0", "end")
        self.translation.insert("1.0", result["translation"] or "No corpus-backed translation is available for this reading.\n\nStatus: " + result["translation_status"] + "\nProvenance: " + str(result["provenance"] or "none"))
        self.status.config(text=f"{result['translation_status']} · confidence={result['confidence']} · corpus={result['corpus_id'] or 'none'}")

    def _speak(self, text: str, language: str, mode: str) -> None:
        if not text.strip():
            self.status.config(text=f"No {mode} text to read"); return
        if not self._voice.available():
            self.status.config(text="Install pyttsx3 for desktop voice playback"); return
        try:
            self._voice.speak(text, language=language, rate=int(self.rate.get()))
            self.status.config(text=f"Voice: {mode}")
        except Exception as exc:
            self.status.config(text=f"Voice error: {exc}")

    def speak_transliteration(self) -> None:
        self._speak(self.transliteration.get("1.0", "end-1c"), "en", "transliteration")

    def speak_translation(self) -> None:
        self._speak(self.translation.get("1.0", "end-1c"), self.target.get(), "translation")

    def stop_voice(self) -> None:
        self._voice.stop(); self.status.config(text="Voice stopped")


def main() -> None:
    ThamudicTranslatorApp().mainloop()

if __name__ == "__main__": main()
