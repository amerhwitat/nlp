#!/usr/bin/env python3
"""
PDF → image OCR (no Tesseract) → neural summarizer GUI

This script:
- Lets you pick a PDF and extracts images from pages (PyMuPDF / fitz).
- Runs OCR on images using EasyOCR (no Tesseract).
- Shows extracted OCR text, lines (one per list item), and image previews.
- Computes TextRank pseudo-labels (TF-IDF + PageRank), trains a small MLP regressor
  to reproduce those scores, and produces an extractive summary (top-k sentences).
- Saves OCR text line-by-line to a text file.

Requirements
- Python 3.8+
- Install dependencies:
    pip install pymupdf pillow easyocr numpy pandas scikit-learn networkx
  (EasyOCR requires PyTorch; pip will install a CPU build by default. For GPU,
  install a matching torch+cuda package first.)
"""

from __future__ import annotations
import io
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext

from PIL import Image, ImageTk
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# OCR backend: EasyOCR (no Tesseract)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


# -------------------------
# Utilities
# -------------------------


def extract_images_from_pdf(pdf_path: str, max_images_per_page: int = 4) -> list[Image.Image]:
    """
    Extract images from PDF pages using PyMuPDF (fitz).
    If a page has no embedded images, render the page to an image and include it.
    """
    images = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        imglist = page.get_images(full=True)
        count = 0
        for imginfo in imglist:
            if count >= max_images_per_page:
                break
            xref = imginfo[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                images.append(im)
                count += 1
            except Exception:
                continue
        # fallback: render page to image if no embedded images found
        if not imglist:
            try:
                zoom = 1.25
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(im)
            except Exception:
                pass
    doc.close()
    return images


def ocr_with_easyocr(images: list[Image.Image], langs: list[str] | None = None, gpu: bool = False) -> list[str]:
    """
    Run OCR on a list of PIL images using EasyOCR.
    Returns a list of extracted text strings (one per image).
    """
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR is not installed. Install with: pip install easyocr")
    # choose languages: default to English if not provided
    langs = langs or ["en"]
    # create reader (this may download models on first run)
    reader = easyocr.Reader(langs, gpu=gpu)
    results = []
    for im in images:
        # EasyOCR accepts numpy arrays (H,W,3)
        arr = np.array(im.convert("RGB"))
        try:
            ocr_result = reader.readtext(arr, detail=0)  # detail=0 returns list of strings
            text = "\n".join([t for t in ocr_result if t and t.strip()])
        except Exception:
            # fallback: try with detail=1 and join
            try:
                raw = reader.readtext(arr, detail=1)
                text = "\n".join([r[1] for r in raw if r and r[1]])
            except Exception:
                text = ""
        results.append(text)
    return results


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter using punctuation heuristics.
    """
    import re
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 400:
            subs = [s.strip() for s in p.split(",") if s.strip()]
            out.extend(subs)
        else:
            out.append(p)
    out = [s for s in out if len(s) > 20]
    return out


def compute_textrank_scores(sentences: list[str], tfidf_matrix=None) -> np.ndarray:
    """
    Build similarity graph between sentences (cosine on TF-IDF) and run PageRank.
    Returns array of scores aligned with sentences.
    """
    if not sentences:
        return np.array([])
    if tfidf_matrix is None:
        vect = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix = vect.fit_transform(sentences)
    sim = (tfidf_matrix @ tfidf_matrix.T).toarray()
    np.fill_diagonal(sim, 0.0)
    G = nx.from_numpy_array(sim)
    try:
        pr = nx.pagerank_numpy(G, weight="weight")
    except Exception:
        pr = nx.pagerank(G, weight="weight")
    scores = np.array([pr.get(i, 0.0) for i in range(len(sentences))], dtype=float)
    return scores


# -------------------------
# GUI Application
# -------------------------


class PDFSummarizerNoTesseract(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF OCR + Neural Summarizer (no Tesseract)")
        self.geometry("1100x760")

        # state
        self.pdf_path = None
        self.images: list[Image.Image] = []
        self.ocr_text = ""
        self.ocr_lines: list[str] = []
        self.sentences: list[str] = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tr_scores = None
        self.model = None

        # OCR options
        self.langs_var = tk.StringVar(value="en")
        self.gpu_var = tk.BooleanVar(value=False)

        self._image_thumbs: list[ImageTk.PhotoImage] = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Button(top, text="Select PDF", command=self.select_pdf).pack(side="left", padx=6)
        ttk.Button(top, text="Extract & OCR (EasyOCR)", command=self.extract_and_ocr_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Save OCR Lines", command=self.save_ocr_lines).pack(side="left", padx=6)
        ttk.Label(top, text="Languages (comma):").pack(side="left", padx=(12, 4))
        ttk.Entry(top, textvariable=self.langs_var, width=12).pack(side="left")
        ttk.Checkbutton(top, text="Use GPU", variable=self.gpu_var).pack(side="left", padx=8)

        ttk.Button(top, text="Compute TextRank", command=self.compute_textrank_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Train Neural Model", command=self.train_model_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Generate Summary", command=self.generate_summary).pack(side="left", padx=6)
        ttk.Button(top, text="Save Summary", command=self.save_summary).pack(side="left", padx=6)

        ttk.Label(top, text="Summary length:").pack(side="left", padx=(12, 4))
        self.summary_k = tk.IntVar(value=5)
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.summary_k, width=4).pack(side="left")

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8)
        self.progress = ttk.Progressbar(status, length=420, mode="determinate")
        self.progress.pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(pan)
        pan.add(left_frame, weight=1)
        right_frame = ttk.Frame(pan, width=420)
        pan.add(right_frame, weight=0)

        ttk.Label(left_frame, text="OCR Text (from images)").pack(anchor="w")
        self.ocr_textbox = scrolledtext.ScrolledText(left_frame, height=18)
        self.ocr_textbox.pack(fill="both", expand=True, pady=(4, 8))

        ttk.Label(left_frame, text="OCR Lines (double-click to copy)").pack(anchor="w")
        self.lines_listbox = tk.Listbox(left_frame, height=10)
        self.lines_listbox.pack(fill="both", expand=False, pady=(4, 8))
        self.lines_listbox.bind("<Double-Button-1>", self.copy_line_to_clipboard)

        ttk.Label(left_frame, text="Extracted images (click to preview OCR)").pack(anchor="w")
        self.images_canvas = tk.Canvas(left_frame, height=140, bg="white")
        self.images_canvas.pack(fill="x", pady=(4, 8))
        self.images_canvas.bind("<Button-1>", self.on_images_canvas_click)

        ttk.Label(right_frame, text="Sentences (detected)").pack(anchor="w")
        self.sent_list = tk.Listbox(right_frame, height=12)
        self.sent_list.pack(fill="both", expand=False, pady=(4, 8))

        ttk.Label(right_frame, text="Summary (top sentences)").pack(anchor="w")
        self.summary_box = scrolledtext.ScrolledText(right_frame, height=12)
        self.summary_box.pack(fill="both", expand=True, pady=(4, 8))

        ttk.Label(right_frame, text="Log").pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(right_frame, height=6, state="disabled")
        self.log_box.pack(fill="both", expand=False, pady=(4, 8))

    # -------------------------
    # UI helpers
    # -------------------------
    def log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, text: str, progress=None, maximum=None):
        self.status_var.set(text)
        if maximum is not None:
            self.progress['maximum'] = maximum
        if progress is not None:
            self.progress['value'] = progress
        self.update_idletasks()

    # -------------------------
    # Actions
    # -------------------------
    def select_pdf(self):
        path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        self.pdf_path = path
        self.log(f"Selected PDF: {os.path.basename(path)}")
        self.set_status("PDF selected")

    def extract_and_ocr_thread(self):
        if not self.pdf_path:
            messagebox.showinfo("No PDF", "Please select a PDF first.")
            return
        threading.Thread(target=self._extract_and_ocr, daemon=True).start()

    def _extract_and_ocr(self):
        self.set_status("Extracting images...", progress=0, maximum=1)
        self.log("Extracting images from PDF...")
        try:
            images = extract_images_from_pdf(self.pdf_path)
        except Exception as e:
            self.log(f"Failed to extract images: {e}")
            self.set_status("Idle")
            return
        self.images = images
        self._render_image_thumbnails()
        self.log(f"Extracted {len(images)} images")
        # OCR images using EasyOCR
        langs = [l.strip() for l in self.langs_var.get().split(",") if l.strip()]
        gpu = bool(self.gpu_var.get())
        self.set_status("Running OCR on images...", progress=0, maximum=max(1, len(images)))
        try:
            ocr_texts = ocr_with_easyocr(images, langs=langs or ["en"], gpu=gpu)
        except Exception as e:
            self.log(f"EasyOCR failed: {e}")
            self.set_status("Idle")
            return
        all_text = []
        for i, txt in enumerate(ocr_texts, start=1):
            all_text.append(txt)
            self.set_status(f"OCR image {i}/{len(images)}", progress=i, maximum=len(images))
            time.sleep(0.02)
        self.ocr_text = "\n\n".join(all_text).strip()
        # split into lines
        self.ocr_lines = [line.strip() for line in self.ocr_text.splitlines() if line.strip()]
        self._update_lines_listbox()
        # update OCR textbox
        self.ocr_textbox.delete("1.0", "end")
        self.ocr_textbox.insert("1.0", self.ocr_text)
        # split into sentences
        self.sentences = split_into_sentences(self.ocr_text)
        self.sent_list.delete(0, "end")
        for s in self.sentences:
            self.sent_list.insert("end", s[:120].replace("\n", " "))
        self.set_status("OCR complete", progress=0)
        self.log("OCR complete and sentences extracted")

    def _render_image_thumbnails(self):
        self.images_canvas.delete("all")
        self._image_thumbs = []
        x = 4
        pad = 6
        h = 120
        for i, im in enumerate(self.images):
            try:
                ratio = im.width / im.height
                w = int(h * ratio)
                thumb = im.copy().resize((w, h), Image.LANCZOS)
                tkimg = ImageTk.PhotoImage(thumb)
                self._image_thumbs.append(tkimg)
                self.images_canvas.create_image(x, 4, anchor="nw", image=tkimg)
                self.images_canvas.create_rectangle(x - 1, 3, x + w + 1, 4 + h + 1, outline="black")
                self.images_canvas.create_text(x + 6, 4 + h + 6, anchor="nw", text=str(i + 1), fill="blue")
                x += w + pad
            except Exception:
                continue

    def on_images_canvas_click(self, event):
        x = event.x
        pad = 6
        h = 120
        pos = 4
        for i, im in enumerate(self.images):
            ratio = im.width / im.height
            w = int(h * ratio)
            if pos <= x <= pos + w:
                # show OCR preview for that image
                try:
                    langs = [l.strip() for l in self.langs_var.get().split(",") if l.strip()]
                    gpu = bool(self.gpu_var.get())
                    txts = ocr_with_easyocr([im], langs=langs or ["en"], gpu=gpu)
                    txt = txts[0] if txts else ""
                except Exception as e:
                    txt = f"OCR failed: {e}"
                PreviewWindow(self, im, txt)
                return
            pos += w + pad

    # -------------------------
    # OCR lines helpers
    # -------------------------
    def _update_lines_listbox(self):
        self.lines_listbox.delete(0, "end")
        for i, line in enumerate(self.ocr_lines, start=1):
            display = line if len(line) <= 200 else line[:197] + "..."
            self.lines_listbox.insert("end", f"{i}: {display}")

    def copy_line_to_clipboard(self, event=None):
        sel = self.lines_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        line = self.ocr_lines[idx]
        self.clipboard_clear()
        self.clipboard_append(line)
        self.log("Copied line to clipboard")

    def save_ocr_lines(self):
        if not self.ocr_lines:
            messagebox.showinfo("No OCR lines", "Run OCR first to extract text lines.")
            return
        path = filedialog.asksaveasfilename(title="Save OCR lines", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for line in self.ocr_lines:
                    fh.write(line.replace("\r", "").replace("\n", " ") + "\n")
            self.log(f"Saved {len(self.ocr_lines)} OCR lines to {path}")
            messagebox.showinfo("Saved", f"Saved {len(self.ocr_lines)} lines to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save OCR lines: {e}")

    # -------------------------
    # TextRank, training, summary
    # -------------------------
    def compute_textrank_thread(self):
        if not self.sentences:
            messagebox.showinfo("No sentences", "Run OCR/extraction first.")
            return
        threading.Thread(target=self._compute_textrank, daemon=True).start()

    def _compute_textrank(self):
        self.set_status("Computing TF-IDF and TextRank...", progress=0, maximum=1)
        self.log("Computing TF-IDF matrix...")
        vect = TfidfVectorizer(max_features=4000, stop_words="english")
        X = vect.fit_transform(self.sentences)
        self.tfidf_vectorizer = vect
        self.tfidf_matrix = X
        self.log("Computing TextRank scores (PageRank on sentence graph)...")
        scores = compute_textrank_scores(self.sentences, tfidf_matrix=X)
        self.tr_scores = scores
        top_idx = np.argsort(-scores)[: min(5, len(scores))]
        summary = "\n\n".join([self.sentences[i] for i in sorted(top_idx)])
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", summary)
        self.set_status("TextRank computed")
        self.log("TextRank labels ready (used as pseudo-labels for training)")

    def train_model_thread(self):
        if not self.sentences or self.tfidf_matrix is None or self.tr_scores is None:
            messagebox.showinfo("Missing data", "Run OCR and compute TextRank first.")
            return
        threading.Thread(target=self._train_model, daemon=True).start()

    def _train_model(self):
        self.set_status("Training neural model...", progress=0, maximum=1)
        self.log("Preparing training data...")
        X = self.tfidf_matrix.toarray()
        y = self.tr_scores
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        self.log("Training MLP regressor (this may take a moment)...")
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42))
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            self.model = model
            self.log(f"Training complete. Test R^2 score: {score:.3f}")
            self.set_status("Model trained")
        except Exception as e:
            self.log(f"Training failed: {e}")
            self.set_status("Idle")

    def generate_summary(self):
        if not self.sentences:
            messagebox.showinfo("No sentences", "Run OCR first.")
            return
        k = max(1, int(self.summary_k.get()))
        if self.model is None:
            if self.tr_scores is None:
                messagebox.showinfo("No model or TextRank", "Compute TextRank or train the model first.")
                return
            scores = self.tr_scores
            self.log("Using TextRank scores (no trained model available).")
        else:
            X = self.tfidf_vectorizer.transform(self.sentences).toarray()
            try:
                scores = self.model.predict(X)
            except Exception as e:
                self.log(f"Prediction failed: {e}")
                scores = self.tr_scores if self.tr_scores is not None else np.zeros(len(self.sentences))
        idxs = np.argsort(-scores)
        topk = sorted(idxs[:min(k, len(idxs))])
        summary = "\n\n".join([self.sentences[i] for i in topk])
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", summary)
        self.log(f"Generated summary with {len(topk)} sentences")
        self.set_status("Summary generated")

    def save_summary(self):
        text = self.summary_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("No summary", "Generate a summary first.")
            return
        path = filedialog.asksaveasfilename(title="Save summary", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self.log(f"Saved summary to {path}")
            messagebox.showinfo("Saved", f"Saved summary to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


class PreviewWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, image: Image.Image, ocr_text: str):
        super().__init__(parent)
        self.title("Image OCR preview")
        self.geometry("700x600")
        img_frame = ttk.Frame(self)
        img_frame.pack(fill="both", expand=False, padx=8, pady=8)
        w = 640
        ratio = image.width / image.height
        h = int(w / ratio)
        tkimg = ImageTk.PhotoImage(image.resize((w, h), Image.LANCZOS))
        lbl = tk.Label(img_frame, image=tkimg)
        lbl.image = tkimg
        lbl.pack()
        ttk.Label(self, text="OCR text for this image").pack(anchor="w", padx=8)
        txt = scrolledtext.ScrolledText(self, height=12)
        txt.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        txt.insert("1.0", ocr_text)
        txt.configure(state="normal")


def main():
    if not EASYOCR_AVAILABLE:
        # Inform user but still allow them to open the GUI; OCR actions will fail until easyocr is installed.
        print("Warning: easyocr not installed. Install with: pip install easyocr")
    app = PDFSummarizerNoTesseract()
    app.mainloop()


if __name__ == "__main__":
    main()