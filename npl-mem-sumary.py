#!/usr/bin/env python3
"""
Memory-safe PDF OCR + Neural Summarizer GUI (no Tesseract)

Main memory-safety strategies:
- Process PDF pages/images one at a time (streaming).
- Downscale images before OCR (configurable).
- Reuse EasyOCR reader.
- Use HashingVectorizer (sparse) to avoid building large vocabularies.
- Cap number of sentences used for TextRank to limit similarity matrix size.
- Use SGDRegressor for incremental training (partial_fit) to avoid storing full dense matrices.
- Explicitly close images and call gc.collect() after heavy steps.
"""

from __future__ import annotations
import io
import os
import time
import threading
import gc
import math
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext

from PIL import Image, ImageTk, ImageOps
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# OCR backend: EasyOCR (no Tesseract)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# -------------------------
# Configuration (tweak to fit your machine)
# -------------------------
MAX_IMAGE_DIM = 1600          # downscale images so max(width,height) <= this
MAX_SENTENCES = 1200          # cap sentences used for TextRank to limit memory
HASHING_N_FEATURES = 2 ** 16  # features for HashingVectorizer (sparse)
SGD_BATCH_SIZE = 256          # batch size for incremental training
SVD_DIM = 128                 # reduce TF features before similarity (keeps memory bounded)
# -------------------------

# -------------------------
# Utilities
# -------------------------


def extract_images_from_pdf_stream(pdf_path: str):
    """
    Generator: yields PIL.Image objects for each page.
    If page has embedded images, yields them first (one by one).
    If no embedded images, yields a rendered page image.
    This avoids storing all images in memory.
    """
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            imglist = page.get_images(full=True)
            if imglist:
                for imginfo in imglist:
                    xref = imginfo[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    try:
                        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        yield im
                    except Exception:
                        continue
            else:
                # render page to image
                try:
                    zoom = 1.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    yield im
                except Exception:
                    continue
    finally:
        doc.close()


def downscale_image(im: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """Return a downscaled copy of image with max dimension <= max_dim."""
    w, h = im.size
    max_wh = max(w, h)
    if max_wh <= max_dim:
        return im.copy()
    scale = max_dim / float(max_wh)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return im.resize((new_w, new_h), Image.LANCZOS)


def ocr_with_easyocr_batch(images: list[Image.Image], reader, langs: list[str] | None = None, gpu: bool = False) -> list[str]:
    """
    OCR a small batch of PIL images using an existing EasyOCR reader instance.
    Returns list of text strings.
    """
    results = []
    for im in images:
        try:
            arr = np.array(im.convert("RGB"))
            txts = reader.readtext(arr, detail=0)
            text = "\n".join([t for t in txts if t and t.strip()])
        except Exception:
            # fallback to detail=1
            try:
                raw = reader.readtext(arr, detail=1)
                text = "\n".join([r[1] for r in raw if r and r[1]])
            except Exception:
                text = ""
        results.append(text)
    return results


def split_into_sentences(text: str) -> list[str]:
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


def compute_textrank_limited(sentences: list[str], vectorizer, svd: TruncatedSVD | None, max_sentences: int = MAX_SENTENCES):
    """
    Compute TextRank scores but limit the number of sentences used.
    - If sentences > max_sentences, keep the longest sentences (heuristic).
    - Use HashingVectorizer (sparse) and optional SVD to reduce memory for similarity matrix.
    """
    if not sentences:
        return np.array([]), np.arange(0)
    n = len(sentences)
    if n > max_sentences:
        # keep longest sentences (heuristic) to bound memory
        lengths = np.array([len(s) for s in sentences])
        idx_keep = np.argsort(-lengths)[:max_sentences]
        idx_keep = np.sort(idx_keep)
        kept = [sentences[i] for i in idx_keep]
    else:
        idx_keep = np.arange(n)
        kept = sentences

    X = vectorizer.transform(kept)  # sparse
    if svd is not None:
        X_reduced = svd.transform(X)
        sim = cosine_similarity(X_reduced)
    else:
        # compute sparse dot product then convert to dense similarity matrix (may be large)
        sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)
    G = nx.from_numpy_array(sim)
    try:
        pr = nx.pagerank_numpy(G, weight="weight")
    except Exception:
        pr = nx.pagerank(G, weight="weight")
    scores = np.array([pr.get(i, 0.0) for i in range(len(kept))], dtype=float)
    # map scores back to original sentence indices (0 for dropped sentences)
    full_scores = np.zeros(n, dtype=float)
    full_scores[idx_keep] = scores
    return full_scores, idx_keep


# -------------------------
# GUI Application (memory-safe)
# -------------------------


class PDFSummarizerMemorySafe(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF OCR + Summarizer (memory-safe, no Tesseract)")
        self.geometry("1100x760")

        # state
        self.pdf_path = None
        self.ocr_text = ""
        self.ocr_lines = []
        self.sentences = []
        self.vectorizer = HashingVectorizer(n_features=HASHING_N_FEATURES, alternate_sign=False, norm='l2', binary=False)
        self.svd = TruncatedSVD(n_components=SVD_DIM) if SVD_DIM and SVD_DIM < HASHING_N_FEATURES else None
        self.tr_scores = None
        self.model = None  # SGDRegressor pipeline will be created on training
        self.scaler = StandardScaler()
        self.reader = None  # EasyOCR reader (reused)
        self.langs_var = tk.StringVar(value="en")
        self.gpu_var = tk.BooleanVar(value=False)
        self._image_thumb_refs = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Select PDF", command=self.select_pdf).pack(side="left", padx=6)
        ttk.Button(top, text="Extract & OCR (stream)", command=self.extract_and_ocr_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Save OCR Lines", command=self.save_ocr_lines).pack(side="left", padx=6)
        ttk.Label(top, text="Langs:").pack(side="left", padx=(12, 4))
        ttk.Entry(top, textvariable=self.langs_var, width=12).pack(side="left")
        ttk.Checkbutton(top, text="GPU", variable=self.gpu_var).pack(side="left", padx=6)
        ttk.Button(top, text="Compute TextRank", command=self.compute_textrank_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Train (incremental)", command=self.train_model_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Generate Summary", command=self.generate_summary).pack(side="left", padx=6)
        ttk.Button(top, text="Save Summary", command=self.save_summary).pack(side="left", padx=6)
        ttk.Label(top, text="Summary length:").pack(side="left", padx=(12, 4))
        self.summary_k = tk.IntVar(value=5)
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.summary_k, width=4).pack(side="left")

        status = ttk.Frame(self); status.pack(fill="x", padx=8)
        self.progress = ttk.Progressbar(status, length=420, mode="determinate"); self.progress.pack(side="left", padx=(0,8))
        self.status_var = tk.StringVar(value="Idle"); ttk.Label(status, textvariable=self.status_var).pack(side="left")

        pan = ttk.PanedWindow(self, orient="horizontal"); pan.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(pan); pan.add(left, weight=1)
        right = ttk.Frame(pan, width=420); pan.add(right, weight=0)

        ttk.Label(left, text="OCR Text").pack(anchor="w")
        self.ocr_textbox = scrolledtext.ScrolledText(left, height=18); self.ocr_textbox.pack(fill="both", expand=True, pady=(4,8))
        ttk.Label(left, text="OCR Lines").pack(anchor="w")
        self.lines_listbox = tk.Listbox(left, height=10); self.lines_listbox.pack(fill="both", expand=False, pady=(4,8))
        self.lines_listbox.bind("<Double-Button-1>", self.copy_line_to_clipboard)
        ttk.Label(left, text="Page thumbnails (streamed)").pack(anchor="w")
        self.images_canvas = tk.Canvas(left, height=140, bg="white"); self.images_canvas.pack(fill="x", pady=(4,8))
        self.images_canvas.bind("<Button-1>", self.on_images_canvas_click)

        ttk.Label(right, text="Sentences (detected)").pack(anchor="w")
        self.sent_list = tk.Listbox(right, height=12); self.sent_list.pack(fill="both", expand=False, pady=(4,8))
        ttk.Label(right, text="Summary").pack(anchor="w")
        self.summary_box = scrolledtext.ScrolledText(right, height=12); self.summary_box.pack(fill="both", expand=True, pady=(4,8))
        ttk.Label(right, text="Log").pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(right, height=6, state="disabled"); self.log_box.pack(fill="both", expand=False, pady=(4,8))

    # -------------------------
    # Helpers
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

    def select_pdf(self):
        path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        self.pdf_path = path
        self.log(f"Selected PDF: {os.path.basename(path)}")
        self.set_status("PDF selected")

    # -------------------------
    # OCR pipeline (streamed)
    # -------------------------
    def extract_and_ocr_thread(self):
        if not self.pdf_path:
            messagebox.showinfo("No PDF", "Select a PDF first.")
            return
        threading.Thread(target=self._extract_and_ocr_stream, daemon=True).start()

    def _ensure_reader(self):
        if self.reader is None:
            langs = [l.strip() for l in self.langs_var.get().split(",") if l.strip()]
            if not langs:
                langs = ["en"]
            gpu = bool(self.gpu_var.get())
            self.log("Initializing EasyOCR reader (may download models)...")
            self.reader = easyocr.Reader(langs, gpu=gpu)

    def _extract_and_ocr_stream(self):
        if not EASYOCR_AVAILABLE:
            messagebox.showerror("Missing dependency", "easyocr is not installed. Install with: pip install easyocr")
            return
        self.set_status("Streaming pages and OCR...", progress=0, maximum=1)
        self.log("Starting streamed extraction and OCR (memory-safe).")
        # prepare reader once
        try:
            self._ensure_reader()
        except Exception as e:
            self.log(f"Failed to init OCR reader: {e}")
            return

        # We'll process pages/images one by one and append OCR text lines incrementally.
        all_text_parts = []
        thumb_x = 4
        pad = 6
        self.images_canvas.delete("all")
        self._image_thumb_refs.clear()
        page_count = 0
        try:
            for im in extract_images_from_pdf_stream(self.pdf_path):
                page_count += 1
                # downscale to limit memory and speed up OCR
                small = downscale_image(im, max_dim=MAX_IMAGE_DIM)
                # OCR single image
                texts = ocr_with_easyocr_batch([small], self.reader)
                txt = texts[0] if texts else ""
                all_text_parts.append(txt)
                # show a small thumbnail (keep only references, not full images)
                try:
                    ratio = small.width / small.height
                    h = 120
                    w = int(h * ratio)
                    thumb = small.resize((w, h), Image.LANCZOS)
                    tkimg = ImageTk.PhotoImage(thumb)
                    self._image_thumb_refs.append(tkimg)
                    self.images_canvas.create_image(thumb_x, 4, anchor="nw", image=tkimg)
                    self.images_canvas.create_rectangle(thumb_x - 1, 3, thumb_x + w + 1, 4 + h + 1, outline="black")
                    self.images_canvas.create_text(thumb_x + 6, 4 + h + 6, anchor="nw", text=str(page_count), fill="blue")
                    thumb_x += w + pad
                except Exception:
                    pass
                # cleanup large objects
                try:
                    small.close()
                except Exception:
                    pass
                try:
                    im.close()
                except Exception:
                    pass
                del small, im
                gc.collect()
                self.set_status(f"OCR pages processed: {page_count}", progress=page_count, maximum=page_count + 1)
            # join OCR text
            self.ocr_text = "\n\n".join(all_text_parts).strip()
            # split into lines and sentences
            self.ocr_lines = [line.strip() for part in all_text_parts for line in part.splitlines() if line.strip()]
            self.sentences = split_into_sentences(self.ocr_text)
            # update UI (on main thread)
            self.after(0, self._update_ui_after_ocr)
            self.log(f"OCR complete: pages processed {page_count}, lines {len(self.ocr_lines)}, sentences {len(self.sentences)}")
            self.set_status("OCR complete")
        except Exception as e:
            self.log(f"Stream OCR failed: {e}")
            self.set_status("Idle")
        finally:
            # ensure reader persists for reuse; free temporary lists
            del all_text_parts
            gc.collect()

    def _update_ui_after_ocr(self):
        self.ocr_textbox.delete("1.0", "end")
        self.ocr_textbox.insert("1.0", self.ocr_text)
        self.lines_listbox.delete(0, "end")
        for i, line in enumerate(self.ocr_lines, start=1):
            display = line if len(line) <= 200 else line[:197] + "..."
            self.lines_listbox.insert("end", f"{i}: {display}")
        self.sent_list.delete(0, "end")
        for s in self.sentences:
            self.sent_list.insert("end", s[:120].replace("\n", " "))

    def on_images_canvas_click(self, event):
        # thumbnails are small; we won't map exact click to page here to keep memory low.
        # Instead, show a message that thumbnails are previews only.
        messagebox.showinfo("Preview", "Thumbnails are previews only. Use OCR text area to inspect content.")

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
            messagebox.showinfo("No OCR lines", "Run OCR first.")
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
    # TextRank (memory-limited)
    # -------------------------
    def compute_textrank_thread(self):
        if not self.sentences:
            messagebox.showinfo("No sentences", "Run OCR first.")
            return
        threading.Thread(target=self._compute_textrank_limited, daemon=True).start()

    def _compute_textrank_limited(self):
        self.set_status("Computing TextRank (limited)...")
        self.log("Vectorizing sentences (HashingVectorizer)...")
        # HashingVectorizer returns sparse matrix; we can optionally reduce with SVD to limit memory
        X = self.vectorizer.transform(self.sentences)
        svd = None
        if self.svd is not None:
            # fit SVD on a sample to avoid huge memory; use small sample if sentences large
            n_samples = min(X.shape[0], 2000)
            try:
                sample_idx = np.linspace(0, X.shape[0] - 1, n_samples, dtype=int)
                X_sample = X[sample_idx]
                self.svd.fit(X_sample)
                svd = self.svd
            except Exception as e:
                self.log(f"SVD fit failed: {e}; proceeding without SVD")
                svd = None
        self.log("Computing TextRank scores (bounded by MAX_SENTENCES)...")
        scores, kept_idx = compute_textrank_limited(self.sentences, self.vectorizer, svd, max_sentences=MAX_SENTENCES)
        self.tr_scores = scores
        # show top few sentences
        top_idx = np.argsort(-scores)[: min(5, len(scores))]
        summary = "\n\n".join([self.sentences[i] for i in sorted(top_idx)])
        self.after(0, lambda: self.summary_box.delete("1.0", "end"))
        self.after(0, lambda: self.summary_box.insert("1.0", summary))
        self.set_status("TextRank computed")
        self.log("TextRank computed (limited).")
        # cleanup
        del X, svd
        gc.collect()

    # -------------------------
    # Incremental training (SGDRegressor)
    # -------------------------
    def train_model_thread(self):
        if not self.sentences or self.tr_scores is None:
            messagebox.showinfo("Missing data", "Run OCR and compute TextRank first.")
            return
        threading.Thread(target=self._train_incremental, daemon=True).start()

    def _train_incremental(self):
        self.set_status("Training incremental model...")
        self.log("Preparing incremental training pipeline (HashingVectorizer + SGDRegressor)...")
        # Build pipeline manually: vectorizer -> scaler -> SGDRegressor
        # We'll train in batches to avoid storing dense matrices.
        n = len(self.sentences)
        indices = np.arange(n)
        # shuffle indices for training
        indices = shuffle(indices, random_state=42)
        # create regressor
        reg = SGDRegressor(max_iter=1000, tol=1e-3)
        # scaler will be fit incrementally using partial_fit on batches
        scaler = StandardScaler(with_mean=False)  # with_mean=False to support sparse input
        batch_size = SGD_BATCH_SIZE
        first_pass = True
        # We'll use HashingVectorizer to produce sparse X; convert to dense for scaler/regressor partial_fit
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_idx = indices[start:end]
            batch_sent = [self.sentences[i] for i in batch_idx]
            y_batch = self.tr_scores[batch_idx]
            X_batch_sparse = self.vectorizer.transform(batch_sent)
            # convert to dense (batch_size x features) - this is the main memory hotspot but bounded by batch_size
            X_batch = X_batch_sparse.toarray()
            # fit scaler incrementally
            if first_pass:
                scaler.partial_fit(X_batch)
            else:
                scaler.partial_fit(X_batch)
            X_batch_scaled = scaler.transform(X_batch)
            # partial_fit regressor
            if first_pass:
                # need to provide classes for regressors? partial_fit for regressor doesn't need classes
                reg.partial_fit(X_batch_scaled, y_batch)
                first_pass = False
            else:
                reg.partial_fit(X_batch_scaled, y_batch)
            # cleanup
            del X_batch_sparse, X_batch, X_batch_scaled
            gc.collect()
            self.set_status(f"Training batches: {end}/{n}")
        # store pipeline components
        self.model = {"regressor": reg, "scaler": scaler}
        self.log("Incremental training complete.")
        self.set_status("Model trained")
        gc.collect()

    # -------------------------
    # Summary generation
    # -------------------------
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
            self.log("Using TextRank scores (no trained model).")
        else:
            # vectorize in batches to avoid huge dense matrix
            n = len(self.sentences)
            batch_size = SGD_BATCH_SIZE
            scores = np.zeros(n, dtype=float)
            reg = self.model["regressor"]
            scaler = self.model["scaler"]
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                batch_sent = self.sentences[start:end]
                Xs = self.vectorizer.transform(batch_sent).toarray()
                Xs_scaled = scaler.transform(Xs)
                try:
                    preds = reg.predict(Xs_scaled)
                except Exception:
                    preds = np.zeros(Xs_scaled.shape[0])
                scores[start:end] = preds
                del Xs, Xs_scaled
                gc.collect()
        idxs = np.argsort(-scores)
        topk = sorted(idxs[:min(k, len(idxs))])
        summary = "\n\n".join([self.sentences[i] for i in topk])
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", summary)
        self.log(f"Generated summary with {len(topk)} sentences")
        self.set_status("Summary generated")
        gc.collect()

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

# -------------------------
# Run
# -------------------------
def main():
    if not EASYOCR_AVAILABLE:
        print("Warning: easyocr not installed. Install with: pip install easyocr")
    app = PDFSummarizerMemorySafe()
    app.mainloop()

if __name__ == "__main__":
    main()