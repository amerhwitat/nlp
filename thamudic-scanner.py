#!/usr/bin/env python3
"""
Thamudic template parser and variation inspector

- Splits horizontal strips into segments (variants).
- Groups variants by label extracted from filename.
- Lets user review and edit English equivalents for each Thamudic label.
- Saves mapping CSV with columns: label, english, variant_files...
"""

from __future__ import annotations
import os
import math
import json
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from PIL import Image, ImageOps, ImageFilter, ImageTk
import numpy as np
import pandas as pd
import threading
import typing as T

# -----------------------
# Configuration
# -----------------------
TEMPLATE_MAPPING_CSV = "template_mapping.csv"
VARIANT_OUTPUT_DIR = "template_variants"
os.makedirs(VARIANT_OUTPUT_DIR, exist_ok=True)

# -----------------------
# Utilities
# -----------------------


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def normalize_vec(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(np.float32).ravel()
    mean = v.mean() if v.size else 0.0
    std = v.std() if v.size else 1.0
    if std < 1e-6:
        std = 1.0
    v = (v - mean) / std
    norm = np.linalg.norm(v) + 1e-12
    return v / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return -1.0
    return float(np.dot(a, b))


# -----------------------
# Horizontal strip splitter
# -----------------------
def split_horizontal_strip(img: Image.Image, min_width: int = 8, gap_threshold: float = 0.98) -> list[Image.Image]:
    """
    Split an image that contains multiple glyph variants arranged horizontally.
    Strategy:
      - Convert to binary (adaptive-ish) and compute column-wise "ink" density.
      - Find long runs of near-zero density (gaps) to separate segments.
      - Return list of cropped PIL images (tight vertical crop).
    """
    # convert to grayscale array
    arr = np.array(ImageOps.autocontrast(img).resize(img.size))
    # normalize to 0..1
    arrn = (arr.astype(np.float32) - arr.min()) / (arr.max() - arr.min() + 1e-9)
    # invert so ink ~1
    ink = 1.0 - arrn
    # column density
    col_density = ink.mean(axis=0)  # shape (W,)
    W = col_density.shape[0]
    # find columns that are "empty" (low ink)
    empty = col_density < (col_density.max() * 0.05 + 1e-6)
    # find runs of empty columns
    runs = []
    start = None
    for i, v in enumerate(empty):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, W - 1))
    # compute candidate split positions as midpoints of sufficiently long empty runs
    splits = []
    min_gap = max(3, int(W * 0.01))
    for s, e in runs:
        if (e - s + 1) >= min_gap:
            splits.append((s + e) // 2)
    # if no splits found, try a weaker threshold using relative gap detection
    if not splits:
        # find local minima in col_density
        from scipy.signal import find_peaks  # optional; fallback if not available
        try:
            peaks, _ = find_peaks(-col_density, distance=max(2, W // 20))
            splits = peaks.tolist()
        except Exception:
            # fallback: split by equal-width chunks if image is wide
            if W > 120:
                n = max(2, W // 60)
                splits = [int(W * i / n) for i in range(1, n)]
    # build segment x ranges
    xs = [0] + splits + [W]
    segments = []
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if x2 - x1 < min_width:
            continue
        crop = img.crop((x1, 0, x2, img.height))
        # tight vertical crop to remove top/bottom whitespace
        arrc = np.array(ImageOps.autocontrast(crop))
        row_density = (1.0 - (arrc.astype(np.float32) - arrc.min()) / (arrc.max() - arrc.min() + 1e-9)).mean(axis=1)
        non_empty_rows = np.where(row_density > 0.02)[0]
        if non_empty_rows.size:
            y1, y2 = int(non_empty_rows[0]), int(non_empty_rows[-1]) + 1
            crop = crop.crop((0, y1, crop.width, y2))
        segments.append(crop)
    # if no segments found, return the original as single segment
    if not segments:
        return [img]
    return segments


# -----------------------
# Template loader and grouping
# -----------------------
def parse_label_from_filename(fname: str) -> tuple[str, str]:
    """
    Try to extract (label, english) from filename.
    Examples:
      - alif_A_var1.png -> ("alif", "A")
      - U+10A80-alif-A.png -> ("alif", "A")
      - thamudic_alif.png -> ("alif", "")
    Returns (label, english) where english may be empty string.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = re_split_tokens(base)
    label = ""
    english = ""
    # prefer first token as label, last token as english if ASCII letters
    if parts:
        label = parts[0]
        if len(parts) >= 2:
            last = parts[-1]
            if last.isascii() and any(ch.isalpha() for ch in last):
                english = last
            elif len(parts) >= 3:
                # maybe middle token is english
                mid = parts[1]
                if mid.isascii() and any(ch.isalpha() for ch in mid):
                    english = mid
    return label, english


def re_split_tokens(s: str) -> list[str]:
    import re
    return [p for p in re.split(r'[_\-\s\.]+', s) if p]


def load_templates_folder(folder: str, resize=(48, 48)) -> dict[str, dict]:
    """
    Load templates from folder. For each file:
      - if image is wider than tall and appears to be a horizontal strip, split into segments
      - extract label and english from filename
      - store each variant as an entry with vector for matching
    Returns groups: {label: {"english": str, "variants": [variant_info,...]}}
    variant_info: {"path":..., "image":PIL.Image, "vec":np.array, "score":None}
    """
    groups: dict[str, dict] = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            continue
        path = os.path.join(folder, fname)
        try:
            img = load_image(path)
        except Exception:
            continue
        # decide if horizontal strip: width significantly larger than height
        is_strip = img.width >= img.height * 1.6 and img.width > 80
        segments = [img]
        if is_strip:
            try:
                segments = split_horizontal_strip(img)
            except Exception:
                segments = [img]
        label, english = parse_label_from_filename(fname)
        if not label:
            # fallback to filename base
            label = os.path.splitext(fname)[0]
        # ensure group exists
        if label not in groups:
            groups[label] = {"english": english or "", "variants": []}
        # add each segment as variant
        for i, seg in enumerate(segments):
            # normalize and vectorize
            seg_small = seg.resize(resize, Image.LANCZOS)
            arr = np.array(ImageOps.autocontrast(seg_small), dtype=np.float32)
            vec = normalize_vec(arr)
            variant_name = f"{os.path.splitext(fname)[0]}_v{i}"
            # save a copy of variant to disk for traceability
            out_name = os.path.join(VARIANT_OUTPUT_DIR, f"{variant_name}.png")
            try:
                seg_small.convert("L").save(out_name)
            except Exception:
                out_name = path  # fallback
            groups[label]["variants"].append({
                "source_file": path,
                "variant_file": out_name,
                "image": seg_small,
                "vec": vec,
                "score": None
            })
        # if english empty but group had previous english, keep previous; else set
        if not groups[label]["english"] and english:
            groups[label]["english"] = english
    return groups


# -----------------------
# Similarity analysis
# -----------------------
def compute_pairwise_similarities(group: dict) -> list[tuple[int, int, float]]:
    """
    For a group, compute pairwise cosine similarities between variants.
    Returns list of (i, j, score) sorted descending by score.
    """
    variants = group["variants"]
    n = len(variants)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            a = variants[i]["vec"]
            b = variants[j]["vec"]
            s = cosine_sim(a, b)
            results.append((i, j, s))
    results.sort(key=lambda x: -x[2])
    return results


# -----------------------
# GUI Application
# -----------------------
class TemplateInspector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thamudic Template Inspector")
        self.geometry("1100x720")
        self.groups: dict[str, dict] = {}
        self.current_label: str | None = None
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Load Templates", command=self.load_templates_dialog).pack(side="left", padx=4)
        ttk.Button(top, text="Load Mapping CSV", command=self.load_mapping_csv).pack(side="left", padx=4)
        ttk.Button(top, text="Save Mapping CSV", command=self.save_mapping_csv).pack(side="left", padx=4)
        ttk.Button(top, text="Export Variants", command=self.export_variants).pack(side="left", padx=4)
        ttk.Button(top, text="Compute Similarities", command=self.compute_similarities_current).pack(side="left", padx=4)

        mid = ttk.Frame(self); mid.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(mid, width=260); left.pack(side="left", fill="y")
        ttk.Label(left, text="Template Groups").pack(anchor="w")
        self.group_list = tk.Listbox(left, width=36, height=30)
        self.group_list.pack(fill="y", expand=True)
        self.group_list.bind("<<ListboxSelect>>", self.on_group_select)

        right = ttk.Frame(mid); right.pack(side="left", fill="both", expand=True, padx=(8,0))
        # top area: label and english edit
        top_right = ttk.Frame(right); top_right.pack(fill="x")
        ttk.Label(top_right, text="Label:").pack(side="left")
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(top_right, textvariable=self.label_var, width=24)
        self.label_entry.pack(side="left", padx=(4, 12))
        ttk.Label(top_right, text="English:").pack(side="left")
        self.english_var = tk.StringVar()
        self.english_entry = ttk.Entry(top_right, textvariable=self.english_var, width=24)
        self.english_entry.pack(side="left", padx=(4, 12))
        ttk.Button(top_right, text="Apply Edits", command=self.apply_label_edits).pack(side="left")

        # center area: variants canvas
        center = ttk.Frame(right); center.pack(fill="both", expand=True, pady=(8,0))
        self.canvas = tk.Canvas(center, bg="white")
        self.canvas.pack(fill="both", expand=True)
        # bottom: variant list and actions
        bottom = ttk.Frame(right); bottom.pack(fill="x", pady=(8,0))
        ttk.Label(bottom, text="Variants").pack(anchor="w")
        self.variant_list = tk.Listbox(bottom, height=6)
        self.variant_list.pack(fill="x", expand=True)
        btns = ttk.Frame(bottom); btns.pack(fill="x", pady=(6,0))
        ttk.Button(btns, text="Remove Variant", command=self.remove_variant).pack(side="left", padx=4)
        ttk.Button(btns, text="Mark Bad", command=self.mark_variant_bad).pack(side="left", padx=4)
        ttk.Button(btns, text="Show Pairwise Similarities", command=self.show_pairwise_sim).pack(side="left", padx=4)

        # log
        logf = ttk.Frame(self); logf.pack(fill="x", padx=6, pady=(6,8))
        ttk.Label(logf, text="Log").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(logf, height=6, state="disabled")
        self.log_area.pack(fill="both", expand=True)

    def log(self, text: str):
        self.log_area.configure(state="normal")
        self.log_area.insert("end", f"{time.strftime('%H:%M:%S')} - {text}\n")
        self.log_area.see("end")
        self.log_area.configure(state="disabled")

    # -----------------------
    # Actions
    # -----------------------
    def load_templates_dialog(self):
        folder = filedialog.askdirectory(title="Select templates folder")
        if not folder:
            return
        self.log(f"Loading templates from {folder} ...")
        # load in background to keep UI responsive
        threading.Thread(target=self._load_templates, args=(folder,), daemon=True).start()

    def _load_templates(self, folder: str):
        groups = load_templates_folder(folder, resize=(64, 64))
        # merge with existing groups (append variants)
        for label, info in groups.items():
            if label not in self.groups:
                self.groups[label] = {"english": info.get("english", ""), "variants": []}
            # append variants
            for v in info["variants"]:
                self.groups[label]["variants"].append(v)
        # update UI on main thread
        self.after(0, self._refresh_group_list)
        self.log(f"Loaded {len(groups)} labels (groups). Total groups now: {len(self.groups)}")

    def _refresh_group_list(self):
        self.group_list.delete(0, "end")
        for label in sorted(self.groups.keys()):
            english = self.groups[label].get("english", "")
            display = f"{label}  [{english}]" if english else label
            self.group_list.insert("end", display)

    def on_group_select(self, event=None):
        sel = self.group_list.curselection()
        if not sel:
            return
        idx = sel[0]
        label = sorted(self.groups.keys())[idx]
        self.current_label = label
        info = self.groups[label]
        self.label_var.set(label)
        self.english_var.set(info.get("english", ""))
        self._refresh_variant_list()
        self._draw_variants()

    def _refresh_variant_list(self):
        self.variant_list.delete(0, "end")
        if not self.current_label:
            return
        variants = self.groups[self.current_label]["variants"]
        for i, v in enumerate(variants):
            fname = os.path.basename(v.get("variant_file", v.get("source_file", "")))
            self.variant_list.insert("end", f"{i}: {fname}")

    def _draw_variants(self):
        self.canvas.delete("all")
        if not self.current_label:
            return
        variants = self.groups[self.current_label]["variants"]
        if not variants:
            return
        # draw thumbnails horizontally
        pad = 8
        thumb_h = min(160, max(48, int(self.canvas.winfo_height() * 0.6)))
        x = pad
        y = pad
        images_refs = []
        for i, v in enumerate(variants):
            img = v["image"].resize((int(thumb_h * v["image"].width / v["image"].height), thumb_h), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(img.convert("RGB"))
            images_refs.append(tkimg)
            self.canvas.create_image(x, y, anchor="nw", image=tkimg)
            self.canvas.create_rectangle(x - 1, y - 1, x + img.width + 1, y + img.height + 1, outline="black")
            self.canvas.create_text(x + 4, y + img.height + 4, anchor="nw", text=str(i), fill="blue")
            x += img.width + pad
        # keep references to avoid GC
        self.canvas.images = images_refs

    def apply_label_edits(self):
        if not self.current_label:
            return
        new_label = self.label_var.get().strip()
        new_english = self.english_var.get().strip()
        if not new_label:
            messagebox.showerror("Error", "Label cannot be empty")
            return
        # rename group if label changed
        if new_label != self.current_label:
            if new_label in self.groups:
                messagebox.showerror("Error", "Label already exists")
                return
            self.groups[new_label] = self.groups.pop(self.current_label)
            self.current_label = new_label
        self.groups[self.current_label]["english"] = new_english
        self._refresh_group_list()
        self.log(f"Updated label '{self.current_label}' english='{new_english}'")

    def remove_variant(self):
        sel = self.variant_list.curselection()
        if not sel or not self.current_label:
            return
        idx = sel[0]
        variants = self.groups[self.current_label]["variants"]
        if 0 <= idx < len(variants):
            removed = variants.pop(idx)
            self._refresh_variant_list()
            self._draw_variants()
            self.log(f"Removed variant {os.path.basename(removed.get('variant_file'))} from {self.current_label}")

    def mark_variant_bad(self):
        sel = self.variant_list.curselection()
        if not sel or not self.current_label:
            return
        idx = sel[0]
        variants = self.groups[self.current_label]["variants"]
        if 0 <= idx < len(variants):
            variants[idx]["bad"] = True
            self.log(f"Marked variant {idx} as bad for {self.current_label}")

    def compute_similarities_current(self):
        if not self.current_label:
            messagebox.showinfo("Info", "Select a group first")
            return
        group = self.groups[self.current_label]
        sims = compute_pairwise_similarities(group)
        if not sims:
            messagebox.showinfo("Info", "Not enough variants to compute similarities")
            return
        # show top similarities in a small window
        win = tk.Toplevel(self)
        win.title(f"Pairwise similarities for {self.current_label}")
        txt = scrolledtext.ScrolledText(win, width=60, height=20)
        txt.pack(fill="both", expand=True)
        for i, j, s in sims[:200]:
            txt.insert("end", f"{i} <-> {j} : {s:.4f}\n")
        txt.configure(state="disabled")

    def show_pairwise_sim(self):
        self.compute_similarities_current()

    def load_mapping_csv(self):
        path = filedialog.askopenfilename(title="Select mapping CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            for _, r in df.iterrows():
                label = r.get("label") or r.get("char") or r.get("template")
                english = r.get("english") or r.get("transliteration") or r.get("meaning") or ""
                if label:
                    if label not in self.groups:
                        self.groups[label] = {"english": english, "variants": []}
                    else:
                        if english:
                            self.groups[label]["english"] = english
            self._refresh_group_list()
            self.log(f"Loaded mapping CSV {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mapping CSV: {e}")

    def save_mapping_csv(self):
        path = filedialog.asksaveasfilename(title="Save mapping CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        rows = []
        for label, info in self.groups.items():
            english = info.get("english", "")
            variant_files = ";".join([os.path.basename(v.get("variant_file", v.get("source_file", ""))) for v in info["variants"]])
            rows.append({"label": label, "english": english, "variants": variant_files})
        try:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
            self.log(f"Saved mapping to {path}")
            messagebox.showinfo("Saved", f"Saved mapping to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mapping: {e}")

    def export_variants(self):
        # export all variant images into a chosen folder grouped by label
        folder = filedialog.askdirectory(title="Export variants to folder")
        if not folder:
            return
        for label, info in self.groups.items():
            labdir = os.path.join(folder, label)
            os.makedirs(labdir, exist_ok=True)
            for i, v in enumerate(info["variants"]):
                out = os.path.join(labdir, f"{label}_v{i}.png")
                try:
                    v["image"].convert("L").save(out)
                except Exception:
                    pass
        self.log(f"Exported variants to {folder}")
        messagebox.showinfo("Exported", f"Exported variants to {folder}")

# -----------------------
# Run
# -----------------------
def main():
    app = TemplateInspector()
    app.mainloop()


if __name__ == "__main__":
    main()