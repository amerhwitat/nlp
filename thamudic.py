#!/usr/bin/env python3
"""
Thamudic Image Reader — show image + word-by-word transliteration with on-screen log and progress

Features
- Load images (local) or a folder
- Scan images for glyph clusters (no OpenCV)
- Group glyphs into words, OCR each word region (pytesseract) and extract Thamudic sequences
- Transliterate using a user-editable mapping CSV (or simple unknown marker)
- Display image with highlighted word boxes and show original + transliteration in a table
- On-screen log and progress bar; threaded processing so UI stays responsive
- Save results to CSV

Requirements
- Python 3.8+ (3.13 recommended)
- pip install pillow numpy pytesseract pandas beautifulsoup4 requests
- Tesseract OCR installed and on PATH for pytesseract to work (optional but recommended)
"""

import os
import io
import time
import json
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from PIL import Image, ImageOps, ImageFilter, ImageTk
import numpy as np
import pytesseract
import re
import pandas as pd

# --- Configuration and files ---
OUT_DIR = "thamudic_images"
MAPPING_CSV = "thamudic_char_map.csv"
OUTPUT_CSV = "thamudic_translations.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# Unicode regex for Ancient North Arabian (Thamudic) U+10A80–U+10A9F
THAMUDIC_RE = re.compile(r'[\U00010A80-\U00010A9F]+')

# --- Image processing helpers (no OpenCV) ---


def preprocess_image_for_detection(pil_img, block=32):
    """Grayscale, autocontrast, median filter, block adaptive thresholding."""
    im = pil_img.convert("L")
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.MedianFilter(3))
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for y in range(0, h, block):
        for x in range(0, w, block):
            by = arr[y:y + block, x:x + block]
            if by.size == 0:
                continue
            m = int(np.mean(by))
            th = max(10, m - 12)
            out[y:y + block, x:x + block] = (by > th) * 255
    return Image.fromarray(out.astype(np.uint8))


def connected_components_boxes(binary_arr, min_area=30):
    """Find connected components in binary 2D numpy array (0/255) and return bounding boxes."""
    h, w = binary_arr.shape
    visited = np.zeros((h, w), dtype=bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if visited[y, x] or binary_arr[y, x] == 0:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            xs = []
            ys = []
            while stack:
                sx, sy = stack.pop()
                xs.append(sx); ys.append(sy)
                for nx, ny in ((sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and binary_arr[ny, nx] != 0:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if not xs:
                continue
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if area >= min_area:
                boxes.append((x1, y1, x2, y2))
    return boxes


def group_boxes_into_lines(boxes, y_tol=14):
    """Group boxes into horizontal lines by center Y proximity."""
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    lines = []
    for b in boxes_sorted:
        cy = (b[1] + b[3]) / 2
        placed = False
        for line in lines:
            ly = np.mean([(bb[1] + bb[3]) / 2 for bb in line])
            if abs(cy - ly) <= y_tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    for line in lines:
        line.sort(key=lambda bb: bb[0])
    return lines


def group_line_boxes_into_words(line_boxes, gap_threshold=18):
    """Group boxes in a line into words by horizontal gap threshold."""
    if not line_boxes:
        return []
    words = []
    current = [line_boxes[0]]
    for prev, cur in zip(line_boxes, line_boxes[1:]):
        gap = cur[0] - prev[2]
        if gap <= gap_threshold:
            current.append(cur)
        else:
            words.append(current)
            current = [cur]
    words.append(current)
    return words


# --- OCR and transliteration helpers ---


def ocr_crop_get_thamudic(pil_crop):
    """Run pytesseract on crop and extract Thamudic sequences (best-effort)."""
    try:
        txt = pytesseract.image_to_string(pil_crop, lang="ara+eng", config="--psm 6")
    except Exception:
        txt = pytesseract.image_to_string(pil_crop, config="--psm 6")
    seqs = THAMUDIC_RE.findall(txt)
    if seqs:
        return " ".join(seqs)
    # fallback: return any non-whitespace characters (may include glyph placeholders)
    cleaned = "".join(ch for ch in txt if not ch.isspace())
    return cleaned.strip()


def load_mapping(csv_path=MAPPING_CSV):
    """Load char->transliteration mapping CSV (char,transliteration)."""
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, r in df.iterrows():
            ch = r.get("char", "")
            tr = r.get("transliteration", "")
            if ch:
                mapping[ch] = tr
    except Exception:
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                for r in rdr:
                    if not r:
                        continue
                    ch = r[0].strip()
                    tr = r[1].strip() if len(r) > 1 else ""
                    if ch:
                        mapping[ch] = tr
        except Exception:
            pass
    return mapping


def save_mapping(mapping, csv_path=MAPPING_CSV):
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["char", "transliteration"])
            for ch, tr in mapping.items():
                writer.writerow([ch, tr])
    except Exception:
        pass


def transliterate_word(word, mapping):
    out = []
    for ch in word:
        if ch.isspace():
            out.append(" ")
        else:
            out.append(mapping.get(ch, "?"))
    return "".join(out)


# --- GUI application ---


class ThamudicReaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thamudic Reader — Image + Translation")
        self.geometry("1100x720")
        self.image_paths = []
        self.current_index = -1
        self.current_image_pil = None
        self.current_image_tk = None
        self.current_boxes = []  # list of (x1,y1,x2,y2)
        self.words = []  # list of dicts: {source, translit, note, bbox, image}
        self.mapping = load_mapping()
        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Load Images", command=self.load_images).pack(side="left", padx=4)
        ttk.Button(top, text="Load Folder", command=self.load_folder).pack(side="left", padx=4)
        ttk.Button(top, text="Scan Current Image", command=self.scan_current_image_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Scan All Images", command=self.scan_all_images_thread).pack(side="left", padx=6)
        ttk.Button(top, text="Load Mapping CSV", command=self.load_mapping_file).pack(side="left", padx=6)
        ttk.Button(top, text="Save Mapping CSV", command=self.save_mapping_file).pack(side="left", padx=6)
        ttk.Button(top, text="Save Results CSV", command=self.save_results).pack(side="left", padx=6)

        # Progress bar and status
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=8)
        self.progress = ttk.Progressbar(status_frame, length=420, mode="determinate")
        self.progress.pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")

        # Main area: left image, right table
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # Left: image canvas and navigation
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=False)
        self.canvas = tk.Canvas(left, width=640, height=540, bg="black")
        self.canvas.pack()
        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=6)
        ttk.Button(nav, text="Prev", command=self.prev_image).pack(side="left", padx=4)
        ttk.Button(nav, text="Next", command=self.next_image).pack(side="left", padx=4)
        ttk.Button(nav, text="Show Boxes", command=self.redraw_boxes).pack(side="left", padx=4)
        self.image_label = ttk.Label(left, text="No image loaded")
        self.image_label.pack()

        # Right: table of words and preview
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        cols = ("original", "transliteration", "note")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("original", text="Original (Thamudic)")
        self.tree.heading("transliteration", text="Transliteration")
        self.tree.heading("note", text="Note / Translation")
        self.tree.column("original", width=200)
        self.tree.column("transliteration", width=180)
        self.tree.column("note", width=220)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        preview_frame = ttk.Frame(right)
        preview_frame.pack(fill="x", pady=6)
        ttk.Label(preview_frame, text="Word preview").pack(anchor="w")
        self.preview_canvas = tk.Canvas(preview_frame, width=220, height=80, bg="white")
        self.preview_canvas.pack()

        # Bottom: log
        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=False, padx=8, pady=(6, 8))
        ttk.Label(bottom, text="Log").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(bottom, height=8, state="disabled")
        self.log_area.pack(fill="both", expand=True)

    # --- logging and status helpers ---

    def log(self, text):
        self.log_area.configure(state="normal")
        self.log_area.insert("end", f"{time.strftime('%H:%M:%S')} - {text}\n")
        self.log_area.see("end")
        self.log_area.configure(state="disabled")

    def set_status(self, text, progress=None, maximum=None):
        self.status_var.set(text)
        if maximum is not None:
            self.progress['maximum'] = maximum
        if progress is not None:
            self.progress['value'] = progress
        self.update_idletasks()

    # --- file loading ---

    def load_images(self):
        paths = filedialog.askopenfilenames(title="Select images", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")])
        if not paths:
            return
        for p in paths:
            if p not in self.image_paths:
                self.image_paths.append(p)
        if self.current_index == -1 and self.image_paths:
            self.current_index = 0
            self.load_current_image()
        self.log(f"Loaded {len(paths)} images")

    def load_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        added = 0
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                path = os.path.join(folder, fname)
                if path not in self.image_paths:
                    self.image_paths.append(path)
                    added += 1
        if self.current_index == -1 and self.image_paths:
            self.current_index = 0
            self.load_current_image()
        self.log(f"Added {added} images from folder")

    def load_mapping_file(self):
        path = filedialog.askopenfilename(title="Select mapping CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            mapping = {}
            for _, r in df.iterrows():
                ch = r.get("char", "")
                tr = r.get("transliteration", "")
                if ch:
                    mapping[ch] = tr
            self.mapping = mapping
            self.log(f"Loaded mapping ({len(mapping)} entries) from {os.path.basename(path)}")
            messagebox.showinfo("Mapping loaded", f"Loaded {len(mapping)} mapping entries")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mapping: {e}")

    def save_mapping_file(self):
        path = filedialog.asksaveasfilename(title="Save mapping CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["char", "transliteration"])
                for ch, tr in self.mapping.items():
                    writer.writerow([ch, tr])
            self.log(f"Saved mapping to {path}")
            messagebox.showinfo("Saved", f"Mapping saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mapping: {e}")

    # --- image navigation and display ---

    def load_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return
        path = self.image_paths[self.current_index]
        try:
            pil = Image.open(path).convert("RGB")
            self.current_image_pil = pil
            self.display_image(pil)
            self.image_label.config(text=os.path.basename(path))
            self.log(f"Loaded image {os.path.basename(path)}")
        except Exception as e:
            self.log(f"Failed to open image {path}: {e}")

    def display_image(self, pil):
        cw = 640; ch = 540
        iw, ih = pil.size
        scale = min(cw / iw, ch / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)
        resized = pil.resize((new_w, new_h), Image.LANCZOS)
        self.current_image_tk = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image_tk)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self._display_scale = scale

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = max(0, self.current_index - 1)
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
        self.load_current_image()

    # --- scanning and processing (threaded) ---

    def scan_current_image_thread(self):
        threading.Thread(target=self.scan_current_image, daemon=True).start()

    def scan_all_images_thread(self):
        threading.Thread(target=self.scan_all_images, daemon=True).start()

    def scan_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            messagebox.showinfo("Info", "No image loaded")
            return
        path = self.image_paths[self.current_index]
        self.set_status("Scanning image...", progress=0, maximum=1)
        self.log(f"Scanning {os.path.basename(path)}")
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            self.log(f"Failed to open image: {e}")
            self.set_status("Idle", progress=0)
            return
        pre = preprocess_image_for_detection(pil)
        arr = np.array(pre)
        bin_arr = (arr > 127).astype(np.uint8) * 255
        boxes = connected_components_boxes(bin_arr, min_area=40)
        lines = group_boxes_into_lines(boxes, y_tol=18)
        words_found = []
        for line in lines:
            words = group_line_boxes_into_words(line, gap_threshold=20)
            for word_boxes in words:
                x1 = min(b[0] for b in word_boxes)
                y1 = min(b[1] for b in word_boxes)
                x2 = max(b[2] for b in word_boxes)
                y2 = max(b[3] for b in word_boxes)
                pad = 4
                iw, ih = pil.size
                cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad); cx2 = min(iw, x2 + pad); cy2 = min(ih, y2 + pad)
                crop = pil.crop((cx1, cy1, cx2, cy2))
                text = ocr_crop_get_thamudic(crop)
                if not text:
                    continue
                # split tokens and keep those with Thamudic chars
                tokens = re.split(r'[\s\.,;:\-\—\(\)\[\]\"\'\u200e\u200f]+', text)
                for t in tokens:
                    if not t:
                        continue
                    if THAMUDIC_RE.search(t):
                        translit = transliterate_word(t, self.mapping)
                        entry = {"source": t, "translit": translit, "note": "", "image": path, "bbox": (cx1, cy1, cx2, cy2)}
                        words_found.append(entry)
        # update UI
        self.current_boxes = [w["bbox"] for w in words_found]
        # append to global words list and refresh tree
        added = 0
        for w in words_found:
            self.words.append(w); added += 1
        self.refresh_tree()
        self.redraw_boxes()
        self.log(f"Scan complete: {added} words added")
        self.set_status("Idle", progress=0)

    def scan_all_images(self):
        total = len(self.image_paths)
        if total == 0:
            messagebox.showinfo("Info", "No images loaded")
            return
        self.set_status("Scanning all images...", progress=0, maximum=total)
        all_added = 0
        for i, path in enumerate(self.image_paths, start=1):
            self.set_status(f"Scanning {os.path.basename(path)} ({i}/{total})", progress=i-1, maximum=total)
            self.log(f"[{i}/{total}] Scanning {os.path.basename(path)}")
            try:
                pil = Image.open(path).convert("RGB")
            except Exception as e:
                self.log(f"  failed to open: {e}")
                continue
            pre = preprocess_image_for_detection(pil)
            arr = np.array(pre)
            bin_arr = (arr > 127).astype(np.uint8) * 255
            boxes = connected_components_boxes(bin_arr, min_area=40)
            lines = group_boxes_into_lines(boxes, y_tol=18)
            words_found = []
            for line in lines:
                words = group_line_boxes_into_words(line, gap_threshold=20)
                for word_boxes in words:
                    x1 = min(b[0] for b in word_boxes)
                    y1 = min(b[1] for b in word_boxes)
                    x2 = max(b[2] for b in word_boxes)
                    y2 = max(b[3] for b in word_boxes)
                    pad = 4
                    iw, ih = pil.size
                    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad); cx2 = min(iw, x2 + pad); cy2 = min(ih, y2 + pad)
                    crop = pil.crop((cx1, cy1, cx2, cy2))
                    text = ocr_crop_get_thamudic(crop)
                    if not text:
                        continue
                    tokens = re.split(r'[\s\.,;:\-\—\(\)\[\]\"\'\u200e\u200f]+', text)
                    for t in tokens:
                        if not t:
                            continue
                        if THAMUDIC_RE.search(t):
                            translit = transliterate_word(t, self.mapping)
                            entry = {"source": t, "translit": translit, "note": "", "image": path, "bbox": (cx1, cy1, cx2, cy2)}
                            words_found.append(entry)
            for w in words_found:
                self.words.append(w); all_added += 1
            self.set_status(f"Scanned {i}/{total}", progress=i, maximum=total)
            self.log(f"  found {len(words_found)} words in {os.path.basename(path)}")
        self.refresh_tree()
        self.log(f"All scans complete. Total words added: {all_added}")
        self.set_status("Idle", progress=0)

    # --- UI updates: tree and preview ---

    def refresh_tree(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for i, e in enumerate(self.words):
            self.tree.insert("", "end", iid=str(i), values=(e["source"], e["translit"], e.get("note", "")))

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        entry = self.words[idx]
        # show preview crop
        img_path = entry.get("image")
        bbox = entry.get("bbox")
        if img_path and bbox:
            try:
                pil = Image.open(img_path).convert("RGB")
                crop = pil.crop(bbox).resize((220, 80), Image.LANCZOS)
                tkimg = ImageTk.PhotoImage(crop)
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(0, 0, anchor="nw", image=tkimg)
                self.preview_canvas.image = tkimg  # keep reference
            except Exception:
                pass

    def on_tree_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        col = self.tree.identify_column(event.x)
        col_index = int(col.replace("#", "")) - 1
        if col_index not in (0, 1, 2):
            return
        x, y, width, height = self.tree.bbox(item, column=col)
        value = self.tree.set(item, column=self.tree["columns"][col_index])
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus_set()

        def on_commit(event=None):
            new_val = entry.get()
            entry.destroy()
            idx = int(item)
            if col_index == 0:
                self.words[idx]["source"] = new_val
                self.words[idx]["translit"] = transliterate_word(new_val, self.mapping)
            elif col_index == 1:
                self.words[idx]["translit"] = new_val
            else:
                self.words[idx]["note"] = new_val
            self.refresh_tree()

        entry.bind("<Return>", on_commit)
        entry.bind("<FocusOut>", on_commit)

    # --- drawing boxes on image canvas ---

    def redraw_boxes(self):
        if not self.current_image_pil:
            return
        self.display_image(self.current_image_pil)
        # draw boxes for words that belong to current image
        path = self.image_paths[self.current_index] if 0 <= self.current_index < len(self.image_paths) else None
        if not path:
            return
        scale = getattr(self, "_display_scale", 1.0)
        for i, e in enumerate(self.words):
            if e.get("image") != path:
                continue
            bbox = e.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            rx1 = int(x1 * scale); ry1 = int(y1 * scale); rx2 = int(x2 * scale); ry2 = int(y2 * scale)
            self.canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="lime", width=2)
            self.canvas.create_text(rx1 + 4, ry1 + 4, anchor="nw", text=e["source"], fill="yellow", font=("Arial", 10))

    # --- save results ---

    def save_results(self):
        path = filedialog.asksaveasfilename(title="Save results CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["original", "transliteration", "note", "image", "bbox"])
                for e in self.words:
                    writer.writerow([e["source"], e["translit"], e.get("note", ""), e.get("image", ""), json.dumps(e.get("bbox", ""))])
            self.log(f"Saved {len(self.words)} entries to {path}")
            messagebox.showinfo("Saved", f"Saved {len(self.words)} entries to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    # --- utility: add mapping entry quickly ---

    def add_mapping_entry(self):
        ch = tk.simpledialog.askstring("Character", "Enter Thamudic character (single):", parent=self)
        if not ch:
            return
        tr = tk.simpledialog.askstring("Transliteration", f"Enter transliteration for {ch}:", parent=self)
        if tr is None:
            tr = ""
        self.mapping[ch] = tr
        save_mapping(self.mapping, MAPPING_CSV)
        messagebox.showinfo("Saved", f"Mapping saved to {MAPPING_CSV}")
        self.log(f"Added mapping: {ch} -> {tr}")

    # --- end ---


def save_mapping(mapping, csv_path=MAPPING_CSV):
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["char", "transliteration"])
            for ch, tr in mapping.items():
                writer.writerow([ch, tr])
    except Exception:
        pass


if __name__ == "__main__":
    app = ThamudicReaderApp()
    # add a small menu for mapping quick add
    menubar = tk.Menu(app)
    app.config(menu=menubar)
    tools = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Tools", menu=tools)
    tools.add_command(label="Add mapping entry", command=app.add_mapping_entry)
    app.mainloop()