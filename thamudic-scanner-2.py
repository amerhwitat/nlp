#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thamudic AI Scanner – FULLY INTEGRATED SINGLE FILE

NO OCR
NO TESSERACT
VISION + ML + HUMAN-IN-THE-LOOP

pip install pillow numpy pandas scikit-learn torch torchvision requests beautifulsoup4
"""

# ===============================
# Imports
# ===============================
import os, io, time, json, threading, math, csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageTk

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Optional torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ===============================
# Paths
# ===============================
BASE_DIR = os.getcwd()
IMAGE_DIR = "images"
GLYPH_DIR = "glyphs"
MODEL_DIR = "model"
LABELS_CSV = "glyph_labels.csv"
CLUSTER_JSON = "clusters.json"

for d in (IMAGE_DIR, GLYPH_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ===============================
# Utility functions
# ===============================
def confidence_to_color(c):
    if c >= 0.85: return "lime"
    if c >= 0.6: return "yellow"
    return "red"

# ===============================
# Image processing (NO OCR)
# ===============================
def preprocess_for_blobs(pil):
    im = pil.convert("L")
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.MedianFilter(3))
    arr = np.array(im)
    h, w = arr.shape
    out = np.zeros_like(arr)
    block = 32
    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = arr[y:y+block, x:x+block]
            if blk.size == 0: continue
            th = max(10, int(np.mean(blk)) - 12)
            out[y:y+block, x:x+block] = (blk > th) * 255
    return Image.fromarray(out.astype(np.uint8))

def connected_components_boxes(arr, min_area=40):
    h, w = arr.shape
    visited = np.zeros_like(arr, bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if visited[y,x] or arr[y,x] == 0: continue
            stack=[(x,y)]
            xs, ys = [], []
            visited[y,x]=True
            while stack:
                cx,cy=stack.pop()
                xs.append(cx); ys.append(cy)
                for nx,ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                    if 0<=nx<w and 0<=ny<h and not visited[ny,nx] and arr[ny,nx]:
                        visited[ny,nx]=True
                        stack.append((nx,ny))
            if xs:
                x1,x2,y1,y2=min(xs),max(xs),min(ys),max(ys)
                if (x2-x1)*(y2-y1)>=min_area:
                    boxes.append((x1,y1,x2,y2))
    return boxes

def infer_reading_order(boxes):
    xs=[b[0] for b in boxes]; ys=[b[1] for b in boxes]
    if np.std(xs)>np.std(ys):
        return "RTL"
    return "VERTICAL"

# ===============================
# ML Model
# ===============================
if TORCH_AVAILABLE:
    class GlyphNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.conv=nn.Sequential(
                nn.Conv2d(1,16,3,1,1),nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,1,1),nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc=nn.Linear(32*8*8,n)
        def forward(self,x):
            x=self.conv(x)
            x=x.view(x.size(0),-1)
            return self.fc(x)

# ===============================
# Dataset
# ===============================
if TORCH_AVAILABLE:
    class GlyphDataset(Dataset):
        def __init__(self,csv_path):
            df=pd.read_csv(csv_path)
            self.paths=df["path"].tolist()
            self.labels=df["label"].astype("category")
            self.map=dict(enumerate(self.labels.cat.categories))
            self.ids=self.labels.cat.codes.values
        def __len__(self): return len(self.paths)
        def __getitem__(self,i):
            im=Image.open(self.paths[i]).convert("L").resize((32,32))
            x=torch.tensor(np.array(im)/255.).float().unsqueeze(0)
            return x,self.ids[i]

# ===============================
# GUI Application
# ===============================
class ThamudicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thamudic AI Scanner")
        self.geometry("1200x750")

        self.images=[]
        self.words=[]
        self.glyph_index=[]
        self.model=None
        self.labels_map={}

        self._build_ui()

    # ---------------------------
    # UI
    # ---------------------------
    def _build_ui(self):
        top=ttk.Frame(self); top.pack(fill="x")
        ttk.Button(top,text="Load Images",command=self.load_images).pack(side="left")
        ttk.Button(top,text="Scan",command=self.scan_thread).pack(side="left")
        ttk.Button(top,text="Cluster",command=self.cluster_thread).pack(side="left")
        ttk.Button(top,text="Retrain Model",command=self.retrain_thread).pack(side="left")

        self.canvas=tk.Canvas(self,width=800,height=550,bg="black")
        self.canvas.pack(side="left")

        right=ttk.Frame(self); right.pack(side="left",fill="both",expand=True)
        self.tree=ttk.Treeview(right,columns=("img","glyphs"),show="headings")
        self.tree.heading("img",text="Image")
        self.tree.heading("glyphs",text="Glyphs")
        self.tree.pack(fill="both",expand=True)

        self.log=ScrolledText(right,height=8)
        self.log.pack(fill="x")

    def log_msg(self,msg):
        self.log.insert("end",msg+"\n")
        self.log.see("end")

    # ---------------------------
    # Actions
    # ---------------------------
    def load_images(self):
        files=filedialog.askopenfilenames(filetypes=[("Images","*.png *.jpg *.jpeg")])
        self.images=list(files)
        self.log_msg(f"Loaded {len(files)} images")

    def scan_thread(self):
        threading.Thread(target=self.scan_images,daemon=True).start()

    def scan_images(self):
        for path in self.images:
            pil=Image.open(path).convert("RGB")
            pre=preprocess_for_blobs(pil)
            arr=np.array(pre)
            boxes=connected_components_boxes(arr)
            order=infer_reading_order(boxes)
            boxes=sorted(boxes,key=lambda b:b[0],reverse=(order=="RTL"))
            for i,b in enumerate(boxes):
                gp=os.path.join(GLYPH_DIR,f"{os.path.basename(path)}_{i}.png")
                pil.crop(b).resize((32,32)).save(gp)
                self.glyph_index.append((gp,b))
            self.log_msg(f"Scanned {os.path.basename(path)}")

    def cluster_thread(self):
        threading.Thread(target=self.cluster_worker,daemon=True).start()

    def cluster_worker(self):
        paths=[p for p,_ in self.glyph_index]
        X=[]
        for p in paths:
            im=np.array(Image.open(p).convert("L").resize((32,32)))/255.
            X.append(im.flatten())
        X=np.array(X)
        pca=PCA(32).fit_transform(X)
        km=KMeans(n_clusters=min(32,len(X))).fit(pca)
        with open(CLUSTER_JSON,"w",encoding="utf-8") as f:
            json.dump(dict(zip(paths,km.labels_.tolist())),f,ensure_ascii=False)
        self.log_msg("Clustering complete")

    def retrain_thread(self):
        if not TORCH_AVAILABLE:
            self.log_msg("Torch not available")
            return
        threading.Thread(target=self.retrain_worker,daemon=True).start()

    def retrain_worker(self):
        ds=GlyphDataset(LABELS_CSV)
        self.model=GlyphNet(len(ds.map))
        opt=torch.optim.Adam(self.model.parameters(),1e-3)
        dl=DataLoader(ds,batch_size=32,shuffle=True)
        for _ in range(5):
            for x,y in dl:
                opt.zero_grad()
                loss=F.cross_entropy(self.model(x),y)
                loss.backward()
                opt.step()
        torch.save(self.model.state_dict(),os.path.join(MODEL_DIR,"model.pt"))
        self.log_msg("Model retrained")

# ===============================
# Main
# ===============================
if __name__=="__main__":
    app=ThamudicApp()
    app.mainloop()
