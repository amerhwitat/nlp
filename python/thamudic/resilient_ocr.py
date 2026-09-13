"""Resilient native OCR execution for Thamudic/NLP scanners."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT=max(5,int(os.environ.get("THAMUDIC_OCR_TIMEOUT","45")))
IMAGE_SUFFIXES={".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
MAX_IMAGE_DIMENSION=max(1000,int(os.environ.get("THAMUDIC_MAX_IMAGE_DIM","2400")))

def _env():
    env=os.environ.copy(); env.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
    for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): env.setdefault(n,"1")
    return env

def _creation_kwargs():
    return {"creationflags":getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)} if os.name=="nt" else {"start_new_session":True}

def terminate_worker(proc: subprocess.Popen):
    if proc.poll() is not None:return
    try:
        if os.name=="nt": subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5,check=False)
        else:
            proc.terminate()
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: proc.kill()
    except Exception:
        try: proc.kill()
        except Exception: pass
    try: proc.wait(timeout=3)
    except Exception: pass

def prepare_image(path: Path):
    try:
        from PIL import Image,ImageOps,ImageEnhance
        image=ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if max(image.size)>MAX_IMAGE_DIMENSION:
            scale=MAX_IMAGE_DIMENSION/float(max(image.size)); image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
        image=ImageEnhance.Contrast(image).enhance(1.12)
        fd,name=tempfile.mkstemp(prefix="thamudic-ocr-",suffix=".png"); os.close(fd); image.save(name,"PNG",optimize=True)
        return Path(name),True
    except Exception:return path,False

def run_worker(worker: Path,args:list[str],timeout:int=DEFAULT_TIMEOUT):
    proc=None
    try:
        proc=subprocess.Popen([sys.executable,str(worker),*args],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",errors="replace",env=_env(),**_creation_kwargs())
        try: stdout,stderr=proc.communicate(timeout=max(1,int(timeout)))
        except subprocess.TimeoutExpired:
            terminate_worker(proc); return {"ok":False,"error_type":"TimeoutError","error":f"Native worker timed out after {int(timeout)}s","timed_out":True,"recoverable":True,"provider":"easyocr-subprocess"}
        lines=(stdout or "").strip().splitlines(); payload=None
        if lines:
            try: payload=json.loads(lines[-1])
            except json.JSONDecodeError: pass
        if not payload:
            detail=(stderr or "").strip(); return {"ok":False,"error_type":"WorkerProtocolError","error":"Native worker exited without valid JSON"+(f": {detail[-1000:]}" if detail else ""),"recoverable":True,"provider":"easyocr-subprocess"}
        if not payload.get("ok"): payload["recoverable"]=True
        return payload
    except OSError as exc:return {"ok":False,"error_type":type(exc).__name__,"error":str(exc),"recoverable":True}
    finally:
        if proc is not None and proc.poll() is None: terminate_worker(proc)

def ocr_image(path,worker,languages=None,timeout=DEFAULT_TIMEOUT):
    source=Path(path).expanduser(); prepared,temp=prepare_image(source) if source.suffix.casefold() in IMAGE_SUFFIXES else (source,False)
    try:
        result=run_worker(Path(worker),[str(prepared),"--languages",",".join(languages or ["en","ar"])],timeout)
        result.setdefault("provider","easyocr-subprocess"); result["source_file"]=str(source); result.setdefault("ocr_available",bool(result.get("ok") and result.get("text"))); result.setdefault("ocr_timed_out",bool(result.get("timed_out"))); result.setdefault("ocr_recoverable",True)
        if result.get("timed_out"): result["ocr_error"]=result.get("error","Native OCR worker timed out")
        return result
    finally:
        if temp:
            try: prepared.unlink(missing_ok=True)
            except Exception: pass

def pdf_page(path,page,output,worker,timeout=120):
    return run_worker(Path(worker),["--render-pdf",str(path),"--page",str(page),"--output",str(output)],timeout)
