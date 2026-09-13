#!/usr/bin/env python3
"""Standalone all-in-one Ancient North Arabian / Thamudic scanner.

This file intentionally consolidates the runtime Python functionality used by the
modular ``python/thamudic`` package: Unicode inspection, Thamudic transliteration,
corpus-backed translation, ancient-script metadata, universal translation capability,
translation history/integrity, PDF reporting, optional voice, and a Tkinter GUI.

The modular package remains the canonical importable library. This file is designed
for users who want one copyable/runnable Python implementation. Data registries are
loaded from the repository when present and have a small built-in fallback for the
core Ancient North Arabian scripts.

No Tesseract or camel_tools dependency is required. Unknown ancient readings are
never fabricated: translation is reported as unavailable/provider-required.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys, tempfile, unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

# ---------------------------------------------------------------------------
# Paths / embedded fallback metadata
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "source_languages"

FIRST, LAST = 0x10A80, 0x10A9F
_NAMES = ["HEH","LAM","HAH","MEEM","QAF","WAW","ES-2","REH","BEH","TEH","ES-1","KAF","NOON","KHAH","SAD","ES-3","FEH","ALEF","AIN","DAD","GEEM","DAL","GHAIN","TAH","ZAIN","THAL","YEH","THEH","ZAH"]
_TRANS = ["h","l","ḥ","m","q","w","s2","r","b","t","s1","k","n","ḫ","ṣ","s3","f","ʼ","ʽ","ḍ","g","d","ġ","ṭ","z","ḏ","y","ṯ","ẓ"]
CHARACTERS = tuple({"codepoint":cp,"character":chr(cp),"name":f"OLD NORTH ARABIAN LETTER {n}","transliteration":t,"utf8":chr(cp).encode("utf-8"),"utf8_hex":chr(cp).encode("utf-8").hex(" ").upper()} for cp,n,t in zip(range(FIRST,0x10A9D),_NAMES,_TRANS)) + tuple({"codepoint":cp,"character":chr(cp),"name":f"OLD NORTH ARABIAN NUMBER {n}","transliteration":t,"utf8":chr(cp).encode("utf-8"),"utf8_hex":chr(cp).encode("utf-8").hex(" ").upper()} for cp,n,t in ((0x10A9D,"ONE","1"),(0x10A9E,"TEN","10"),(0x10A9F,"TWENTY","20")))
BY_CHARACTER = {x["character"]:x for x in CHARACTERS}
BY_CODEPOINT = {x["codepoint"]:x for x in CHARACTERS}
VARIANT_FORMS = ("Dadanitic","Safaitic","Hismaic","Taymanitic","Minaic","Thamudic B")

def is_old_north_arabian(value: int | str) -> bool:
    cp = ord(value) if isinstance(value,str) else value
    return FIRST <= cp <= LAST

def utf8_bytes(value: int | str) -> bytes:
    cp = ord(value) if isinstance(value,str) else value
    return chr(cp).encode("utf-8")

def transliterate(text: str) -> str:
    return "".join(BY_CHARACTER[c]["transliteration"] if c in BY_CHARACTER else c for c in text)

def extract(text: str) -> str:
    return "".join(c for c in text if is_old_north_arabian(c))

# ---------------------------------------------------------------------------
# Source-language scanner
# ---------------------------------------------------------------------------
_FALLBACK_RANGES = {
    "ancient-egyptian": [(0x13000,0x1342F)], "greek": [(0x0370,0x03FF),(0x1F00,0x1FFF)],
    "latin": [(0x0041,0x005A),(0x0061,0x007A),(0x1E00,0x1EFF)], "ancient-north-arabian": [(FIRST,LAST)],
    "old-south-arabian": [(0x10A60,0x10A7F)], "phoenician": [(0x10900,0x1091F)],
    "ancient-hebrew": [(0x0590,0x05FF)], "aramaic": [(0x10840,0x1085F)], "ugaritic": [(0x10380,0x1039F)],
    "old-persian": [(0x103A0,0x103DF)], "coptic": [(0x2C80,0x2CFF)], "old-turkic": [(0x10C00,0x10C4F)],
    "chinese": [(0x3400,0x4DBF),(0x4E00,0x9FFF)], "japanese": [(0x3040,0x30FF),(0x4E00,0x9FFF)],
}

def _load_json(name: str) -> Any | None:
    p = DATA / name
    try: return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None

def _profiles() -> dict[str,dict[str,Any]]:
    data = _load_json("ancient_classical_unicode.json")
    if data and "profiles" in data: return {x["id"]:x for x in data["profiles"]}
    return {k:{"id":k,"ranges":[f"U+{a:04X}-U+{b:04X}" for a,b in v]} for k,v in _FALLBACK_RANGES.items()}

def _parse_range(value: str) -> tuple[int,int]:
    a,b=value.removeprefix("U+").split("-U+"); return int(a,16),int(b,16)

def scan_source_language(text: str, language: str | None = None) -> dict[str,Any]:
    profiles=_profiles(); aliases={"egyptian":"ancient-egyptian","ancient-greek":"greek","classical-latin":"latin","hebrew":"ancient-hebrew","ona":"ancient-north-arabian"}
    key=aliases.get(language.casefold().replace(" ","-") if language else "",language.casefold().replace(" ","-") if language else None)
    candidates=(key,) if key else tuple(profiles)
    if key and key not in profiles: raise ValueError(f"unsupported source language: {language}")
    chars=[]; counts={k:0 for k in candidates}
    for i,ch in enumerate(text):
        cp=ord(ch); matches=[]
        for k in candidates:
            if any(a<=cp<=b for a,b in (_parse_range(r) for r in profiles[k].get("ranges",[]))): matches.append(k); counts[k]+=1
        if matches: chars.append({"index":i,"character":ch,"codepoint":f"U+{cp:04X}","decimal":cp,"name":unicodedata.name(ch,"UNNAMED"),"utf8":" ".join(f"{b:02X}" for b in ch.encode()),"utf8_bytes":list(ch.encode()),"normalized_nfc":unicodedata.normalize("NFC",ch),"matches":matches})
    ranked=sorted(counts.items(),key=lambda x:x[1],reverse=True)
    return {"text":text,"encoding":"UTF-8","unicode_normalization":"NFC","requested_language":language,"detected_languages":[k for k,c in ranked if c],"ambiguous_script_overlap":any(len(x["matches"])>1 for x in chars),"counts":counts,"characters":chars,"matched_character_count":len(chars),"supported_source_languages":list(profiles)}

def encode_utf8(text: str)->dict[str,Any]:
    raw=text.encode("utf-8"); return {"text":text,"encoding":"UTF-8","bytes":list(raw),"hex":" ".join(f"{b:02X}" for b in raw)}

# ---------------------------------------------------------------------------
# Evidence-backed Ancient North Arabian translation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusEntry:
    identifier:str; script:str; transliteration:str; translations:dict[str,str]; source_url:str; confidence:str="scholarly"

OCIANA_SEED=(
 CorpusEntry("TIJ 503","Safaitic","ytm bn ʿbny w wgm ʿl- ḫll -h",{"en":"Ytm son ʿbny and he grieved for his friend","ar":"يتم بن عبني وحزن على صديقه"},"https://ociana.osu.edu/inscriptions/2400"),
 CorpusEntry("AH 311","Dadanitic","bḏkrh wdd ḏ{h}k",{"en":"Bḏkrh loves {Ḏhk}","ar":"بذَكرَه يحب {ذهك}"},"https://ociana.osu.edu/inscriptions/13954"),
 CorpusEntry("Is.H 806","Thamudic B","l ḍtm h- s¹fr w h- frs¹",{"en":"By Ḍtm are the inscription and the horse","ar":"لِضَتم النقش والحصان"},"https://ociana.osu.edu/inscriptions/5826"),
 CorpusEntry("GETham 2","Thamudic B","l (l)hn wdd ns²l ḏ ʿtq",{"en":"By Ḏ son of . (Llhn) greets Ns²l who was freed","ar":"من Ḏ بن . (للهن) يحيّي نس²ل الذي أُعتق"},"https://ociana.osu.edu/inscriptions/44105"),
)
class TranslationProvider(Protocol):
    def lookup(self,transliteration:str,script:str)->CorpusEntry|None: ...
class InMemoryCorpus:
    def __init__(self,entries:Iterable[CorpusEntry]=OCIANA_SEED):
        self.entries=tuple(entries); self.index={(re.sub(r"\s+"," ",e.transliteration.casefold().strip()),e.script.casefold()):e for e in self.entries}
    def lookup(self,transliteration:str,script:str="")->CorpusEntry|None:
        key=(re.sub(r"\s+"," ",transliteration.casefold().strip()),script.casefold())
        if key in self.index:return self.index[key]
        candidates=[e for (t,_),e in self.index.items() if t==key[0]]
        return candidates[0] if len(candidates)==1 else None

def translate(text:str,script:str="Dadanitic",target_language:str="en",provider:TranslationProvider|None=None)->dict[str,Any]:
    target="ar" if target_language.casefold().startswith("ar") else "en"; tr=transliterate(text); entry=(provider or InMemoryCorpus()).lookup(tr,script)
    return {"source_text":text,"script":script,"transliteration":tr,"target_language":target,"translation":entry.translations.get(target) if entry else None,"translation_status":"corpus_match" if entry and target in entry.translations else "not_available","confidence":entry.confidence if entry else "unknown","corpus_id":entry.identifier if entry else None,"provenance":entry.source_url if entry else None}

def supported_targets(): return ("en","ar")

# ---------------------------------------------------------------------------
# Script registry and universal translation capability
# ---------------------------------------------------------------------------
_FALLBACK_META={
 "ancient-north-arabian":{"id":"ancient-north-arabian","name":"Ancient North Arabian","iso639":["xna"],"scripts":["Safaitic","Dadanitic","Hismaic","Taymanitic","Thamudic B/C/D"],"direction":"rtl","variations":["Safaitic","Dadanitic","Hismaic","Taymanitic","Thamudic B"],"translation_modes":["script-to-transliteration","transliteration-to-translation"],"dating":"First millennium BCE to late antiquity (corpus-dependent)","geographic_scope":"Arabian Peninsula and adjacent regions","script_type":"abjad"},
 "ancient-egyptian":{"id":"ancient-egyptian","name":"Ancient Egyptian","iso639":["egy"],"scripts":["Egyptian Hieroglyphs","Hieratic","Demotic"],"direction":"rtl-or-ltr","variations":["Old Egyptian","Middle Egyptian","Late Egyptian","Demotic"],"translation_modes":["hieroglyph-to-transliteration","transliteration-to-translation"],"script_type":"mixed"},
 "akkadian":{"id":"akkadian","name":"Akkadian","iso639":["akk"],"scripts":["Sumero-Akkadian Cuneiform"],"direction":"ltr","variations":["Old Babylonian","Neo-Babylonian","Old Assyrian","Neo-Assyrian"],"translation_modes":["cuneiform-to-transliteration","transliteration-to-translation"],"script_type":"logo-syllabic"},
 "sumerian":{"id":"sumerian","name":"Sumerian","iso639":["sux"],"scripts":["Sumero-Akkadian Cuneiform"],"direction":"ltr","variations":["Early Dynastic","Ur III","Old Babylonian literary"],"translation_modes":["cuneiform-to-transliteration","transliteration-to-translation"],"script_type":"logo-syllabic"},
 "phoenician":{"id":"phoenician","name":"Phoenician","iso639":["phn"],"scripts":["Phoenician"],"direction":"rtl","variations":["Phoenician","Punic","Neo-Punic"],"translation_modes":["script-to-transliteration","transliteration-to-translation"],"script_type":"abjad"},
 "ancient-hebrew":{"id":"ancient-hebrew","name":"Ancient Hebrew","iso639":["hbo"],"scripts":["Paleo-Hebrew","Hebrew"],"direction":"rtl","variations":["Paleo-Hebrew","Biblical Hebrew","Second Temple Hebrew"],"translation_modes":["script-to-transliteration","transliteration-to-translation"],"script_type":"abjad"},
 "aramaic":{"id":"aramaic","name":"Aramaic","iso639":["arc"],"scripts":["Imperial Aramaic","Nabataean","Syriac"],"direction":"rtl","variations":["Imperial","Biblical","Palmyrene","Nabataean","Syriac"],"translation_modes":["script-to-transliteration","transliteration-to-translation"],"script_type":"abjad"},
 "greek":{"id":"greek","name":"Greek","iso639":["grc"],"scripts":["Greek"],"direction":"ltr","variations":["Mycenaean","Archaic","Classical","Koine","Hellenistic"],"translation_modes":["classical-text-to-translation"],"script_type":"alphabet"},
 "latin":{"id":"latin","name":"Latin","iso639":["lat"],"scripts":["Latin"],"direction":"ltr","variations":["Old Latin","Classical","Late","Medieval","Epigraphic"],"translation_modes":["classical-text-to-translation"],"script_type":"alphabet"},
 "chinese":{"id":"chinese","name":"Classical Chinese","iso639":["lzh","zho"],"scripts":["Han"],"direction":"ltr","variations":["Oracle Bone","Bronze","Seal","Clerical","Literary Chinese"],"translation_modes":["han-to-reading","reading-to-translation"],"script_type":"logographic"},
 "japanese":{"id":"japanese","name":"Historical Japanese","iso639":["jpn"],"scripts":["Han","Kana"],"direction":"rtl-or-ltr","variations":["Man'yogana","Classical Japanese","Kyujitai"],"translation_modes":["han-to-reading","reading-to-translation"],"script_type":"mixed"},
}
def load_alphabet_registry():
    data=_load_json("ancient_language_alphabets.json"); meta=_load_json("ancient_script_metadata.json")
    if data and "languages" in data:
        extra=(meta or {}).get("languages",{}); out={}
        for x in data["languages"]:
            p=dict(x); p.update(extra.get(x["id"],{})); out[x["id"]]=p
        return out
    return dict(_FALLBACK_META)
def supported_alphabet_languages(): return tuple(load_alphabet_registry())
def language_profile(language:str):
    p=load_alphabet_registry(); key=language.casefold().replace(" ","-"); aliases={"egyptian":"ancient-egyptian","hebrew":"ancient-hebrew","ancient-greek":"greek","classical-latin":"latin","ona":"ancient-north-arabian"}; key=aliases.get(key,key)
    if key not in p: raise ValueError(f"unsupported alphabet language: {language}")
    return p[key]
def variations(language): return tuple(language_profile(language).get("variations",()))
def translation_capabilities(language): return tuple(language_profile(language).get("translation_modes",()))
def translation_directions(language):
    modes=set(translation_capabilities(language)); src=any(any(s in m for s in ("script-to-transliteration","cuneiform-to-transliteration","hieroglyph-to-transliteration","han-to-reading")) for m in modes); tr="transliteration-to-translation" in modes or "reading-to-translation" in modes
    return {"source_to_transliteration":src,"transliteration_to_translation":tr,"source_to_translation":any("translation" in m and "to-translation" in m for m in modes) or (src and tr),"translation_to_source_retrieval":any("translation-to-" in m for m in modes)}
@dataclass(frozen=True)
class TranslationResult:
    source:str; source_language:str; source_form:str; target_language:str; target_form:str|None; transliteration:str|None; translation:str|None; status:str; confidence:float; provider:str; provenance:str|None=None
    def as_dict(self): return asdict(self)
def translate_ancient(source,source_language,target_language,source_form="script",provider=None,request_metadata=None,log=True):
    if not source.strip(): raise ValueError("source is required")
    p=language_profile(source_language); directions=translation_directions(source_language); key={"script":"source_to_translation","transliteration":"transliteration_to_translation","translation":"translation_to_source_retrieval"}.get(source_form.casefold())
    if key is None: raise ValueError("source_form must be script, transliteration, or translation")
    result=provider.translate(source,p["id"],source_form,target_language.casefold()) if provider else None
    if result is None: result=TranslationResult(source,p["id"],source_form,target_language.casefold(),None,source if source_form!="translation" else None,None,"provider_required" if directions.get(key) else "direction_not_registered",0.0,"none")
    if log: append_record(make_record(source=result.source,source_language=result.source_language,source_form=result.source_form,target_language=result.target_language,target_form=result.target_form,transliteration=result.transliteration,translation=result.translation,status=result.status,confidence=result.confidence,provider=result.provider,provenance=result.provenance,script_metadata=build_script_summary(result.source_language),request_metadata=request_metadata or {}))
    return result

def translation_matrix(): return {x:translation_directions(x) for x in supported_alphabet_languages()}

# ---------------------------------------------------------------------------
# Translation history / integrity
# ---------------------------------------------------------------------------
@dataclass
class TranslationLogRecord:
    timestamp:str; source:str; source_language:str; source_form:str; target_language:str; target_form:str|None; transliteration:str|None; translation:str|None; status:str; confidence:float; provider:str; provenance:str|None=None; script_metadata:dict[str,Any]=field(default_factory=dict); request_metadata:dict[str,Any]=field(default_factory=dict); record_hash:str=""
    def as_dict(self):
        d=asdict(self); d["record_hash"]=compute_record_hash(d); return d
def compute_record_hash(data):
    d=dict(data); d.pop("record_hash",None); return hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def make_record(**kw):
    r=TranslationLogRecord(timestamp=datetime.now(timezone.utc).isoformat(),**kw); r.record_hash=compute_record_hash(asdict(r)); return r
def default_log_path(): return Path(os.getenv("THAMUDIC_TRANSLATION_LOG","translation_logs/translations.jsonl"))
def append_record(record,path=None):
    p=Path(path) if path else default_log_path(); p.parent.mkdir(parents=True,exist_ok=True); p.open("a",encoding="utf-8").write(json.dumps(record.as_dict(),ensure_ascii=False,sort_keys=True)+"\n"); return p
def read_records(path=None):
    p=Path(path) if path else default_log_path()
    if not p.exists(): return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
def verify_records(records):
    invalid=[]
    for r in records:
        if r.get("record_hash")!=compute_record_hash(r): invalid.append(str(r.get("record_hash","")))
    return {"total":len(records),"valid":len(records)-len(invalid),"invalid":len(invalid),"invalid_hashes":invalid}

# ---------------------------------------------------------------------------
# Script summaries / PDF exports
# ---------------------------------------------------------------------------
def build_script_summary(language):
    p=language_profile(language); return {"language_id":p["id"],"name":p.get("name",p["id"]),"iso639":p.get("iso639",[]),"original_script":p.get("scripts",p.get("script",[])),"script_family":p.get("script_family"),"script_type":p.get("script_type"),"writing_direction":p.get("direction"),"writing_direction_description":p.get("writing_direction_description"),"unicode_blocks":p.get("unicode_blocks",[]),"variations":list(variations(language)),"dating":p.get("dating"),"dating_status":p.get("dating_status"),"geographic_scope":p.get("geographic_scope"),"materials":p.get("materials",[]),"related_scripts":p.get("related_scripts",[]),"translation_modes":list(translation_capabilities(language)),"translation_directions":translation_directions(language),"transliteration_systems":p.get("transliteration_systems",[]),"notes":p.get("notes"),"evidence_policy":"Separate attested original text, scholarly transliteration, corpus-backed translation, model output, and reconstructed/uncertain readings."}
def build_script_report(language,original_text,target_language="en"):
    if not original_text.strip(): raise ValueError("original_text is required")
    r=translate_ancient(original_text,language,target_language,log=True).as_dict(); return {"report_version":"1.1","script_information":build_script_summary(language),"original_text":original_text,"source_language":language,"target_language":target_language,"transliteration":r.get("transliteration"),"translation":r.get("translation"),"translation_status":r.get("status"),"confidence":r.get("confidence"),"provider":r.get("provider"),"provenance":r.get("provenance"),"translation_result":r}
def _pdf_bytes(title,records=None,report=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as e: raise RuntimeError("PDF export requires ReportLab") from e
    font="Helvetica"
    for candidate in (os.getenv("THAMUDIC_PDF_FONT",""),"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","C:/Windows/Fonts/arial.ttf","/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if candidate and Path(candidate).is_file():
            try: pdfmetrics.registerFont(TTFont("ThamudicUnicode",candidate)); font="ThamudicUnicode"; break
            except Exception: pass
    fd,path=tempfile.mkstemp(suffix=".pdf"); os.close(fd); doc=SimpleDocTemplate(path,pagesize=A4,rightMargin=40,leftMargin=40,topMargin=44,bottomMargin=38,title=title,author="amerhwitat/nlp"); styles=getSampleStyleSheet(); styles["Title"].fontName=font; styles["BodyText"].fontName=font; small=ParagraphStyle("small",parent=styles["BodyText"],fontName=font,fontSize=8,leading=10); story=[Paragraph(title,styles["Title"])]
    def add(k,v): story.append(Paragraph(f"<b>{str(k).replace('_',' ').title()}:</b> {str(v).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace(chr(10),'<br/>')}",small))
    if report:
        for k,v in report.items(): add(k,json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2) if isinstance(v,(dict,list)) else v)
    else:
        rows=list(records or []); story.append(Paragraph(f"Records: {len(rows)}",small))
        for i,r in enumerate(rows,1):
            story.append(Paragraph(f"Record {i}",styles["Heading2"])); data=[[Paragraph("Field",small),Paragraph("Value",small)]]
            for k,v in r.items(): data.append([Paragraph(str(k),small),Paragraph(str(v).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace(chr(10),'<br/>'),small)])
            t=Table(data,colWidths=[125,390],repeatRows=1); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.25,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP") ])); story += [t,Spacer(1,12)]
    doc.build(story); raw=Path(path).read_bytes(); Path(path).unlink(missing_ok=True); return raw
def export_history_pdf(path=None,output="translation-history.pdf"):
    records=read_records(path); out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(_pdf_bytes("Ancient Language Translation History",records=records)); return out

def export_report_pdf(report,output):
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(_pdf_bytes("Ancient Script Report",report=report)); return out

# ---------------------------------------------------------------------------
# Voice capability
# ---------------------------------------------------------------------------
def available_tts_backends():
    out=["browser-speech-synthesis"]
    try: import pyttsx3; out.append("pyttsx3")
    except ImportError: pass
    return out
def voice_speak(text,language="en",mode="translation"):
    if not text.strip(): raise ValueError("text is required")
    if mode=="original": return {"status":"pronunciation_provider_required","native_ancient_tts":False,"backends":available_tts_backends()}
    return {"status":"ready","language":language,"mode":mode,"backends":available_tts_backends()}

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ThamudicScannerApp:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk=tk; self.ttk=ttk; self.root=tk.Tk(); self.root.title("Thamudic / Ancient Script Scanner — All-in-One"); self.root.geometry("1180x900"); self._engine=None; self._build()
    def _build(self):
        tk,ttk=self.tk,self.ttk; f=ttk.Frame(self.root,padding=10); f.pack(fill="both",expand=True)
        ttk.Label(f,text="Original script / inscription / scholarly transliteration").pack(anchor="w"); self.source=tk.Text(f,height=7,wrap="word"); self.source.pack(fill="x")
        c=ttk.Frame(f); c.pack(fill="x",pady=8); ttk.Label(c,text="Script").pack(side="left"); self.script=ttk.Combobox(c,values=list(supported_alphabet_languages()),width=25,state="readonly"); self.script.set("ancient-north-arabian"); self.script.pack(side="left",padx=5); ttk.Label(c,text="Target").pack(side="left",padx=5); self.target=ttk.Combobox(c,values=list(supported_targets()),width=8,state="readonly"); self.target.set("en"); self.target.pack(side="left"); ttk.Button(c,text="Scan",command=self.scan).pack(side="left",padx=5); ttk.Button(c,text="Translate",command=self.translate).pack(side="left",padx=5); ttk.Button(c,text="Script metadata",command=self.metadata).pack(side="left",padx=5); ttk.Button(c,text="History PDF",command=self.history_pdf).pack(side="left",padx=5); ttk.Button(c,text="Print history",command=self.print_history).pack(side="left",padx=5)
        panes=ttk.Frame(f); panes.pack(fill="both",expand=True); ttk.Label(panes,text="Transliteration / scan").pack(anchor="w"); self.trans=tk.Text(panes,height=6); self.trans.pack(fill="x"); ttk.Label(panes,text="Translation / report").pack(anchor="w",pady=(8,0)); self.out=tk.Text(panes,wrap="word"); self.out.pack(fill="both",expand=True); self.status=ttk.Label(f,text="Ready"); self.status.pack(anchor="w",pady=5)
    def _set(self,w,text): w.delete("1.0","end"); w.insert("1.0",text)
    def scan(self):
        text=self.source.get("1.0","end-1c"); r=scan_source_language(text,"ancient-north-arabian") if self.script.get()=="ancient-north-arabian" else scan_source_language(text,self.script.get()); self._set(self.trans,json.dumps(r,ensure_ascii=False,indent=2)); self.status.config(text=f"Matched characters: {r['matched_character_count']}")
    def translate(self):
        text=self.source.get("1.0","end-1c").strip()
        if not text: self.status.config(text="Enter source text first"); return
        r=translate(text,self.script.get(),self.target.get()); self._set(self.trans,r["transliteration"]); self._set(self.out,json.dumps(r,ensure_ascii=False,indent=2)); self.status.config(text=f"{r['translation_status']} · confidence={r['confidence']}")
    def metadata(self): self._set(self.out,json.dumps(build_script_summary(self.script.get()),ensure_ascii=False,indent=2))
    def history_pdf(self):
        from tkinter import filedialog, messagebox
        out=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")],initialfile="translation-history.pdf")
        if out:
            try: export_history_pdf(output=out); self.status.config(text=f"PDF written: {out}")
            except Exception as e: messagebox.showerror("PDF export",str(e))
    def print_history(self):
        from tkinter import filedialog, messagebox
        out=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")],initialfile="translation-history-print.pdf")
        if not out:return
        try:
            p=export_history_pdf(output=out)
            if sys.platform.startswith("win"): os.startfile(str(p),"print")
            elif sys.platform=="darwin": os.system(f'lpr "{p}"')
            else: os.system(f'lpr "{p}"')
            self.status.config(text=f"Print submitted: {p}")
        except Exception as e: messagebox.showerror("Print",str(e))
    def run(self): self.root.mainloop()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap=argparse.ArgumentParser(description="All-in-one Thamudic / Ancient Script scanner")
    ap.add_argument("--gui",action="store_true",help="start Tkinter GUI")
    ap.add_argument("--text",default="",help="text to scan/translate")
    ap.add_argument("--script",default="ancient-north-arabian")
    ap.add_argument("--target",default="en",choices=["en","ar"])
    ap.add_argument("--scan",action="store_true")
    ap.add_argument("--translate",action="store_true")
    ap.add_argument("--history-pdf",metavar="PATH")
    ap.add_argument("--verify-history",action="store_true")
    args=ap.parse_args(argv)
    if args.gui or not any((args.text,args.history_pdf,args.verify_history)): return ThamudicScannerApp().run()
    if args.history_pdf: print(export_history_pdf(output=args.history_pdf)); return 0
    if args.verify_history: print(json.dumps(verify_records(read_records()),ensure_ascii=False,indent=2)); return 0
    if args.scan: print(json.dumps(scan_source_language(args.text,args.script),ensure_ascii=False,indent=2))
    if args.translate: print(json.dumps(translate(args.text,args.script,args.target),ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
