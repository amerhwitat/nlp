#!/usr/bin/env python3
"""Python-only web application for the Thamudic research platform.

This replaces the planned Next.js/TypeScript layer with Flask and a shared
SQLite ObjectDatabase. It is intentionally self-contained: HTML/CSS is
rendered from Python templates so the application logic remains Python-first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, render_template_string

from ancient_objects_db import ObjectDatabase
from ancient_script_registry import SCRIPTS
from historical_periods import PERIODS
from thamudic_scanner import scan_image


APP_CSS = """
:root { --bg:#10151c; --panel:#18212c; --panel2:#202c38; --text:#edf2f7; --muted:#9eacba; --accent:#d5a84b; --line:#304050; }
* { box-sizing:border-box; } body { margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--text); }
.app { display:grid; grid-template-columns:250px 1fr; min-height:100vh; }
.sidebar { background:#0c1117; border-right:1px solid var(--line); padding:22px 14px; position:sticky; top:0; height:100vh; }
.brand { font-size:20px; font-weight:800; padding:4px 10px 20px; } .brand span { color:var(--accent); }
.nav a { display:block; color:var(--muted); text-decoration:none; padding:11px 12px; border-radius:8px; margin:3px 0; }
.nav a:hover,.nav a.active { background:var(--panel2); color:var(--text); }
.main { padding:26px 30px; max-width:1500px; width:100%; margin:auto; }
.header { display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:24px; }
.header h1 { margin:0; font-size:28px; } .status { color:#86c995; font-size:13px; }
.grid { display:grid; grid-template-columns:repeat(4,minmax(160px,1fr)); gap:14px; margin-bottom:22px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; } .metric { font-size:30px; font-weight:800; margin-top:7px; } .label { color:var(--muted); font-size:13px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; margin-bottom:18px; }
.toolbar { display:flex; gap:10px; flex-wrap:wrap; } input,select,button,textarea { background:#0f171f; color:var(--text); border:1px solid var(--line); border-radius:7px; padding:10px; } button { cursor:pointer; background:var(--panel2); } button.primary { background:#8e6b25; border-color:#b88e35; }
table { width:100%; border-collapse:collapse; } th,td { padding:10px; border-bottom:1px solid var(--line); text-align:left; font-size:13px; } th { color:var(--muted); }
pre { white-space:pre-wrap; word-break:break-word; background:#0c1117; padding:14px; border-radius:8px; overflow:auto; }
@media(max-width:900px){ .app{grid-template-columns:1fr}.sidebar{height:auto;position:relative}.grid{grid-template-columns:repeat(2,1fr)} }
"""

LAYOUT = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }} · Thamudic Scanner</title><style>{{ css }}</style></head>
<body><div class="app"><aside class="sidebar"><div class="brand">THAMUDIC <span>SCANNER</span></div><nav class="nav">
<a href="/">Dashboard</a><a href="/scanner">Scanner</a><a href="/translator">Translator</a><a href="/artifacts">Historical Objects</a><a href="/inscriptions">Inscriptions</a><a href="/scripts">Ancient Scripts</a><a href="/sources">Sources & Rights</a><a href="/database">Database</a><a href="/research">Research</a></nav></aside><main class="main">{{ body|safe }}</main></div></body></html>
"""


def create_app(db_path: str | Path = "ancient_objects.sqlite") -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = str(db_path)

    def db() -> ObjectDatabase:
        return ObjectDatabase(app.config["DB_PATH"])

    def page(title: str, body: str, **context: Any):
        return render_template_string(LAYOUT, title=title, css=APP_CSS,
                                      body=render_template_string(body, **context))

    @app.get("/")
    def dashboard():
        with db() as database:
            stats = database.statistics()
            recent = database.list_objects(limit=8)
        body = """
        <div class="header"><div><h1>Thamudic Research Dashboard</h1><div class="status">SQLite evidence store · Python application</div></div></div>
        <section class="grid">{% for label,key in [('Objects','objects'),('Annotations','annotations'),('Sources','sources')] %}<div class="card"><div class="label">{{label}}</div><div class="metric">{{stats[key]}}</div></div>{% endfor %}<div class="card"><div class="label">Schema</div><div class="metric">v{{stats.schema_version}}</div></div></section>
        <section class="panel"><h2>Recent research records</h2><table><tr><th>ID</th><th>Title</th><th>Period</th><th>Script</th><th>Type</th></tr>{% for r in recent %}<tr><td>{{r.id}}</td><td>{{r.title}}</td><td>{{r.period_name or r.period_key or ''}}</td><td>{{r.script_key or ''}}</td><td>{{r.object_type or ''}}</td></tr>{% endfor %}</table></section>
        """
        return page("Dashboard", body, stats=stats, recent=recent)

    @app.get("/api/objects")
    def objects_api():
        with db() as database:
            rows = database.list_objects(query=request.args.get("q", ""),
                                         period_key=request.args.get("period_key") or None,
                                         script_key=request.args.get("script_key") or None,
                                         object_type=request.args.get("object_type") or None,
                                         country=request.args.get("country") or None,
                                         limit=int(request.args.get("limit", "200")))
        return jsonify({"objects": rows, "count": len(rows)})

    @app.get("/api/objects/<object_id>")
    def object_api(object_id: str):
        with db() as database:
            try:
                return jsonify(database.get_object(object_id))
            except KeyError:
                return jsonify({"error": "object not found"}), 404

    @app.get("/artifacts")
    def artifacts():
        q = request.args.get("q", "")
        with db() as database:
            rows = database.list_objects(query=q, object_type="artifact", limit=200)
            if not rows:
                rows = database.list_objects(query=q, limit=200)
        body = """
        <div class="header"><div><h1>Historical Objects & Artifacts</h1><div class="status">Searchable evidence catalog</div></div></div>
        <section class="panel"><form class="toolbar"><input name="q" value="{{q}}" placeholder="Search title, translation, tags…"><button class="primary">Search</button></form></section>
        <section class="panel"><table><tr><th>Record</th><th>Title</th><th>Period</th><th>Script</th><th>Location</th><th>Source</th></tr>{% for r in rows %}<tr><td>{{r.id}}</td><td>{{r.title}}</td><td>{{r.period_name or r.period_key or ''}}</td><td>{{r.script_key or ''}}</td><td>{{r.site or r.country or ''}}</td><td>{{r.source_name or ''}}</td></tr>{% endfor %}</table></section>
        """
        return page("Historical Objects", body, rows=rows, q=q)

    @app.get("/inscriptions")
    def inscriptions():
        with db() as database:
            rows = database.list_objects(object_type="inscription", limit=200)
        body = """<div class="header"><div><h1>Inscriptions</h1><div class="status">Human-reviewed readings and scanner evidence</div></div></div><section class="panel"><table><tr><th>Record</th><th>Title</th><th>Script</th><th>Transliteration</th><th>English</th><th>Confidence</th></tr>{% for r in rows %}<tr><td>{{r.id}}</td><td>{{r.title}}</td><td>{{r.script_key or ''}}</td><td>{{r.transliteration or ''}}</td><td>{{r.translation_en or ''}}</td><td>{{r.confidence or ''}}</td></tr>{% endfor %}</table></section>"""
        return page("Inscriptions", body, rows=rows)

    @app.get("/scripts")
    def scripts():
        body = """<div class="header"><div><h1>Ancient Scripts</h1><div class="status">Script registry</div></div></div><section class="grid">{% for key,s in scripts.items() %}<div class="card"><div class="label">{{key}}</div><div class="metric">{{s.name}}</div><p>{{s.description}}</p></div>{% endfor %}</section>"""
        return page("Ancient Scripts", body, scripts=SCRIPTS)

    @app.get("/sources")
    def sources():
        with db() as database:
            rows = [dict(r) for r in database.conn.execute("SELECT * FROM sources ORDER BY name COLLATE NOCASE")]
        body = """<div class="header"><div><h1>Sources & Rights</h1><div class="status">Provenance and media-rights tracking</div></div></div><section class="panel"><table><tr><th>Name</th><th>Homepage</th><th>Rights</th><th>Image policy</th></tr>{% for r in rows %}<tr><td>{{r.name or ''}}</td><td>{{r.homepage or ''}}</td><td>{{r.rights_policy or ''}}</td><td>{{r.image_policy or ''}}</td></tr>{% endfor %}</table></section>"""
        return page("Sources", body, rows=rows)

    @app.get("/database")
    def database_page():
        with db() as database:
            stats = database.statistics()
        body = """<div class="header"><div><h1>Database</h1><div class="status">Canonical SQLite persistence with JSON/CSV/SQL interchange</div></div></div><section class="grid">{% for label,key in [('Objects','objects'),('Annotations','annotations'),('Sources','sources')] %}<div class="card"><div class="label">{{label}}</div><div class="metric">{{stats[key]}}</div></div>{% endfor %}</section><section class="panel"><pre>{{stats|tojson(indent=2)}}</pre></section>"""
        return page("Database", body, stats=stats)

    @app.get("/research")
    def research():
        body = """<div class="header"><div><h1>Research Workspace</h1><div class="status">Evidence → interpretation → provenance</div></div></div><section class="panel"><h2>Research workflow</h2><ol><li>Import photograph or PDF.</li><li>Normalize and segment inscription glyphs.</li><li>Review candidate components and confidence.</li><li>Record transliteration and Arabic/English notes.</li><li>Attach sources, provenance and rights.</li><li>Persist the record in SQLite.</li><li>Export JSON/CSV/SQL for external systems.</li></ol></section>"""
        return page("Research", body)

    @app.route("/scanner", methods=["GET", "POST"])
    def scanner():
        result = None
        error = None
        if request.method == "POST":
            upload = request.files.get("image")
            if not upload or not upload.filename:
                error = "Choose an image file."
            else:
                import tempfile
                suffix = Path(upload.filename).suffix or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
                    upload.save(fh.name)
                    try:
                        result = scan_image(Path(fh.name))
                    except Exception as exc:
                        error = str(exc)
        body = """<div class="header"><div><h1>Thamudic Scanner</h1><div class="status">Python image normalization + connected-component evidence</div></div></div><section class="panel"><form method="post" enctype="multipart/form-data" class="toolbar"><input type="file" name="image" accept="image/*"><button class="primary">Scan image</button></form>{% if error %}<p>{{error}}</p>{% endif %}{% if result %}<h2>Scan result</h2><p>{{result.glyphs|length}} candidate components · {{result.recognition_status}}</p><pre>{{result|tojson(indent=2)}}</pre>{% endif %}</section>"""
        return page("Scanner", body, result=result, error=error)

    @app.get("/translator")
    def translator():
        body = """<div class="header"><div><h1>Thamudic Translator Workspace</h1><div class="status">Evidence-aware translation, not automatic scholarly certification</div></div></div><section class="panel"><form method="post" action="/api/translate"><textarea name="text" rows="8" style="width:100%" placeholder="Enter transliteration or Unicode text…"></textarea><br><button class="primary">Analyze</button></form></section>"""
        return page("Translator", body)

    @app.post("/api/translate")
    def translate_api():
        text = request.form.get("text", "") if request.form else ""
        return jsonify({"input": text, "status": "human_review_required", "note": "Translation is not asserted from segmentation alone."})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
