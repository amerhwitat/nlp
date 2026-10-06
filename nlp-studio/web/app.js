/* NLP / Ancient Script Studio — single-page app.
 *
 * Runs in two modes:
 *   - backend mode: talks to the FastAPI service under /api/*.
 *   - static mode:  fully client-side (Tesseract.js OCR + bundled JSON
 *                    lookup tables generated from the Python registry).
 * The mode is auto-detected from /api/health.
 */
(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const ONA_VARIANTS = ['Dadanitic', 'Safaitic', 'Hismaic', 'Taymanitic', 'Minaic', 'Thamudic B'];
  const STEP_ORDER = ['upload', 'ocr', 'scan', 'transliterate', 'translate'];

  const state = {
    mode: 'unknown',
    file: null,
    activeTab: 'upload',
    static: null,
    result: null,
    stepTimer: null,
  };

  const els = {
    modeBadge: $('#modeBadge'),
    fileInput: $('#fileInput'),
    cameraInput: $('#cameraInput'),
    cameraBtn: $('#cameraBtn'),
    dropzone: $('#dropzone'),
    pickedFile: $('#pickedFile'),
    cameraPicked: $('#cameraPicked'),
    cameraPreview: $('#cameraPreview'),
    textInput: $('#textInput'),
    scriptSelect: $('#scriptSelect'),
    languageSelect: $('#languageSelect'),
    targetSelect: $('#targetSelect'),
    runBtn: $('#runBtn'),
    progressCard: $('#progressCard'),
    progressLabel: $('#progressLabel'),
    progressPct: $('#progressPct'),
    progressFill: $('#progressFill'),
    steps: $('#steps'),
    resultsCard: $('#resultsCard'),
    exportButtons: $('#exportButtons'),
    originalText: $('#originalText'),
    transliterationText: $('#transliterationText'),
    translationText: $('#translationText'),
    translationBadges: $('#translationBadges'),
    metadata: $('#metadata'),
    historyCard: $('#historyCard'),
    historyList: $('#historyList'),
    historyRefresh: $('#historyRefresh'),
  };

  /* ---------- utilities ---------- */

  function toast(message) {
    let node = $('.toast');
    if (!node) {
      node = document.createElement('div');
      node.className = 'toast';
      document.body.appendChild(node);
    }
    node.textContent = message;
    node.classList.add('is-visible');
    clearTimeout(node._t);
    node._t = setTimeout(() => node.classList.remove('is-visible'), 2600);
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function setBadge(text, cls) {
    els.modeBadge.textContent = text;
    els.modeBadge.className = 'mode-badge ' + (cls || '');
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) return resolve();
      const s = document.createElement('script');
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('failed to load ' + src));
      document.head.appendChild(s);
    });
  }

  function downloadBlob(content, mime, filename) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function downloadUrl(url, filename) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('export failed');
      const blob = await res.blob();
      downloadBlob(blob, blob.type || 'application/octet-stream', filename);
    } catch (err) {
      toast(err.message || 'Export failed');
    }
  }

  /* ---------- static data ---------- */

  async function detectMode() {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 2500);
      const res = await fetch('/api/health', { signal: ctrl.signal });
      clearTimeout(timer);
      if (res.ok) {
        state.mode = 'backend';
        setBadge('Backend connected', 'is-online');
        return;
      }
    } catch (_) { /* fall through to static mode */ }
    state.mode = 'static';
    setBadge('Offline · static mode', 'is-offline');
    await loadStaticData();
  }

  async function loadStaticData() {
    const [ona, registry, corpus, profiles] = await Promise.all([
      fetch('./data/ona_chars.json').then((r) => r.json()),
      fetch('./data/alphabet_registry.json').then((r) => r.json()),
      fetch('./data/translation_corpus.json').then((r) => r.json()),
      fetch('./data/source_language_profiles.json').then((r) => r.json()),
    ]);
    state.static = { ona, registry, corpus, profiles };
  }

  /* ---------- controls ---------- */

  function fillSelect(select, options, selected) {
    select.innerHTML = '';
    for (const opt of options) {
      const el = document.createElement('option');
      el.value = opt.value;
      el.textContent = opt.label;
      if (opt.value === selected) el.selected = true;
      select.appendChild(el);
    }
  }

  async function populateControls() {
    let variants = ONA_VARIANTS;
    let languages = [{ id: 'ancient-north-arabian', name: 'Ancient North Arabian' }];

    if (state.mode === 'backend') {
      try {
        const data = await fetch('/api/languages').then((r) => r.json());
        if (data.ona_scripts && data.ona_scripts.length) variants = data.ona_scripts;
        if (data.source_languages && data.source_languages.length) {
          languages = data.source_languages.map((l) => ({ id: l.id, name: l.name }));
        }
      } catch (_) { /* keep defaults */ }
    } else if (state.static) {
      if (state.static.ona.variant_forms.length) variants = state.static.ona.variant_forms;
      languages = Object.entries(state.static.registry).map(([id, p]) => ({ id, name: p.name || id }));
    }

    fillSelect(els.scriptSelect, variants.map((v) => ({ value: v, label: v })), 'Dadanitic');
    fillSelect(els.languageSelect, languages.map((l) => ({ value: l.id, label: l.name })), 'ancient-north-arabian');
  }

  /* ---------- tabs ---------- */

  function switchTab(name) {
    state.activeTab = name;
    $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.tab === name));
    $$('.tab-panel').forEach((p) => p.classList.toggle('is-active', p.dataset.panel === name));
  }

  /* ---------- file handling ---------- */

  function acceptFile(file) {
    const ok = /\.(png|jpe?g|webp|bmp|tiff?|pdf)$/i.test(file.name) || /image\//.test(file.type);
    if (!ok) { toast('Unsupported file type. Use PNG, JPG, WEBP, BMP, TIFF, or PDF.'); return false; }
    if (file.size > 10 * 1024 * 1024) { toast('File too large (max 10 MB).'); return false; }
    state.file = file;
    els.pickedFile.hidden = false;
    els.pickedFile.innerHTML = `<span class="name">${escapeHtml(file.name)}</span><span>${(file.size / 1024).toFixed(1)} KB</span>`;
    return true;
  }

  /* ---------- static OCR / scan / translate ---------- */

  function staticTransliterate(text) {
    const byChar = state.static.ona.by_character;
    return Array.from(text).map((ch) => (byChar[ch] !== undefined ? byChar[ch] : ch)).join('');
  }

  function staticScan(text) {
    const profiles = state.static.profiles;
    const ids = Object.keys(profiles);
    const ranges = {};
    for (const id of ids) {
      ranges[id] = (profiles[id].ranges || []).map((r) => {
        const parts = r.replace(/^U\+/, '').split('-U+');
        return { start: parseInt(parts[0], 16), end: parseInt(parts[1], 16) };
      });
    }
    const counts = {}; ids.forEach((id) => { counts[id] = 0; });
    const onaInfo = {};
    (state.static.ona.characters || []).forEach((c) => { onaInfo[c.character] = c; });
    const characters = [];
    Array.from(text).forEach((ch, index) => {
      const cp = ch.codePointAt(0);
      const matches = ids.filter((id) => ranges[id].some((r) => r.start <= cp && cp <= r.end));
      if (matches.length) {
        matches.forEach((id) => { counts[id] += 1; });
        const info = onaInfo[ch] || {};
        characters.push({
          index, character: ch,
          codepoint: 'U+' + cp.toString(16).toUpperCase().padStart(4, '0'),
          name: info.name || '', transliteration: info.transliteration || staticTransliterate(ch), matches,
        });
      }
    });
    const detected = ids.filter((id) => counts[id] > 0);
    return { text, encoding: 'UTF-8', detected_languages: detected, counts, characters, matched_character_count: characters.length };
  }

  function latinize(value) {
    return String(value).replace(/[–—]/g, '-').toLowerCase().replace(/\s+/g, ' ').trim();
  }

  function staticTranslate(text, script, target) {
    const transliteration = staticTransliterate(text);
    const key = latinize(transliteration);
    const corpus = state.static.corpus;
    let entry = corpus.find((e) => latinize(e.transliteration) === key && e.script.toLowerCase() === script.toLowerCase());
    if (!entry) {
      const candidates = corpus.filter((e) => latinize(e.transliteration) === key);
      if (candidates.length === 1) entry = candidates[0];
    }
    const targetLang = target.startsWith('ar') ? 'ar' : 'en';
    const translation = entry ? (entry.translations[targetLang] || null) : null;
    return {
      source_text: text, script, transliteration, target_language: targetLang,
      translation, translation_status: translation ? 'corpus_match' : 'not_available',
      confidence: entry ? entry.confidence : 'unknown',
      corpus_id: entry ? entry.identifier : null,
      provenance: entry ? entry.source_url : null,
      provider: entry ? 'corpus' : 'none',
      source_language: els.languageSelect.value,
    };
  }

  function staticSummary(languageId) {
    const p = state.static.registry[languageId] || state.static.registry['ancient-north-arabian'] || {};
    return {
      language_id: languageId, name: p.name || languageId, iso639: p.iso639 || [],
      original_script: p.scripts || p.script || [], script_family: p.script_family,
      script_type: p.script_type, writing_direction: p.direction,
      writing_direction_description: p.writing_direction_description,
      unicode_blocks: p.unicode_blocks || [], variations: p.variations || [],
      dating: p.dating, dating_status: p.dating_status, geographic_scope: p.region || p.geographic_scope,
      materials: p.materials || [], related_scripts: p.related_scripts || [],
      transliteration_systems: p.transliteration_systems || [], notes: p.notes,
    };
  }

  function staticOCR(file, onProgress) {
    return new Promise((resolve, reject) => {
      if (typeof Tesseract === 'undefined') { reject(new Error('Tesseract.js failed to load (no backend and no CDN access).')); return; }
      Tesseract.recognize(file, 'eng', {
        logger: (m) => {
          if (m.status === 'recognizing text' && typeof m.progress === 'number') onProgress(m.progress);
        },
      }).then((result) => {
        const text = (result.data.text || '').trim();
        resolve({ ok: !!text, text, engine: 'tesseract.js', confidence: (result.data.confidence || 0) / 100 });
      }).catch(reject);
    });
  }

  /* ---------- progress ---------- */

  function resetProgress() {
    if (state.stepTimer) clearInterval(state.stepTimer);
    els.progressFill.style.width = '0%';
    els.progressPct.textContent = '0%';
    els.progressLabel.textContent = 'Starting…';
    $$('#steps li').forEach((li) => li.className = '');
  }

  function setStep(name, status) {
    const li = document.querySelector(`#steps li[data-step="${name}"]`);
    if (li) li.className = status; // '', 'is-active', 'is-done'
  }

  function setProgress(pct, label) {
    const p = Math.max(0, Math.min(100, Math.round(pct)));
    els.progressFill.style.width = p + '%';
    els.progressPct.textContent = p + '%';
    if (label) els.progressLabel.textContent = label;
  }

  function startStepAnimation() {
    let i = 0;
    setStep(STEP_ORDER[0], 'is-active');
    state.stepTimer = setInterval(() => {
      if (i >= STEP_ORDER.length - 1) { clearInterval(state.stepTimer); return; }
      setStep(STEP_ORDER[i], 'is-done');
      i += 1;
      setStep(STEP_ORDER[i], 'is-active');
    }, 700);
  }

  function finishSteps() {
    if (state.stepTimer) clearInterval(state.stepTimer);
    STEP_ORDER.forEach((s) => setStep(s, 'is-done'));
    setProgress(100, 'Complete');
  }

  /* ---------- pipeline ---------- */

  async function backendTranslate(text) {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        script: els.scriptSelect.value,
        source_language: els.languageSelect.value,
        target_language: els.targetSelect.value,
      }),
    });
    if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
    return res.json();
  }

  async function backendUpload(file) {
    const fd = new FormData();
    fd.append('file', file, file.name);
    fd.append('script', els.scriptSelect.value);
    fd.append('source_language', els.languageSelect.value);
    fd.append('target_language', els.targetSelect.value);
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(((await res.json()).detail) || res.statusText);
    return res.json();
  }

  async function staticPipeline() {
    let text = els.textInput.value.trim();
    let ocr = null;
    if (state.activeTab !== 'text') {
      if (!state.file) throw new Error('Choose an image or PDF first.');
      setProgress(0, 'OCR (Tesseract.js)…');
      setStep('upload', 'is-done'); setStep('ocr', 'is-active');
      ocr = await staticOCR(state.file, (p) => setProgress(p * 100, 'OCR (Tesseract.js)…'));
      text = ocr.text;
      if (!text) throw new Error('OCR found no text in this image.');
    } else {
      setStep('upload', 'is-done'); setStep('ocr', 'is-done');
    }
    setStep('scan', 'is-active'); setStep('ocr', 'is-done');
    const scan = staticScan(text);
    setStep('scan', 'is-done'); setStep('transliterate', 'is-active');
    const translation = staticTranslate(text, els.scriptSelect.value, els.targetSelect.value);
    setStep('transliterate', 'is-done'); setStep('translate', 'is-active');
    const summary = staticSummary(els.languageSelect.value);
    return {
      title: 'NLP / Ancient Script Studio',
      result_id: null,
      script: els.scriptSelect.value,
      source_language: els.languageSelect.value,
      target_language: els.targetSelect.value,
      upload: ocr ? { original_name: state.file.name, size_bytes: state.file.size } : null,
      ocr, scan, translation, script_information: summary,
      workflow: 'upload -> OCR -> scan -> transliterate -> translate',
    };
  }

  async function run() {
    if (state.activeTab === 'text' && !els.textInput.value.trim()) { toast('Type or paste some inscription text.'); return; }
    els.resultsCard.hidden = true;
    resetProgress();
    els.progressCard.hidden = false;
    startStepAnimation();
    try {
      let payload;
      if (state.mode === 'backend') {
        if (state.activeTab === 'text') {
          setProgress(10, 'Translating…');
          payload = await backendTranslate(els.textInput.value.trim());
        } else {
          if (!state.file) throw new Error('Choose an image or PDF first.');
          setProgress(10, 'Uploading & OCR…');
          payload = await backendUpload(state.file);
        }
      } else {
        payload = await staticPipeline();
      }
      finishSteps();
      state.result = payload;
      renderResults(payload);
      els.progressCard.hidden = true;
      els.resultsCard.hidden = false;
      els.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      if (state.mode === 'backend') loadHistory();
    } catch (err) {
      els.progressCard.hidden = true;
      toast(err.message || 'Something went wrong.');
    }
  }

  /* ---------- results ---------- */

  function badge(label, cls) {
    return `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
  }

  function renderResults(payload) {
    const translation = payload.translation || {};
    const original = (payload.scan && payload.scan.text) || translation.source_text || (payload.ocr && payload.ocr.text) || '';
    const transliteration = translation.transliteration || '';
    const translated = translation.translation || '';

    els.originalText.textContent = original || '—';
    els.transliterationText.textContent = transliteration || '—';
    els.translationText.textContent = translated || 'No translation available (no corpus match).';

    const badges = [];
    if (translation.confidence) badges.push(badge('confidence: ' + translation.confidence, 'confidence'));
    badges.push(badge('provider: ' + (translation.provider || payload.ocr?.engine || 'none'), 'provider'));
    if (translation.translation_status) badges.push(badge('status: ' + translation.translation_status, translation.translation_status === 'not_available' ? 'warn' : ''));
    if (translation.corpus_id) badges.push(badge('corpus: ' + translation.corpus_id, ''));
    if (translation.provenance) badges.push(badge(`<a href="${escapeHtml(translation.provenance)}" target="_blank" rel="noopener">provenance ↗</a>`, ''));
    els.translationBadges.innerHTML = badges.join('');

    renderMetadata(payload);
    renderExportButtons(payload);
  }

  function renderMetadata(payload) {
    const info = payload.script_information || {};
    const scan = payload.scan || {};
    const items = [
      ['Script', info.name || payload.source_language],
      ['Writing direction', info.writing_direction + (info.writing_direction_description ? ' — ' + info.writing_direction_description : '')],
      ['Script family', info.script_family || '—'],
      ['Script type', info.script_type || '—'],
      ['Original script(s)', (info.original_script || []).join(', ') || '—'],
      ['Unicode blocks', (info.unicode_blocks || []).join(', ') || '—'],
      ['Dating', info.dating || '—'],
      ['Dating status', info.dating_status || '—'],
      ['Region', info.geographic_scope || '—'],
      ['Materials', (info.materials || []).join(', ') || '—'],
      ['Related scripts', (info.related_scripts || []).join(', ') || '—'],
      ['Transliteration systems', (info.transliteration_systems || []).join(', ') || '—'],
      ['Detected languages', (scan.detected_languages || []).join(', ') || 'none'],
      ['Matched characters', String(scan.matched_character_count ?? 0)],
    ];
    const grid = items.map(([k, v]) => `<div class="meta-item"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div></div>`).join('');

    let table = '';
    const chars = (scan.characters || []).slice(0, 60);
    if (chars.length) {
      table = `<table class="char-table"><thead><tr><th>Character</th><th>Codepoint</th><th>Transliteration</th><th>Name</th></tr></thead><tbody>` +
        chars.map((c) => `<tr><td class="glyph">${escapeHtml(c.character)}</td><td>${escapeHtml(c.codepoint)}</td><td>${escapeHtml(c.transliteration || '')}</td><td>${escapeHtml(c.name || '')}</td></tr>`).join('') +
        `</tbody></table>`;
    }
    els.metadata.innerHTML = `<div class="meta-grid">${grid}</div>${table}`;
  }

  function renderExportButtons(payload) {
    const formats = [
      ['json', 'JSON'], ['md', 'Markdown'], ['txt', 'TXT'], ['pdf', 'PDF'],
    ];
    els.exportButtons.innerHTML = formats.map(([fmt, label]) =>
      `<button class="btn btn-secondary btn-sm" data-export="${fmt}">${label} ⤓</button>`).join('');
    $$('#exportButtons [data-export]').forEach((btn) => {
      btn.addEventListener('click', () => exportResult(btn.dataset.export, payload));
    });
  }

  function flatten(value) {
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function txtify(payload) {
    const title = payload.title || 'NLP / Ancient Script Studio';
    const lines = [`${title} — report`, ''];
    for (const [k, v] of Object.entries(payload)) lines.push(`${k}: ${flatten(v)}`);
    return lines.join('\n') + '\n';
  }

  function mdify(payload) {
    const title = payload.title || 'NLP / Ancient Script Studio';
    const lines = [`# ${title} — report`, ''];
    for (const [k, v] of Object.entries(payload)) lines.push(`- **${k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}**: ${flatten(v)}`);
    return lines.join('\n') + '\n';
  }

  function clientPdf(payload) {
    loadScript('https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js')
      .then(() => {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF({ unit: 'pt', format: 'a4' });
        const title = payload.title || 'NLP / Ancient Script Studio';
        doc.setFontSize(18); doc.text(title, 40, 50);
        doc.setFontSize(10);
        let y = 80;
        for (const [k, v] of Object.entries(payload)) {
          const text = `${k}: ${flatten(v)}`;
          const lines = doc.splitTextToSize(text, 515);
          for (const line of lines) {
            if (y > 800) { doc.addPage(); y = 50; }
            doc.text(line, 40, y);
            y += 13;
          }
          y += 4;
        }
        doc.save('nlp-studio-result.pdf');
      })
      .catch(() => toast('PDF export unavailable offline.'));
  }

  function exportResult(format, payload) {
    if (!payload) return;
    const base = 'nlp-studio-result';
    if (format === 'json') downloadBlob(JSON.stringify(payload, null, 2), 'application/json', base + '.json');
    else if (format === 'txt') downloadBlob(txtify(payload), 'text/plain', base + '.txt');
    else if (format === 'md') downloadBlob(mdify(payload), 'text/markdown', base + '.md');
    else if (format === 'pdf') {
      if (state.mode === 'backend' && payload.result_id) {
        downloadUrl(`/api/export/result/${payload.result_id}?format=pdf`, `nlp-studio-${payload.result_id}.pdf`);
      } else {
        clientPdf(payload);
      }
    }
  }

  /* ---------- copy / TTS ---------- */

  function copyText(text) {
    if (!text) { toast('Nothing to copy.'); return; }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => toast('Copied to clipboard.')).catch(() => fallbackCopy(text));
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('Copied to clipboard.'); } catch (_) { toast('Copy failed.'); }
    ta.remove();
  }

  function speak(text, lang) {
    if (!text) { toast('Nothing to speak.'); return; }
    if (!('speechSynthesis' in window)) { toast('Text-to-speech not available.'); return; }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang || 'en';
    window.speechSynthesis.speak(utterance);
  }

  /* ---------- history ---------- */

  async function loadHistory() {
    if (state.mode !== 'backend') return;
    try {
      const data = await fetch('/api/history').then((r) => r.json());
      els.historyCard.hidden = false;
      const records = (data.records || []).slice(-20).reverse();
      if (!records.length) {
        els.historyList.innerHTML = '<li class="empty">No translations recorded yet.</li>';
        return;
      }
      els.historyList.innerHTML = records.map((r) => {
        const src = r.source || '';
        const tr = r.translation || '—';
        const meta = [r.target_language, r.status, r.provider].filter(Boolean).join(' · ');
        return `<li><span class="src">${escapeHtml(src)}</span><span class="tr">${escapeHtml(tr)}</span><span class="meta">${escapeHtml(meta)}</span></li>`;
      }).join('');
    } catch (_) { /* history is non-critical */ }
  }

  /* ---------- wiring ---------- */

  function wireEvents() {
    $$('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

    els.dropzone.addEventListener('click', () => els.fileInput.click());
    els.dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') els.fileInput.click(); });
    ['dragenter', 'dragover'].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add('is-dragover'); }));
    ['dragleave', 'drop'].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove('is-dragover'); }));
    els.dropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) { acceptFile(file); switchTab('upload'); }
    });

    els.fileInput.addEventListener('change', () => {
      if (els.fileInput.files[0]) acceptFile(els.fileInput.files[0]);
    });

    if (els.cameraBtn) els.cameraBtn.addEventListener('click', () => els.cameraInput.click());
    els.cameraInput.addEventListener('change', () => {
      const file = els.cameraInput.files[0];
      if (!file) return;
      state.file = file;
      els.cameraPicked.hidden = false;
      els.cameraPicked.innerHTML = `<span class="name">${escapeHtml(file.name)}</span><span>${(file.size / 1024).toFixed(1)} KB</span>`;
      const url = URL.createObjectURL(file);
      els.cameraPreview.hidden = false;
      els.cameraPreview.src = url;
    });

    els.runBtn.addEventListener('click', run);
    els.historyRefresh.addEventListener('click', loadHistory);

    // result action buttons (copy / speak)
    document.addEventListener('click', (e) => {
      const copyBtn = e.target.closest('[data-copy]');
      if (copyBtn) {
        const kind = copyBtn.dataset.copy;
        const payload = state.result;
        if (!payload) return;
        if (kind === 'original') copyText((payload.scan && payload.scan.text) || (payload.translation && payload.translation.source_text) || '');
        else if (kind === 'transliteration') copyText(payload.translation && payload.translation.transliteration);
        else if (kind === 'translation') copyText(payload.translation && payload.translation.translation);
        return;
      }
      const speakBtn = e.target.closest('[data-speak]');
      if (speakBtn) {
        const kind = speakBtn.dataset.speak;
        const payload = state.result;
        if (!payload) return;
        const target = payload.target_language || 'en';
        if (kind === 'translation') speak((payload.translation && payload.translation.translation) || '', target.startsWith('ar') ? 'ar' : 'en');
        else if (kind === 'transliteration') speak((payload.translation && payload.translation.transliteration) || '', 'en');
        else if (kind === 'original') toast('Native ancient-script speech is not available — use transliteration/translation playback.');
      }
    });
  }

  /* ---------- boot ---------- */

  async function boot() {
    wireEvents();
    await detectMode();
    await populateControls();
    if (state.mode === 'backend') loadHistory();
  }

  boot();
})();
