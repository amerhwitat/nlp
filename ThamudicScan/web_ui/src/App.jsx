import { useEffect, useMemo, useState } from 'react'
import FileDropzone from './components/FileDropzone'
import ProgressPanel from './components/ProgressPanel'
import ResultsTable from './components/ResultsTable'
import StatsPanel from './components/StatsPanel'
import {
  browserSpeak, exportScriptReport, exportScriptSummary, exportSession, exportTranslationLog, getAlphabetLanguages, getScriptSummary, getSession,
  getTranslationLog, getVoiceCapabilities, pauseVoice, resumeVoice, scanFile, scanSourceLanguage, scanText, stopVoice,
  subscribeProgress, translateAncient, translateText, validateText, voiceSpeak,
} from './api'

const EXAMPLE = '𐪀𐪁𐪂 𐪃𐪄'

export default function App() {
  const [text, setText] = useState(EXAMPLE); const [keywords, setKeywords] = useState('')
  const [script, setScript] = useState('Dadanitic'); const [targetLanguage, setTargetLanguage] = useState('en')
  const [sourceLanguage, setSourceLanguage] = useState('ancient-north-arabian'); const [sourceForm, setSourceForm] = useState('script')
  const [translation, setTranslation] = useState(null); const [sourceScan, setSourceScan] = useState(null)
  const [sessionId, setSessionId] = useState(''); const [progress, setProgress] = useState(null); const [summary, setSummary] = useState(null)
  const [scriptSummary, setScriptSummary] = useState(null); const [results, setResults] = useState([]); const [validation, setValidation] = useState(null)
  const [languages, setLanguages] = useState([]); const [voice, setVoice] = useState(null); const [logInfo, setLogInfo] = useState(null)
  const [busy, setBusy] = useState(false); const [translationBusy, setTranslationBusy] = useState(false); const [sourceScanBusy, setSourceScanBusy] = useState(false); const [summaryBusy, setSummaryBusy] = useState(false); const [error, setError] = useState('')
  const keywordList = useMemo(() => keywords.split(',').map((value) => value.trim()).filter(Boolean), [keywords])
  useEffect(() => { getAlphabetLanguages().then((r) => setLanguages(r.languages || [])).catch(() => {}); getVoiceCapabilities().then(setVoice).catch(() => {}); getTranslationLog().then(setLogInfo).catch(() => {}) }, [])
  useEffect(() => { if (!sessionId) return undefined; return subscribeProgress(sessionId, setProgress, () => {}) }, [sessionId])
  async function handleScan() { setBusy(true); setError(''); setProgress(null); try { const r = await scanText({ text, keywords: keywordList, source: 'text-input' }); setSessionId(r.session_id); setResults(r.results); setSummary(r.summary); setValidation(await validateText(text)); setLogInfo(await getTranslationLog()) } catch (e) { setError(e.message) } finally { setBusy(false) } }
  async function handleTranslate() { setTranslationBusy(true); setError(''); try { setTranslation(await translateText(text, script, targetLanguage)); setLogInfo(await getTranslationLog()) } catch (e) { setError(e.message) } finally { setTranslationBusy(false) } }
  async function handleAncientTranslate() { setTranslationBusy(true); setError(''); try { setTranslation(await translateAncient(text, sourceLanguage, targetLanguage, sourceForm)); setLogInfo(await getTranslationLog()) } catch (e) { setError(e.message) } finally { setTranslationBusy(false) } }
  async function handleSourceScan() { setSourceScanBusy(true); setError(''); try { setSourceScan(await scanSourceLanguage(text, sourceLanguage)) } catch (e) { setError(e.message) } finally { setSourceScanBusy(false) } }
  async function handleSummary() { setSummaryBusy(true); setError(''); try { setScriptSummary(await getScriptSummary(sourceLanguage || script.toLowerCase())) } catch (e) { setError(e.message) } finally { setSummaryBusy(false) } }
  async function handleFile(file) { setBusy(true); setError(''); setProgress(null); try { const r = await scanFile(file); setSessionId(r.session_id); setResults(r.results); setSummary(r.summary) } catch (e) { setError(e.message) } finally { setBusy(false) } }
  async function reopenSession() { const value = window.prompt('Session ID'); if (!value) return; setBusy(true); setError(''); try { const r = await getSession(value); setSessionId(value); setResults(r.results); setSummary(r.session); const last = r.events?.at(-1); if (last) setProgress(last) } catch (e) { setError(e.message) } finally { setBusy(false) } }
  function speak(textToSpeak, lang, mode) { if (!textToSpeak) return; const browserReady = browserSpeak(textToSpeak, lang); if (!browserReady) voiceSpeak(textToSpeak, lang, mode).catch((e) => setError(e.message)) }
  const reportPayload = { original_text: text, source_language: sourceLanguage, target_language: targetLanguage }

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">𐪀</span><div><strong>Ancient Script & NLP Scanner</strong><small>Unicode · transliteration · translation · script history · voice</small></div></div><nav><button>File</button><button>Tools</button><button>Help</button></nav></header>
    <main>
      <section className="hero"><div><p className="eyebrow">UNICODE · UTF-8 · NLP · EPIGRAPHY · VOICE</p><h1>Read the script.<br /><span>Preserve the evidence.</span></h1><p>Inspect original characters, historical variants, writing direction, dating, related scripts, transliteration and evidence-backed translation without silently inventing missing readings.</p></div><div className="hero-glyphs" aria-hidden="true">𐪀𐪁𐪂</div></section>
      <section className="workspace-grid">
        <div className="panel controls-panel">
          <div className="panel-heading"><span>Source text</span><span className="status-dot">● Ready</span></div>
          <label>Keywords <input value={keywords} onChange={(e) => setKeywords(e.target.value)} placeholder="e.g. h, l, inscription" /></label>
          <label>Original script / text <textarea value={text} onChange={(e) => setText(e.target.value)} rows={7} spellCheck="false" /></label>
          <div className="button-row"><button className="primary" disabled={busy || !text.trim()} onClick={handleScan}>{busy ? 'Scanning…' : 'Scan text'}</button><button disabled={busy} onClick={reopenSession}>Reopen session</button></div>
          <FileDropzone onFile={handleFile} disabled={busy} />{error && <div className="error-box" role="alert">{error}</div>}
        </div>
        <aside className="panel tools-panel">
          <div className="panel-heading">Ancient language / script</div>
          <label>Language<select value={sourceLanguage} onChange={(e) => setSourceLanguage(e.target.value)}><option value="">Auto detect</option>{languages.map((id) => <option key={id} value={id}>{id.replaceAll('-', ' ')}</option>)}</select></label>
          <label>Input form<select value={sourceForm} onChange={(e) => setSourceForm(e.target.value)}><option value="script">Original script</option><option value="transliteration">Transliteration</option></select></label>
          <div className="button-row"><button disabled={sourceScanBusy || !text.trim()} onClick={handleSourceScan}>{sourceScanBusy ? 'Scanning…' : 'Scan + UTF-8'}</button><button disabled={summaryBusy || !sourceLanguage} onClick={handleSummary}>{summaryBusy ? 'Loading…' : 'Script summary'}</button></div>
          {sourceScan && <div className="validation-result"><strong>Detected</strong><div>{sourceScan.detected_languages?.join(', ') || 'No requested profile matched'}</div><strong>Encoding</strong><div>{sourceScan.encoding} · {sourceScan.unicode_normalization}</div><small>{sourceScan.matched_character_count} matched characters · overlap: {sourceScan.ambiguous_script_overlap ? 'yes' : 'no'}</small></div>}
          <div className="panel-heading">Translation / transliteration</div>
          <label>Target language<select value={targetLanguage} onChange={(e) => setTargetLanguage(e.target.value)}><option value="en">English</option><option value="ar">Arabic</option><option value="el">Greek</option><option value="la">Latin</option></select></label>
          <div className="button-row"><button disabled={translationBusy || !text.trim() || !sourceLanguage} onClick={handleAncientTranslate}>{translationBusy ? 'Translating…' : 'Translate selected language'}</button><button disabled={translationBusy || !text.trim()} onClick={handleTranslate}>ONA baseline</button></div>
          {translation && <div className="validation-result"><strong>Original / source</strong><div>{translation.source || text}</div><strong>Transliteration</strong><div dir="ltr">{translation.transliteration || '—'}</div><strong>Translation</strong><div dir={targetLanguage === 'ar' ? 'rtl' : 'ltr'}>{translation.translation || 'No corpus-backed translation is available.'}</div><small>Status: {translation.translation_status || translation.status} · Confidence: {translation.confidence ?? '—'}</small>{translation.provenance && <small><a href={translation.provenance} target="_blank" rel="noreferrer">Corpus provenance</a></small>}</div>}
          <div className="panel-heading">Voice controls</div>
          <div className="button-row"><button onClick={() => speak(text, '', 'original')}>▶ Original</button><button onClick={() => speak(translation?.transliteration, 'en', 'transliteration')}>▶ Transliteration</button><button onClick={() => speak(translation?.translation, targetLanguage, 'translation')}>▶ Translation</button></div>
          <div className="button-row"><button onClick={pauseVoice}>Pause</button><button onClick={resumeVoice}>Resume</button><button onClick={stopVoice}>Stop</button></div>
          {voice && <small>Voice backends: {voice.tts_backends?.join(', ') || 'browser'} · native ancient TTS requires a pronunciation provider.</small>}
          <div className="panel-heading">Complete script report</div>
          {scriptSummary && <div className="validation-result"><strong>{scriptSummary.name}</strong><div>{scriptSummary.original_script}</div><div>Direction: {scriptSummary.writing_direction}</div><div>Dating: {scriptSummary.dating}</div><div>Region: {scriptSummary.geographic_scope}</div><div>Related: {(scriptSummary.related_scripts || []).join(', ')}</div><div>Variants: {(scriptSummary.variations || []).join(', ')}</div><div>Transliteration: {(scriptSummary.transliteration_systems || []).join(', ')}</div><div className="button-row"><button onClick={() => exportScriptSummary(sourceLanguage, 'json')}>Metadata JSON</button><button onClick={() => exportScriptSummary(sourceLanguage, 'pdf')}>Metadata PDF</button><button onClick={() => exportScriptReport(reportPayload, 'json')}>Full JSON</button><button onClick={() => exportScriptReport(reportPayload, 'md')}>Full Markdown</button><button onClick={() => exportScriptReport(reportPayload, 'txt')}>Full TXT</button><button onClick={() => exportScriptReport(reportPayload, 'pdf')}>Full PDF</button></div></div>}
          <div className="panel-heading">Translation history</div>
          {logInfo && <div className="validation-result"><strong>{logInfo.records?.length || 0} records</strong><div>Integrity: {logInfo.verification?.valid || 0} valid / {logInfo.verification?.invalid || 0} invalid</div><small>{logInfo.path}</small><div className="button-row"><button onClick={() => exportTranslationLog('json')}>JSON</button><button onClick={() => exportTranslationLog('jsonl')}>JSONL</button><button onClick={() => exportTranslationLog('txt')}>TXT</button><button onClick={() => exportTranslationLog('pdf')}>PDF</button></div></div>}
          <div className="panel-heading">Validation</div><button disabled={busy || !text.trim()} onClick={async () => { try { setValidation(await validateText(text)); setError('') } catch (e) { setError(e.message) } }}>Validate Unicode</button>
          {validation && <div className="validation-result"><div className="validation-glyphs" dir="ltr">{validation.characters.join('')}</div><strong>{validation.count} recognized characters</strong><small>{validation.codepoints.map((cp) => `U+${cp.toString(16).toUpperCase()}`).join(' · ')}</small></div>}
        </aside>
      </section>
      <ProgressPanel progress={progress} /><StatsPanel summary={summary} validation={validation} /><ResultsTable results={results} />
      {sessionId && <section className="export-bar"><div><strong>Session</strong><code>{sessionId}</code></div><div className="button-row"><button onClick={() => exportSession(sessionId, 'csv')}>Export CSV</button><button onClick={() => exportSession(sessionId, 'json')}>Export JSON</button><button onClick={() => exportSession(sessionId, 'pdf')}>Export PDF</button></div></section>}
    </main>
    <footer>Ancient-script dates are approximate research metadata; Unicode identity, historical dating, transliteration and translation are separate evidence layers.</footer>
  </div>
}
