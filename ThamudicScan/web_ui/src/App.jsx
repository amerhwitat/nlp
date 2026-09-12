import { useEffect, useMemo, useState } from 'react'
import FileDropzone from './components/FileDropzone'
import ProgressPanel from './components/ProgressPanel'
import ResultsTable from './components/ResultsTable'
import StatsPanel from './components/StatsPanel'
import { exportSession, getSession, scanFile, scanText, subscribeProgress, validateText } from './api'

const EXAMPLE = '𐪀𐪁𐪂 𐪃𐪄'

export default function App() {
  const [text, setText] = useState(EXAMPLE)
  const [keywords, setKeywords] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [progress, setProgress] = useState(null)
  const [summary, setSummary] = useState(null)
  const [results, setResults] = useState([])
  const [validation, setValidation] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const keywordList = useMemo(() => keywords.split(',').map((value) => value.trim()).filter(Boolean), [keywords])

  useEffect(() => {
    if (!sessionId) return undefined
    return subscribeProgress(sessionId, setProgress, () => {})
  }, [sessionId])

  async function handleScan() {
    setBusy(true); setError(''); setProgress(null)
    try {
      const response = await scanText({ text, keywords: keywordList, source: 'text-input' })
      setSessionId(response.session_id); setResults(response.results); setSummary(response.summary)
      setValidation(await validateText(text))
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  async function handleFile(file) {
    setBusy(true); setError(''); setProgress(null)
    try {
      const response = await scanFile(file)
      setSessionId(response.session_id); setResults(response.results); setSummary(response.summary)
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  async function reopenSession() {
    const value = window.prompt('Session ID')
    if (!value) return
    setBusy(true); setError('')
    try {
      const response = await getSession(value)
      setSessionId(value); setResults(response.results); setSummary(response.session)
      const last = response.events?.at(-1)
      if (last) setProgress(last)
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">𐪀</span><div><strong>Thamudic Scanner</strong><small>Ancient North Arabian research workspace</small></div></div>
        <nav><button>File</button><button>Tools</button><button>Help</button></nav>
      </header>

      <main>
        <section className="hero">
          <div><p className="eyebrow">UNICODE · UTF-8 · RESEARCH</p><h1>Read the script.<br /><span>Preserve the evidence.</span></h1><p>Scan inscriptions and UTF-8 research material with canonical Old North Arabian extraction and transliteration.</p></div>
          <div className="hero-glyphs" aria-hidden="true">𐪀𐪁𐪂</div>
        </section>

        <section className="workspace-grid">
          <div className="panel controls-panel">
            <div className="panel-heading"><span>Scan source</span><span className="status-dot">● Ready</span></div>
            <label>Keywords <input value={keywords} onChange={(event) => setKeywords(event.target.value)} placeholder="e.g. h, l, inscription" /></label>
            <label>Inscription / text <textarea value={text} onChange={(event) => setText(event.target.value)} rows={7} spellCheck="false" /></label>
            <div className="button-row"><button className="primary" disabled={busy || !text.trim()} onClick={handleScan}>{busy ? 'Scanning…' : 'Scan text'}</button><button disabled={busy} onClick={reopenSession}>Reopen session</button></div>
            <FileDropzone onFile={handleFile} disabled={busy} />
            {error && <div className="error-box" role="alert">{error}</div>}
          </div>
          <aside className="panel tools-panel">
            <div className="panel-heading">Validation</div>
            <p>Check recognized characters and code points before or after a scan.</p>
            <button disabled={busy || !text.trim()} onClick={async () => { try { setValidation(await validateText(text)); setError('') } catch (err) { setError(err.message) } }}>Validate Unicode</button>
            {validation && <div className="validation-result"><div className="validation-glyphs" dir="ltr">{validation.characters.join('')}</div><strong>{validation.count} recognized characters</strong><small>{validation.codepoints.map((cp) => `U+${cp.toString(16).toUpperCase()}`).join(' · ')}</small></div>}
          </aside>
        </section>

        <ProgressPanel progress={progress} />
        <StatsPanel summary={summary} validation={validation} />
        <ResultsTable results={results} />

        {sessionId && <section className="export-bar"><div><strong>Session</strong><code>{sessionId}</code></div><div className="button-row"><button onClick={() => exportSession(sessionId, 'csv')}>Export CSV</button><button onClick={() => exportSession(sessionId, 'json')}>Export JSON</button></div></section>}
      </main>
      <footer>Canonical range: U+10A80–U+10A9F · Recognition confidence is a model/scanner measure, not historical certainty.</footer>
    </div>
  )
}
