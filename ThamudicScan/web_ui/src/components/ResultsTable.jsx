export default function ResultsTable({ results }) {
  return (
    <section className="panel results-panel">
      <div className="panel-heading"><span>Recognition results</span><span>{results.length} result{results.length === 1 ? '' : 's'}</span></div>
      {results.length === 0 ? (
        <div className="empty-state">No matching Old North Arabian text yet.</div>
      ) : (
        <div className="results-grid">
          {results.map((result) => (
            <article className="result-card" key={result.id}>
              <div className="result-top"><span>{result.source}</span><span>{Math.round((result.confidence || 0) * 100)}% recognition confidence</span></div>
              <div className="ona-text" lang="und" dir="ltr">{result.text}</div>
              <div className="translit" dir="ltr">{result.transliteration}</div>
              <div className="result-meta">
                <span>{result.language}</span>
                <span>{result.script_variant}</span>
                <span>{(result.codepoints || []).map((cp) => `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`).join(' · ')}</span>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
