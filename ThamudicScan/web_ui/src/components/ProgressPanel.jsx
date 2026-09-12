export default function ProgressPanel({ progress }) {
  if (!progress) return null
  return (
    <section className="panel progress-panel" aria-live="polite">
      <div className="panel-heading"><span>Scan progress</span><strong>{progress.progress ?? 0}%</strong></div>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${progress.progress ?? 0}%` }} /></div>
      <div className="progress-meta">
        <span>{progress.status}</span>
        <span>{progress.processed_count ?? 0} processed</span>
        <span>{progress.match_count ?? 0} matches</span>
      </div>
      {progress.source && <div className="source-line">Source: {progress.source}</div>}
      {progress.message && <div className="event-line">{progress.message}</div>}
    </section>
  )
}
