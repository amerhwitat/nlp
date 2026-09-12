export default function StatsPanel({ summary, validation }) {
  return (
    <section className="stats-grid">
      <div className="stat-card"><span>Processed</span><strong>{summary?.processed_count ?? 0}</strong></div>
      <div className="stat-card"><span>Matches</span><strong>{summary?.match_count ?? 0}</strong></div>
      <div className="stat-card"><span>ONA code points</span><strong>{validation?.count ?? 0}</strong></div>
      <div className="stat-card"><span>Unicode range</span><strong>10A80–10A9F</strong></div>
    </section>
  )
}
