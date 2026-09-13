const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  const type = response.headers.get('content-type') || ''
  return type.includes('application/json') ? response.json() : response.blob()
}

const postJson = (path, payload) => request(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
})

export const analyzeArtifact = (payload) => postJson('/artifacts/analyze', payload)
export const analyzeAncientArtifact = (payload) => postJson('/artifacts/analyze-ancient', payload)
export const searchArtifacts = ({ q = '', sourceLanguage = '', scriptVariant = '', limit = 100 } = {}) =>
  request(`/artifacts/search?q=${encodeURIComponent(q)}&source_language=${encodeURIComponent(sourceLanguage)}&script_variant=${encodeURIComponent(scriptVariant)}&limit=${encodeURIComponent(limit)}`)
export const getArtifact = (id) => request(`/artifacts/${encodeURIComponent(id)}`)
export const getArtifactStats = () => request('/artifacts/stats')
export const analyzeArtifactUpload = (file, script = 'Dadanitic', targetLanguage = 'en') => {
  const form = new FormData(); form.append('file', file)
  return request(`/artifacts/analyze-upload?script=${encodeURIComponent(script)}&target_language=${encodeURIComponent(targetLanguage)}`, { method: 'POST', body: form })
}
export const exportArtifactsJson = () => window.open(`${API_BASE}/artifacts/export/json`, '_blank', 'noopener,noreferrer')
