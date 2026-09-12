const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json()
}

export function scanText(payload) {
  return request('/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function scanFile(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/scan_file', { method: 'POST', body: form })
}

export function validateText(text) {
  return request('/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
}

export function getSession(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`)
}

export function subscribeProgress(sessionId, onEvent, onError) {
  const source = new EventSource(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/events`)
  source.onmessage = (event) => {
    try { onEvent(JSON.parse(event.data)) } catch (error) { onError?.(error) }
  }
  source.onerror = (error) => onError?.(error)
  return () => source.close()
}

export function exportSession(sessionId, format) {
  const url = `${API_BASE}/export/${encodeURIComponent(sessionId)}?format=${encodeURIComponent(format)}`
  window.open(url, '_blank', 'noopener,noreferrer')
}
