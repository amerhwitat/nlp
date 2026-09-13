const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
async function request(path, options = {}) { const response = await fetch(`${API_BASE}${path}`, options); if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status})`) }; return response.json() }
const postJson = (path, payload) => request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
export function scanText(payload) { return postJson('/scan', payload) }
export function scanFile(file) { const form = new FormData(); form.append('file', file); return request('/scan_file', { method: 'POST', body: form }) }
export function validateText(text) { return postJson('/validate', { text }) }
export function translateText(text, script = 'Dadanitic', targetLanguage = 'en') { return postJson('/translate', { text, script, target_language: targetLanguage }) }
export function translateAncient(text, sourceLanguage, targetLanguage, sourceForm = 'script') { return postJson('/translate_ancient', { text, source_language: sourceLanguage, target_language: targetLanguage, source_form: sourceForm }) }
export function scanSourceLanguage(text, language = '') { return postJson('/scan_language', { text, language: language || null }) }
export function getAlphabetLanguages() { return request('/alphabet-languages') }
export function getAlphabetProfile(language) { return request(`/alphabet-languages/${encodeURIComponent(language)}`) }
export function getScriptSummary(language) { return request(`/script-summary/${encodeURIComponent(language)}`) }
export function exportScriptSummary(language, format = 'json') { window.open(`${API_BASE}/script-summary/${encodeURIComponent(language)}/export?format=${encodeURIComponent(format)}`, '_blank', 'noopener,noreferrer') }
export function scriptReport(payload) { return postJson('/script-report', payload) }
export async function exportScriptReport(payload, format = 'json') {
  const response = await fetch(`${API_BASE}/script-report/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...payload, format }) })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status})`) }
  const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `ancient-script-report.${format}`; link.click(); URL.revokeObjectURL(url)
}
export function getVoiceCapabilities() { return request('/voice/capabilities') }
export function voiceSpeak(text, language, mode = 'translation') { return postJson('/voice/speak', { text, language, mode }) }
export function browserSpeak(text, language = '') { if (!('speechSynthesis' in window)) return false; window.speechSynthesis.cancel(); const utterance = new SpeechSynthesisUtterance(text); if (language) utterance.lang = language; window.speechSynthesis.speak(utterance); return true }
export function stopVoice() { if ('speechSynthesis' in window) window.speechSynthesis.cancel() }
export function pauseVoice() { if ('speechSynthesis' in window) window.speechSynthesis.pause() }
export function resumeVoice() { if ('speechSynthesis' in window) window.speechSynthesis.resume() }
export function getSession(sessionId) { return request(`/sessions/${encodeURIComponent(sessionId)}`) }
export function subscribeProgress(sessionId, onEvent, onError) { const source = new EventSource(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/events`); source.onmessage = (event) => { try { onEvent(JSON.parse(event.data)) } catch (error) { onError?.(error) } }; source.onerror = (error) => onError?.(error); return () => source.close() }
export function exportSession(sessionId, format) { window.open(`${API_BASE}/export/${encodeURIComponent(sessionId)}?format=${encodeURIComponent(format)}`, '_blank', 'noopener,noreferrer') }
