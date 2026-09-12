export class SpeechController {
  constructor() {
    this.synthesis = typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null;
    this.current = null;
  }

  available() {
    return Boolean(this.synthesis && typeof SpeechSynthesisUtterance !== 'undefined');
  }

  voices() {
    return this.available() ? this.synthesis.getVoices() : [];
  }

  play(text, options = {}) {
    if (!this.available() || !text) return false;
    this.stop();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = Number(options.rate ?? 1);
    utterance.pitch = Number(options.pitch ?? 1);
    utterance.volume = Number(options.volume ?? 1);
    if (options.lang) utterance.lang = options.lang;
    if (options.voice) utterance.voice = options.voice;
    this.current = utterance;
    this.synthesis.speak(utterance);
    return true;
  }

  pause() { if (this.available()) this.synthesis.pause(); }
  resume() { if (this.available()) this.synthesis.resume(); }
  stop() { if (this.available()) this.synthesis.cancel(); this.current = null; }
  replay(text, options = {}) { return this.play(text, options); }
}
