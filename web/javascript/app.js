import { SpeechController } from './voice.js';

const input = document.querySelector('#input');
const transliteration = document.querySelector('#transliteration');
const translation = document.querySelector('#translation');
const out = document.querySelector('#out');
const speech = new SpeechController();

function bindVoiceControls(container) {
  const target = document.querySelector(`#${container.dataset.voiceTarget}`);
  const voiceSelect = container.querySelector('[data-control="voice"]');
  const lang = container.querySelector('[data-control="lang"]');
  const rate = container.querySelector('[data-control="rate"]');
  const pitch = container.querySelector('[data-control="pitch"]');
  const volume = container.querySelector('[data-control="volume"]');
  const status = container.querySelector('[data-status]');

  const refreshVoices = () => {
    voiceSelect.replaceChildren();
    const voices = speech.voices();
    voices.forEach((voice) => {
      const option = document.createElement('option');
      option.value = voice.voiceURI;
      option.textContent = `${voice.name} (${voice.lang})`;
      voiceSelect.append(option);
    });
    status.textContent = speech.available()
      ? `Speech ready (${voices.length} voice${voices.length === 1 ? '' : 's'})`
      : 'Speech synthesis unavailable; use reference audio/phoneme backend';
  };

  const options = () => ({
    voice: speech.voices().find((voice) => voice.voiceURI === voiceSelect.value),
    lang: lang.value,
    rate: rate.value,
    pitch: pitch.value,
    volume: volume.value,
  });

  container.querySelector('[data-action="play"]').addEventListener('click', () => speech.play(target.value, options()));
  container.querySelector('[data-action="pause"]').addEventListener('click', () => speech.pause());
  container.querySelector('[data-action="resume"]').addEventListener('click', () => speech.resume());
  container.querySelector('[data-action="stop"]').addEventListener('click', () => speech.stop());
  container.querySelector('[data-action="replay"]').addEventListener('click', () => speech.replay(target.value, options()));
  refreshVoices();
  if (speech.synthesis) speech.synthesis.addEventListener('voiceschanged', refreshVoices);
}

document.querySelectorAll('[data-voice-target]').forEach(bindVoiceControls);

input.addEventListener('input', () => {
  const text = input.value;
  transliteration.value ||= text;
  out.textContent = JSON.stringify({
    characters: [...text].length,
    words: text.trim() ? text.trim().split(/\s+/u).length : 0,
    lines: text ? text.split(/\r?\n/u).length : 0,
    speechAvailable: speech.available(),
  }, null, 2);
});
