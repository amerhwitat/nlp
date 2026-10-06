"""Translation service: deterministic offline corpus + optional remote provider hook.

The offline path reuses ``python.thamudic.ancient_translation`` (OCIANA seed
corpus, evidence-first transliteration) and every request is recorded in the
reusable ``python.thamudic.translation_log`` history.

The remote provider hook is *only* enabled when ``ANCIENT_TRANSLATION_API_KEY``
is present in the environment. Without that key the service is fully offline and
never calls an external API.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from . import config
from python.thamudic.ancient_translation import translate as corpus_translate
from python.thamudic.translation_log import append_record, make_record

# Confidence -> numeric score used only for the durable history log.
_CONFIDENCE_SCORE = {"scholarly": 1.0, "model": 0.8, "unknown": 0.0, "provider": 0.7}


def _confidence_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return _CONFIDENCE_SCORE.get(str(value).casefold(), 0.0)


def _remote_translate(text: str, transliteration: str, script: str, target_language: str) -> dict[str, Any] | None:
    """Generic HTTP translation hook. Enabled only when an API key is configured."""
    if not config.provider_enabled():
        return None
    url = config.TRANSLATION_API_URL or "https://api.openai.com/v1/chat/completions"
    prompt = (
        f"Translate the following Ancient North Arabian inscription ({script}) "
        f"from its scholarly transliteration into {target_language}. "
        f"Return only the translation.\n\n"
        f"Original: {text}\nTransliteration: {transliteration}"
    )
    body = {
        "model": config.TRANSLATION_MODEL or "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You translate ancient inscriptions. Answer with only the translation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.TRANSLATION_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # provider failure never breaks the request
        return {
            "translation": None,
            "provider": "remote",
            "provenance": url,
            "confidence": "model",
            "error": f"{type(exc).__name__}: {exc}",
        }

    content: str | None = None
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
        if content is None and isinstance(payload.get("translation"), str):
            content = payload["translation"]
    if isinstance(payload, str):
        content = payload
    content = (content or "").strip()
    return {
        "translation": content or None,
        "provider": "remote",
        "provenance": url,
        "confidence": "model",
    }


def translate_text(
    text: str,
    *,
    script: str = "Dadanitic",
    source_language: str = "ancient-north-arabian",
    target_language: str = "en",
    request_metadata: dict[str, Any] | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Transliterate + translate, preferring the offline corpus, then the optional provider."""
    if not text or not text.strip():
        raise ValueError("text is required")

    result = corpus_translate(text, script=script, target_language=target_language)

    provider = "corpus"
    if not result.get("translation"):
        remote = _remote_translate(text, result.get("transliteration") or text, script, target_language)
        if remote and remote.get("translation"):
            result["translation"] = remote["translation"]
            result["translation_status"] = "provider"
            result["confidence"] = remote.get("confidence", "model")
            result["provider"] = "remote"
            result["provenance"] = remote.get("provenance")
        else:
            result["provider"] = "none"
    else:
        result["provider"] = "corpus"

    result["source_language"] = source_language

    if log:
        _log_result(result, source_language, target_language, request_metadata)

    return result


def _log_result(result: dict[str, Any], source_language: str, target_language: str, request_metadata: dict[str, Any] | None) -> None:
    status = str(result.get("translation_status") or "unknown")
    provider = str(result.get("provider") or "none")
    record = make_record(
        source=str(result.get("source_text") or ""),
        source_language=source_language,
        source_form="script",
        target_language=target_language,
        target_form=result.get("translation"),
        transliteration=result.get("transliteration"),
        translation=result.get("translation"),
        status=status,
        confidence=_confidence_score(result.get("confidence")),
        provider=provider,
        provenance=result.get("provenance"),
        request_metadata=request_metadata or {},
    )
    append_record(record, config.DATA_DIR / "translations.jsonl")
