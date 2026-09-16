"""Pure-Python helpers used by the unified Tkinter GUI."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from crypto_research_lab import digest_text
from p2p_transport import PeerMessage
from rnn_llm_module import CryptoResearchAssistant


def build_digest_report(text: str) -> dict[str, str]:
    return digest_text(text)


def build_analysis_report(text: str) -> dict[str, Any]:
    return CryptoResearchAssistant().analyze(text)


def build_peer_message(sender: str, kind: str, payload: dict[str, Any]) -> PeerMessage:
    return PeerMessage.create(sender, kind, payload)


def validate_public_address(address: str) -> dict[str, str | bool]:
    """Validate the shape of a public Bitcoin or EVM address without network calls."""
    value = address.strip()
    if re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        return {"valid": True, "network": "EVM-compatible", "format": "hex-20-byte"}
    if re.fullmatch(r"(bc1|tb1)[ac-hj-np-z02-9]{11,87}", value.lower()):
        return {"valid": True, "network": "Bitcoin", "format": "Bech32/Bech32m-shaped"}
    if re.fullmatch(r"[123mn2][1-9A-HJ-NP-Za-km-z]{25,40}", value):
        return {"valid": True, "network": "Bitcoin-like", "format": "Base58Check-shaped"}
    return {"valid": False, "network": "unknown", "format": "unrecognized public-address shape"}


def format_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
