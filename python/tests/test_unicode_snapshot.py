import json
from pathlib import Path

from ancient_languages.unicode_registry import codepoint_to_utf8, load_unicode_registry


ROOT = Path(__file__).parents[2]
SNAPSHOT = ROOT / "data" / "ancient_languages" / "unicode_snapshot.json"


def test_snapshot_version_and_representative_scripts():
    registry = load_unicode_registry(str(SNAPSHOT), "18.0.0-draft")
    assert registry.lookup(0x05D0).script == "Hebr"
    assert registry.lookup(0x03B1).script == "Grek"
    assert registry.lookup(0x2C81).script == "Copt"
    assert registry.lookup(0x0710).script == "Syrc"


def test_snapshot_utf8_matches_codepoint():
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for item in payload["records"]:
        cp = int(item["codepoint"][2:], 16)
        assert item["utf8"] == codepoint_to_utf8(cp).hex(" ").upper()


def test_invalid_codepoint_rejected():
    try:
        codepoint_to_utf8(0x110000)
    except ValueError:
        return
    raise AssertionError("out-of-range code point must fail")
