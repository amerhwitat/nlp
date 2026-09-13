from python.thamudic.script_summary import build_script_summary, export_script_summary
from python.thamudic.voice import VoiceRequest, speak, voice_control_commands


def test_script_summary_contains_core_metadata():
    summary = build_script_summary("ancient-egyptian")
    assert summary["original_script"]
    assert summary["writing_direction"]
    assert summary["variations"]
    assert "related_scripts" in summary
    assert "translation_directions" in summary


def test_summary_exports_json_and_markdown():
    body, media_type, filename = export_script_summary("greek", "json")
    assert "Classical Greek" in body
    assert media_type.startswith("application/json")
    assert filename.endswith(".json")
    body, media_type, filename = export_script_summary("greek", "md")
    assert "#" in body
    assert media_type.startswith("text/markdown")
    assert filename.endswith(".md")


def test_voice_original_is_explicitly_provider_gated():
    result = speak(VoiceRequest("𓂀", "ancient-egyptian", mode="original"))
    assert result["native_ancient_tts"] is False
    assert result["status"] == "pronunciation_provider_required"
    assert "speak translation" in voice_control_commands()
