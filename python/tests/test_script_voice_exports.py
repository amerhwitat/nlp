from python.thamudic.script_summary import build_script_summary, export_script_summary
from python.thamudic.voice import VoiceRequest, speak, voice_control_commands


def test_script_summary_contains_historical_metadata_fields():
    summary = build_script_summary("ancient-egyptian")
    assert summary["name"] == "Ancient Egyptian"
    assert summary["original_script"]
    assert "dating" in summary
    assert "related_scripts" in summary
    assert "writing_direction" in summary


def test_script_summary_exports_json_markdown_and_text():
    for fmt, media_type in (("json", "application/json"), ("md", "text/markdown"), ("txt", "text/plain")):
        body, actual_media_type, filename = export_script_summary("greek", fmt)
        assert body
        assert actual_media_type.startswith(media_type)
        assert filename.endswith(f"-script-summary.{fmt}")


def test_voice_modes_never_fake_native_ancient_pronunciation():
    result = speak(VoiceRequest(text="𓂀", language="ancient-egyptian", mode="original"))
    assert result["status"] == "pronunciation_provider_required"
    assert result["native_ancient_tts"] is False


def test_voice_controls_are_available():
    commands = voice_control_commands()
    assert {"speak original", "speak transliteration", "speak translation", "pause", "resume", "stop", "repeat"}.issubset(commands)
