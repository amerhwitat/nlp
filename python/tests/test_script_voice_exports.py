from python.thamudic.script_summary import build_script_report, build_script_summary, export_script_report, export_script_summary
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


def test_script_report_preserves_original_transliteration_and_translation_layers():
    report = build_script_report("ancient-north-arabian", "𐪀", "en")
    assert report["original_text"] == "𐪀"
    assert "transliteration" in report
    assert "translation" in report
    assert "script_information" in report


def test_script_report_exports_all_formats():
    for fmt in ("json", "md", "txt"):
        body, _, filename = export_script_report("greek", "ἀ", "en", fmt)
        assert body and filename.endswith(f"-script-report.{fmt}")


def test_voice_modes_never_fake_native_ancient_pronunciation():
    result = speak(VoiceRequest(text="𓂀", language="ancient-egyptian", mode="original"))
    assert result["status"] == "pronunciation_provider_required"
    assert result["native_ancient_tts"] is False


def test_voice_controls_are_available():
    commands = voice_control_commands()
    assert {"speak original", "speak transliteration", "speak translation", "pause", "resume", "stop", "repeat"}.issubset(commands)
