"""Python facade for Thamudic, Ancient North Arabian, and ancient-language NLP."""
from .old_north_arabian import BY_CHARACTER, BY_CODEPOINT, CHARACTERS, FIRST, LAST, VARIANT_FORMS, is_old_north_arabian, transliterate as transliterate_ona, utf8_bytes
from .ancient_translation import CorpusEntry, InMemoryCorpus, OCIANA_SEED, supported_targets, translate, transliterate_source
from .source_language_scanner import encode_utf8, scan_source_language, supported_source_languages
from .ancient_alphabet_registry import load_alphabet_registry, supported_alphabet_languages, language_profile, variations, translation_capabilities, translation_directions
from .universal_translation import TranslationResult, TranslationProvider, translate_ancient, translation_matrix
from .script_summary import build_script_summary, build_script_report, export_script_summary, export_script_report
from .voice import VoiceRequest, speak as voice_speak, available_tts_backends, voice_control_commands, speech_recognition_capability
from .translation_log import TranslationLogRecord, make_record, append_record, read_records, verify_records, export_records, default_log_path

is_thamudic = is_old_north_arabian

def extract(text: str) -> str: return ''.join(ch for ch in text if is_old_north_arabian(ch))
def transliterate(text: str, mapping: dict[int, str] | None = None) -> str:
    if mapping is not None: return ''.join(mapping.get(ord(ch), '?' if is_old_north_arabian(ch) else ch) for ch in text)
    return transliterate_ona(text)

__all__ = [
    "BY_CHARACTER", "BY_CODEPOINT", "CHARACTERS", "FIRST", "LAST", "VARIANT_FORMS", "CorpusEntry", "InMemoryCorpus", "OCIANA_SEED", "extract", "is_old_north_arabian", "is_thamudic", "supported_targets", "translate", "transliterate", "transliterate_source", "supported_source_languages", "scan_source_language", "encode_utf8", "utf8_bytes", "load_alphabet_registry", "supported_alphabet_languages", "language_profile", "variations", "translation_capabilities", "translation_directions", "TranslationResult", "TranslationProvider", "translate_ancient", "translation_matrix", "build_script_summary", "build_script_report", "export_script_summary", "export_script_report", "VoiceRequest", "voice_speak", "available_tts_backends", "voice_control_commands", "speech_recognition_capability", "TranslationLogRecord", "make_record", "append_record", "read_records", "verify_records", "export_records", "default_log_path",
]
