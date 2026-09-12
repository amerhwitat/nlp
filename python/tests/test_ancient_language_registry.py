from pathlib import Path

from ancient_languages.registry import load_registry


ROOT = Path(__file__).parents[2] / "data" / "ancient_languages"


def test_coptic_and_greek_scripts_are_distinct():
    registry = load_registry(str(ROOT))
    assert registry.get_script("coptic").unicode_script == "Copt"
    assert registry.get_script("grek").unicode_script == "Grek"
    assert registry.get_script("coptic").id != registry.get_script("grek").id


def test_hebrew_and_aramaic_stages_exist():
    registry = load_registry(str(ROOT))
    assert registry.get_stage("hebrew.ancient").language_id == "hebrew"
    assert registry.get_stage("hebrew.modern").language_id == "hebrew"
    assert registry.get_stage("aramaic.imperial").language_id == "aramaic"
    assert registry.get_stage("neo_aramaic").language_id == "aramaic"


def test_polytonic_greek_is_orthography_not_language_identity():
    registry = load_registry(str(ROOT))
    profile = registry.get_orthography("greek.polytonic")
    assert profile["script_id"] == "grek"
    assert "greek.classical" in profile["language_stage_ids"]


def test_registry_validation_has_no_errors():
    registry = load_registry(str(ROOT))
    assert registry.validate() == []
