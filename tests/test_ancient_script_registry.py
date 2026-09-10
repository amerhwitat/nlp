from ancient_script_registry import SCRIPTS, get_script, list_scripts


def test_old_north_arabian_profile():
    profile = get_script("old_north_arabian")
    assert profile.unicode_range == "U+10A80-U+10A9F"
    assert profile.direction == "RTL"


def test_expected_ancient_varieties_are_registered():
    for key in ("thamudic_b", "taymanitic", "hismaic", "himaitic", "safaitic", "dadanitic", "sabaic", "phoenician", "nabataean"):
        assert key in SCRIPTS


def test_registry_is_serializable():
    rows = list_scripts()
    assert rows
    assert all("name" in row and "unicode_range" in row for row in rows)
