from python.thamudic import FIRST, LAST, utf8_bytes


def test_canonical_old_north_arabian_range_and_utf8():
    assert FIRST == 0x10A80
    assert LAST == 0x10A9F
    assert utf8_bytes(FIRST) == chr(FIRST).encode('utf-8')
    assert utf8_bytes(LAST) == chr(LAST).encode('utf-8')


def test_surrounding_codepoints_are_not_misclassified():
    from python.thamudic import is_old_north_arabian
    assert not is_old_north_arabian(FIRST - 1)
    assert not is_old_north_arabian(LAST + 1)
