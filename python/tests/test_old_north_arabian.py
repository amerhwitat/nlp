import unittest
from thamudic.old_north_arabian import CHARACTERS, FIRST, LAST, is_old_north_arabian, utf8_bytes, transliterate

class OldNorthArabianTests(unittest.TestCase):
    def test_full_unicode_range(self):
        self.assertEqual(len(CHARACTERS), 32)
        self.assertEqual(CHARACTERS[0]["codepoint"], FIRST)
        self.assertEqual(CHARACTERS[-1]["codepoint"], LAST)

    def test_utf8(self):
        self.assertEqual(utf8_bytes(0x10A80), bytes.fromhex("F0 90 AA 80"))
        self.assertEqual(utf8_bytes(0x10A9F), bytes.fromhex("F0 90 AA 9F"))

    def test_detection_and_transliteration(self):
        self.assertTrue(is_old_north_arabian("𐪀"))
        self.assertFalse(is_old_north_arabian("ا"))
        self.assertEqual(transliterate("𐪀𐪁𐪂"), "hlḥ")

if __name__ == "__main__":
    unittest.main()
