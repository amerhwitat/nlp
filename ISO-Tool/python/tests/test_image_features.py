import tempfile
import unittest
from pathlib import Path
from iso_tool.image_profiles import get_profile
from iso_tool.iso_inspect import inspect_iso

class ImageFeatureTests(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(get_profile('bios-uefi').udf_version, '1.02')
        with self.assertRaises(ValueError): get_profile('missing')

    def test_non_iso_is_safe(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.bin'; p.write_bytes(b'not-an-iso')
            info=inspect_iso(p)
            self.assertFalse(info.iso9660)
            self.assertFalse(info.el_torito)

if __name__ == '__main__': unittest.main()
