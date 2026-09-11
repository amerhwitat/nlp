import struct
import tempfile
import unittest
from pathlib import Path

from iso_tool.advanced_inspect import SECTOR, inspect_advanced
from iso_tool.image_profiles import get_profile
from iso_tool.reproducible import source_date_epoch


class AdvancedInspectionTests(unittest.TestCase):
    def make_iso(self):
        data = bytearray(SECTOR * 24)
        data[510:512] = b"\x55\xAA"
        pvd = SECTOR * 16
        data[pvd + 1:pvd + 6] = b"CD001"
        data[pvd + 6] = 1
        data[pvd + 40:pvd + 45] = b"TEST "
        data[pvd + 132:pvd + 136] = struct.pack("<I", 0)
        data[pvd + 158:pvd + 162] = struct.pack("<I", 20)
        data[pvd + 166:pvd + 170] = struct.pack("<I", 2048)
        term = SECTOR * 17
        data[term + 1:term + 6] = b"CD001"
        data[term] = 255
        return data

    def test_detects_iso_and_mbr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.iso"
            path.write_bytes(self.make_iso())
            result = inspect_advanced(path)
            self.assertTrue(result.iso9660)
            self.assertTrue(result.mbr)
            self.assertEqual(result.descriptors[0].volume_id, "TEST")
            self.assertEqual(len(result.sha256), 64)

    def test_profiles(self):
        profile = get_profile("reproducible-bios-uefi")
        self.assertTrue(profile.reproducible)
        self.assertTrue(profile.large_image_boot_order)
        self.assertEqual(profile.udf_version, "1.02")

    def test_source_date_epoch_optional(self):
        value = source_date_epoch()
        self.assertTrue(value is None or isinstance(value, int))


if __name__ == "__main__":
    unittest.main()
