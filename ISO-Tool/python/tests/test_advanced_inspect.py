import struct
import tempfile
import unittest
from pathlib import Path

from iso_tool.advanced_inspect import SECTOR, inspect_advanced
from iso_tool.image import build_iso_command
from iso_tool.image_profiles import get_profile
from iso_tool.reproducible import source_date_epoch


class AdvancedInspectionTests(unittest.TestCase):
    def make_iso(self):
        data = bytearray(SECTOR * 40)
        data[510:512] = b"\x55\xAA"
        pvd = SECTOR * 16
        data[pvd + 1:pvd + 6] = b"CD001"
        data[pvd + 6] = 1
        data[pvd + 40:pvd + 45] = b"TEST "
        data[pvd + 158:pvd + 162] = struct.pack("<I", 20)
        data[pvd + 166:pvd + 170] = struct.pack("<I", 2048)
        data[pvd + 71:pvd + 75] = struct.pack("<I", 19)
        data[pvd + 7:pvd + 16] = b"EL TORITO"
        term = SECTOR * 17
        data[term + 1:term + 6] = b"CD001"
        data[term] = 255
        catalog = SECTOR * 19
        data[catalog] = 1
        data[catalog + 1] = 0
        data[catalog + 30:catalog + 32] = b"\x55\xAA"
        entry = catalog + 32
        data[entry] = 0x88
        data[entry + 2:entry + 4] = struct.pack("<H", 0x7C0)
        data[entry + 4] = 0
        data[entry + 6:entry + 8] = struct.pack("<H", 4)
        data[entry + 8:entry + 12] = struct.pack("<I", 21)
        words = list(struct.unpack("<16H", data[catalog:catalog + 32]))
        words[7] = 0
        words[7] = (-sum(words)) & 0xFFFF
        data[catalog:catalog + 32] = struct.pack("<16H", *words)
        return data

    def test_detects_iso_mbr_and_eltorito(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.iso"
            path.write_bytes(self.make_iso())
            result = inspect_advanced(path)
            self.assertTrue(result.iso9660)
            self.assertTrue(result.mbr)
            self.assertEqual(result.descriptors[0].volume_id, "TEST")
            self.assertEqual(len(result.boot_entries), 1)
            self.assertEqual(result.boot_entries[0].platform_id, 0)
            self.assertEqual(result.boot_entries[0].load_segment, 0x7C0)
            self.assertFalse(any("checksum" in warning for warning in result.warnings))
            self.assertEqual(len(result.sha256), 64)

    def test_profiles(self):
        profile = get_profile("reproducible-bios-uefi")
        self.assertTrue(profile.reproducible)
        self.assertTrue(profile.large_image_boot_order)
        self.assertEqual(profile.udf_version, "1.02")

    def test_source_date_epoch_optional(self):
        value = source_date_epoch()
        self.assertTrue(value is None or isinstance(value, int))

    def test_xorriso_command_uses_efi_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "EFI/BOOT").mkdir(parents=True)
            (root / "boot/bios").mkdir(parents=True)
            (root / "EFI/BOOT/BOOTX64.EFI").write_bytes(b"efi")
            (root / "boot/bios/first_stage.bin").write_bytes(b"bios")
            cmd = build_iso_command(root, root / "out.iso", profile="bios-uefi", backend="xorriso")
            self.assertIn("-e", cmd)
            self.assertIn("EFI/BOOT/BOOTX64.EFI", cmd)
            self.assertIn("-eltorito-alt-boot", cmd)


if __name__ == "__main__":
    unittest.main()
