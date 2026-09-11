import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from python.analyzer import analyze_bytes, analyze_path
from python.flashtool import DeviceInfo, FlashPlan, Transport, preflight


class AnalyzerTests(unittest.TestCase):
    def test_android_boot_magic(self):
        result = analyze_bytes(b"ANDROID!" + b"\0" * 64)
        self.assertTrue(result.android_magic)
        self.assertEqual(result.kind, "android-boot-family")

    def test_sparse_header(self):
        header = struct.pack("<IHHHHIIII", 0xED26FF3A, 1, 0, 28, 12, 4096, 2, 1)
        result = analyze_bytes(header)
        self.assertTrue(result.sparse)
        self.assertEqual(result.sparse_block_size, 4096)
        self.assertEqual(result.sparse_total_blocks, 2)

    def test_crau_header(self):
        payload = b"CrAU" + struct.pack(">QQI", 2, 1234, 56) + b"\0" * 64
        result = analyze_bytes(payload)
        self.assertTrue(result.ota)
        self.assertEqual(result.payload_version, 2)
        self.assertEqual(result.manifest_size, 1234)
        self.assertEqual(result.manifest_signature_size, 56)
        self.assertEqual(result.manifest_offset, 80)

    def test_avb_header_fields(self):
        header = bytearray(256)
        header[0:4] = b"AVB0"
        struct.pack_into(">II", header, 4, 1, 2)
        struct.pack_into(">I", header, 28, 1)
        struct.pack_into(">Q", header, 104, 4096)
        struct.pack_into(">Q", header, 112, 77)
        struct.pack_into(">I", header, 124, 3)
        result = analyze_bytes(bytes(header))
        self.assertTrue(result.avb)
        self.assertEqual(result.avb_required_major, 1)
        self.assertEqual(result.avb_required_minor, 2)
        self.assertEqual(result.avb_algorithm, 1)
        self.assertEqual(result.avb_descriptors_size, 4096)
        self.assertEqual(result.avb_rollback_index, 77)
        self.assertEqual(result.avb_rollback_location, 3)

    def test_ota_zip_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "update.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("payload.bin", b"CrAU" + b"\0" * 32)
                zf.writestr("payload_properties.txt", "FILE_SIZE=36\n")
                zf.writestr("META-INF/com/android/metadata", "post-build=demo\n")
            result = analyze_path(path)
            self.assertEqual(result["kind"], "ota-zip")
            self.assertTrue(any("payload.bin" in n for n in result["notes"]))

    def test_locked_write_preflight(self):
        device = DeviceInfo(transport=Transport.FASTBOOT, bootloader_unlocked=False)
        plan = FlashPlan("boot", "boot.img", dry_run=False)
        result = preflight(device, plan)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "locked-write")


if __name__ == "__main__":
    unittest.main()
