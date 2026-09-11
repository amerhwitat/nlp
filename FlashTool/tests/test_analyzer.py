import io
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from python.analyzer import analyze_bytes, analyze_path


class AnalyzerTests(unittest.TestCase):
    def test_android_boot_magic(self):
        result = analyze_bytes(b"ANDROID!" + b"\0" * 64)
        self.assertTrue(result.android_magic)
        self.assertEqual(result.kind, "android-boot-family")

    def test_sparse_header(self):
        header = struct.pack("<IHHHHIIII", 0xED26FF3A, 1, 0, 28, 12, 4096, 2, 1)
        result = analyze_bytes(header)
        self.assertTrue(result.sparse)
        self.assertEqual(result.kind, "sparse")

    def test_crau_header(self):
        payload = b"CrAU" + struct.pack(">QQI", 2, 1234, 56) + b"\0" * 64
        result = analyze_bytes(payload)
        self.assertTrue(result.ota)
        self.assertEqual(result.payload_version, 2)
        self.assertEqual(result.manifest_size, 1234)
        self.assertEqual(result.manifest_signature_size, 56)

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


if __name__ == "__main__":
    unittest.main()
