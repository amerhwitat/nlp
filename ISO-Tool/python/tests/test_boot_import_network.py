import tempfile
import unittest
from pathlib import Path

from iso_tool.boot_import import inspect_image, import_boot_sector
from iso_tool.entrypoints import list_entry_points


class BootImportTests(unittest.TestCase):
    def test_inspect_and_import(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "source.img"
            dst = Path(td) / "stage" / "boot.bin"
            src.write_bytes(b"A" * 510 + b"\x55\xaa")
            info = inspect_image(src)
            self.assertTrue(info.bootable)
            import_boot_sector(src, dst)
            self.assertEqual(dst.stat().st_size, 512)

    def test_entry_points(self):
        ids = {item["id"] for item in list_entry_points()}
        self.assertIn("build-compiled-images", ids)
        self.assertIn("build-iso", ids)
        self.assertIn("import-boot-image", ids)


if __name__ == "__main__":
    unittest.main()
