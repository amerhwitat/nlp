import tempfile
import unittest
from pathlib import Path
from iso_tool.boot_validator import select_with_fallback

class BootValidatorTests(unittest.TestCase):
    def test_falls_back_when_primary_artifact_missing(self):
        menu={'default':'primary','entries':[
            {'id':'primary','firmware':['bios'],'biosLoadAddress':'0x7C00','artifact':'/boot/missing.bin','fallback':['secondary']},
            {'id':'secondary','firmware':['bios'],'biosLoadAddress':'0x7C00','artifact':'/boot/ok.bin','fallback':[]}]}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'boot').mkdir(); (root/'boot/ok.bin').write_bytes(b'ok')
            selected,attempts=select_with_fallback(menu,root)
            self.assertEqual(selected,'secondary'); self.assertEqual(attempts[0].status,'unavailable'); self.assertEqual(attempts[1].status,'eligible')

if __name__=='__main__': unittest.main()
