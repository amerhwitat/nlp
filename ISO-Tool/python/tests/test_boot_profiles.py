import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class BootProfileTests(unittest.TestCase):
    def test_menu_has_bios_7c00_and_uefi_contract(self):
        menu=json.loads((ROOT/'boot/menu.json').read_text(encoding='utf-8'))
        first=menu['entries'][0]
        self.assertEqual(first['biosLoadAddress'],'0x7C00')
        self.assertEqual(first['uefiEntry'],'efi_main')
        self.assertTrue(menu['fallbackOnValidationFailure'])
        self.assertIn('0x8000',menu['customLoaderAddressNote'])

if __name__=='__main__': unittest.main()
