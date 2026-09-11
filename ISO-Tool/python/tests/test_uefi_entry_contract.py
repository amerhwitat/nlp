import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

class UefiEntryContractTests(unittest.TestCase):
    def test_efi_entry_is_not_bios_interrupt_code(self):
        text=(ROOT/'boot/uefi/entry.c').read_text(encoding='utf-8')
        self.assertIn('efi_main',text)
        self.assertIn('EFI_SYSTEM_TABLE',text)
        self.assertNotIn('int 0x10',text)
        self.assertNotIn('int 0x16',text)
        self.assertIn('no universal 0x8000',text)

if __name__=='__main__': unittest.main()
