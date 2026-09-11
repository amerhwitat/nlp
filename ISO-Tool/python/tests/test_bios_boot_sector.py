import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

class BiosBootSectorTests(unittest.TestCase):
    def test_source_declares_7c00_and_bios_interrupts(self):
        text=(ROOT/'boot/bios/first_stage.asm').read_text(encoding='utf-8')
        self.assertIn('ORG 0x7C00',text)
        self.assertIn('int 0x10',text)
        self.assertIn('int 0x16',text)
        self.assertIn('dw 0xAA55',text)

if __name__=='__main__': unittest.main()
