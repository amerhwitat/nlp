import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.flashtool import DeviceInfo, FlashPlan, Transport, validate_plan


class FlashToolTests(unittest.TestCase):
    def test_dry_run_locked_device_is_allowed(self):
        d = DeviceInfo(transport=Transport.FASTBOOT, bootloader_unlocked=False)
        self.assertTrue(validate_plan(d, FlashPlan("boot", "boot.img", True)))

    def test_write_locked_device_is_blocked(self):
        d = DeviceInfo(transport=Transport.FASTBOOT, bootloader_unlocked=False)
        self.assertFalse(validate_plan(d, FlashPlan("boot", "boot.img", False)))

    def test_no_transport_is_blocked(self):
        d = DeviceInfo(transport=Transport.NONE, bootloader_unlocked=True)
        self.assertFalse(validate_plan(d, FlashPlan("boot", "boot.img", False)))


if __name__ == '__main__':
    unittest.main()
