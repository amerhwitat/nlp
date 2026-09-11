from dataclasses import dataclass
from enum import Enum

class Transport(str, Enum):
    NONE = "none"
    ADB = "adb"
    FASTBOOT = "fastboot"
    FASTBOOTD = "fastbootd"

@dataclass
class DeviceInfo:
    serial: str = ""
    product: str = ""
    state: str = ""
    ram_bytes: int = 0
    storage_bytes: int = 0
    bootloader_unlocked: bool = False
    transport: Transport = Transport.NONE

@dataclass
class FlashPlan:
    partition: str
    image_path: str
    dry_run: bool = True
    verify: bool = True

def validate_plan(device: DeviceInfo, plan: FlashPlan) -> bool:
    if not plan.partition or not plan.image_path:
        return False
    if device.transport is Transport.NONE:
        return False
    return plan.dry_run or device.bootloader_unlocked

if __name__ == "__main__":
    print("Chimera II FlashTool: inspect/validate/dry-run mode")
