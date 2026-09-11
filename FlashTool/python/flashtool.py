from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class Transport(str, Enum):
    NONE = "none"
    ADB = "adb"
    FASTBOOT = "fastboot"
    FASTBOOTD = "fastbootd"

class Slot(str, Enum):
    UNKNOWN = "unknown"
    A = "a"
    B = "b"
    NON_AB = "non-ab"

@dataclass
class DeviceInfo:
    serial: str = ""
    product: str = ""
    state: str = ""
    ram_bytes: int = 0
    storage_bytes: int = 0
    bootloader_unlocked: bool = False
    transport: Transport = Transport.NONE
    slot: Slot = Slot.UNKNOWN
    supports_dynamic_partitions: bool = False
    supports_fastbootd: bool = False

@dataclass
class FlashPlan:
    partition: str
    image_path: str
    dry_run: bool = True
    verify: bool = True
    expected_sha256: str = ""
    target_slot: Slot = Slot.UNKNOWN


def validate_plan(device: DeviceInfo, plan: FlashPlan) -> bool:
    if device is None or plan is None:
        return False
    if not plan.partition or not plan.image_path:
        return False
    if device.transport is Transport.NONE:
        return False
    if not plan.dry_run and not device.bootloader_unlocked:
        return False
    if plan.target_slot not in (Slot.UNKNOWN, Slot.NON_AB, device.slot):
        return False
    return True


def preflight(device: DeviceInfo, plan: FlashPlan, partition_size: int = 0) -> dict:
    if not validate_plan(device, plan):
        code = "locked-write" if not plan.dry_run and not device.bootloader_unlocked else "invalid-input"
        if device.transport is Transport.NONE:
            code = "no-transport"
        return {"ok": False, "code": code}

    image_size = 0
    path = Path(plan.image_path)
    if path.exists() and path.is_file():
        image_size = path.stat().st_size
    if partition_size and image_size and image_size > partition_size:
        return {"ok": False, "code": "image-too-large", "image_size": image_size, "partition_size": partition_size}

    return {
        "ok": True,
        "code": "ok",
        "dry_run": plan.dry_run,
        "verify": plan.verify,
        "requires_confirmation": True,
        "image_size": image_size,
    }

if __name__ == "__main__":
    print("Chimera II FlashTool: inspect/validate/dry-run mode")
