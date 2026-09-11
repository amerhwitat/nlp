"""Isolated BIOS/UEFI boot test orchestration.

Only disposable VM/emulator processes are launched. Imported boot code is never
executed directly by the host application.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil, subprocess

@dataclass
class BootTestResult:
    firmware: str
    status: str
    evidence: str
    command: list[str] | None = None

def assemble_bios(source: Path, output: Path, nasm: str = "nasm") -> BootTestResult:
    exe=shutil.which(nasm)
    if not exe:return BootTestResult("bios","unverified","NASM not installed")
    try:
        cp=subprocess.run([exe,"-f","bin",str(source),"-o",str(output)],capture_output=True,text=True,timeout=30,check=True); data=output.read_bytes()
        if len(data)!=512 or data[-2:]!=b"\x55\xaa":return BootTestResult("bios","failed","assembled output is not a 512-byte boot sector with 55AA signature")
        return BootTestResult("bios","assembled",f"NASM produced 512-byte boot sector; stdout={cp.stdout.strip()}")
    except Exception as exc:return BootTestResult("bios","failed",f"{type(exc).__name__}: {exc}")

def qemu_bios_test(image: Path, timeout: int=10, qemu: str="qemu-system-x86_64") -> BootTestResult:
    exe=shutil.which(qemu)
    if not exe:return BootTestResult("bios","unverified","QEMU not installed")
    command=[exe,"-machine","pc","-display","none","-serial","stdio","-no-reboot","-no-shutdown","-drive",f"format=raw,file={image}"]
    try:
        cp=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        return BootTestResult("bios","emulated" if cp.returncode==0 else "failed",f"exit={cp.returncode}; stdout={cp.stdout[-2000:]}; stderr={cp.stderr[-2000:]}",command)
    except subprocess.TimeoutExpired:
        return BootTestResult("bios","timeout","QEMU remained running until timeout; boot success requires expected console/serial evidence",command)
    except Exception as exc:return BootTestResult("bios","failed",f"{type(exc).__name__}: {exc}",command)

def uefi_test_availability(ovmf_code: Path|None, qemu: str="qemu-system-x86_64") -> BootTestResult:
    if not shutil.which(qemu):return BootTestResult("uefi","unverified","QEMU not installed")
    if not ovmf_code or not Path(ovmf_code).is_file():return BootTestResult("uefi","unverified","OVMF firmware path not configured")
    return BootTestResult("uefi","ready","QEMU and OVMF are available; execute only in disposable VM harness")
