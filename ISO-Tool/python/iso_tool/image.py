from __future__ import annotations
from pathlib import Path
import shutil, subprocess
from .image_profiles import get_profile

def find_backend():
    for n in ('xorriso','xorrisofs','oscdimg'):
        p=shutil.which(n)
        if p: return p
    return None

def create_iso(staging: Path, output: Path, label='ISO_TOOL', profile='data'):
    backend=find_backend()
    if not backend: raise RuntimeError('No supported ISO backend found (xorriso/xorrisofs/oscdimg).')
    cfg=get_profile(profile)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    name=Path(backend).name.lower()
    if name in ('xorriso','xorrisofs'):
        cmd=[backend,'-as','mkisofs','-iso-level','3','-V',label]
        if cfg.udf_version: cmd += ['-udf']
        if cfg.bios_boot:
            boot=Path(staging)/cfg.bios_boot
            if boot.is_file(): cmd += ['-b',str(boot),'-no-emul-boot','-boot-load-size','4','-boot-info-table']
        if cfg.uefi_boot:
            efi=Path(staging)/cfg.uefi_boot
            if efi.is_file(): cmd += ['-eltorito-alt-boot','-e','-no-emul-boot','-b',str(efi)]
        cmd += ['-o',str(output),str(staging)]
    else:
        cmd=[backend,'-l','-m','*','-o',str(output),str(staging)]
        if cfg.bios_boot and cfg.uefi_boot:
            b=Path(staging)/cfg.bios_boot; e=Path(staging)/cfg.uefi_boot
            if b.is_file() and e.is_file(): cmd[1:1]=['-bootdata:2#p0,e,b'+str(b)+'#pEF,e,b'+str(e)]
        elif cfg.bios_boot and (Path(staging)/cfg.bios_boot).is_file(): cmd[1:1]=['-b'+str(Path(staging)/cfg.bios_boot),'-p0']
        elif cfg.uefi_boot and (Path(staging)/cfg.uefi_boot).is_file(): cmd[1:1]=['-b'+str(Path(staging)/cfg.uefi_boot),'-pEF']
    return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=3600)
