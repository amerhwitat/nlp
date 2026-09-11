; ISO-Tool BIOS first-stage boot sector.
; Conventional BIOS handoff loads the sector at physical 0000:7C00.
; BIOS interrupts are valid here while still in real mode.
BITS 16
ORG 0x7C00

start:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    mov si, msg
.print:
    lodsb
    test al, al
    jz .menu
    mov ah, 0x0E
    int 0x10              ; BIOS teletype output
    jmp .print

.menu:
    xor ah, ah
    int 0x16              ; BIOS keyboard input
    cmp al, '1'
    je .primary
    cmp al, '2'
    je .fallback
    jmp .menu

.primary:
    ; First entry: handoff to the configured Chimera/next-stage loader.
    ; ISO-Tool replaces this marker with a validated stage-2 sector/layout.
    jmp .fallback

.fallback:
    mov si, fallback_msg
.fprint:
    lodsb
    test al, al
    jz .halt
    mov ah, 0x0E
    int 0x10
    jmp .fprint
.halt:
    cli
    hlt
    jmp .halt

msg db 13,10,'ISO-Tool BIOS: 1=primary  2=fallback',13,10,0
fallback_msg db 13,10,'Primary unavailable; fallback selected.',13,10,0

times 510-($-$$) db 0
dw 0xAA55
