# BIOS first-stage

`first_stage.asm` is a minimal 512-byte NASM real-mode first-stage template.

- Conventional BIOS load/handoff address: physical `0x7C00`.
- Uses BIOS `INT 10h` for text output and `INT 16h` for keyboard input.
- Ends with the BIOS boot signature `55 AA`.
- The first menu can select a primary or fallback path.

This is a first-stage template, not a complete operating-system loader. ISO-Tool must replace/connect the handoff area only after the selected next-stage artifact has passed validation.
