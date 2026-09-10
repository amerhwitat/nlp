#!/usr/bin/env python3
"""Backward-compatible launcher for the professional Thamudic desktop app."""
from thamudic_desktop import ThamudicDesktop

AncientResearchApp = ThamudicDesktop

if __name__ == "__main__":
    AncientResearchApp().mainloop()
