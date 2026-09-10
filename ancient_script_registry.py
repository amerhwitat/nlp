#!/usr/bin/env python3
"""Research metadata registry for Ancient Arabian and related scripts.

This registry is deliberately conservative: it identifies script families and
Unicode ranges but does not pretend that an image can be translated solely
from a label. Recognition and translation remain evidence-assisted workflows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ScriptProfile:
    key: str
    name: str
    family: str
    unicode_range: str
    direction: str
    status: str
    notes: str


SCRIPTS: Dict[str, ScriptProfile] = {
    "old_north_arabian": ScriptProfile("old_north_arabian", "Old North Arabian", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "Unicode", "Umbrella script family; use a specific variety when evidence permits."),
    "thamudic_b": ScriptProfile("thamudic_b", "Thamudic B", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "research", "OCIANA retains B as a Thamudic pending variety."),
    "thamudic_c": ScriptProfile("thamudic_c", "Thamudic C", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "research", "OCIANA retains C as a Thamudic pending variety."),
    "thamudic_d": ScriptProfile("thamudic_d", "Thamudic D", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "research", "OCIANA retains D as a Thamudic pending variety."),
    "taymanitic": ScriptProfile("taymanitic", "Taymanitic (formerly Thamudic A)", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "Unicode", "Distinct script/language historically called Thamudic A."),
    "hismaic": ScriptProfile("hismaic", "Hismaic (formerly Thamudic E)", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "Unicode", "Distinct Ancient North Arabian variety associated with the Hisma."),
    "himaitic": ScriptProfile("himaitic", "Himaitic (formerly Thamudic F)", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "research", "Use corpus provenance and scholarly classification."),
    "safaitic": ScriptProfile("safaitic", "Safaitic", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "Unicode", "Large corpus of Ancient North Arabian graffiti."),
    "dadanitic": ScriptProfile("dadanitic", "Dadanitic", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "Unicode", "Oasis North Arabian script/language associated with Dadan."),
    "dumaitic": ScriptProfile("dumaitic", "Dumaitic", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "research", "Ancient North Arabian variety; keep corpus provenance."),
    "hasaitic": ScriptProfile("hasaitic", "Hasaitic", "Ancient North Arabian", "U+10A80-U+10A9F", "RTL", "research", "Ancient Arabian epigraphic variety; identification requires evidence."),
    "sabaic": ScriptProfile("sabaic", "Sabaic", "Ancient South Arabian", "U+10A60-U+10A7F", "RTL", "Unicode", "Ancient South Arabian script; separate from Ancient North Arabian."),
    "minaeic": ScriptProfile("minaeic", "Minaic", "Ancient South Arabian", "U+10A60-U+10A7F", "RTL", "Unicode", "Ancient South Arabian epigraphic tradition."),
    "qatabanic": ScriptProfile("qatabanic", "Qatabanic", "Ancient South Arabian", "U+10A60-U+10A7F", "RTL", "Unicode", "Ancient South Arabian epigraphic tradition."),
    "hadramitic": ScriptProfile("hadramitic", "Hadramitic", "Ancient South Arabian", "U+10A60-U+10A7F", "RTL", "Unicode", "Ancient South Arabian epigraphic tradition."),
    "phoenician": ScriptProfile("phoenician", "Phoenician", "Northwest Semitic", "U+10900-U+1091F", "RTL", "Unicode", "Separate alphabetic tradition; do not conflate with ANA."),
    "aramaic": ScriptProfile("aramaic", "Imperial/Ancient Aramaic", "Northwest Semitic", "U+10840-U+1085F", "RTL", "Unicode", "Use period and variety metadata when classifying inscriptions."),
    "nabataean": ScriptProfile("nabataean", "Nabataean", "Northwest Semitic", "U+10880-U+108AF", "RTL", "Unicode", "Historical script with important links to later Arabic palaeography."),
}


def list_scripts() -> List[dict]:
    return [asdict(SCRIPTS[k]) for k in sorted(SCRIPTS)]


def get_script(key: str) -> ScriptProfile:
    return SCRIPTS[key]


if __name__ == "__main__":
    for item in list_scripts():
        print(f"{item['key']}: {item['name']} [{item['unicode_range']}]")
