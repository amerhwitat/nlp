from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import cos, pi, sin
from typing import Dict, List, Tuple

@dataclass
class Evidence:
    id: str
    source: str
    kind: str = "observed"
    confidence: float = 1.0
    notes: str = ""

@dataclass
class Character:
    id: str
    role: str
    x: float
    y: float
    heading: float = 0.0
    activity: str = "idle"
    evidence_ids: List[str] = field(default_factory=list)

@dataclass
class EventNode:
    id: str
    label: str
    start: datetime
    duration_s: float
    actors: List[str] = field(default_factory=list)
    location: Tuple[float, float] = (0.0, 0.0)
    evidence_ids: List[str] = field(default_factory=list)

@dataclass
class HistoricalScene:
    id: str
    title: str
    start: datetime
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    uncertainty_years: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    characters: List[Character] = field(default_factory=list)
    events: List[EventNode] = field(default_factory=list)
    environment: Dict[str, float] = field(default_factory=dict)
    sky: Dict[str, object] = field(default_factory=dict)
    tensor128: List[float] = field(default_factory=lambda: [0.0] * 128)

    def add_event(self, event: EventNode) -> None:
        self.events.append(event)

    def add_character(self, character: Character) -> None:
        self.characters.append(character)

    def set_tensor_domain(self, domain: int, values: List[float]) -> None:
        if not 0 <= domain < 8 or len(values) != 16:
            raise ValueError("domain must be 0..7 and contain 16 values")
        self.tensor128[domain * 16:(domain + 1) * 16] = values

    def simulate(self, seconds: float, step: float = 0.25) -> List[dict]:
        if step <= 0 or seconds < 0:
            raise ValueError("seconds must be >= 0 and step must be > 0")
        frames = []
        t = 0.0
        while t <= seconds + 1e-9:
            frame_chars = []
            for c in self.characters:
                phase = (c.heading * pi / 180.0) + t * 0.15
                speed = 0.5 if c.activity not in {"idle", "sleep"} else 0.0
                frame_chars.append({"id": c.id, "x": c.x + cos(phase) * speed * t,
                                    "y": c.y + sin(phase) * speed * t,
                                    "heading": c.heading, "activity": c.activity})
            frames.append({"time_s": round(t, 4), "characters": frame_chars})
            t += step
        return frames

    def event_window(self, at: datetime, horizon_s: float = 3600.0) -> List[EventNode]:
        end = at + timedelta(seconds=horizon_s)
        return [e for e in self.events if e.start < end and e.start + timedelta(seconds=e.duration_s) >= at]
