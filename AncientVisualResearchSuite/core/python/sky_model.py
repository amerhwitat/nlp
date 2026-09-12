from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import cos, pi, sin

@dataclass
class SkyObject:
    name: str
    right_ascension_h: float
    declination_deg: float
    magnitude: float = 0.0

@dataclass
class SkySnapshot:
    local_sidereal_hours: float
    altitude_deg: dict[str, float]
    azimuth_deg: dict[str, float]

class HistoricalSky:
    """Deterministic lightweight sky layer.

    It is intended for scene visualization and data plumbing. For publication-grade
    archaeoastronomy, connect an authoritative ephemeris/astronomy engine and retain
    its version, inputs and uncertainty in the scene evidence graph.
    """
    def snapshot(self, when: datetime, latitude_deg: float, longitude_deg: float,
                 objects: list[SkyObject]) -> SkySnapshot:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        day = when.timestamp() / 86400.0
        lst = (18.697374558 + 24.06570982441908 * (day - 10957.5) + longitude_deg / 15.0) % 24.0
        lat = latitude_deg * pi / 180.0
        altitude, azimuth = {}, {}
        for obj in objects:
            ha = ((lst - obj.right_ascension_h) * 15.0) * pi / 180.0
            dec = obj.declination_deg * pi / 180.0
            alt = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(ha)
            alt = max(-1.0, min(1.0, alt))
            alt_deg = __import__('math').asin(alt) * 180.0 / pi
            # Visualization-grade azimuth; replace with a high-precision ephemeris when needed.
            az = (__import__('math').atan2(sin(ha), cos(ha) * sin(lat) - __import__('math').tan(dec) * cos(lat)) * 180.0 / pi + 180.0) % 360.0
            altitude[obj.name] = alt_deg
            azimuth[obj.name] = az
        return SkySnapshot(lst, altitude, azimuth)
