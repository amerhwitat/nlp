from dataclasses import dataclass
from math import exp

@dataclass
class EnvironmentState:
    temperature_c: float = 20.0
    rain_mm: float = 0.0
    humidity: float = 0.5
    visibility_km: float = 20.0
    terrain_class: str = 'semi_arid'
    vegetation_index: float = 0.3
    water_level_m: float = 0.0
    wind_mps: float = 2.0
    cloud_fraction: float = 0.0

    def advance(self, hours: float) -> 'EnvironmentState':
        if hours < 0: raise ValueError('hours must be non-negative')
        cooling = exp(-hours / 12.0)
        return EnvironmentState(
            temperature_c=20 + (self.temperature_c-20)*cooling,
            rain_mm=self.rain_mm,
            humidity=max(0,min(1,self.humidity + self.rain_mm*0.005*hours)),
            visibility_km=max(1,self.visibility_km*(1-self.cloud_fraction*.2)),
            terrain_class=self.terrain_class,
            vegetation_index=max(0,min(1,self.vegetation_index + self.rain_mm*.0005*hours)),
            water_level_m=self.water_level_m + self.rain_mm*.001,
            wind_mps=self.wind_mps,
            cloud_fraction=max(0,min(1,self.cloud_fraction + self.humidity*.01*hours)),
        )
