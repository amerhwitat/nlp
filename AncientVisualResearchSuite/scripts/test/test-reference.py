from datetime import datetime, timezone
from core.python.historical_scene import HistoricalScene, Character, EventNode, Evidence
from core.python.sky_model import HistoricalSky, SkyObject
from core.python.character_generator import CharacterGenerator

scene=HistoricalScene('t','test',datetime(1900,1,1,tzinfo=timezone.utc),24,45,evidence=[Evidence('e1','test')])
scene.add_character(Character('c','traveler',0,0,0,'walking'))
scene.add_event(EventNode('e','arrival',scene.start,10,['c']))
assert len(scene.simulate(1,.5)) == 3
sky=HistoricalSky().snapshot(scene.start,scene.latitude,scene.longitude,[SkyObject('test',5,20)])
assert 'test' in sky.altitude_deg
assert CharacterGenerator().generate('traveler',['e1']).id.startswith('char-')
print('AVRS Python reference tests OK')
