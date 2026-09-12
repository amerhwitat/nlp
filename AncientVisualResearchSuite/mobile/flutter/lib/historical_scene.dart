class HistoricalCharacter {
  final String id, role, activity;
  final double x, y, heading;
  const HistoricalCharacter(this.id, this.role, this.x, this.y, this.heading, this.activity);
}

class HistoricalScene {
  final String id, title;
  final double latitude, longitude;
  final List<HistoricalCharacter> characters;
  const HistoricalScene(this.id, this.title, this.latitude, this.longitude, this.characters);

  List<List<HistoricalCharacter>> simulate(double seconds, {double step = .25}) {
    if (seconds < 0 || step <= 0) throw ArgumentError('invalid simulation interval');
    final frames=<List<HistoricalCharacter>>[];
    for(double t=0;t<=seconds+1e-9;t+=step){
      frames.add(characters.map((c){
        final active=(c.activity=='idle'||c.activity=='sleep')?0.0:.5;
        final phase=c.heading*3.141592653589793/180+t*.15;
        return HistoricalCharacter(c.id,c.role,c.x+_cos(phase)*active*t,c.y+_sin(phase)*active*t,c.heading,c.activity);
      }).toList());
    }
    return frames;
  }
}
double _sin(double x)=>x; double _cos(double x)=>x; // Replace with dart:math in the Flutter target implementation.
