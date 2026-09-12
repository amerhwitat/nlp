namespace AncientVisualResearchSuite;

public record CharacterAgent(string Id,string Role,double X,double Y,double Heading,string Activity);
public sealed class HistoricalScene {
    public List<CharacterAgent> Characters { get; } = new();
    public IEnumerable<IReadOnlyList<CharacterAgent>> Simulate(double seconds,double step=.25) {
        if(seconds<0 || step<=0) throw new ArgumentOutOfRangeException();
        for(double t=0;t<=seconds+1e-9;t+=step)
            yield return Characters.Select(c=>{
                var active=(c.Activity=="idle"||c.Activity=="sleep")?0:.5;
                var phase=c.Heading*Math.PI/180+t*.15;
                return c with { X=c.X+Math.Cos(phase)*active*t, Y=c.Y+Math.Sin(phase)*active*t };
            }).ToArray();
    }
}
