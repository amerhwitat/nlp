package com.amerhwitat.avrs;
import java.util.*;

public final class HistoricalScene {
  public record CharacterAgent(String id,String role,double x,double y,double heading,String activity) {}
  private final List<CharacterAgent> characters;
  public HistoricalScene(List<CharacterAgent> characters){ this.characters=new ArrayList<>(characters); }
  public List<List<CharacterAgent>> simulate(double seconds,double step){
    if(seconds<0||step<=0) throw new IllegalArgumentException("invalid interval");
    List<List<CharacterAgent>> frames=new ArrayList<>();
    for(double t=0;t<=seconds+1e-9;t+=step){
      List<CharacterAgent> frame=new ArrayList<>();
      for(var c:characters){ double active=(c.activity().equals("idle")||c.activity().equals("sleep"))?0:.5; double phase=Math.toRadians(c.heading())+t*.15;
        frame.add(new CharacterAgent(c.id(),c.role(),c.x()+Math.cos(phase)*active*t,c.y()+Math.sin(phase)*active*t,c.heading(),c.activity())); }
      frames.add(frame);
    }
    return frames;
  }
}
