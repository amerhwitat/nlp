export type EvidenceClass = 'observed'|'supported'|'inferred'|'speculative'|'visualization';
export interface Evidence { id:string; source:string; kind:EvidenceClass; confidence:number; notes?:string }
export interface Character { id:string; role:string; x:number; y:number; heading:number; activity:string }
export interface EventNode { id:string; label:string; start:string; durationSeconds:number; actors:string[] }
export interface HistoricalScene {
  id:string; title:string; start:string; latitude:number; longitude:number; elevationM:number;
  evidence:Evidence[]; characters:Character[]; events:EventNode[]; environment:Record<string,number>;
  sky:Record<string,unknown>; tensor128:number[];
}

export function simulate(scene:HistoricalScene, seconds:number, step=0.25) {
  const frames:any[]=[];
  for(let t=0;t<=seconds+1e-9;t+=step){
    frames.push({timeSeconds:Number(t.toFixed(4)), characters:scene.characters.map(c=>{
      const active = c.activity==='idle'||c.activity==='sleep' ? 0 : .5;
      const phase=c.heading*Math.PI/180+t*.15;
      return {id:c.id,x:c.x+Math.cos(phase)*active*t,y:c.y+Math.sin(phase)*active*t,heading:c.heading};
    })});
  }
  return frames;
}
