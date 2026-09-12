import * as THREE from 'three';

export interface StarPoint { name:string; rightAscensionHours:number; declinationDegrees:number; magnitude:number }

/** Build a local celestial sphere for a scene snapshot. Feed positions from the authoritative ephemeris adapter for research use. */
export function buildSky(stars:StarPoint[], radius=100){
  const geometry=new THREE.BufferGeometry();
  const positions:number[]=[];
  for(const s of stars){
    const ra=s.rightAscensionHours*Math.PI/12;
    const dec=s.declinationDegrees*Math.PI/180;
    positions.push(radius*Math.cos(dec)*Math.cos(ra), radius*Math.sin(dec), radius*Math.cos(dec)*Math.sin(ra));
  }
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  const material=new THREE.PointsMaterial({size:0.8});
  return new THREE.Points(geometry,material);
}
