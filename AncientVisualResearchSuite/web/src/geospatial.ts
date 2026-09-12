export interface GeoEntity { id:string; longitude:number; latitude:number; height:number; label:string; start?:string; end?:string }

/** Convert a historical scene into a neutral entity list suitable for CesiumJS/CZML/3D Tiles adapters. */
export function toGeoEntities(scene:any):GeoEntity[]{
  const entities:GeoEntity[]=[];
  for(const c of scene.characters ?? []) entities.push({id:c.id,longitude:scene.longitude,latitude:scene.latitude,height:scene.elevationM+(c.y??0),label:c.role});
  for(const e of scene.events ?? []) entities.push({id:e.id,longitude:scene.longitude,latitude:scene.latitude,height:scene.elevationM,label:e.label,start:e.start,end:new Date(new Date(e.start).getTime()+e.durationSeconds*1000).toISOString()});
  return entities;
}
