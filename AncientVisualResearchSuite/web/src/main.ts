import * as THREE from 'three';
import { HistoricalScene, simulate } from './historical_scene';

const scene: HistoricalScene = {
  id:'demo-event', title:'Evidence-led historical scene', start:'1900-01-01T20:00:00Z',
  latitude:24.0, longitude:45.0, elevationM:500, evidence:[], characters:[], events:[],
  environment:{temperatureC:22, visibilityKm:20}, sky:{season:'winter', night:true}, tensor128:Array(128).fill(0)
};

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
const world = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, .1, 10000);
camera.position.set(4,4,8); camera.lookAt(0,0,0);
world.add(new THREE.HemisphereLight(0xffffff,0x334455,1.4));
const ground = new THREE.GridHelper(30,30); world.add(ground);

export function loadScene(s:HistoricalScene){
  scene.characters=s.characters; scene.events=s.events; scene.evidence=s.evidence;
  console.info('historical scene', s.title, 'frames', simulate(s, 10).length);
}

addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
(function loop(){requestAnimationFrame(loop);renderer.render(world,camera)})();
