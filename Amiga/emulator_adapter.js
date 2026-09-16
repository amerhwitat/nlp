/* Adapter for the locally vendored SAE browser engine. */
(function(){
  const CORE=['prototypes.js','utils.js','dms.js','config.js','roms.js','memory.js','autoconf.js','expansion.js','events.js','gayle.js','ide.js','filesys.js','hardfile.js','dongle.js','input.js','serpar.js','custom.js','blitter.js','copper.js','playfield.js','video.js','audio.js','cia.js','disk.js','rtc.js','m68k.js','cpu.js','amiga.js'];
  const A=window.ChimeraAmiga={
    engine:null,cfg:null,running:false,files:{},
    log(m){const e=document.getElementById('log');if(e){e.textContent+=`[${new Date().toLocaleTimeString()}] ${m}\n`;e.scrollTop=e.scrollHeight;}},
    loadScript(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(new Error(src));document.head.appendChild(s);});},
    async load(){try{for(const f of CORE)await this.loadScript('vendor/sae/sae/'+f);if(typeof window.ScriptedAmigaEmulator!=='function')throw new Error('ScriptedAmigaEmulator not exported');this.engine=new window.ScriptedAmigaEmulator();this.cfg=this.engine.getConfig();this.cfg.video.id='screenHost';this.cfg.video.enabled=true;this.log('SAE core loaded locally and configuration initialized.');return true;}catch(e){this.log('SAE core not loaded: '+e.message);return false;}},
    readFile(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsBinaryString(file);});},
    async start(){if(!this.engine||!this.cfg){this.log('No emulator core. Run fetch_upstream.sh/ps1 first.');return false;}try{const model=document.getElementById('model').value;if(this.cfg.model)this.cfg.model=model;if(this.cfg.chipset)this.cfg.chipset.ntsc=document.getElementById('video').value==='ntsc';if(!this.cfg.memory.rom.size){this.log('Select a Kickstart ROM before starting.');return false;}const err=this.engine.start();if(typeof err!=='undefined'&&err!==0){this.log('SAE start error: '+err);return false;}this.running=true;document.getElementById('placeholder').style.display='none';this.log('SAE emulator started.');return true;}catch(e){this.log('Engine start error: '+e.message);return false;}},
    pause(){if(this.engine?.pause)this.engine.pause();this.running=false;this.log('Paused.');},
    reset(){if(this.engine?.reset)this.engine.reset();this.log('Reset requested.');},
    async file(name,file){this.files[name]=file;this.log(`${name}: ${file.name} (${file.size} bytes) selected.`);if(!this.cfg)return;const data=await this.readFile(file);if(name==='ROM'){this.cfg.memory.rom.name=file.name;this.cfg.memory.rom.data=data;this.cfg.memory.rom.size=file.size;}if(name==='DF0'||name==='DF1'){const n=name==='DF0'?0:1;const f=this.cfg.floppy.drive[n].file;f.name=file.name;f.data=data;f.size=file.size;if(this.running&&this.engine.insert)this.engine.insert(n);}if(name==='HDF')this.log('HDF selected; configure hardfile mapping in the emulator core before start.');},
    fullscreen(){document.getElementById('screenHost')?.requestFullscreen?.()}
  };
})();
