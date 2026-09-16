/* Adapter for the locally vendored SAE browser engine. */
(function(){
  const CORE = ['prototypes.js','utils.js','dms.js','config.js','roms.js','memory.js','autoconf.js','expansion.js','events.js','gayle.js','ide.js','filesys.js','hardfile.js','dongle.js','input.js','serpar.js','custom.js','blitter.js','copper.js','playfield.js','video.js','audio.js','cia.js','disk.js','rtc.js','m68k.js','cpu.js','amiga.js'];
  const A = window.ChimeraAmiga = {
    engine:null, cfg:null, running:false, files:{},
    log(message){const el=document.getElementById('log');if(el){el.textContent+=`[${new Date().toLocaleTimeString()}] ${message}\n`;el.scrollTop=el.scrollHeight;}},
    loadScript(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(new Error(src));document.head.appendChild(s);});},
    async load(){
      const base='vendor/sae/sae/';
      try{
        for(const file of CORE) await this.loadScript(base+file);
        if(typeof window.ScriptedAmigaEmulator!=='function') throw new Error('ScriptedAmigaEmulator not exported');
        this.log('SAE core loaded locally.');
        return true;
      }catch(e){this.log('SAE core not loaded: '+e.message);return false;}
    },
    readFile(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsBinaryString(file);});},
    async start(){
      if(typeof window.ScriptedAmigaEmulator!=='function'){this.log('No local emulator core. Run fetch_upstream.sh/ps1 first.');return false;}
      try{
        this.engine=new window.ScriptedAmigaEmulator();
        this.cfg=this.engine.getConfig();
        this.cfg.video.id='screenHost';
        this.cfg.video.enabled=true;
        const model=document.getElementById('model').value;
        if(this.cfg.model) this.cfg.model=model;
        if(document.getElementById('video').value==='ntsc' && this.cfg.chipset) this.cfg.chipset.ntsc=true;
        if(!this.cfg.memory.rom.size){this.log('Select a Kickstart ROM before starting.');return false;}
        const err=this.engine.start();
        if(typeof err!=='undefined' && err!==0){this.log('SAE start error: '+err);return false;}
        this.running=true;document.getElementById('placeholder').style.display='none';this.log('SAE emulator started.');return true;
      }catch(e){this.log('Engine start error: '+e.message);return false;}
    },
    pause(){if(this.engine?.pause) this.engine.pause();this.running=false;this.log('Paused.');},
    reset(){if(this.engine?.reset) this.engine.reset();this.log('Reset requested.');},
    async file(name,file){
      this.files[name]=file; this.log(`${name}: ${file.name} (${file.size} bytes) selected.`);
      if(!this.cfg)return;
      const data=await this.readFile(file);
      if(name==='ROM'){this.cfg.memory.rom.name=file.name;this.cfg.memory.rom.data=data;this.cfg.memory.rom.size=file.size;if(window.crc32)this.cfg.memory.rom.crc32=window.crc32(data);}
      if(name==='DF0'||name==='DF1'){const n=name==='DF0'?0:1;const f=this.cfg.floppy.drive[n].file;f.name=file.name;f.data=data;f.size=file.size;if(window.crc32)f.crc32=window.crc32(data);if(this.running&&this.engine.insert)this.engine.insert(n);}
      if(name==='HDF')this.log('HDF selected; hardfile mapping depends on the selected SAE configuration.');
    },
    fullscreen(){document.getElementById('screenHost')?.requestFullscreen?.();}
  };
})();
