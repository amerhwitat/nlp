/* Adapter for a locally vendored SAE/vAmigaWeb-compatible browser engine. */
(function(){
  const A = window.ChimeraAmiga = {
    engine: null, running:false, files:{},
    log(message){ const el=document.getElementById('log'); if(el){ el.textContent += `[${new Date().toLocaleTimeString()}] ${message}\n`; el.scrollTop=el.scrollHeight; } },
    async load(){
      const candidates=['vendor/sae/index.js','vendor/vamigaweb/vAmigaWeb_player.js'];
      for(const src of candidates){
        try { await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=reject;document.head.appendChild(s)}); this.log(`Loaded engine candidate: ${src}`); return true; } catch(e) {}
      }
      this.log('No local emulator bundle detected. The UI is ready for the vendored engine.');
      return false;
    },
    async start(canvas){
      this.running=true;
      /* SAE exposes ScriptedAmigaEmulator in its browser build. Keep this adapter
         intentionally tolerant because upstream builds have changed APIs. */
      if(window.ScriptedAmigaEmulator){
        try{
          this.engine=new window.ScriptedAmigaEmulator();
          if(typeof this.engine.start==='function') this.engine.start();
          this.log('SAE engine started.');
          return true;
        }catch(e){this.log('Engine start error: '+e.message);}
      }
      this.log('Adapter running in integration mode; no emulator core is installed yet.');
      return false;
    },
    pause(){ if(this.engine?.pause) this.engine.pause(); this.running=false; this.log('Paused.'); },
    reset(){ if(this.engine?.reset) this.engine.reset(); this.log('Reset requested.'); },
    async file(name,file){ this.files[name]=file; this.log(`${name}: ${file.name} (${file.size} bytes) selected locally.`); },
    fullscreen(){ document.getElementById('screenHost')?.requestFullscreen?.(); }
  };
})();
