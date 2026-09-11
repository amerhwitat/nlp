from __future__ import annotations
import math, json

class TinyRNN:
    """Small dependency-free recurrent sequence model for build-step hints."""
    def __init__(self, hidden=8, seed=7):
        self.hidden=hidden; self.wxh=[0.05*(i+1) for i in range(hidden)]; self.whh=[0.02*(i+1) for i in range(hidden)]; self.why=[0.03*(i+1) for i in range(hidden)]; self.bh=[0.0]*hidden; self.by=0.0
    def _step(self,x,h): return [math.tanh(x*self.wxh[i]+h[i]*self.whh[i]+self.bh[i]) for i in range(self.hidden)]
    def score(self, sequence):
        h=[0.0]*self.hidden
        for x in sequence: h=self._step(float(x),h)
        return 1/(1+math.exp(-sum(self.why[i]*h[i] for i in range(self.hidden))-self.by))
    def train(self,sequences,labels,epochs=10,lr=0.01):
        # Lightweight deterministic online update; suitable for small build traces.
        for _ in range(max(1,epochs)):
            for seq,y in zip(sequences,labels):
                pred=self.score(seq); err=float(y)-pred; h=[0.0]*self.hidden
                for x in seq: h=self._step(float(x),h)
                for i in range(self.hidden): self.why[i]+=lr*err*h[i]
                self.by-=lr*err
        return self
    def predict_next(self, steps): return self.score([float(x) for x in steps])

class NeuralEngine:
    def __init__(self): self.rnn=TinyRNN(); self.backend='builtin-rnn'
    def train_from_build_sequences(self,sequences,labels=None):
        labels=labels or [1]*len(sequences); self.rnn.train(sequences,labels); return {'backend':self.backend,'samples':len(sequences)}
    def recommend(self,features):
        score=self.rnn.predict_next(features); return {'confidence':round(score,4),'mode':'recommendation','authorization_required':True}
    def optional_torch_info(self):
        try:
            import torch
            return {'available':True,'version':torch.__version__,'rnn':'torch.nn.RNN','transformer':'torch.nn.Transformer'}
        except Exception as exc: return {'available':False,'reason':str(exc)}
    def save(self,path):
        json.dump({'backend':self.backend,'hidden':self.rnn.hidden,'wxh':self.rnn.wxh,'whh':self.rnn.whh,'why':self.rnn.why,'bh':self.rnn.bh,'by':self.rnn.by},open(path,'w',encoding='utf-8'),indent=2)
