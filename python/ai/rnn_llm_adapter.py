"""Aurora-compatible RNN/LLM bridge for NLP tools.
Inference is provider-injected; no credentials or network access are embedded.
"""
class RNNLLMAdapter:
    def __init__(self, model=None):
        self.model=model
        self.events=[]
    def learn_event(self,event):
        self.events.append({k:event[k] for k in ("page","element","action","timestamp") if k in event})
        self.events=self.events[-2000:]
    def reply(self,prompt):
        if self.model is None:
            return {"text":"RNN/LLM provider unavailable; local browser learner may be used.","provider":"unavailable","confidence":0.0}
        return {"text":str(self.model(prompt)),"provider":"injected-model","confidence":1.0}
