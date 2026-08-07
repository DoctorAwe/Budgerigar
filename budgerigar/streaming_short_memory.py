from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch

@dataclass(frozen=True)
class ShortMemoryConfig:
    sample_rate:int=16000
    tick_samples:int=80
    hidden_dim:int=160
    recurrent_layers:int=3
    output_classes:int=11

def create_short_memory_model(config=ShortMemoryConfig()):
    torch,nn,_=require_torch()
    class ChunkEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.network=nn.Sequential(nn.Conv1d(1,32,9,4,4),nn.SiLU(),nn.Conv1d(32,64,7,4,3),nn.SiLU(),nn.Conv1d(64,96,5,2,2),nn.SiLU(),nn.Flatten(),nn.LazyLinear(config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU())
        def forward(self,x):return self.network(x.unsqueeze(1))
    class StreamingShortMemory(nn.Module):
        def __init__(self):
            super().__init__();self.config=config;self.encoder=ChunkEncoder();self.memory=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.norm=nn.LayerNorm(config.hidden_dim);self.output=nn.Linear(config.hidden_dim,config.output_classes)
        def initial_state(self,batch,device=None):
            device=device or next(self.parameters()).device;return torch.zeros(config.recurrent_layers,batch,config.hidden_dim,device=device)
        def stream_step(self,tick,state=None):
            if tick.ndim!=2 or tick.shape[-1]!=config.tick_samples:raise ValueError(f"expected [batch,{config.tick_samples}] tick")
            if state is None:state=self.initial_state(len(tick),tick.device)
            value=self.encoder(tick).unsqueeze(1);value,state=self.memory(value,state);value=self.norm(value[:,0]);logits=self.output(value);return logits,state,1-logits.softmax(-1)[:,0]
        def forward(self,ticks,state=None):
            batch,length,samples=ticks.shape;encoded=self.encoder(ticks.reshape(batch*length,samples)).view(batch,length,-1);value,state=self.memory(encoded,state);logits=self.output(self.norm(value));return logits,state,1-logits.softmax(-1)[...,0]
        def export_config(self):return asdict(config)
    return StreamingShortMemory()

class WaveformTickBuffer:
    """Accept arbitrary waveform chunks and emit complete 5 ms ticks."""
    def __init__(self,tick_samples=80):self.tick_samples=tick_samples;self.pending=None
    def push(self,waveform):
        torch,_,_=require_torch();waveform=waveform.flatten();self.pending=waveform if self.pending is None else torch.cat([self.pending,waveform]);count=len(self.pending)//self.tick_samples
        if not count:return waveform.new_empty((0,self.tick_samples))
        ready=self.pending[:count*self.tick_samples].view(count,self.tick_samples);self.pending=self.pending[count*self.tick_samples:];return ready
    def reset(self):self.pending=None
