from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch

@dataclass(frozen=True)
class ShortMemoryConfig:
    sample_rate:int=16000
    tick_samples:int=160
    cochlear_bands:int=32
    cochlear_kernel:int=129
    subframes:int=4
    hidden_dim:int=160
    recurrent_layers:int=3
    output_classes:int=11

def create_short_memory_model(config=ShortMemoryConfig()):
    torch,nn,_=require_torch()
    if config.tick_samples%config.subframes:raise ValueError("tick_samples must divide evenly into subframes")
    class CochlearFrontEnd(nn.Module):
        """Continuous causal filter bank with rectification, compression and adaptation."""
        def __init__(self):
            super().__init__();self.filters=nn.Conv1d(1,config.cochlear_bands,config.cochlear_kernel,bias=False)
            half=(config.cochlear_kernel-1)//2;t=torch.arange(-half,half+1,dtype=torch.float32)
            centers=torch.logspace(torch.log10(torch.tensor(80.)),torch.log10(torch.tensor(config.sample_rate*.45)),config.cochlear_bands);impulse=[]
            for center in centers:
                low=(center/1.22/config.sample_rate).clamp_min(30/config.sample_rate);high=(center*1.22/config.sample_rate).clamp_max(.49)
                band=(2*high*torch.sinc(2*high*t)-2*low*torch.sinc(2*low*t))*torch.hamming_window(config.cochlear_kernel);impulse.append(band/band.norm().clamp_min(1e-6))
            with torch.no_grad():self.filters.weight.copy_(torch.stack(impulse).unsqueeze(1))
            self.projection=nn.Sequential(nn.Linear(config.cochlear_bands*config.subframes*2,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU())
        def initial_state(self,batch,device,dtype):return torch.zeros(batch,1,config.cochlear_kernel-1,device=device,dtype=dtype),torch.zeros(batch,config.cochlear_bands,device=device,dtype=dtype)
        def forward(self,waveform,state=None):
            if state is None:state=self.initial_state(len(waveform),waveform.device,waveform.dtype)
            cache,previous=state;joined=torch.cat([cache,waveform.unsqueeze(1)],-1);response=self.filters(joined).abs().clamp_min(1e-6).pow(.3)
            ticks=waveform.shape[-1]//config.tick_samples;per_subframe=config.tick_samples//config.subframes
            activity=response.view(len(waveform),config.cochlear_bands,ticks,config.subframes,per_subframe).mean(-1);flat=activity.flatten(2)
            delta=(flat-torch.cat([previous.unsqueeze(-1),flat[:,:,:-1]],-1)).view_as(activity)
            features=torch.cat([activity,delta],1).permute(0,2,1,3).flatten(2);new_state=joined[:,:,-(config.cochlear_kernel-1):],flat[:,:,-1]
            return self.projection(features),new_state
    class StreamingShortMemory(nn.Module):
        def __init__(self):
            super().__init__();self.config=config;self.cochlea=CochlearFrontEnd();self.memory=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.norm=nn.LayerNorm(config.hidden_dim);self.output=nn.Linear(config.hidden_dim,config.output_classes)
        def initial_state(self,batch,device=None,dtype=None):
            parameter=next(self.parameters());device=device or parameter.device;dtype=dtype or parameter.dtype
            return torch.zeros(config.recurrent_layers,batch,config.hidden_dim,device=device,dtype=dtype),self.cochlea.initial_state(batch,device,dtype)
        def stream_step(self,tick,state=None):
            if tick.ndim!=2 or tick.shape[-1]!=config.tick_samples:raise ValueError(f"expected [batch,{config.tick_samples}] tick")
            if state is None:state=self.initial_state(len(tick),tick.device,tick.dtype)
            memory_state,cochlear_state=state;encoded,cochlear_state=self.cochlea(tick,cochlear_state);value,memory_state=self.memory(encoded,memory_state);logits=self.output(self.norm(value[:,0]));return logits,(memory_state,cochlear_state),1-logits.softmax(-1)[:,0]
        def forward(self,ticks,state=None):
            batch,length,samples=ticks.shape
            if state is None:state=self.initial_state(batch,ticks.device,ticks.dtype)
            memory_state,cochlear_state=state;encoded,cochlear_state=self.cochlea(ticks.reshape(batch,length*samples),cochlear_state);value,memory_state=self.memory(encoded,memory_state);logits=self.output(self.norm(value));return logits,(memory_state,cochlear_state),1-logits.softmax(-1)[...,0]
        def export_config(self):return asdict(config)
    return StreamingShortMemory()

class WaveformTickBuffer:
    """Accept arbitrary waveform chunks and emit complete high-level ticks."""
    def __init__(self,tick_samples=160):self.tick_samples=tick_samples;self.pending=None
    def push(self,waveform):
        torch,_,_=require_torch();waveform=waveform.flatten();self.pending=waveform if self.pending is None else torch.cat([self.pending,waveform]);count=len(self.pending)//self.tick_samples
        if not count:return waveform.new_empty((0,self.tick_samples))
        ready=self.pending[:count*self.tick_samples].view(count,self.tick_samples);self.pending=self.pending[count*self.tick_samples:];return ready
    def reset(self):self.pending=None
