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
    hidden_dim:int=96
    token_layers:int=4
    attention_heads:int=4
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
            super().__init__();self.config=config;self.cochlea=CochlearFrontEnd()
            self.initial_tokens=nn.Parameter(torch.zeros(config.token_layers,config.hidden_dim));self.level_embedding=nn.Parameter(torch.randn(config.token_layers,config.hidden_dim)*.02)
            self.token_norm=nn.LayerNorm(config.hidden_dim);self.token_attention=nn.MultiheadAttention(config.hidden_dim,config.attention_heads,batch_first=True)
            self.update_gate=nn.Linear(config.hidden_dim*3,config.hidden_dim);self.update_candidate=nn.Linear(config.hidden_dim*3,config.hidden_dim)
            self.refinement=nn.Sequential(nn.LayerNorm(config.hidden_dim),nn.Linear(config.hidden_dim,config.hidden_dim*2),nn.SiLU(),nn.Linear(config.hidden_dim*2,config.hidden_dim))
            self.read_score=nn.Linear(config.hidden_dim,1);self.output=nn.Linear(config.hidden_dim,config.output_classes);self.content=nn.Linear(config.hidden_dim,10)
            # Start as slow, nearly identity memory; learning may still open the gate.
            nn.init.constant_(self.update_gate.bias,-3.0)
            nn.init.zeros_(self.refinement[-1].weight);nn.init.zeros_(self.refinement[-1].bias)
        def initial_state(self,batch,device=None,dtype=None):
            parameter=next(self.parameters());device=device or parameter.device;dtype=dtype or parameter.dtype
            tokens=self.initial_tokens.to(device=device,dtype=dtype).unsqueeze(0).expand(batch,-1,-1).clone();return tokens,self.cochlea.initial_state(batch,device,dtype)
        def _update_tokens(self,evidence,tokens):
            represented=self.token_norm(tokens+self.level_embedding.unsqueeze(0));mask=torch.ones(config.token_layers,config.token_layers,dtype=torch.bool,device=tokens.device).triu(1)
            context=self.token_attention(represented,represented,represented,attn_mask=mask,need_weights=False)[0]
            acoustic=evidence.unsqueeze(1).expand(-1,config.token_layers,-1);combined=torch.cat([tokens,context,acoustic+self.level_embedding.unsqueeze(0)],-1)
            gate=self.update_gate(combined).sigmoid();candidate=self.update_candidate(combined).tanh();tokens=self.token_norm(tokens+gate*(candidate-tokens));return tokens+self.refinement(tokens)
        def _read(self,tokens):
            weight=self.read_score(tokens).squeeze(-1).softmax(-1);return (tokens*weight.unsqueeze(-1)).sum(1)
        def stream_step(self,tick,state=None):
            if tick.ndim!=2 or tick.shape[-1]!=config.tick_samples:raise ValueError(f"expected [batch,{config.tick_samples}] tick")
            if state is None:state=self.initial_state(len(tick),tick.device,tick.dtype)
            tokens,cochlear_state=state;encoded,cochlear_state=self.cochlea(tick,cochlear_state);tokens=self._update_tokens(encoded[:,0],tokens);value=self._read(tokens);logits=self.output(value);diagnostics={"emission_probability":1-logits.softmax(-1)[:,0],"content_logits":self.content(value),"token_states":tokens};return logits,(tokens,cochlear_state),diagnostics
        def forward(self,ticks,state=None):
            batch,length,samples=ticks.shape
            if state is None:state=self.initial_state(batch,ticks.device,ticks.dtype)
            tokens,cochlear_state=state;encoded,cochlear_state=self.cochlea(ticks.reshape(batch,length*samples),cochlear_state);logits=[];content=[]
            for tick in range(length):
                tokens=self._update_tokens(encoded[:,tick],tokens);value=self._read(tokens);logits.append(self.output(value));content.append(self.content(value))
            logits=torch.stack(logits,1);diagnostics={"emission_probability":1-logits.softmax(-1)[...,0],"content_logits":torch.stack(content,1),"token_states":tokens};return logits,(tokens,cochlear_state),diagnostics
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
