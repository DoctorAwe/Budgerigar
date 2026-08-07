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
    local_layers:int=2
    local_kernel:int=7
    reconstruction_slots:int=4
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
    class CausalLocalEncoder(nn.Module):
        """Build phonetic-scale evidence before it is written into long-lived memory."""
        def __init__(self):
            super().__init__();self.convolutions=nn.ModuleList([nn.Conv1d(config.hidden_dim,config.hidden_dim,config.local_kernel) for _ in range(config.local_layers)]);self.norms=nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.local_layers)])
        def initial_state(self,batch,device,dtype):return tuple(torch.zeros(batch,config.hidden_dim,config.local_kernel-1,device=device,dtype=dtype) for _ in self.convolutions)
        def forward(self,features,state=None):
            if state is None:state=self.initial_state(len(features),features.device,features.dtype)
            x=features.transpose(1,2);new_state=[]
            for convolution,norm,cache in zip(self.convolutions,self.norms,state):
                joined=torch.cat([cache,x],-1);filtered=convolution(joined).transpose(1,2);x=torch.nn.functional.silu(norm(filtered)+x.transpose(1,2)).transpose(1,2);new_state.append(joined[:,:,-(config.local_kernel-1):])
            return x.transpose(1,2),tuple(new_state)
    class StreamingShortMemory(nn.Module):
        def __init__(self):
            super().__init__();self.config=config;self.cochlea=CochlearFrontEnd();self.local_encoder=CausalLocalEncoder()
            self.initial_tokens=nn.Parameter(torch.zeros(config.token_layers,config.hidden_dim));self.level_embedding=nn.Parameter(torch.randn(config.token_layers,config.hidden_dim)*.02)
            self.token_norms=nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.token_layers)]);self.token_attentions=nn.ModuleList([nn.MultiheadAttention(config.hidden_dim,config.attention_heads,batch_first=True) for _ in range(config.token_layers)])
            self.update_gates=nn.ModuleList([nn.Linear(config.hidden_dim*4,config.hidden_dim) for _ in range(config.token_layers)]);self.update_candidates=nn.ModuleList([nn.Linear(config.hidden_dim*4,config.hidden_dim) for _ in range(config.token_layers)])
            self.refinements=nn.ModuleList([nn.Sequential(nn.LayerNorm(config.hidden_dim),nn.Linear(config.hidden_dim,config.hidden_dim*2),nn.SiLU(),nn.Linear(config.hidden_dim*2,config.hidden_dim)) for _ in range(config.token_layers)])
            self.content_query=nn.Parameter(torch.randn(1,config.hidden_dim)*.02);self.reconstruction_queries=nn.Parameter(torch.randn(config.reconstruction_slots,config.hidden_dim)*.02)
            self.decoder_attention=nn.MultiheadAttention(config.hidden_dim,config.attention_heads,batch_first=True);self.decoder_norm=nn.LayerNorm(config.hidden_dim);self.decoder_refinement=nn.Sequential(nn.Linear(config.hidden_dim,config.hidden_dim*2),nn.SiLU(),nn.Linear(config.hidden_dim*2,config.hidden_dim))
            self.content=nn.Linear(config.hidden_dim,10);self.reconstruction=nn.Linear(config.hidden_dim,config.hidden_dim);self.emission=nn.Linear(config.hidden_dim,1)
            for gate,refinement in zip(self.update_gates,self.refinements):
                nn.init.constant_(gate.bias,-2.5);nn.init.zeros_(refinement[-1].weight);nn.init.zeros_(refinement[-1].bias)
        def initial_state(self,batch,device=None,dtype=None):
            parameter=next(self.parameters());device=device or parameter.device;dtype=dtype or parameter.dtype
            tokens=self.initial_tokens.to(device=device,dtype=dtype).unsqueeze(0).expand(batch,-1,-1).clone();return tokens,self.cochlea.initial_state(batch,device,dtype),self.local_encoder.initial_state(batch,device,dtype)
        def _update_tokens(self,evidence,tokens):
            old=tokens;updated=[]
            for level,(norm,attention,gate_layer,candidate_layer,refinement) in enumerate(zip(self.token_norms,self.token_attentions,self.update_gates,self.update_candidates,self.refinements)):
                query=norm(old[:,level]+self.level_embedding[level]).unsqueeze(1);sources=norm(old[:,:level+1]+self.level_embedding[:level+1].unsqueeze(0));context=attention(query,sources,sources,need_weights=False)[0].squeeze(1)
                lower=evidence if level==0 else updated[-1];combined=torch.cat([old[:,level],context,lower,evidence],-1);gate=gate_layer(combined).sigmoid();candidate=candidate_layer(combined).tanh();value=old[:,level]+gate*(candidate-old[:,level]);updated.append(value+refinement(value))
            return torch.stack(updated,1)
        def decode_memory(self,tokens):
            queries=torch.cat([self.content_query,self.reconstruction_queries],0).unsqueeze(0).expand(len(tokens),-1,-1);decoded=self.decoder_attention(queries,tokens,tokens,need_weights=False)[0];decoded=self.decoder_norm(queries+decoded);decoded=self.decoder_norm(decoded+self.decoder_refinement(decoded));read=decoded[:,0]
            return self.content(read),self.reconstruction(decoded[:,1:]),read
        def _outputs(self,tokens):
            content,reconstruction,read=self.decode_memory(tokens);emit=self.emission(read).sigmoid().squeeze(-1);digit=content.softmax(-1);probability=torch.cat([(1-emit).unsqueeze(-1),emit.unsqueeze(-1)*digit],-1);logits=probability.clamp_min(1e-7).log()
            return logits,{"emission_probability":emit,"content_logits":content,"reconstruction":reconstruction}
        def stream_step(self,tick,state=None):
            if tick.ndim!=2 or tick.shape[-1]!=config.tick_samples:raise ValueError(f"expected [batch,{config.tick_samples}] tick")
            if state is None:state=self.initial_state(len(tick),tick.device,tick.dtype)
            tokens,cochlear_state,local_state=state;encoded,cochlear_state=self.cochlea(tick,cochlear_state);encoded,local_state=self.local_encoder(encoded,local_state);tokens=self._update_tokens(encoded[:,0],tokens);logits,diagnostics=self._outputs(tokens);diagnostics["token_states"]=tokens;diagnostics["encoded_features"]=encoded;return logits,(tokens,cochlear_state,local_state),diagnostics
        def forward(self,ticks,state=None):
            batch,length,samples=ticks.shape
            if state is None:state=self.initial_state(batch,ticks.device,ticks.dtype)
            tokens,cochlear_state,local_state=state;encoded,cochlear_state=self.cochlea(ticks.reshape(batch,length*samples),cochlear_state);encoded,local_state=self.local_encoder(encoded,local_state);token_history=[]
            for tick in range(length):
                tokens=self._update_tokens(encoded[:,tick],tokens);token_history.append(tokens)
            token_history=torch.stack(token_history,1);flat=token_history.reshape(batch*length,config.token_layers,config.hidden_dim);flat_logits,flat_diagnostics=self._outputs(flat);logits=flat_logits.view(batch,length,-1);diagnostics={"emission_probability":flat_diagnostics["emission_probability"].view(batch,length),"content_logits":flat_diagnostics["content_logits"].view(batch,length,-1),"token_states":tokens,"token_history":token_history,"encoded_features":encoded};return logits,(tokens,cochlear_state,local_state),diagnostics
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
