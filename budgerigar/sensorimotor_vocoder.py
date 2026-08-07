from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class SensorimotorVocoderConfig:
    sample_rate:int=16000
    tick_samples:int=160
    hidden_dim:int=128
    source_dim:int=3
    articulation_dim:int=32
    controller_layers:int=2
    filter_taps:int=63


def create_sensorimotor_vocoder(config=SensorimotorVocoderConfig()):
    torch,nn,functional=require_torch()
    class SensorimotorVocoder(nn.Module):
        def __init__(self):
            super().__init__();self.config=config
            self.hearing=nn.Sequential(nn.Linear(config.tick_samples,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.analysis=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=2,batch_first=True)
            self.source_head=nn.Linear(config.hidden_dim,config.source_dim);self.articulation_head=nn.Linear(config.hidden_dim,config.articulation_dim)
            self.articulation_projection=nn.Sequential(nn.Linear(config.articulation_dim,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU())
            self.motor_controller=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.controller_layers,batch_first=True);self.filter_kernel=nn.Linear(config.hidden_dim,config.filter_taps);self.filter_gain=nn.Linear(config.hidden_dim,1);nn.init.zeros_(self.filter_kernel.weight);nn.init.zeros_(self.filter_kernel.bias)
        def analyze(self,ticks,state=None):
            encoded=self.hearing(ticks);encoded,state=self.analysis(encoded,state);return self.source_head(encoded),self.articulation_head(encoded).tanh(),state
        def source_parameters(self,source):
            voiced=source[...,2].sigmoid();return {"drive":source[...,0].sigmoid(),"f0_hz":60+340*source[...,1].sigmoid(),"voiced":voiced,"breath":1-voiced}
        def render(self,source,articulation,state=None,phase=None):
            parameters=self.source_parameters(source);batch,ticks,_=source.shape
            if phase is None:pitch_phase=torch.zeros(batch,device=source.device,dtype=source.dtype);sample_offset=0
            else:pitch_phase,sample_offset=phase
            if state is None:motor_state=None;excitation_cache=torch.zeros(batch,config.filter_taps-1,device=source.device,dtype=source.dtype)
            else:motor_state,excitation_cache=state
            positions=torch.arange(config.tick_samples,device=source.device,dtype=source.dtype).view(1,1,-1);conditioning=self.articulation_projection(articulation);controlled,motor_state=self.motor_controller(conditioning,motor_state);raw_kernel=self.filter_kernel(controlled).tanh();delta=torch.zeros_like(raw_kernel);delta[...,-1]=1;kernel=delta+raw_kernel;kernel=kernel/kernel.abs().sum(-1,keepdim=True).clamp_min(1e-4);gain=.25+3.75*self.filter_gain(controlled).sigmoid()
            omega=2*torch.pi*parameters["f0_hz"]/config.sample_rate;starts=pitch_phase.unsqueeze(-1)+torch.cat([torch.zeros(batch,1,device=source.device,dtype=source.dtype),torch.cumsum(omega[:,:-1]*config.tick_samples,-1)],-1);angles=starts.unsqueeze(-1)+omega.unsqueeze(-1)*positions;harmonic=(torch.sin(angles)+.35*torch.sin(2*angles)+.15*torch.sin(3*angles))/1.5;absolute=torch.arange(sample_offset,sample_offset+ticks*config.tick_samples,device=source.device,dtype=source.dtype);hashed=torch.sin(absolute*12.9898)*43758.5453;noise=((hashed-torch.floor(hashed))*2-1).view(1,ticks,config.tick_samples);voice=parameters["voiced"].unsqueeze(-1);excitation=(voice*harmonic+(1-voice)*noise).flatten(1);joined=torch.cat([excitation_cache,excitation],-1);windows=joined.unfold(-1,config.filter_taps,1);sample_kernels=kernel.repeat_interleave(config.tick_samples,1);filtered=(windows*sample_kernels).sum(-1).view(batch,ticks,config.tick_samples);drive=parameters["drive"].unsqueeze(-1);wave=(1.5*drive*gain*filtered).tanh();pitch_phase=(pitch_phase+(omega*config.tick_samples).sum(-1)).remainder(2*torch.pi);phase=(pitch_phase,sample_offset+ticks*config.tick_samples);state=(motor_state,joined[:,-(config.filter_taps-1):])
            return wave,state,phase,parameters
        def forward(self,ticks,analysis_state=None,motor_state=None,phase=None):
            source,articulation,analysis_state=self.analyze(ticks,analysis_state);wave,motor_state,phase,parameters=self.render(source,articulation,motor_state,phase);return wave,{"source_controls":source,"articulation_controls":articulation,"source_parameters":parameters,"analysis_state":analysis_state,"motor_state":motor_state,"phase":phase}
        def export_config(self):return asdict(config)
    return SensorimotorVocoder()
