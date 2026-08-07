from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class SensorimotorVocoderConfig:
    sample_rate:int=16000
    tick_samples:int=160
    hidden_dim:int=128
    source_dim:int=8
    articulation_dim:int=32
    controller_layers:int=2


def create_sensorimotor_vocoder(config=SensorimotorVocoderConfig()):
    torch,nn,functional=require_torch()
    class SensorimotorVocoder(nn.Module):
        def __init__(self):
            super().__init__();self.config=config
            self.hearing=nn.Sequential(nn.Linear(config.tick_samples,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.analysis=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=2,batch_first=True)
            self.source_head=nn.Linear(config.hidden_dim,config.source_dim);self.articulation_head=nn.Linear(config.hidden_dim,config.articulation_dim)
            self.source_projection=nn.Sequential(nn.Linear(config.source_dim,config.hidden_dim),nn.SiLU());self.articulation_projection=nn.Sequential(nn.Linear(config.articulation_dim,config.hidden_dim),nn.SiLU())
            self.motor_controller=nn.GRU(config.hidden_dim*2+1,config.hidden_dim,num_layers=config.controller_layers,batch_first=True);self.residual=nn.Linear(config.hidden_dim,config.tick_samples)
        def analyze(self,ticks,state=None):
            encoded=self.hearing(ticks);encoded,state=self.analysis(encoded,state);return self.source_head(encoded),self.articulation_head(encoded).tanh(),state
        def source_parameters(self,source):
            return {"drive":source[...,0].sigmoid(),"f0_hz":60+340*source[...,1].sigmoid(),"voiced":source[...,2].sigmoid(),"breath":source[...,3].sigmoid()}
        def render(self,source,articulation,state=None,phase=None):
            parameters=self.source_parameters(source);batch,ticks,_=source.shape;phase=torch.zeros(batch,device=source.device,dtype=source.dtype) if phase is None else phase;positions=torch.arange(config.tick_samples,device=source.device,dtype=source.dtype).view(1,1,-1)
            conditioning=torch.cat([self.source_projection(source),self.articulation_projection(articulation),parameters["drive"].unsqueeze(-1)],-1);controlled,state=self.motor_controller(conditioning,state);residual=self.residual(controlled).tanh()
            omega=2*torch.pi*parameters["f0_hz"]/config.sample_rate;starts=phase.unsqueeze(-1)+torch.cat([torch.zeros(batch,1,device=source.device,dtype=source.dtype),torch.cumsum(omega[:,:-1]*config.tick_samples,-1)],-1);angles=starts.unsqueeze(-1)+omega.unsqueeze(-1)*positions;harmonic=(torch.sin(angles)+.35*torch.sin(2*angles)+.15*torch.sin(3*angles))/1.5;voice=parameters["voiced"].unsqueeze(-1);breath=parameters["breath"].unsqueeze(-1);drive=parameters["drive"].unsqueeze(-1);wave=(drive*(voice*harmonic+(1-voice+.35*breath)*residual)).tanh();phase=(phase+(omega*config.tick_samples).sum(-1)).remainder(2*torch.pi)
            return wave,state,phase,parameters
        def forward(self,ticks,analysis_state=None,motor_state=None,phase=None):
            source,articulation,analysis_state=self.analyze(ticks,analysis_state);wave,motor_state,phase,parameters=self.render(source,articulation,motor_state,phase);return wave,{"source_controls":source,"articulation_controls":articulation,"source_parameters":parameters,"analysis_state":analysis_state,"motor_state":motor_state,"phase":phase}
        def export_config(self):return asdict(config)
    return SensorimotorVocoder()
