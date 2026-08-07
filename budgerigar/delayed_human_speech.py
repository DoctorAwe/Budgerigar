from __future__ import annotations
from dataclasses import asdict,dataclass
from .human_speech_model import HumanSpeechConfig,create_human_speech_model
from .neural_echo import require_torch


@dataclass(frozen=True)
class DelayedHumanSpeechConfig:
    human:HumanSpeechConfig=HumanSpeechConfig()
    memory_dim:int=192
    memory_layers:int=2


def create_delayed_human_speech(config=DelayedHumanSpeechConfig()):
    torch,nn,_=require_torch()
    class DelayedHumanSpeech(nn.Module):
        def __init__(self):
            super().__init__();self.config=config;self.human=create_human_speech_model(config.human);self.memory_input=nn.Sequential(nn.Linear(config.human.latent_dim,config.memory_dim),nn.LayerNorm(config.memory_dim),nn.SiLU());self.memory=nn.GRU(config.memory_dim,config.memory_dim,config.memory_layers,batch_first=True);self.recall=nn.Sequential(nn.Linear(config.memory_dim,config.memory_dim),nn.SiLU(),nn.Linear(config.memory_dim,config.human.latent_dim),nn.Tanh())
        def forward(self,ticks,state=None):
            auditory_state=memory_state=motor_state=phase=None
            if state is not None:auditory_state,memory_state,motor_state,phase=state
            sensation,auditory_state=self.human.encode(ticks,auditory_state);remembered,memory_state=self.memory(self.memory_input(sensation),memory_state);recall=self.recall(remembered);waveform,motor_state,phase,diagnostics=self.human.synthesize(recall,motor_state,phase);diagnostics.update({"sensation":sensation,"memory_trajectory":remembered,"recall":recall});return waveform,(auditory_state,memory_state,motor_state,phase),diagnostics
        def stream_step(self,tick,state=None):
            waveform,state,diagnostics=self.forward(tick.unsqueeze(1),state);return waveform[:,0],state,diagnostics
        def export_config(self):return {"human":asdict(config.human),"memory_dim":config.memory_dim,"memory_layers":config.memory_layers}
    return DelayedHumanSpeech()
