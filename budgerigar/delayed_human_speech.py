from __future__ import annotations
from dataclasses import asdict,dataclass
from .human_speech_model import HumanSpeechConfig,create_human_speech_model
from .neural_echo import require_torch
from .streaming_short_memory import ShortMemoryConfig,create_short_memory_model


@dataclass(frozen=True)
class DelayedHumanSpeechConfig:
    human:HumanSpeechConfig=HumanSpeechConfig()
    semantic:ShortMemoryConfig=ShortMemoryConfig(hidden_dim=96,token_layers=4,acoustic_slots=0)


def create_delayed_human_speech(config=DelayedHumanSpeechConfig()):
    torch,nn,_=require_torch()
    if config.human.subframes!=config.semantic.subframes:raise ValueError("human and semantic subframes must match")
    class DelayedHumanSpeech(nn.Module):
        def __init__(self):
            super().__init__();self.config=config;self.human=create_human_speech_model(config.human);self.semantic=create_short_memory_model(config.semantic);self.sensation_adapter=nn.Sequential(nn.Linear(config.human.latent_dim*config.human.subframes,config.semantic.hidden_dim),nn.LayerNorm(config.semantic.hidden_dim),nn.SiLU());self.recall_projection=nn.Sequential(nn.Linear(config.semantic.hidden_dim,config.semantic.hidden_dim*2),nn.SiLU(),nn.Linear(config.semantic.hidden_dim*2,config.human.latent_dim*config.human.subframes),nn.Tanh())
        def remember(self,ticks,auditory_state=None,tokens=None):
            with torch.no_grad():sensation,auditory_state=self.human.encode(ticks,auditory_state)
            batch,length,_=ticks.shape;evidence=self.sensation_adapter(sensation.reshape(batch,length,-1));tokens=self.semantic.initial_tokens.unsqueeze(0).expand(batch,-1,-1).clone() if tokens is None else tokens;reads=[];token_history=[];content=[]
            for index in range(length):
                tokens=self.semantic._update_tokens(evidence[:,index],tokens);logits,_,read=self.semantic.decode_memory(tokens);reads.append(read);content.append(logits);token_history.append(tokens)
            reads=torch.stack(reads,1);diagnostics={"sensation":sensation,"semantic_evidence":evidence,"token_states":tokens,"token_history":torch.stack(token_history,1),"content_logits":torch.stack(content,1),"memory_reads":reads};return reads,(auditory_state,tokens),diagnostics
        def forward(self,ticks,state=None):
            auditory_state=tokens=motor_state=phase=None
            if state is not None:auditory_state,tokens,motor_state,phase=state
            reads,(auditory_state,tokens),diagnostics=self.remember(ticks,auditory_state,tokens);batch,length,_=ticks.shape;recall=self.recall_projection(reads).view(batch,length*config.human.subframes,config.human.latent_dim);waveform,motor_state,phase,synthesis=self.human.synthesize(recall,motor_state,phase);diagnostics.update(synthesis);diagnostics["recall"]=recall;return waveform,(auditory_state,tokens,motor_state,phase),diagnostics
        def stream_step(self,tick,state=None):
            waveform,state,diagnostics=self.forward(tick.unsqueeze(1),state);return waveform[:,0],state,diagnostics
        def export_config(self):return {"human":asdict(config.human),"semantic":asdict(config.semantic)}
    return DelayedHumanSpeech()
