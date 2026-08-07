from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class UnifiedStreamingConfig:
    sample_rate:int=16000
    tick_samples:int=160
    subframes:int=4
    hidden_dim:int=128
    latent_dim:int=40
    recurrent_layers:int=2
    synthesis_overlap:int=2


def create_unified_streaming_autoencoder(config=UnifiedStreamingConfig()):
    torch,nn,_=require_torch()
    if config.tick_samples%config.subframes:raise ValueError("tick_samples must divide evenly into subframes")
    if config.synthesis_overlap!=2:raise ValueError("current streaming overlap-add decoder requires synthesis_overlap=2")
    subframe_samples=config.tick_samples//config.subframes
    class UnifiedStreamingAutoencoder(nn.Module):
        def __init__(self):
            super().__init__();self.config=config
            frame_samples=subframe_samples*config.synthesis_overlap;self.hearing=nn.Sequential(nn.Linear(subframe_samples,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.encoder=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.local_latent=nn.Linear(config.hidden_dim,config.latent_dim);self.context_latent=nn.Linear(config.hidden_dim,config.latent_dim);self.latent_projection=nn.Sequential(nn.Linear(config.latent_dim,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.decoder=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.local_waveform=nn.Sequential(nn.Linear(config.latent_dim,config.hidden_dim),nn.SiLU(),nn.Linear(config.hidden_dim,frame_samples));self.context_waveform=nn.Linear(config.hidden_dim,frame_samples);self.source_probe=nn.Linear(config.hidden_dim,3)
        def encode(self,ticks,state=None):
            batch,length,samples=ticks.shape;subframes=ticks.view(batch,length*config.subframes,subframe_samples);heard=self.hearing(subframes);encoded,state=self.encoder(heard,state);latent=(self.local_latent(heard)+.25*self.context_latent(encoded)).tanh();return latent,state,encoded
        def decode(self,latent,state=None):
            if state is None:recurrent_state=None;overlap_tail=None
            elif isinstance(state,tuple):recurrent_state,overlap_tail=state
            else:recurrent_state,overlap_tail=state,None
            decoded,recurrent_state=self.decoder(self.latent_projection(latent),recurrent_state);frames=self.local_waveform(latent)+.25*self.context_waveform(decoded);head,tail=frames[...,:subframe_samples],frames[...,subframe_samples:];overlap_tail=head.new_zeros(len(head),subframe_samples) if overlap_tail is None else overlap_tail;previous=torch.cat([overlap_tail.unsqueeze(1),tail[:,:-1]],1);waveform=(head+previous).tanh();batch=latent.shape[0];return waveform.reshape(batch,-1,config.tick_samples),(recurrent_state,tail[:,-1])
        def probe_source(self,encoded):
            batch,subframes,_=encoded.shape;pooled=encoded.view(batch,-1,config.subframes,config.hidden_dim).mean(2);raw=self.source_probe(pooled);voiced=raw[...,2].sigmoid();return {"drive":raw[...,0].sigmoid(),"f0_hz":60+340*raw[...,1].sigmoid(),"voiced":voiced}
        def stream_step(self,tick,state=None):
            if tick.ndim!=2 or tick.shape[-1]!=config.tick_samples:raise ValueError(f"expected [batch,{config.tick_samples}]")
            encoder_state,decoder_state=(None,None) if state is None else state;latent,encoder_state,encoded=self.encode(tick.unsqueeze(1),encoder_state);waveform,decoder_state=self.decode(latent,decoder_state);diagnostics={"latent":latent,"source_probe":self.probe_source(encoded)};return waveform[:,0],(encoder_state,decoder_state),diagnostics
        def forward(self,ticks,state=None):
            encoder_state,decoder_state=(None,None) if state is None else state;latent,encoder_state,encoded=self.encode(ticks,encoder_state);waveform,decoder_state=self.decode(latent,decoder_state);return waveform,(encoder_state,decoder_state),{"latent":latent,"source_probe":self.probe_source(encoded)}
        def export_config(self):return asdict(config)
    return UnifiedStreamingAutoencoder()
