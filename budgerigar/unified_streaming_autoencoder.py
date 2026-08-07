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
    postfilter_channels:int=16
    postfilter_kernel:int=9


def create_unified_streaming_autoencoder(config=UnifiedStreamingConfig()):
    torch,nn,_=require_torch()
    if config.tick_samples%config.subframes:raise ValueError("tick_samples must divide evenly into subframes")
    subframe_samples=config.tick_samples//config.subframes
    class UnifiedStreamingAutoencoder(nn.Module):
        def __init__(self):
            super().__init__();self.config=config
            self.hearing=nn.Sequential(nn.Linear(subframe_samples,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.encoder=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.local_latent=nn.Linear(config.hidden_dim,config.latent_dim);self.context_latent=nn.Linear(config.hidden_dim,config.latent_dim);self.latent_projection=nn.Sequential(nn.Linear(config.latent_dim,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.decoder=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.local_waveform=nn.Sequential(nn.Linear(config.latent_dim,config.hidden_dim),nn.SiLU(),nn.Linear(config.hidden_dim,subframe_samples));self.context_waveform=nn.Linear(config.hidden_dim,subframe_samples);self.postfilter_in=nn.Conv1d(1,config.postfilter_channels,config.postfilter_kernel);self.postfilter_out=nn.Conv1d(config.postfilter_channels,1,config.postfilter_kernel);nn.init.zeros_(self.postfilter_out.weight);nn.init.zeros_(self.postfilter_out.bias);self.source_probe=nn.Linear(config.hidden_dim,3)
        def encode(self,ticks,state=None):
            batch,length,samples=ticks.shape;subframes=ticks.view(batch,length*config.subframes,subframe_samples);heard=self.hearing(subframes);encoded,state=self.encoder(heard,state);latent=(self.local_latent(heard)+.25*self.context_latent(encoded)).tanh();return latent,state,encoded
        def decode(self,latent,state=None):
            if state is None:recurrent_state=None;postfilter_state=None
            elif isinstance(state,tuple):recurrent_state,postfilter_state=state
            else:recurrent_state,postfilter_state=state,None
            decoded,recurrent_state=self.decoder(self.latent_projection(latent),recurrent_state);raw=(self.local_waveform(latent)+.25*self.context_waveform(decoded)).tanh();batch=latent.shape[0];raw=raw.reshape(batch,1,-1);cache_samples=config.postfilter_kernel-1
            if postfilter_state is None:raw_cache=raw.new_zeros(batch,1,cache_samples);hidden_cache=raw.new_zeros(batch,config.postfilter_channels,cache_samples)
            else:raw_cache,hidden_cache=postfilter_state
            joined=torch.cat([raw_cache,raw],-1);hidden=torch.nn.functional.silu(self.postfilter_in(joined));joined_hidden=torch.cat([hidden_cache,hidden],-1);correction=self.postfilter_out(joined_hidden);waveform=(raw+.1*correction).clamp(-1,1);postfilter_state=(joined[:,:,-cache_samples:],joined_hidden[:,:,-cache_samples:]);return waveform.reshape(batch,-1,config.tick_samples),(recurrent_state,postfilter_state)
        def probe_source(self,encoded):
            batch,subframes,_=encoded.shape;pooled=encoded.view(batch,-1,config.subframes,config.hidden_dim).mean(2);raw=self.source_probe(pooled);voiced=raw[...,2].sigmoid();return {"drive":raw[...,0].sigmoid(),"f0_hz":60+340*raw[...,1].sigmoid(),"voiced":voiced}
        def stream_step(self,tick,state=None):
            if tick.ndim!=2 or tick.shape[-1]!=config.tick_samples:raise ValueError(f"expected [batch,{config.tick_samples}]")
            encoder_state,decoder_state=(None,None) if state is None else state;latent,encoder_state,encoded=self.encode(tick.unsqueeze(1),encoder_state);waveform,decoder_state=self.decode(latent,decoder_state);diagnostics={"latent":latent,"source_probe":self.probe_source(encoded)};return waveform[:,0],(encoder_state,decoder_state),diagnostics
        def forward(self,ticks,state=None):
            encoder_state,decoder_state=(None,None) if state is None else state;latent,encoder_state,encoded=self.encode(ticks,encoder_state);waveform,decoder_state=self.decode(latent,decoder_state);return waveform,(encoder_state,decoder_state),{"latent":latent,"source_probe":self.probe_source(encoded)}
        def export_config(self):return asdict(config)
    return UnifiedStreamingAutoencoder()
