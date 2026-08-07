from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class HumanSpeechConfig:
    sample_rate:int=16000
    tick_samples:int=160
    subframes:int=4
    cochlear_bands:int=32
    cochlear_kernel:int=129
    hidden_dim:int=128
    latent_dim:int=16
    recurrent_layers:int=2
    synthesis_frame_samples:int=160
    harmonics:int=16


def create_human_speech_model(config=HumanSpeechConfig()):
    torch,nn,functional=require_torch();hop=config.tick_samples//config.subframes;bins=config.synthesis_frame_samples//2+1
    if config.tick_samples%config.subframes:raise ValueError("tick_samples must divide into subframes")
    if config.synthesis_frame_samples%hop:raise ValueError("synthesis frame must be a multiple of the hop")
    class HumanSpeechModel(nn.Module):
        def __init__(self):
            super().__init__();self.config=config;self.cochlea=nn.Conv1d(1,config.cochlear_bands,config.cochlear_kernel,bias=False)
            half=(config.cochlear_kernel-1)//2;t=torch.arange(-half,half+1,dtype=torch.float32);centers=torch.logspace(torch.log10(torch.tensor(80.)),torch.log10(torch.tensor(config.sample_rate*.45)),config.cochlear_bands);impulse=[]
            for center in centers:
                low=(center/1.22/config.sample_rate).clamp_min(30/config.sample_rate);high=(center*1.22/config.sample_rate).clamp_max(.49);band=(2*high*torch.sinc(2*high*t)-2*low*torch.sinc(2*low*t))*torch.hamming_window(config.cochlear_kernel);impulse.append(band/band.norm().clamp_min(1e-6))
            with torch.no_grad():self.cochlea.weight.copy_(torch.stack(impulse).unsqueeze(1))
            self.auditory_projection=nn.Sequential(nn.Linear(config.cochlear_bands*2,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.auditory_context=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.sensation=nn.Sequential(nn.Linear(config.hidden_dim,config.latent_dim),nn.Tanh());self.motor_projection=nn.Sequential(nn.Linear(config.latent_dim,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.SiLU());self.motor=nn.GRU(config.hidden_dim,config.hidden_dim,num_layers=config.recurrent_layers,batch_first=True);self.larynx=nn.Linear(config.hidden_dim,4);self.vocal_tract=nn.Linear(config.hidden_dim,bins);self.register_buffer("synthesis_window",torch.hann_window(config.synthesis_frame_samples),persistent=False)
        def encode(self,ticks,state=None):
            batch,length,_=ticks.shape
            if state is None:cache=ticks.new_zeros(batch,1,config.cochlear_kernel-1);previous=ticks.new_zeros(batch,config.cochlear_bands);context_state=None
            else:cache,previous,context_state=state
            waveform=ticks.flatten(1);joined=torch.cat([cache,waveform.unsqueeze(1)],-1);response=self.cochlea(joined).abs().clamp_min(1e-6).pow(.3);steps=length*config.subframes;activity=response.view(batch,config.cochlear_bands,steps,hop).mean(-1);delta=activity-torch.cat([previous.unsqueeze(-1),activity[:,:,:-1]],-1);features=torch.cat([activity,delta],1).transpose(1,2);represented,context_state=self.auditory_context(self.auditory_projection(features),context_state);latent=self.sensation(represented);state=(joined[:,:,-(config.cochlear_kernel-1):],activity[:,:,-1],context_state);return latent,state
        def _controls(self,motor):
            raw=self.larynx(motor);voiced=raw[...,2].sigmoid();return {"pressure":raw[...,0].sigmoid(),"f0_hz":60+340*raw[...,1].sigmoid(),"voiced":voiced,"aperiodicity":raw[...,3].sigmoid()}
        def synthesize(self,latent,state=None,phase=None):
            batch,steps,_=latent.shape
            if state is None:motor_state=None;overlap=latent.new_zeros(batch,config.synthesis_frame_samples-hop)
            else:motor_state,overlap=state
            if phase is None:pitch_phase=latent.new_zeros(batch);sample_offset=0
            else:pitch_phase,sample_offset=phase
            motor,motor_state=self.motor(self.motor_projection(latent),motor_state);controls=self._controls(motor);raw_envelope=(2*self.vocal_tract(motor).tanh()).exp();frequencies=torch.linspace(0,config.sample_rate/2,bins,device=latent.device,dtype=latent.dtype);low_cut=frequencies/(frequencies+35);low_cut[0]=0;high_shelf=(1+(frequencies/3000).square()).rsqrt();envelope=raw_envelope*low_cut*high_shelf;envelope=envelope/envelope.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4);positions=torch.arange(config.synthesis_frame_samples,device=latent.device,dtype=latent.dtype).view(1,1,-1);omega=2*torch.pi*controls["f0_hz"]/config.sample_rate;starts=pitch_phase.unsqueeze(-1)+torch.cat([latent.new_zeros(batch,1),torch.cumsum(omega[:,:-1]*hop,-1)],-1);angles=starts.unsqueeze(-1)+omega.unsqueeze(-1)*positions;orders=torch.arange(1,config.harmonics+1,device=latent.device,dtype=latent.dtype);weights=orders.pow(-1.7);allowed=(controls["f0_hz"].unsqueeze(-1)*orders)<(config.sample_rate*.48);weighted=weights.view(1,1,-1)*allowed;harmonic=(torch.sin(angles.unsqueeze(-1)*orders)*weighted.unsqueeze(-2)).sum(-1)/weighted.sum(-1,keepdim=True).clamp_min(1);absolute=torch.arange(sample_offset,sample_offset+steps*hop,hop,device=latent.device,dtype=latent.dtype).view(1,steps,1)+positions;hashed=torch.sin(absolute*12.9898)*43758.5453;previous_hashed=torch.sin((absolute-1)*12.9898)*43758.5453;white=(hashed-torch.floor(hashed))*2-1;previous_white=(previous_hashed-torch.floor(previous_hashed))*2-1;noise=(white-.95*previous_white)/1.95;voiced=controls["voiced"].unsqueeze(-1);aperiodic=controls["aperiodicity"].unsqueeze(-1);excitation=voiced*harmonic+(1-voiced+.15*aperiodic)*noise;spectrum=torch.fft.rfft(excitation.float(),n=config.synthesis_frame_samples)*envelope.float();frames=torch.fft.irfft(spectrum,n=config.synthesis_frame_samples).to(latent.dtype)*self.synthesis_window.to(latent);frames=frames*(2*controls["pressure"].unsqueeze(-1));outputs=[]
            for index in range(steps):
                frame=frames[:,index];outputs.append(frame[:,:hop]+overlap[:,:hop]);overlap=torch.cat([overlap[:,hop:],overlap.new_zeros(batch,hop)],-1)+frame[:,hop:]
            waveform=torch.stack(outputs,1).tanh();pitch_phase=(pitch_phase+(omega*hop).sum(-1)).remainder(2*torch.pi);phase=(pitch_phase,sample_offset+steps*hop);return waveform.reshape(batch,-1,config.tick_samples),(motor_state,overlap),phase,{"motor_state":motor,"controls":controls,"vocal_tract_envelope":envelope}
        def stream_step(self,tick,state=None):
            auditory_state=motor_state=phase=None
            if state is not None:auditory_state,motor_state,phase=state
            latent,auditory_state=self.encode(tick.unsqueeze(1),auditory_state);waveform,motor_state,phase,diagnostics=self.synthesize(latent,motor_state,phase);diagnostics["sensation"]=latent;return waveform[:,0],(auditory_state,motor_state,phase),diagnostics
        def forward(self,ticks,state=None):
            auditory_state=motor_state=phase=None
            if state is not None:auditory_state,motor_state,phase=state
            latent,auditory_state=self.encode(ticks,auditory_state);waveform,motor_state,phase,diagnostics=self.synthesize(latent,motor_state,phase);diagnostics["sensation"]=latent;return waveform,(auditory_state,motor_state,phase),diagnostics
        def export_config(self):return asdict(config)
    return HumanSpeechModel()
