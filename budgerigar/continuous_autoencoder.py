from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch

@dataclass(frozen=True)
class ContinuousAutoencoderConfig:
    sample_rate:int=24000
    base_channels:int=32
    latent_dim:int=192
    strides:tuple[int,...]=(4,4,5,4)
    residual_layers:int=2

def create_continuous_autoencoder(config=ContinuousAutoencoderConfig()):
    torch,nn,functional=require_torch()
    class CausalConv(nn.Module):
        def __init__(self,source,target,kernel,stride=1):super().__init__();self.left=kernel-stride;self.conv=nn.Conv1d(source,target,kernel,stride=stride)
        def forward(self,x):return self.conv(functional.pad(x,(self.left,0)))
    class ChannelNorm(nn.Module):
        def __init__(self,channels):super().__init__();self.norm=nn.LayerNorm(channels)
        def forward(self,x):return self.norm(x.transpose(1,2)).transpose(1,2)
    class Residual(nn.Module):
        def __init__(self,channels,dilation):
            super().__init__();self.left=2*dilation;self.conv=nn.Conv1d(channels,channels,3,dilation=dilation);self.norm=ChannelNorm(channels)
        def forward(self,x):return x+functional.silu(self.norm(self.conv(functional.pad(x,(self.left,0)))))
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();channels=config.base_channels;layers=[CausalConv(1,channels,7)]
            for stride in config.strides:
                layers.extend(Residual(channels,2**level) for level in range(config.residual_layers));target=min(channels*2,config.latent_dim)
                layers.extend([CausalConv(channels,target,2*stride,stride),ChannelNorm(target),nn.SiLU()]);channels=target
            self.network=nn.Sequential(*layers);self.projection=nn.Conv1d(channels,config.latent_dim,1);self.context=nn.GRU(config.latent_dim,config.latent_dim,batch_first=True)
        def forward(self,waveform):
            value=self.projection(self.network(waveform.unsqueeze(1))).transpose(1,2);return self.context(value)[0]
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__();self.context=nn.GRU(config.latent_dim,config.latent_dim,batch_first=True);channels=config.latent_dim;self.blocks=nn.ModuleList()
            for stride in reversed(config.strides):
                target=max(config.base_channels,channels//2);layers=[CausalConv(channels,target,2*stride),ChannelNorm(target),nn.SiLU()]
                layers.extend(Residual(target,2**level) for level in range(config.residual_layers));self.blocks.append(nn.Sequential(*layers));channels=target
            self.output=CausalConv(channels,1,7)
        def forward(self,latent,target_samples):
            value=self.context(latent)[0].transpose(1,2)
            for stride,block in zip(reversed(config.strides),self.blocks):value=block(value.repeat_interleave(stride,-1))
            return self.output(value)[..., :target_samples].squeeze(1).tanh()
    class ContinuousAutoencoder(nn.Module):
        def __init__(self):super().__init__();self.encoder=Encoder();self.decoder=Decoder();self.config=config
        def forward(self,waveform):
            latent=self.encoder(waveform);return self.decoder(latent,waveform.shape[-1]),latent
        def export_config(self):return asdict(config)
    return ContinuousAutoencoder()
