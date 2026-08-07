from __future__ import annotations
from dataclasses import asdict,dataclass
from .neural_echo import require_torch


@dataclass(frozen=True)
class AuditoryEvaluatorConfig:
    sample_rate:int=16000
    n_fft:int=512
    hop_samples:int=160
    frequency_bins:int=128
    channels:int=48
    output_classes:int=10


def create_auditory_evaluator(config=AuditoryEvaluatorConfig()):
    torch,nn,functional=require_torch()
    class FrozenAuditoryEvaluator(nn.Module):
        def __init__(self):
            super().__init__();self.config=config
            self.register_buffer("window",torch.hann_window(config.n_fft),persistent=False)
            c=config.channels;self.encoder=nn.Sequential(nn.Conv2d(1,c,5,padding=2),nn.BatchNorm2d(c),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(c,c*2,3,padding=1),nn.BatchNorm2d(c*2),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(c*2,c*2,3,padding=1),nn.BatchNorm2d(c*2),nn.SiLU());self.classifier=nn.Sequential(nn.Linear(c*2*16,256),nn.SiLU(),nn.Dropout(.2),nn.Linear(256,config.output_classes))
        def features(self,waveform):
            spectrum=torch.stft(waveform,n_fft=config.n_fft,hop_length=config.hop_samples,window=self.window.to(waveform),return_complex=True).abs()[:,:config.frequency_bins].log1p();encoded=self.encoder(spectrum.unsqueeze(1));pooled=functional.adaptive_avg_pool2d(encoded,(4,4));return pooled.flatten(1)
        def forward(self,waveform):return self.classifier(self.features(waveform))
        def export_config(self):return asdict(config)
    return FrozenAuditoryEvaluator()
