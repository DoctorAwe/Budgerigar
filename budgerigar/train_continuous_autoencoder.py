from __future__ import annotations
from dataclasses import asdict,dataclass
import json,math,random
from pathlib import Path
from .continuous_audio_data import WaveformChunkDataset,collate_waveform_chunks
from .continuous_autoencoder import ContinuousAutoencoderConfig,create_continuous_autoencoder
from .neural_echo import require_torch

@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    batch_size:int=4
    learning_rate:float=3e-4
    max_steps:int=400
    epochs:int=30
    max_train_records:int=256
    max_validation_records:int=64
    chunk_samples:int=24000
    gradient_clip:float=1.0
    seed:int=71

def _spectral_loss(torch,predicted,target):
    losses=[]
    for n_fft,hop in ((512,128),(1024,256),(2048,512)):
        window=torch.hann_window(n_fft,device=target.device)
        estimate=torch.stft(predicted,n_fft,hop_length=hop,window=window,return_complex=True).abs();truth=torch.stft(target,n_fft,hop_length=hop,window=window,return_complex=True).abs()
        losses.append((estimate-truth).norm()/truth.norm().clamp_min(1e-6)+(estimate.clamp_min(1e-5).log()-truth.clamp_min(1e-5).log()).abs().mean())
    return sum(losses)/len(losses)

def _si_sdr(torch,predicted,target):
    predicted=predicted-predicted.mean(-1,keepdim=True);target=target-target.mean(-1,keepdim=True)
    scale=(predicted*target).sum(-1,keepdim=True)/target.square().sum(-1,keepdim=True).clamp_min(1e-8);signal=scale*target;noise=predicted-signal
    return 10*torch.log10(signal.square().sum(-1).clamp_min(1e-8)/noise.square().sum(-1).clamp_min(1e-8))

def train_continuous_autoencoder(manifest,output_dir,training=AutoencoderTrainingConfig(),model_config=ContinuousAutoencoderConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed)
    train=WaveformChunkDataset(manifest,"train",model_config.sample_rate,training.chunk_samples,training.max_train_records,True,True)
    validation=WaveformChunkDataset(manifest,"validation",model_config.sample_rate,training.chunk_samples,training.max_validation_records,True,False)
    loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,num_workers=0,collate_fn=collate_waveform_chunks);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_waveform_chunks)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=create_continuous_autoencoder(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,betas=(.8,.99))
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=math.inf;step=0
    for epoch in range(training.epochs):
        model.train();total=count=0
        for waveform,_,_ in train_loader:
            waveform=waveform.to(device);reconstructed,_=model(waveform);wave=functional.l1_loss(reconstructed,waveform);spectral=_spectral_loss(torch,reconstructed.float(),waveform.float());loss=wave+spectral
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),training.gradient_clip);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:print(f"[continuous AE] step={step} loss={float(loss):.4f} wave={float(wave):.4f} spectral={float(spectral):.4f}",flush=True)
            if step>=training.max_steps:break
        model.eval();spectral_values=[];sdr=[];example=None
        with torch.no_grad():
            for waveform,ids,_ in validation_loader:
                waveform=waveform.to(device);reconstructed,latent=model(waveform);spectral_values.append(float(_spectral_loss(torch,reconstructed.float(),waveform.float())));sdr.extend(_si_sdr(torch,reconstructed,waveform).cpu().tolist())
                if example is None:example=(waveform[0].cpu(),reconstructed[0].cpu(),ids[0],tuple(latent.shape))
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_spectral_loss":sum(spectral_values)/len(spectral_values),"validation_si_sdr_db":sum(sdr)/len(sdr)};history.append(metrics);print(json.dumps(metrics),flush=True)
        checkpoint={"architecture":"continuous_waveform_autoencoder","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history};torch.save(checkpoint,output_dir/"last.pt")
        torch.save({"input":example[0],"reconstruction":example[1],"id":example[2],"latent_shape":example[3],"sample_rate":model_config.sample_rate},output_dir/"validation_example.pt")
        if metrics["validation_spectral_loss"]<best:best=metrics["validation_spectral_loss"];torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"continuous_waveform_autoencoder","steps":step,"best_validation_spectral_loss":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
