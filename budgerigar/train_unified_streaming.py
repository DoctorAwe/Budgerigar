from __future__ import annotations
from dataclasses import asdict,dataclass
import json,math,random,time
from pathlib import Path
from .auditory_evaluator import AuditoryEvaluatorConfig,create_auditory_evaluator
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset,collate_short_memory
from .train_auditory_evaluator import robust_waveform_augmentation
from .unified_streaming_autoencoder import UnifiedStreamingConfig,create_unified_streaming_autoencoder


@dataclass(frozen=True)
class UnifiedStreamingTrainingConfig:
    batch_size:int=12
    learning_rate:float=4e-4
    max_steps:int=500
    epochs:int=40
    max_train_records:int=1000
    max_validation_records:int=100
    source_probe_weight:float=.1
    source_probe_every_steps:int=4
    frozen_content_weight:float=.5
    content_every_steps:int=2
    auxiliary_start_step:int=200
    waveform_weight:float=5.0
    derivative_weight:float=1.0
    si_sdr_weight:float=.05
    seed:int=149


def _spectral_loss(torch,predicted,target,resolutions=((256,64),(512,128),(1024,256))):
    losses=[]
    for n_fft,hop in resolutions:
        window=torch.hann_window(n_fft,device=target.device);estimate=torch.stft(predicted,n_fft,hop_length=hop,window=window,return_complex=True).abs();truth=torch.stft(target,n_fft,hop_length=hop,window=window,return_complex=True).abs();losses.append((estimate-truth).norm()/truth.norm().clamp_min(1e-6)+(estimate.clamp_min(1e-5).log()-truth.clamp_min(1e-5).log()).abs().mean())
    return sum(losses)/len(losses)


def _si_sdr(torch,predicted,target):
    predicted=predicted-predicted.mean(-1,keepdim=True);target=target-target.mean(-1,keepdim=True);scale=(predicted*target).sum(-1,keepdim=True)/target.square().sum(-1,keepdim=True).clamp_min(1e-8);signal=scale*target;noise=predicted-signal;return 10*torch.log10(signal.square().sum(-1).clamp_min(1e-8)/noise.square().sum(-1).clamp_min(1e-8))


def _spectral_embedding(torch,functional,waveform):
    window=torch.hann_window(256,device=waveform.device,dtype=waveform.dtype);magnitude=torch.stft(waveform,n_fft=256,hop_length=80,window=window,return_complex=True).abs().log1p();return functional.adaptive_avg_pool2d(magnitude.unsqueeze(1),(32,32)).flatten(1)


def _source_probe_loss(torch,functional,probe,waveform,config):
    ticks=waveform.view(len(waveform),-1,config.tick_samples);energy=ticks.square().mean(-1).sqrt().clamp(0,1);context=functional.pad(waveform,(639,0)).unfold(-1,640,config.tick_samples);context=context-context.mean(-1,keepdim=True);spectrum=torch.fft.rfft(context,n=1024);correlation=torch.fft.irfft(spectrum.abs().square(),n=1024)[...,:320];correlation=correlation/correlation[...,0:1].clamp_min(1e-6);strength,index=correlation[...,40:267].max(-1);pitch=(config.sample_rate/(index+40)).clamp(60,400);voiced=((strength>.25)&(energy>.01)).float();pitch_loss=(functional.l1_loss((probe["f0_hz"]-60)/340,(pitch-60)/340,reduction="none")*voiced).sum()/voiced.sum().clamp_min(1);return functional.l1_loss(probe["drive"],energy)+pitch_loss+functional.binary_cross_entropy(probe["voiced"],voiced)


def train_unified_streaming(manifest,output_dir,evaluator_checkpoint,training=UnifiedStreamingTrainingConfig(),model_config=UnifiedStreamingConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed);dataset=lambda split,limit:ShortMemoryEpisodeDataset(manifest,split,model_config.sample_rate,model_config.tick_samples,max_records=limit);train=dataset("train",training.max_train_records);validation=dataset("validation",training.max_validation_records);loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");payload=torch.load(evaluator_checkpoint,map_location=device,weights_only=False);evaluator=create_auditory_evaluator(AuditoryEvaluatorConfig(**payload["model_config"])).to(device);evaluator.load_state_dict(payload["model"]);evaluator.eval();evaluator.requires_grad_(False);model=create_unified_streaming_autoencoder(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,betas=(.8,.99));output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=math.inf;step=0;log_started=time.perf_counter()
    for epoch in range(training.epochs):
        model.train();total=count=0
        for ticks,_,_,metadata in train_loader:
            ticks=ticks.to(device);waveform=ticks.flatten(1);labels=torch.tensor([item["label"] for item in metadata],device=device);reconstructed,_,diagnostics=model(ticks);reconstructed=reconstructed.flatten(1);weight=1+4*waveform.abs().gt(.01);wave=((reconstructed-waveform).abs()*weight).sum()/weight.sum();derivative=functional.l1_loss(reconstructed[:,1:]-reconstructed[:,:-1],waveform[:,1:]-waveform[:,:-1]);spectral=_spectral_loss(torch,reconstructed.float(),waveform.float(),((256,64),(512,128)));sdr=20*torch.tanh(_si_sdr(torch,reconstructed,waveform)/20).mean();auxiliary=step>=training.auxiliary_start_step;use_probe=auxiliary and (step%training.source_probe_every_steps)==0;probe=_source_probe_loss(torch,functional,diagnostics["source_probe"],waveform,model_config) if use_probe else wave.new_zeros(());use_content=auxiliary and (step%training.content_every_steps)==0;content=functional.cross_entropy(evaluator(robust_waveform_augmentation(torch,reconstructed)),labels) if use_content else wave.new_zeros(());loss=training.waveform_weight*wave+training.derivative_weight*derivative+spectral-training.si_sdr_weight*sdr+training.source_probe_weight*probe+training.frozen_content_weight*content;optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:elapsed=time.perf_counter()-log_started;print(f"[unified M0] step={step} loss={float(loss.detach()):.4f} wave={float(wave.detach()):.4f} spectral={float(spectral.detach()):.4f} si_sdr={float(sdr.detach()):.2f} frozen_content={float(content.detach()):.4f} seconds_per_10={elapsed/(1 if step==1 else 10):.2f}",flush=True);log_started=time.perf_counter()
            if step>=training.max_steps:break
        model.eval();spectral_values=[];shuffled_values=[];mean_values=[];sdr=[];input_separation=[];output_separation=[];real_correct=output_correct=examples=0;example=None
        with torch.no_grad():
            for ticks,_,_,metadata in validation_loader:
                ticks=ticks.to(device);waveform=ticks.flatten(1);labels=torch.tensor([item["label"] for item in metadata],device=device);reconstructed,_,diagnostics=model(ticks);latent=diagnostics["latent"];shuffled=model.decode(latent.roll(1,0))[0];mean=model.decode(latent.mean(0,keepdim=True).expand_as(latent))[0];flat=reconstructed.flatten(1);spectral_values.append(float(_spectral_loss(torch,flat,waveform)));shuffled_values.append(float(_spectral_loss(torch,shuffled.flatten(1),waveform)));mean_values.append(float(_spectral_loss(torch,mean.flatten(1),waveform)));sdr.extend(_si_sdr(torch,flat,waveform).tolist());real_correct+=int((evaluator(waveform).argmax(-1)==labels).sum());output_correct+=int((evaluator(flat).argmax(-1)==labels).sum());examples+=len(labels);input_embedding=_spectral_embedding(torch,functional,waveform);output_embedding=_spectral_embedding(torch,functional,flat);different=labels.ne(labels.roll(1));input_separation.extend((input_embedding-input_embedding.roll(1,0)).abs().mean(-1)[different].tolist());output_separation.extend((output_embedding-output_embedding.roll(1,0)).abs().mean(-1)[different].tolist())
                if example is None:example={"input":waveform[0].cpu(),"reconstruction":flat[0].cpu(),"id":metadata[0]["id"],"sample_rate":model_config.sample_rate}
        correct=sum(spectral_values)/len(spectral_values);shuffled=sum(shuffled_values)/len(shuffled_values);mean=sum(mean_values)/len(mean_values);input_distance=sum(input_separation)/max(len(input_separation),1);output_distance=sum(output_separation)/max(len(output_separation),1);metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_spectral_loss":correct,"validation_latent_shuffled_spectral_loss":shuffled,"validation_latent_mean_spectral_loss":mean,"validation_latent_shuffled_relative_degradation":shuffled/correct-1,"validation_latent_mean_relative_degradation":mean/correct-1,"validation_si_sdr_db":sum(sdr)/len(sdr),"validation_frozen_real_digit_accuracy":real_correct/max(examples,1),"validation_frozen_output_digit_accuracy":output_correct/max(examples,1),"validation_input_separation":input_distance,"validation_output_separation":output_distance,"validation_output_separation_ratio":output_distance/max(input_distance,1e-6)};history.append(metrics);print(json.dumps(metrics),flush=True);checkpoint={"architecture":"unified_continuous_streaming_m0","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"evaluator_checkpoint":str(evaluator_checkpoint),"history":history};torch.save(checkpoint,output_dir/"last.pt");torch.save(example,output_dir/"validation_example.pt");selection=correct+.5*(1-metrics["validation_frozen_output_digit_accuracy"])
        if selection<best:best=selection;torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"unified_continuous_streaming_m0","steps":step,"best_selection_score":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
