from __future__ import annotations
from dataclasses import asdict,dataclass
import json,math,random
from pathlib import Path
from .short_memory_data import ShortMemoryEpisodeDataset,collate_short_memory
from .neural_echo import require_torch
from .sensorimotor_vocoder import SensorimotorVocoderConfig,create_sensorimotor_vocoder


@dataclass(frozen=True)
class SensorimotorTrainingConfig:
    batch_size:int=8
    learning_rate:float=3e-4
    max_steps:int=400
    epochs:int=30
    max_train_records:int=1000
    max_validation_records:int=200
    source_supervision_weight:float=.5
    control_contrastive_weight:float=1.0
    control_margin:float=.02
    smoothness_weight:float=.02
    seed:int=109


def _spectral_loss(torch,predicted,target):
    losses=[]
    for n_fft,hop in ((256,64),(512,128),(1024,256)):
        window=torch.hann_window(n_fft,device=target.device);estimate=torch.stft(predicted,n_fft,hop_length=hop,window=window,return_complex=True).abs();truth=torch.stft(target,n_fft,hop_length=hop,window=window,return_complex=True).abs();losses.append((estimate-truth).norm()/truth.norm().clamp_min(1e-6)+(estimate.clamp_min(1e-5).log()-truth.clamp_min(1e-5).log()).abs().mean())
    return sum(losses)/len(losses)


def _source_targets(torch,waveform,config):
    ticks=waveform.view(len(waveform),-1,config.tick_samples);energy=ticks.square().mean(-1).sqrt().clamp(0,1);context=torch.nn.functional.pad(waveform,(639,0)).unfold(-1,640,config.tick_samples);context=context-context.mean(-1,keepdim=True);spectrum=torch.fft.rfft(context,n=1024);correlation=torch.fft.irfft(spectrum.abs().square(),n=1024)[...,:320];correlation=correlation/correlation[...,0:1].clamp_min(1e-6);region=correlation[...,40:267];strength,index=region.max(-1);lag=index+40;pitch=(config.sample_rate/lag).clamp(60,400);voiced=(strength>.25)&(energy>.01)
    return energy,(pitch-60)/340,voiced.float()


def _source_loss(torch,functional,parameters,waveform,config):
    energy,pitch,voiced=_source_targets(torch,waveform,config);drive=functional.l1_loss(parameters["drive"],energy);pitch_loss=functional.l1_loss((parameters["f0_hz"]-60)/340,pitch,reduction="none");pitch_loss=(pitch_loss*voiced).sum()/voiced.sum().clamp_min(1);voice=functional.binary_cross_entropy(parameters["voiced"],voiced);return drive+pitch_loss+voice


def _oracle_source_controls(torch,waveform,config):
    """Return deterministic renderer logits, leaving no channel for phonetic steganography."""
    energy,pitch,voiced=_source_targets(torch,waveform,config)
    def logit(value):
        value=value.clamp(1e-4,1-1e-4)
        return (value/(1-value)).log()
    return torch.stack((logit(energy),logit(pitch),logit(voiced)),-1)


def train_sensorimotor_vocoder(manifest,output_dir,training=SensorimotorTrainingConfig(),model_config=SensorimotorVocoderConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed)
    train=ShortMemoryEpisodeDataset(manifest,"train",model_config.sample_rate,model_config.tick_samples,max_records=training.max_train_records);validation=ShortMemoryEpisodeDataset(manifest,"validation",model_config.sample_rate,model_config.tick_samples,max_records=training.max_validation_records);loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_generator=torch.Generator().manual_seed(training.seed+1);validation_loader=loader(validation,batch_size=training.batch_size,shuffle=True,generator=validation_generator,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=create_sensorimotor_vocoder(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,betas=(.8,.99));output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=math.inf;step=0
    for epoch in range(training.epochs):
        model.train();total=count=0
        for ticks,_,_,_ in train_loader:
            ticks=ticks.to(device);waveform=ticks.flatten(1);predicted_source,articulation,_=model.analyze(ticks);oracle_source=_oracle_source_controls(torch,waveform,model_config);reconstructed=model.render(oracle_source,articulation)[0].flatten(1);wave=functional.l1_loss(reconstructed,waveform);spectral=_spectral_loss(torch,reconstructed.float(),waveform.float());source=_source_loss(torch,functional,model.source_parameters(predicted_source),waveform,model_config);smooth=(articulation[:,1:]-articulation[:,:-1]).abs().mean();source_swap=model.render(oracle_source.roll(1,0),articulation)[0].flatten(1);articulation_swap=model.render(oracle_source,articulation.roll(1,0))[0].flatten(1);correct_error=(reconstructed-waveform).abs().mean(-1);source_error=(source_swap-waveform).abs().mean(-1);articulation_error=(articulation_swap-waveform).abs().mean(-1);contrastive=(training.control_margin+correct_error-source_error).relu().mean()+(training.control_margin+correct_error-articulation_error).relu().mean();loss=wave+spectral+training.source_supervision_weight*source+training.smoothness_weight*smooth+training.control_contrastive_weight*contrastive
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:print(f"[sensorimotor M0] step={step} loss={float(loss):.4f} wave={float(wave):.4f} spectral={float(spectral):.4f} source={float(source):.4f} control_margin={float(contrastive):.4f}",flush=True)
            if step>=training.max_steps:break
        model.eval();correct=[];source_shuffled=[];articulation_shuffled=[];mean_control=[];example=None
        with torch.no_grad():
            for ticks,_,_,metadata in validation_loader:
                ticks=ticks.to(device);waveform=ticks.flatten(1);predicted_source,articulation,_=model.analyze(ticks);source=_oracle_source_controls(torch,waveform,model_config);reconstructed=model.render(source,articulation)[0];source_swap=model.render(source.roll(1,0),articulation)[0];articulation_swap=model.render(source,articulation.roll(1,0))[0];mean=model.render(source.mean(0,keepdim=True).expand_as(source),articulation.mean(0,keepdim=True).expand_as(articulation))[0]
                for collection,value in ((correct,reconstructed),(source_shuffled,source_swap),(articulation_shuffled,articulation_swap),(mean_control,mean)):collection.append(float(_spectral_loss(torch,value.flatten(1).float(),waveform.float())))
                if example is None:example={"input":waveform[0].cpu(),"reconstruction":reconstructed[0].flatten().cpu(),"oracle_source_controls":source[0].cpu(),"predicted_source_controls":predicted_source[0].cpu(),"articulation_controls":articulation[0].cpu(),"id":metadata[0]["id"],"sample_rate":model_config.sample_rate}
        correct_loss=sum(correct)/len(correct);source_loss=sum(source_shuffled)/len(source_shuffled);articulation_loss=sum(articulation_shuffled)/len(articulation_shuffled);mean_loss=sum(mean_control)/len(mean_control);metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_spectral_loss":correct_loss,"validation_source_shuffled_spectral_loss":source_loss,"validation_articulation_shuffled_spectral_loss":articulation_loss,"validation_mean_control_spectral_loss":mean_loss,"validation_source_relative_degradation":source_loss/correct_loss-1,"validation_articulation_relative_degradation":articulation_loss/correct_loss-1,"validation_mean_relative_degradation":mean_loss/correct_loss-1};history.append(metrics);print(json.dumps(metrics),flush=True);checkpoint={"architecture":"sensorimotor_oracle_source_articulation_m0","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history};torch.save(checkpoint,output_dir/"last.pt");torch.save(example,output_dir/"validation_example.pt")
        if metrics["validation_spectral_loss"]<best:best=metrics["validation_spectral_loss"];torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"sensorimotor_oracle_source_articulation_m0","steps":step,"best_validation_spectral_loss":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
