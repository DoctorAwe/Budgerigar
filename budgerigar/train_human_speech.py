from __future__ import annotations
from dataclasses import asdict,dataclass
import json,math,random,time
from pathlib import Path
from .auditory_evaluator import AuditoryEvaluatorConfig,create_auditory_evaluator
from .human_speech_model import HumanSpeechConfig,create_human_speech_model
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset,collate_short_memory
from .train_auditory_evaluator import robust_waveform_augmentation


@dataclass(frozen=True)
class HumanSpeechTrainingConfig:
    batch_size:int=8
    learning_rate:float=3e-4
    max_steps:int=500
    epochs:int=30
    max_train_records:int=1000
    max_validation_records:int=100
    acoustic_weight:float=1.0
    content_weight:float=.25
    perceptual_weight:float=.5
    control_weight:float=1.0
    envelope_weight:float=.5
    waveform_weight:float=.05
    spectral_balance_weight:float=.5
    seed:int=163


def _acoustic_loss(torch,predicted,target):
    losses=[]
    for n_fft,hop in ((256,64),(512,128),(1024,256)):
        window=torch.hann_window(n_fft,device=target.device);estimate=torch.stft(predicted,n_fft,hop_length=hop,window=window,return_complex=True).abs();truth=torch.stft(target,n_fft,hop_length=hop,window=window,return_complex=True).abs();losses.append((estimate-truth).norm()/truth.norm().clamp_min(1e-6)+(estimate.log1p()-truth.log1p()).abs().mean())
    return sum(losses)/len(losses)


def _source_targets(torch,functional,waveform,config):
    hop=config.tick_samples//config.subframes;energy=waveform.unfold(-1,hop,hop).square().mean(-1).sqrt().clamp(0,1);context=functional.pad(waveform,(639,0)).unfold(-1,640,hop);context=context-context.mean(-1,keepdim=True);spectrum=torch.fft.rfft(context,n=1024);correlation=torch.fft.irfft(spectrum.abs().square(),n=1024)[...,:320];correlation=correlation/correlation[...,0:1].clamp_min(1e-6);strength,index=correlation[...,40:267].max(-1);pitch=(config.sample_rate/(index+40)).clamp(60,400);voiced=((strength>.25)&(energy>.01)).float();return energy,(pitch-60)/340,voiced


def _control_loss(torch,functional,controls,waveform,config):
    energy,pitch,voiced=_source_targets(torch,functional,waveform,config);target_hz=60+340*pitch;pitch_error=functional.smooth_l1_loss(controls["f0_hz"]/100,target_hz/100,reduction="none");pitch_error=(pitch_error*voiced).sum()/voiced.sum().clamp_min(1);active=energy.gt(.01).float();aperiodic=functional.binary_cross_entropy(controls["aperiodicity"],1-voiced,reduction="none");aperiodic=(aperiodic*active).sum()/active.sum().clamp_min(1);return functional.l1_loss(controls["pressure"],energy)+2*pitch_error+functional.binary_cross_entropy(controls["voiced"],voiced)+.5*aperiodic


def _spectral_balance_loss(torch,predicted,target,sample_rate):
    window=torch.hann_window(512,device=target.device,dtype=target.dtype);estimate=torch.stft(predicted,n_fft=512,hop_length=128,window=window,return_complex=True).abs().mean(-1);truth=torch.stft(target,n_fft=512,hop_length=128,window=window,return_complex=True).abs().mean(-1);estimate_distribution=estimate/estimate.sum(-1,keepdim=True).clamp_min(1e-6);truth_distribution=truth/truth.sum(-1,keepdim=True).clamp_min(1e-6);frequencies=torch.fft.rfftfreq(512,1/sample_rate).to(target);estimate_centroid=(estimate_distribution*frequencies).sum(-1);truth_centroid=(truth_distribution*frequencies).sum(-1);high=frequencies>2000;estimate_high=estimate_distribution[:,high].sum(-1);truth_high=truth_distribution[:,high].sum(-1);shape=(estimate_distribution-truth_distribution).abs().sum(-1);centroid=(estimate_centroid.clamp_min(1).log()-truth_centroid.clamp_min(1).log()).abs();high_error=(estimate_high.clamp_min(1e-5).log()-truth_high.clamp_min(1e-5).log()).abs();return (shape+.5*centroid+.25*high_error).mean()


def _energy_envelope(torch,waveform,config):
    hop=config.tick_samples//config.subframes;return waveform.unfold(-1,hop,hop).square().mean(-1).sqrt()


def _spectral_embedding(torch,functional,waveform):
    window=torch.hann_window(256,device=waveform.device,dtype=waveform.dtype);magnitude=torch.stft(waveform,n_fft=256,hop_length=80,window=window,return_complex=True).abs().log1p();return functional.adaptive_avg_pool2d(magnitude.unsqueeze(1),(32,32)).flatten(1)


def train_human_speech(manifest,output_dir,evaluator_checkpoint,training=HumanSpeechTrainingConfig(),model_config=HumanSpeechConfig(),resume_from=None):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed);dataset=lambda split,limit:ShortMemoryEpisodeDataset(manifest,split,model_config.sample_rate,model_config.tick_samples,max_records=limit);train=dataset("train",training.max_train_records);validation=dataset("validation",training.max_validation_records);loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");payload=torch.load(evaluator_checkpoint,map_location=device,weights_only=False);evaluator=create_auditory_evaluator(AuditoryEvaluatorConfig(**payload["model_config"])).to(device);evaluator.load_state_dict(payload["model"]);evaluator.eval();evaluator.requires_grad_(False);model=create_human_speech_model(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,betas=(.8,.99));output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=math.inf;step=0
    parent_history=[]
    if resume_from is not None:
        resumed=torch.load(resume_from,map_location=device,weights_only=False);current=model.state_dict();compatible={name:value for name,value in resumed["model"].items() if name in current and current[name].shape==value.shape};missing=model.load_state_dict(compatible,strict=False);parent_history=list(resumed.get("history",[]));step=int(parent_history[-1]["step"]) if parent_history else 0;print(f"[human speech resume] step={step} parent_history={len(parent_history)} tensors={len(compatible)} new={len(missing.missing_keys)} optimizer=fresh",flush=True)
    started=time.perf_counter()
    for epoch in range(training.epochs):
        model.train();total=count=0
        for ticks,_,_,metadata in train_loader:
            ticks=ticks.to(device);target=ticks.flatten(1);labels=torch.tensor([item["label"] for item in metadata],device=device);output,_,diagnostics=model(ticks);output=output.flatten(1);acoustic=_acoustic_loss(torch,output.float(),target.float());target_energy=_energy_envelope(torch,target,model_config);output_energy=_energy_envelope(torch,output,model_config);envelope=functional.l1_loss(output_energy,target_energy);control=_control_loss(torch,functional,diagnostics["controls"],target,model_config);spectral_balance=_spectral_balance_loss(torch,output.float(),target.float(),model_config.sample_rate);content=functional.cross_entropy(evaluator(robust_waveform_augmentation(torch,output)),labels);target_features=functional.normalize(evaluator.features(target).detach(),dim=-1);output_features=functional.normalize(evaluator.features(output),dim=-1);perceptual=(1-(output_features*target_features).sum(-1)).mean();waveform=functional.l1_loss(output,target);loss=training.acoustic_weight*acoustic+training.envelope_weight*envelope+training.control_weight*control+training.spectral_balance_weight*spectral_balance+training.content_weight*content+training.perceptual_weight*perceptual+training.waveform_weight*waveform;optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:elapsed=time.perf_counter()-started;print(f"[human speech M0] step={step} loss={float(loss.detach()):.4f} acoustic={float(acoustic.detach()):.4f} balance={float(spectral_balance.detach()):.4f} control={float(control.detach()):.4f} content={float(content.detach()):.4f} perceptual={float(perceptual.detach()):.4f} sec/10={elapsed/(1 if step==1 else 10):.2f}",flush=True);started=time.perf_counter()
            if step>=training.max_steps:break
        model.eval();correct=[];shuffled=[];mean_values=[];energy_errors=[];duration_ratios=[];f0_errors=[];predicted_f0=[];target_f0=[];centroid_ratios=[];high_frequency_ratios=[];real_correct=output_correct=examples=0;input_separation=[];output_separation=[];example=None
        with torch.no_grad():
            for ticks,_,_,metadata in validation_loader:
                ticks=ticks.to(device);target=ticks.flatten(1);labels=torch.tensor([item["label"] for item in metadata],device=device);output,_,diagnostics=model(ticks);flat=output.flatten(1);latent=diagnostics["sensation"];wrong=model.synthesize(latent.roll(1,0))[0].flatten(1);average=model.synthesize(latent.mean(0,keepdim=True).expand_as(latent))[0].flatten(1);correct.append(float(_acoustic_loss(torch,flat,target)));shuffled.append(float(_acoustic_loss(torch,wrong,target)));mean_values.append(float(_acoustic_loss(torch,average,target)));target_energy=_energy_envelope(torch,target,model_config);output_energy=_energy_envelope(torch,flat,model_config);energy_errors.append(float(functional.l1_loss(output_energy,target_energy)));duration_ratios.extend(((output_energy>.01).sum(-1)/(target_energy>.01).sum(-1).clamp_min(1)).tolist());_,pitch,voiced=_source_targets(torch,functional,target,model_config);predicted=(diagnostics["controls"]["f0_hz"]-60)/340;mask=voiced.bool();f0_errors.extend(((predicted-pitch).abs()[mask]*340).tolist());predicted_f0.extend((60+340*predicted[mask]).tolist());target_f0.extend((60+340*pitch[mask]).tolist());window=torch.hann_window(512,device=device);target_spectrum=torch.stft(target,n_fft=512,hop_length=128,window=window,return_complex=True).abs();output_spectrum=torch.stft(flat,n_fft=512,hop_length=128,window=window,return_complex=True).abs();frequencies=torch.fft.rfftfreq(512,1/model_config.sample_rate).to(device);target_centroid=(target_spectrum*frequencies.view(1,-1,1)).sum((1,2))/target_spectrum.sum((1,2)).clamp_min(1e-6);output_centroid=(output_spectrum*frequencies.view(1,-1,1)).sum((1,2))/output_spectrum.sum((1,2)).clamp_min(1e-6);centroid_ratios.extend((output_centroid/target_centroid.clamp_min(1)).tolist());high=frequencies>2000;target_high=target_spectrum[:,high].sum((1,2))/target_spectrum.sum((1,2)).clamp_min(1e-6);output_high=output_spectrum[:,high].sum((1,2))/output_spectrum.sum((1,2)).clamp_min(1e-6);high_frequency_ratios.extend((output_high/target_high.clamp_min(1e-6)).tolist());real_correct+=int((evaluator(target).argmax(-1)==labels).sum());output_correct+=int((evaluator(flat).argmax(-1)==labels).sum());examples+=len(labels);input_embedding=_spectral_embedding(torch,functional,target);output_embedding=_spectral_embedding(torch,functional,flat);different=labels.ne(labels.roll(1));input_separation.extend((input_embedding-input_embedding.roll(1,0)).abs().mean(-1)[different].tolist());output_separation.extend((output_embedding-output_embedding.roll(1,0)).abs().mean(-1)[different].tolist())
                if example is None:example={"input":target[0].cpu(),"reconstruction":flat[0].cpu(),"id":metadata[0]["id"],"sample_rate":model_config.sample_rate}
        correct_loss=sum(correct)/len(correct);shuffled_loss=sum(shuffled)/len(shuffled);mean_loss=sum(mean_values)/len(mean_values);input_distance=sum(input_separation)/max(len(input_separation),1);output_distance=sum(output_separation)/max(len(output_separation),1);metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_acoustic_loss":correct_loss,"validation_shuffled_sensation_loss":shuffled_loss,"validation_mean_sensation_loss":mean_loss,"validation_shuffled_relative_degradation":shuffled_loss/correct_loss-1,"validation_mean_relative_degradation":mean_loss/correct_loss-1,"validation_energy_envelope_l1":sum(energy_errors)/len(energy_errors),"validation_voiced_duration_ratio":sum(duration_ratios)/len(duration_ratios),"validation_f0_mae_hz":sum(f0_errors)/max(len(f0_errors),1),"validation_predicted_f0_hz":sum(predicted_f0)/max(len(predicted_f0),1),"validation_target_f0_hz":sum(target_f0)/max(len(target_f0),1),"validation_spectral_centroid_ratio":sum(centroid_ratios)/len(centroid_ratios),"validation_high_frequency_energy_ratio":sum(high_frequency_ratios)/len(high_frequency_ratios),"validation_frozen_real_digit_accuracy":real_correct/max(examples,1),"validation_frozen_output_digit_accuracy":output_correct/max(examples,1),"validation_output_separation_ratio":output_distance/max(input_distance,1e-6)};history.append(metrics);print(json.dumps(metrics),flush=True);checkpoint={"architecture":"human_cochlear_balanced_larynx_m0","model":model.state_dict(),"optimizer":optimizer.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"initialized_from":str(resume_from) if resume_from is not None else None,"parent_history":parent_history,"history":history};torch.save(checkpoint,output_dir/"last.pt");torch.save(example,output_dir/"validation_example.pt");balance_penalty=.001*metrics["validation_f0_mae_hz"]+.1*abs(math.log(max(metrics["validation_spectral_centroid_ratio"],1e-6)))+.05*abs(math.log(max(metrics["validation_high_frequency_energy_ratio"],1e-6)));selection=correct_loss+.5*(1-metrics["validation_frozen_output_digit_accuracy"])+balance_penalty
        if selection<best:best=selection;torch.save(checkpoint,output_dir/"best.pt");torch.save(example,output_dir/"best_validation_example.pt")
        if step>=training.max_steps:break
    report={"architecture":"human_cochlear_balanced_larynx_m0","steps":step,"best_selection_score":best,"initialized_from":str(resume_from) if resume_from is not None else None,"parent_history":parent_history,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
