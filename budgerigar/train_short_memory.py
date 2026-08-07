from __future__ import annotations
from dataclasses import asdict,dataclass
import json,random
from pathlib import Path
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset,collate_short_memory
from .streaming_short_memory import ShortMemoryConfig,create_short_memory_model

@dataclass(frozen=True)
class ShortMemoryTrainingConfig:
    batch_size:int=16
    learning_rate:float=5e-4
    max_steps:int=500
    epochs:int=20
    max_train_records:int=1000
    max_validation_records:int=300
    ctc_weight:float=.25
    blank_weight:float=.25
    content_weight:float=2.0
    reconstruction_weight:float=.5
    retention_weight:float=.5
    emission_weight:float=.1
    timing_weight:float=.25
    timing_margin:float=.5
    cochlear_learning_rate_scale:float=.1
    seed:int=83

def _masks(torch,metadata,ticks,device):
    early=torch.zeros(len(metadata),ticks,dtype=torch.bool,device=device);late=torch.zeros_like(early);labels=[]
    for row,meta in enumerate(metadata):
        early[row,:meta["window_start_tick"]]=True
        late[row,meta["window_end_tick"]+1:]=True;labels.append(meta["label"])
    return early,late,torch.tensor(labels,device=device)

def _window_ctc(torch,functional,logits,metadata,device):
    losses=[]
    for row,meta in enumerate(metadata):
        start,end=meta["window_start_tick"],meta["window_end_tick"]+1;log_probability=logits[row,start:end].log_softmax(-1).unsqueeze(1)
        target=torch.tensor([meta["label"]+1],device=device);length=torch.tensor([end-start],device=device);target_length=torch.ones(1,dtype=torch.long,device=device)
        losses.append(functional.ctc_loss(log_probability,target,length,target_length,blank=0,zero_infinity=True))
    return torch.stack(losses).mean()

def _window_emission_loss(torch,functional,probability,metadata):
    peaks=[]
    for row,meta in enumerate(metadata):peaks.append(probability[row,meta["window_start_tick"]:meta["window_end_tick"]+1].max())
    peaks=torch.stack(peaks);return functional.binary_cross_entropy(peaks,torch.ones_like(peaks)),peaks.mean()

def _relative_timing_loss(torch,probability,metadata,margin):
    early_peaks=[];window_peaks=[]
    for row,meta in enumerate(metadata):
        start,end=meta["window_start_tick"],meta["window_end_tick"]+1
        early_peaks.append(probability[row,:start].max());window_peaks.append(probability[row,start:end].max())
    early_peaks=torch.stack(early_peaks);window_peaks=torch.stack(window_peaks)
    return (margin+early_peaks-window_peaks).relu().mean(),early_peaks.mean(),window_peaks.mean()

def _memory_losses(torch,functional,model,diagnostics,metadata,labels):
    rows=torch.arange(len(metadata),device=labels.device);end_index=torch.tensor([meta["audio_end_tick"] for meta in metadata],device=labels.device);delay_index=torch.tensor([meta["window_end_tick"] for meta in metadata],device=labels.device)
    end_tokens=diagnostics["token_history"][rows,end_index];delay_tokens=diagnostics["token_history"][rows,delay_index];end_content,end_reconstruction,_=model.decode_memory(end_tokens);delay_content,_,_=model.decode_memory(delay_tokens)
    content_loss=(functional.cross_entropy(end_content,labels)+functional.cross_entropy(delay_content,labels))*.5
    retention_loss=functional.kl_div(delay_content.log_softmax(-1),end_content.detach().softmax(-1),reduction="batchmean")
    targets=[]
    for row,meta in enumerate(metadata):
        heard=diagnostics["encoded_features"][row,:meta["audio_end_tick"]+1].transpose(0,1).unsqueeze(0);targets.append(functional.adaptive_avg_pool1d(heard,model.config.reconstruction_slots).squeeze(0).transpose(0,1))
    target=torch.stack(targets).detach();reconstruction_loss=functional.smooth_l1_loss(end_reconstruction,target)
    return content_loss,reconstruction_loss,retention_loss

def train_short_memory(manifest,output_dir,training=ShortMemoryTrainingConfig(),model_config=ShortMemoryConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed)
    train=ShortMemoryEpisodeDataset(manifest,"train",model_config.sample_rate,model_config.tick_samples,max_records=training.max_train_records);validation=ShortMemoryEpisodeDataset(manifest,"validation",model_config.sample_rate,model_config.tick_samples,max_records=training.max_validation_records)
    loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=create_short_memory_model(model_config).to(device)
    filter_parameters=list(model.cochlea.filters.parameters());filter_ids={id(parameter) for parameter in filter_parameters};main_parameters=[parameter for parameter in model.parameters() if id(parameter) not in filter_ids]
    optimizer=torch.optim.AdamW([{"params":main_parameters,"lr":training.learning_rate},{"params":filter_parameters,"lr":training.learning_rate*training.cochlear_learning_rate_scale}])
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=-1.;step=0
    for epoch in range(training.epochs):
        model.train();total=count=0
        for samples,_,valid,metadata in train_loader:
            samples,valid=samples.to(device),valid.to(device);logits,_,diagnostics=model(samples);early,late,labels=_masks(torch,metadata,samples.shape[1],device);early&=valid;late&=valid
            content_loss,reconstruction_loss,retention_loss=_memory_losses(torch,functional,model,diagnostics,metadata,labels);ctc=_window_ctc(torch,functional,logits,metadata,device);emission_loss,peak=_window_emission_loss(torch,functional,diagnostics["emission_probability"],metadata);timing_loss,early_peak,_=_relative_timing_loss(torch,diagnostics["emission_probability"],metadata,training.timing_margin)
            blank_mask=(early|late)&valid;blank_loss=functional.cross_entropy(logits[blank_mask],torch.zeros(int(blank_mask.sum()),dtype=torch.long,device=device));loss=training.ctc_weight*ctc+training.content_weight*content_loss+training.reconstruction_weight*reconstruction_loss+training.retention_weight*retention_loss+training.blank_weight*blank_loss+training.emission_weight*emission_loss+training.timing_weight*timing_loss
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:print(f"[memory decoder] step={step} loss={float(loss):.4f} content={float(content_loss):.4f} reconstruct={float(reconstruction_loss):.4f} retain={float(retention_loss):.4f} timing={float(timing_loss):.4f} pre/window={float(early_peak):.3f}/{float(peak):.3f}",flush=True)
            if step>=training.max_steps:break
        model.eval();end_correct=memory_correct=retained=emission_correct=emitted=examples=early_count=within=0;latencies=[];stream_match=[];window_peaks=[];prewindow_peaks=[];reconstruction_errors=[]
        with torch.no_grad():
            for samples,_,valid,metadata in validation_loader:
                samples,valid=samples.to(device),valid.to(device);logits,_,diagnostics=model(samples);chosen=logits.argmax(-1)
                state=None;parts=[]
                for tick in range(samples.shape[1]):value,state,_=model.stream_step(samples[:,tick],state);parts.append(value)
                stream_match.append(float(torch.stack(parts,1).sub(logits).abs().max()))
                for row,meta in enumerate(metadata):
                    examples+=1;start,end=meta["window_start_tick"],meta["window_end_tick"];label=meta["label"]
                    end_prediction=int(diagnostics["content_logits"][row,meta["audio_end_tick"]].argmax());memory_prediction=int(diagnostics["content_logits"][row,end].argmax());end_correct+=int(end_prediction==label);memory_correct+=int(memory_prediction==label);retained+=int(end_prediction==memory_prediction);prewindow_peaks.append(float(diagnostics["emission_probability"][row,:start].max()));window_peaks.append(float(diagnostics["emission_probability"][row,start:end+1].max()))
                    _,reconstructed,_=model.decode_memory(diagnostics["token_history"][row,meta["audio_end_tick"]].unsqueeze(0));heard=diagnostics["encoded_features"][row,:meta["audio_end_tick"]+1].transpose(0,1).unsqueeze(0);target=functional.adaptive_avg_pool1d(heard,model_config.reconstruction_slots).squeeze(0).transpose(0,1);reconstruction_errors.append(float(functional.l1_loss(reconstructed[0],target)))
                    positions=torch.nonzero(chosen[row].gt(0)&valid[row]).flatten();early_count+=int(bool((positions<start).any()));window=positions[(positions>=start)&(positions<=end)]
                    if len(window):
                        emitted+=1;within+=1;first=int(window[0]);emission_correct+=int(int(chosen[row,first])-1==label);latencies.append((first-meta["audio_end_tick"])*1000*model_config.tick_samples/model_config.sample_rate)
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_end_digit_accuracy":end_correct/max(examples,1),"validation_memory_digit_accuracy":memory_correct/max(examples,1),"validation_memory_retention_rate":retained/max(examples,1),"validation_reconstruction_l1":sum(reconstruction_errors)/max(len(reconstruction_errors),1),"validation_window_emission_accuracy":emission_correct/max(examples,1),"validation_emission_recall":emitted/max(examples,1),"validation_early_emission_rate":early_count/max(examples,1),"validation_mean_latency_ms":sum(latencies)/max(len(latencies),1),"validation_prewindow_peak_probability":sum(prewindow_peaks)/len(prewindow_peaks),"validation_window_peak_probability":sum(window_peaks)/len(window_peaks),"streaming_max_logit_difference":max(stream_match)};history.append(metrics);print(json.dumps(metrics),flush=True)
        architecture="streaming_cochlear_hierarchical_memory_decoder";score=metrics["validation_memory_digit_accuracy"]+metrics["validation_end_digit_accuracy"]-metrics["validation_early_emission_rate"];checkpoint={"architecture":architecture,"model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history};torch.save(checkpoint,output_dir/"last.pt")
        if score>best:best=score;torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"streaming_cochlear_hierarchical_memory_decoder","steps":step,"best_score":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
