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
    blank_weight:float=.25
    content_weight:float=1.0
    seed:int=83

def _masks(torch,metadata,ticks,device):
    early=torch.zeros(len(metadata),ticks,dtype=torch.bool,device=device);hold=torch.zeros_like(early);late=torch.zeros_like(early);labels=[]
    for row,meta in enumerate(metadata):
        early[row,:meta["window_start_tick"]]=True;hold[row,meta["audio_end_tick"]+1:meta["window_end_tick"]+1]=True;late[row,meta["window_end_tick"]+1:]=True;labels.append(meta["label"])
    return early,hold,late,torch.tensor(labels,device=device)

def _window_ctc(torch,functional,logits,metadata,device):
    losses=[]
    for row,meta in enumerate(metadata):
        start,end=meta["window_start_tick"],meta["window_end_tick"]+1;log_probability=logits[row,start:end].log_softmax(-1).unsqueeze(1)
        target=torch.tensor([meta["label"]+1],device=device);length=torch.tensor([end-start],device=device);target_length=torch.ones(1,dtype=torch.long,device=device)
        losses.append(functional.ctc_loss(log_probability,target,length,target_length,blank=0,zero_infinity=True))
    return torch.stack(losses).mean()

def train_short_memory(manifest,output_dir,training=ShortMemoryTrainingConfig(),model_config=ShortMemoryConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed)
    train=ShortMemoryEpisodeDataset(manifest,"train",model_config.sample_rate,model_config.tick_samples,max_records=training.max_train_records);validation=ShortMemoryEpisodeDataset(manifest,"validation",model_config.sample_rate,model_config.tick_samples,max_records=training.max_validation_records)
    loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=create_short_memory_model(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=-1.;step=0
    for epoch in range(training.epochs):
        model.train();total=count=0
        for samples,_,valid,metadata in train_loader:
            samples,valid=samples.to(device),valid.to(device);logits,_,diagnostics=model(samples);early,hold,late,labels=_masks(torch,metadata,samples.shape[1],device);early&=valid;hold&=valid;late&=valid
            expanded=labels.unsqueeze(1).expand_as(hold);content_loss=functional.cross_entropy(diagnostics["content_logits"][hold],expanded[hold]);ctc=_window_ctc(torch,functional,logits,metadata,device)
            blank_mask=(early|late)&valid;blank_loss=functional.cross_entropy(logits[blank_mask],torch.zeros(int(blank_mask.sum()),dtype=torch.long,device=device));loss=ctc+training.content_weight*content_loss+training.blank_weight*blank_loss
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:print(f"[short memory] step={step} loss={float(loss):.4f} ctc={float(ctc):.4f} content={float(content_loss):.4f} blank={float(blank_loss):.4f}",flush=True)
            if step>=training.max_steps:break
        model.eval();memory_correct=emission_correct=emitted=examples=early_count=within=0;latencies=[];stream_match=[]
        with torch.no_grad():
            for samples,_,valid,metadata in validation_loader:
                samples,valid=samples.to(device),valid.to(device);logits,_,diagnostics=model(samples);chosen=logits.argmax(-1)
                state=None;parts=[]
                for tick in range(samples.shape[1]):value,state,_=model.stream_step(samples[:,tick],state);parts.append(value)
                stream_match.append(float(torch.stack(parts,1).sub(logits).abs().max()))
                for row,meta in enumerate(metadata):
                    examples+=1;start,end=meta["window_start_tick"],meta["window_end_tick"];label=meta["label"]
                    memory_prediction=int(diagnostics["content_logits"][row,start:end+1].mean(0).argmax());memory_correct+=int(memory_prediction==label)
                    positions=torch.nonzero(chosen[row].gt(0)&valid[row]).flatten();early_count+=int(bool((positions<start).any()));window=positions[(positions>=start)&(positions<=end)]
                    if len(window):
                        emitted+=1;within+=1;first=int(window[0]);emission_correct+=int(int(chosen[row,first])-1==label);latencies.append((first-meta["audio_end_tick"])*1000*model_config.tick_samples/model_config.sample_rate)
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_memory_digit_accuracy":memory_correct/max(examples,1),"validation_window_emission_accuracy":emission_correct/max(examples,1),"validation_emission_recall":emitted/max(examples,1),"validation_early_emission_rate":early_count/max(examples,1),"validation_mean_latency_ms":sum(latencies)/max(len(latencies),1),"streaming_max_logit_difference":max(stream_match)};history.append(metrics);print(json.dumps(metrics),flush=True)
        score=metrics["validation_window_emission_accuracy"]-metrics["validation_early_emission_rate"];checkpoint={"architecture":"streaming_cochlear_16token_memory_window_ctc","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history};torch.save(checkpoint,output_dir/"last.pt")
        if score>best:best=score;torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"streaming_cochlear_16token_memory_window_ctc","steps":step,"best_score":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
