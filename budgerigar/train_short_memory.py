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
    blank_weight:float=.15
    seed:int=83

def train_short_memory(manifest,output_dir,training=ShortMemoryTrainingConfig(),model_config=ShortMemoryConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed)
    train=ShortMemoryEpisodeDataset(manifest,"train",model_config.sample_rate,model_config.tick_samples,max_records=training.max_train_records);validation=ShortMemoryEpisodeDataset(manifest,"validation",model_config.sample_rate,model_config.tick_samples,max_records=training.max_validation_records)
    loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=create_short_memory_model(model_config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=-1.;step=0
    for epoch in range(training.epochs):
        model.train();total=count=0
        for samples,targets,valid,_ in train_loader:
            samples,targets,valid=samples.to(device),targets.to(device),valid.to(device);logits,_,_=model(samples);token=targets.gt(0)&valid;blank=targets.eq(0)&valid
            token_loss=functional.cross_entropy(logits[token],targets[token]);blank_loss=functional.cross_entropy(logits[blank],targets[blank]);loss=token_loss+training.blank_weight*blank_loss
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:print(f"[short memory] step={step} loss={float(loss):.4f} token={float(token_loss):.4f} blank={float(blank_loss):.4f}",flush=True)
            if step>=training.max_steps:break
        model.eval();correct=emitted=examples=0;early=0;onset=[];stream_match=[]
        with torch.no_grad():
            for samples,_,valid,metadata in validation_loader:
                samples=samples.to(device);logits,_,_=model(samples);chosen=logits.argmax(-1)
                # Verify repeated stream_step calls are numerically identical to batched causal forward.
                state=None;parts=[]
                for tick in range(samples.shape[1]):value,state,_=model.stream_step(samples[:,tick],state);parts.append(value)
                stream_match.append(float(torch.stack(parts,1).sub(logits).abs().max()))
                for row,meta in enumerate(metadata):
                    examples+=1;end=meta["audio_end_tick"];positions=torch.nonzero(chosen[row].gt(0)&valid[row]).flatten()
                    early+=int(bool((positions<=end).any()))
                    after=positions[positions>end]
                    if len(after):
                        emitted+=1;first=int(after[0]);correct+=int(int(chosen[row,first])-1==meta["label"]);onset.append(abs(first-meta["emission_tick"]))
        tick_ms=1000*model_config.tick_samples/model_config.sample_rate;metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_token_accuracy":correct/max(emitted,1),"validation_emission_recall":emitted/max(examples,1),"validation_early_emission_rate":early/max(examples,1),"validation_onset_mae_ms":(sum(onset)/max(len(onset),1))*tick_ms,"streaming_max_logit_difference":max(stream_match)};history.append(metrics);print(json.dumps(metrics),flush=True)
        score=metrics["validation_token_accuracy"]*metrics["validation_emission_recall"]-metrics["validation_early_emission_rate"];checkpoint={"architecture":"streaming_cochlear_short_memory_token","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history};torch.save(checkpoint,output_dir/"last.pt")
        if score>best:best=score;torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"streaming_cochlear_short_memory_token","steps":step,"best_score":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
