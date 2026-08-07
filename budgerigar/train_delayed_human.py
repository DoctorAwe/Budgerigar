from __future__ import annotations
from dataclasses import asdict,dataclass
import json,math,random,time
from pathlib import Path
from .auditory_evaluator import AuditoryEvaluatorConfig,create_auditory_evaluator
from .delayed_human_speech import DelayedHumanSpeechConfig,create_delayed_human_speech
from .delayed_repeat_data import DelayedRepeatDataset,collate_delayed_repeat
from .neural_echo import require_torch
from .train_human_speech import _acoustic_loss


@dataclass(frozen=True)
class DelayedHumanTrainingConfig:
    batch_size:int=6
    learning_rate:float=3e-4
    max_steps:int=500
    epochs:int=30
    max_train_records:int=1000
    max_validation_records:int=100
    thinking_ms:tuple[int,int]=(140,220)
    seed:int=181


def train_delayed_human(manifest,output_dir,human_checkpoint,evaluator_checkpoint,training=DelayedHumanTrainingConfig(),model_config=DelayedHumanSpeechConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed);make=lambda split,limit:DelayedRepeatDataset(manifest,split,model_config.human.sample_rate,model_config.human.tick_samples,training.thinking_ms,max_records=limit);train=make("train",training.max_train_records);validation=make("validation",training.max_validation_records);loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_delayed_repeat);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_delayed_repeat)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");base=torch.load(human_checkpoint,map_location=device,weights_only=False);model=create_delayed_human_speech(model_config).to(device);model.human.load_state_dict(base["model"]);model.human.requires_grad_(False);model.human.eval();evaluation=torch.load(evaluator_checkpoint,map_location=device,weights_only=False);evaluator=create_auditory_evaluator(AuditoryEvaluatorConfig(**evaluation["model_config"])).to(device);evaluator.load_state_dict(evaluation["model"]);evaluator.eval();evaluator.requires_grad_(False);parameters=[value for value in model.parameters() if value.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=training.learning_rate,betas=(.8,.99));output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=math.inf;step=0;started=time.perf_counter()
    for epoch in range(training.epochs):
        model.train();model.human.eval();total=count=0
        for inputs,targets,_,metadata in train_loader:
            inputs=inputs.to(device);targets=targets.to(device);outputs=model(inputs)[0];losses=[]
            for row,item in enumerate(metadata):
                start=item["repeat_start"];end=item["total_ticks"];predicted=outputs[row,start:end].flatten();target=targets[row,start:end].flatten();early=outputs[row,:start].square().mean().sqrt();acoustic=_acoustic_loss(torch,predicted.float(),target.float());target_features=functional.normalize(evaluator.features(target.unsqueeze(0)).detach(),dim=-1);output_features=functional.normalize(evaluator.features(predicted.unsqueeze(0)),dim=-1);perceptual=1-(output_features*target_features).sum(-1).mean();label=torch.tensor([item["label"]],device=device);content=functional.cross_entropy(evaluator(predicted.unsqueeze(0)),label);losses.append(acoustic+.5*perceptual+.25*content+2*early)
            loss=torch.stack(losses).mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:elapsed=time.perf_counter()-started;print(f"[delayed human] step={step} loss={float(loss.detach()):.4f} sec/10={elapsed/(1 if step==1 else 10):.2f}",flush=True);started=time.perf_counter()
            if step>=training.max_steps:break
        model.eval();acoustic_values=[];early_values=[];correct=examples=0;onset_errors=[];example=None
        with torch.no_grad():
            for inputs,targets,_,metadata in validation_loader:
                inputs=inputs.to(device);targets=targets.to(device);outputs=model(inputs)[0]
                for row,item in enumerate(metadata):
                    start=item["repeat_start"];end=item["total_ticks"];predicted=outputs[row,start:end].flatten();target=targets[row,start:end].flatten();acoustic_values.append(float(_acoustic_loss(torch,predicted.float(),target.float())));early_values.append(float(outputs[row,:start].square().mean().sqrt()));correct+=int(evaluator(predicted.unsqueeze(0)).argmax(-1).item()==item["label"]);examples+=1;energy=outputs[row].square().mean(-1).sqrt();positions=torch.nonzero(energy>.01).flatten();onset=int(positions[0]) if len(positions) else end;onset_errors.append(abs(onset-start))
                    if example is None:example={"input":inputs[row,:end].flatten().cpu(),"target":targets[row,:end].flatten().cpu(),"output":outputs[row,:end].flatten().cpu(),"sample_rate":model_config.human.sample_rate,"metadata":item}
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_repeat_acoustic_loss":sum(acoustic_values)/len(acoustic_values),"validation_early_output_rms":sum(early_values)/len(early_values),"validation_output_digit_accuracy":correct/max(examples,1),"validation_onset_mae_ms":10*sum(onset_errors)/len(onset_errors)};history.append(metrics);print(json.dumps(metrics),flush=True);checkpoint={"architecture":"delayed_single_memory_human_speech_s1","model":model.state_dict(),"model_config":model.export_config(),"training_config":asdict(training),"human_checkpoint":str(human_checkpoint),"history":history};torch.save(checkpoint,output_dir/"last.pt");selection=metrics["validation_repeat_acoustic_loss"]+1-metrics["validation_output_digit_accuracy"]+5*metrics["validation_early_output_rms"]
        if selection<best:best=selection;torch.save(checkpoint,output_dir/"best.pt");torch.save(example,output_dir/"best_validation_example.pt")
        if step>=training.max_steps:break
    report={"architecture":"delayed_single_memory_human_speech_s1","steps":step,"best_selection_score":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return report
