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
    semantic_learning_rate_scale:float=.1
    max_steps:int=500
    semantic_warmup_steps:int=250
    epochs:int=30
    max_train_records:int=1000
    max_validation_records:int=100
    thinking_ms:tuple[int,int]=(140,220)
    seed:int=181


def train_delayed_human(manifest,output_dir,human_checkpoint,semantic_checkpoint,evaluator_checkpoint,training=DelayedHumanTrainingConfig(),model_config=DelayedHumanSpeechConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed);make=lambda split,limit:DelayedRepeatDataset(manifest,split,model_config.human.sample_rate,model_config.human.tick_samples,training.thinking_ms,max_records=limit);train=make("train",training.max_train_records);validation=make("validation",training.max_validation_records);loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_delayed_repeat);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_delayed_repeat)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");base=torch.load(human_checkpoint,map_location=device,weights_only=False);memory=torch.load(semantic_checkpoint,map_location=device,weights_only=False);model=create_delayed_human_speech(model_config).to(device);model.human.load_state_dict(base["model"]);model.semantic.load_state_dict(memory["model"]);model.human.requires_grad_(False);model.semantic.requires_grad_(False)
    model.semantic.initial_tokens.requires_grad_(True);model.semantic.level_embedding.requires_grad_(True);model.semantic.content_query.requires_grad_(True)
    semantic_modules=(model.semantic.token_norms,model.semantic.token_attentions,model.semantic.update_gates,model.semantic.update_candidates,model.semantic.refinements,model.semantic.decoder_attention,model.semantic.decoder_norm,model.semantic.decoder_refinement,model.semantic.content)
    for module in semantic_modules:module.requires_grad_(True)
    model.human.eval();model.semantic.eval();evaluation=torch.load(evaluator_checkpoint,map_location=device,weights_only=False);evaluator=create_auditory_evaluator(AuditoryEvaluatorConfig(**evaluation["model_config"])).to(device);evaluator.load_state_dict(evaluation["model"]);evaluator.eval();evaluator.requires_grad_(False);semantic_parameters=[value for value in model.semantic.parameters() if value.requires_grad];new_parameters=[value for name,value in model.named_parameters() if value.requires_grad and not name.startswith("semantic.")];parameters=new_parameters+semantic_parameters;optimizer=torch.optim.AdamW([{"params":new_parameters,"lr":training.learning_rate},{"params":semantic_parameters,"lr":training.learning_rate*training.semantic_learning_rate_scale}],betas=(.8,.99));output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=math.inf;step=0;started=time.perf_counter();print(f"[delayed human] trainable new={sum(x.numel() for x in new_parameters)} semantic={sum(x.numel() for x in semantic_parameters)} semantic_lr={training.learning_rate*training.semantic_learning_rate_scale:g}",flush=True)
    for epoch in range(training.epochs):
        # Gradients must pass through the frozen motor GRU into recall_projection.
        # cuDNN forbids RNN backward after an eval-mode forward, so keep the
        # frozen human module in training execution mode; requires_grad remains
        # False and this architecture contains no dropout or batch norm.
        model.train();model.human.train();model.semantic.eval();total=count=0
        for inputs,targets,_,metadata in train_loader:
            inputs=inputs.to(device);targets=targets.to(device);warmup=step<training.semantic_warmup_steps
            if warmup:_,_,diagnostics=model.remember(inputs);outputs=None
            else:outputs,_,diagnostics=model(inputs)
            losses=[]
            for row,item in enumerate(metadata):
                start=item["repeat_start"];label=torch.tensor([item["label"]],device=device);semantic_end=functional.cross_entropy(diagnostics["content_logits"][row,item["source_ticks"]-1].unsqueeze(0),label);semantic_delay=functional.cross_entropy(diagnostics["content_logits"][row,start-1].unsqueeze(0),label);semantic_loss=.5*(semantic_end+semantic_delay)
                if warmup:losses.append(semantic_loss);continue
                end=item["total_ticks"];predicted=outputs[row,start:end].flatten();target=targets[row,start:end].flatten();early=outputs[row,:start].square().mean().sqrt();early_peak=outputs[row,:start].square().mean(-1).sqrt().max();acoustic=_acoustic_loss(torch,predicted.float(),target.float());target_features=functional.normalize(evaluator.features(target.unsqueeze(0)).detach(),dim=-1);output_features=functional.normalize(evaluator.features(predicted.unsqueeze(0)),dim=-1);perceptual=1-(output_features*target_features).sum(-1).mean();content=functional.cross_entropy(evaluator(predicted.unsqueeze(0)),label);losses.append(acoustic+.5*perceptual+.25*content+.5*semantic_loss+5*early+2*early_peak)
            loss=torch.stack(losses).mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%10==0:elapsed=time.perf_counter()-started;print(f"[delayed human] phase={'semantic' if warmup else 'waveform'} step={step} loss={float(loss.detach()):.4f} sec/10={elapsed/(1 if step==1 else 10):.2f}",flush=True);started=time.perf_counter()
            if step>=training.max_steps:break
        model.eval();acoustic_values=[];early_values=[];correct=semantic_end_correct=semantic_delay_correct=semantic_cleared_correct=examples=0;onset_errors=[];example=None
        with torch.no_grad():
            for inputs,targets,_,metadata in validation_loader:
                inputs=inputs.to(device);targets=targets.to(device);outputs,_,diagnostics=model(inputs)
                for row,item in enumerate(metadata):
                    start=item["repeat_start"];end=item["total_ticks"];predicted=outputs[row,start:end].flatten();target=targets[row,start:end].flatten();label=item["label"];acoustic_values.append(float(_acoustic_loss(torch,predicted.float(),target.float())));early_values.append(float(outputs[row,:start].square().mean().sqrt()));correct+=int(evaluator(predicted.unsqueeze(0)).argmax(-1).item()==label);end_tokens=diagnostics["token_history"][row,item["source_ticks"]-1].unsqueeze(0);delay_tokens=diagnostics["token_history"][row,start-1].unsqueeze(0);semantic_end_correct+=int(model.semantic.decode_memory(end_tokens)[0].argmax(-1).item()==label);semantic_delay_correct+=int(model.semantic.decode_memory(delay_tokens)[0].argmax(-1).item()==label);cleared=model.semantic.initial_tokens.unsqueeze(0);semantic_cleared_correct+=int(model.semantic.decode_memory(cleared)[0].argmax(-1).item()==label);examples+=1;energy=outputs[row].square().mean(-1).sqrt();positions=torch.nonzero(energy>.01).flatten();onset=int(positions[0]) if len(positions) else end;onset_errors.append(abs(onset-start))
                    if example is None:example={"input":inputs[row,:end].flatten().cpu(),"target":targets[row,:end].flatten().cpu(),"output":outputs[row,:end].flatten().cpu(),"sample_rate":model_config.human.sample_rate,"metadata":item}
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_repeat_acoustic_loss":sum(acoustic_values)/len(acoustic_values),"validation_early_output_rms":sum(early_values)/len(early_values),"validation_output_digit_accuracy":correct/max(examples,1),"validation_semantic_end_accuracy":semantic_end_correct/max(examples,1),"validation_semantic_delay_accuracy":semantic_delay_correct/max(examples,1),"validation_semantic_cleared_accuracy":semantic_cleared_correct/max(examples,1),"validation_onset_mae_ms":10*sum(onset_errors)/len(onset_errors)};history.append(metrics);print(json.dumps(metrics),flush=True);checkpoint={"architecture":"delayed_verified_hierarchical_memory_human_speech_s1","model":model.state_dict(),"model_config":model.export_config(),"training_config":asdict(training),"human_checkpoint":str(human_checkpoint),"semantic_checkpoint":str(semantic_checkpoint),"history":history};torch.save(checkpoint,output_dir/"last.pt");selection=metrics["validation_repeat_acoustic_loss"]+1-metrics["validation_output_digit_accuracy"]+5*metrics["validation_early_output_rms"]+max(.7-metrics["validation_semantic_end_accuracy"],0)+max(.7-metrics["validation_semantic_delay_accuracy"],0)+max(metrics["validation_semantic_cleared_accuracy"]-.2,0)
        if selection<best:best=selection;torch.save(checkpoint,output_dir/"best.pt");torch.save(example,output_dir/"best_validation_example.pt")
        if step>=training.max_steps:break
    report={"architecture":"delayed_verified_hierarchical_memory_human_speech_s1","steps":step,"best_selection_score":best,"human_checkpoint":str(human_checkpoint),"semantic_checkpoint":str(semantic_checkpoint),"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return report
