from __future__ import annotations
from dataclasses import asdict,dataclass
import json,random
from pathlib import Path
from .codec_copy_data import CodecCopyEpisodeDataset,collate_codec_copy
from .codec_copy_model import CodecCopyConfig,create_codec_copy_model
from .neural_echo import require_torch

@dataclass(frozen=True)
class CodecCopyTrainingConfig:
    batch_size:int=2
    learning_rate:float=3e-4
    epochs:int=20
    max_steps:int=400
    max_train_records:int=256
    max_validation_records:int=64
    teacher_start:float=0.9
    teacher_end:float=0.35
    voice_weight:float=0.25
    gradient_clip:float=1.0
    seed:int=53

def train_codec_copy(manifest,output_dir,training=CodecCopyTrainingConfig(),model_config=CodecCopyConfig()):
    torch,_,functional=require_torch(); torch.manual_seed(training.seed); random.seed(training.seed)
    train=CodecCopyEpisodeDataset(manifest,"train",max_records=training.max_train_records,preload=True)
    validation=CodecCopyEpisodeDataset(manifest,"validation",max_records=training.max_validation_records,preload=True)
    loader=torch.utils.data.DataLoader
    train_loader=loader(train,batch_size=training.batch_size,shuffle=True,num_workers=0,collate_fn=collate_codec_copy)
    validation_loader=loader(validation,batch_size=training.batch_size,shuffle=False,num_workers=0,collate_fn=collate_codec_copy)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=create_codec_copy_model(model_config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=1e-4)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); history=[]; best=-1.; step=0
    for epoch in range(training.epochs):
        model.train(); total=count=0
        for inputs,targets,voice,valid,_ in train_loader:
            inputs,targets,voice,valid=inputs.to(device),targets.to(device),voice.to(device),valid.to(device)
            progress=min(step/max(training.max_steps-1,1),1.); teacher=training.teacher_start+(training.teacher_end-training.teacher_start)*progress
            predicted,voice_logits,_=model(inputs,targets,teacher); repeat=targets.ge(0)
            token_loss=functional.cross_entropy(predicted[repeat],targets[repeat]); voice_loss=functional.binary_cross_entropy_with_logits(voice_logits[valid],voice[valid])
            loss=token_loss+training.voice_weight*voice_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),training.gradient_clip); optimizer.step()
            step+=1; total+=float(loss.detach()); count+=1
            if step==1 or step%10==0: print(f"[learned repeat] step={step} loss={float(loss):.4f} token={float(token_loss):.4f} voice={float(voice_loss):.4f} teacher={teacher:.3f}",flush=True)
            if step>=training.max_steps: break
        model.eval(); correct=tokens=exact=examples=0; early=[]; repeat_recall=[]
        with torch.no_grad():
            for inputs,targets,voice,valid,_ in validation_loader:
                inputs,targets,voice,valid=inputs.to(device),targets.to(device),voice.to(device),valid.to(device)
                predicted,voice_logits,_=model(inputs); chosen=predicted.argmax(-1); mask=targets.ge(0)
                correct+=int(chosen[mask].eq(targets[mask]).sum()); tokens+=int(mask.sum())
                exact+=int((chosen.eq(targets)|~mask).all((1,2)).sum()); examples+=len(inputs)
                probability=voice_logits.sigmoid(); silent=(voice<.5)&valid; speaking=(voice>=.5)&valid
                early.append(float((probability[silent]>=.5).float().mean())); repeat_recall.append(float((probability[speaking]>=.5).float().mean()))
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_token_accuracy":correct/max(tokens,1),"validation_exact_utterance_rate":exact/max(examples,1),"validation_early_voice_rate":sum(early)/len(early),"validation_repeat_voice_recall":sum(repeat_recall)/len(repeat_recall)}
        history.append(metrics); print(json.dumps(metrics),flush=True)
        checkpoint={"architecture":"fully_learned_neural_repeater","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history}
        torch.save(checkpoint,output_dir/"last.pt")
        if metrics["validation_token_accuracy"]>best: best=metrics["validation_token_accuracy"]; torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps: break
    report={"architecture":"fully_learned_neural_repeater","steps":step,"best_validation_token_accuracy":best,"history":history}
    (output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
