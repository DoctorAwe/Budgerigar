from __future__ import annotations
from dataclasses import asdict,dataclass
import json,math,random
from pathlib import Path
from .auditory_evaluator import AuditoryEvaluatorConfig,create_auditory_evaluator
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset,collate_short_memory


@dataclass(frozen=True)
class AuditoryEvaluatorTrainingConfig:
    batch_size:int=32
    learning_rate:float=5e-4
    max_steps:int=1000
    epochs:int=40
    max_train_records:int=2400
    max_validation_records:int=300
    seed:int=131


def robust_waveform_augmentation(torch,waveform):
    gain=.7+.6*torch.rand(len(waveform),1,device=waveform.device);noise=torch.randn_like(waveform)*(.001+.004*torch.rand(len(waveform),1,device=waveform.device));shift=int(torch.randint(-160,161,(1,),device=waveform.device));shifted=torch.roll(waveform,shift,-1)
    if shift>0:shifted=torch.cat([torch.zeros_like(shifted[:,:shift]),shifted[:,shift:]],-1)
    elif shift<0:shifted=torch.cat([shifted[:,:shift],torch.zeros_like(shifted[:,shift:])],-1)
    if bool(torch.rand((),device=waveform.device)<.5):shifted=torch.nn.functional.interpolate(torch.nn.functional.avg_pool1d(shifted.unsqueeze(1),3,1,1),size=waveform.shape[-1],mode="linear",align_corners=False).squeeze(1)
    return (gain*shifted+noise).clamp(-1,1)


def train_auditory_evaluator(manifest,output_dir,training=AuditoryEvaluatorTrainingConfig(),model_config=AuditoryEvaluatorConfig()):
    torch,_,functional=require_torch();torch.manual_seed(training.seed);random.seed(training.seed);dataset=lambda split,limit:ShortMemoryEpisodeDataset(manifest,split,model_config.sample_rate,160,max_records=limit);train=dataset("train",training.max_train_records);validation=dataset("validation",training.max_validation_records);loader=torch.utils.data.DataLoader;train_loader=loader(train,batch_size=training.batch_size,shuffle=True,collate_fn=collate_short_memory);validation_loader=loader(validation,batch_size=training.batch_size,collate_fn=collate_short_memory)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=create_auditory_evaluator(model_config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training.learning_rate,weight_decay=1e-3);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);history=[];best=-math.inf;step=0
    for epoch in range(training.epochs):
        model.train();total=count=0
        for ticks,_,_,metadata in train_loader:
            waveform=robust_waveform_augmentation(torch,ticks.to(device).flatten(1));labels=torch.tensor([item["label"] for item in metadata],device=device);loss=functional.cross_entropy(model(waveform),labels);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();step+=1;total+=float(loss.detach());count+=1
            if step==1 or step%20==0:print(f"[auditory evaluator] step={step} loss={float(loss.detach()):.4f}",flush=True)
            if step>=training.max_steps:break
        model.eval();clean=robust=examples=0
        with torch.no_grad():
            for ticks,_,_,metadata in validation_loader:
                waveform=ticks.to(device).flatten(1);labels=torch.tensor([item["label"] for item in metadata],device=device);clean+=int((model(waveform).argmax(-1)==labels).sum());robust+=int((model(robust_waveform_augmentation(torch,waveform)).argmax(-1)==labels).sum());examples+=len(labels)
        metrics={"epoch":epoch+1,"step":step,"train_loss":total/max(count,1),"validation_clean_accuracy":clean/max(examples,1),"validation_robust_accuracy":robust/max(examples,1)};history.append(metrics);print(json.dumps(metrics),flush=True);checkpoint={"architecture":"frozen_real_audio_digit_evaluator","model":model.state_dict(),"model_config":asdict(model_config),"training_config":asdict(training),"history":history};torch.save(checkpoint,output_dir/"last.pt");score=min(metrics["validation_clean_accuracy"],metrics["validation_robust_accuracy"])
        if score>best:best=score;torch.save(checkpoint,output_dir/"best.pt")
        if step>=training.max_steps:break
    report={"architecture":"frozen_real_audio_digit_evaluator","steps":step,"best_min_accuracy":best,"history":history};(output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
