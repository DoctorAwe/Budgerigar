from __future__ import annotations
from dataclasses import asdict
import json,math,random
from pathlib import Path
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset,collate_short_memory
from .streaming_short_memory import ShortMemoryConfig,create_short_memory_model


def _load(torch,path,device):
    try:return torch.load(path,map_location=device,weights_only=False)
    except TypeError:return torch.load(path,map_location=device)


def _accuracy(logits,labels):return int(logits.argmax(-1).eq(labels).sum())


def _end_tokens(torch,history,metadata,device):
    rows=torch.arange(len(metadata),device=device);indices=torch.tensor([meta["audio_end_tick"] for meta in metadata],device=device)
    return history[rows,indices]


def _run_logits(model,samples,metadata,device):
    logits,_,diagnostics=model(samples.to(device));rows=require_torch()[0].arange(len(metadata),device=device);indices=require_torch()[0].tensor([meta["audio_end_tick"] for meta in metadata],device=device)
    return diagnostics["content_logits"][rows,indices]


def audit_short_memory(manifest,checkpoint_path,output_path=None,split="validation",batch_size=16,max_records=300,seed=97):
    torch,_,functional=require_torch();torch.manual_seed(seed);random.seed(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");checkpoint=_load(torch,checkpoint_path,device);config=ShortMemoryConfig(**checkpoint["model_config"]);model=create_short_memory_model(config).to(device);model.load_state_dict(checkpoint["model"]);model.eval()
    dataset=ShortMemoryEpisodeDataset(manifest,split,config.sample_rate,config.tick_samples,max_records=max_records);generator=torch.Generator().manual_seed(seed);loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,generator=generator,collate_fn=collate_short_memory)
    fractions=(.25,.5,.75,1.0);delays_ms=(0,100,200,300,500,1000);tail_ms=(100,200);prefix={value:0 for value in fractions};delay={value:0 for value in delays_ms};layer_correct=[0]*config.token_layers;total=end_correct=cleared_correct=shuffled_correct=full_zero_correct=0;tail_correct={value:0 for value in tail_ms};reconstruction={"correct":0.,"shuffled":0.,"mean_template":0.};reconstruction_examples=0
    with torch.no_grad():
        for samples,_,_,metadata in loader:
            samples=samples.to(device);extra=math.ceil(max(delays_ms)*config.sample_rate/(1000*config.tick_samples));required=max(meta["audio_end_tick"]+extra+1 for meta in metadata);samples=functional.pad(samples,(0,0,0,max(0,required-samples.shape[1])));_,_,diagnostics=model(samples);labels=torch.tensor([meta["label"] for meta in metadata],device=device);rows=torch.arange(len(metadata),device=device);total+=len(metadata)
            for fraction in fractions:
                indices=torch.tensor([meta["audio_start_tick"]+max(0,round((meta["audio_end_tick"]-meta["audio_start_tick"]+1)*fraction)-1) for meta in metadata],device=device);prefix[fraction]+=_accuracy(diagnostics["content_logits"][rows,indices],labels)
            for milliseconds in delays_ms:
                offset=round(milliseconds*config.sample_rate/(1000*config.tick_samples));indices=torch.tensor([meta["audio_end_tick"]+offset for meta in metadata],device=device);delay[milliseconds]+=_accuracy(diagnostics["content_logits"][rows,indices],labels)
            tokens=_end_tokens(torch,diagnostics["token_history"],metadata,device);end_logits,reconstructed,_=model.decode_memory(tokens);end_correct+=_accuracy(end_logits,labels)
            initial=model.initial_tokens.unsqueeze(0).expand_as(tokens);cleared_correct+=_accuracy(model.decode_memory(initial)[0],labels);shuffled_correct+=_accuracy(model.decode_memory(tokens.roll(1,0))[0],labels)
            for layer in range(config.token_layers):
                ablated=tokens.clone();ablated[:,layer]=initial[:,layer];layer_correct[layer]+=_accuracy(model.decode_memory(ablated)[0],labels)
            targets=[]
            for row,meta in enumerate(metadata):
                heard=diagnostics["encoded_features"][row,:meta["audio_end_tick"]+1].transpose(0,1).unsqueeze(0);targets.append(functional.adaptive_avg_pool1d(heard,config.reconstruction_slots).squeeze(0).transpose(0,1))
            targets=torch.stack(targets);mean_target=targets.mean(0,keepdim=True).expand_as(targets);reconstruction["correct"]+=float(functional.l1_loss(reconstructed,targets,reduction="sum"));reconstruction["shuffled"]+=float(functional.l1_loss(reconstructed,targets.roll(1,0),reduction="sum"));reconstruction["mean_template"]+=float(functional.l1_loss(reconstructed,mean_target,reduction="sum"));reconstruction_examples+=targets.numel()
            zeroed=samples.clone();
            for row,meta in enumerate(metadata):zeroed[row,:meta["audio_end_tick"]+1]=0
            full_zero_correct+=_accuracy(_run_logits(model,zeroed,metadata,device),labels)
            for milliseconds in tail_ms:
                ablated=samples.clone();ticks=math.ceil(milliseconds*config.sample_rate/(1000*config.tick_samples))
                for row,meta in enumerate(metadata):ablated[row,max(meta["audio_start_tick"],meta["audio_end_tick"]-ticks+1):meta["audio_end_tick"]+1]=0
                tail_correct[milliseconds]+=_accuracy(_run_logits(model,ablated,metadata,device),labels)
    reconstruction={name:value/max(reconstruction_examples,1) for name,value in reconstruction.items()};prefix_accuracy={str(int(value*100)):prefix[value]/total for value in fractions};retention_accuracy={str(value):delay[value]/total for value in delays_ms};semantic_pass=end_correct/total>.8 and delay[1000]/total>.7 and cleared_correct/total<.2 and shuffled_correct/total<.2 and full_zero_correct/total<.2 and prefix_accuracy["100"]>prefix_accuracy["50"]+.3;acoustic_pass=reconstruction["correct"]<reconstruction["shuffled"] and reconstruction["correct"]<reconstruction["mean_template"]
    report={"architecture":checkpoint.get("architecture"),"checkpoint":str(checkpoint_path),"split":split,"examples":total,"model_config":asdict(config),"prefix_digit_accuracy":prefix_accuracy,"silence_retention_accuracy":retention_accuracy,"memory_ablation_accuracy":{"intact":end_correct/total,"all_cleared":cleared_correct/total,"batch_shuffled":shuffled_correct/total,"layer_cleared":{str(index+1):value/total for index,value in enumerate(layer_correct)}},"input_ablation_accuracy":{"full_input_zeroed":full_zero_correct/total,"last_ms_zeroed":{str(value):tail_correct[value]/total for value in tail_ms}},"reconstruction_l1":reconstruction,"reconstruction_margin":{"correct_vs_shuffled":reconstruction["shuffled"]-reconstruction["correct"],"correct_vs_mean_template":reconstruction["mean_template"]-reconstruction["correct"]},"semantic_memory_pass":semantic_pass,"acoustic_memory_pass":acoustic_pass,"audit_pass":semantic_pass and acoustic_pass}
    if output_path is not None:output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True);output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
