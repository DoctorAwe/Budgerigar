from __future__ import annotations
import json,random
from pathlib import Path
from .human_speech_model import HumanSpeechConfig,create_human_speech_model
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset


def render_human_digit_audition(manifest,checkpoint,output_dir,split="validation",separator_ms=350,seed=163):
    """Render one real and one model-generated example for every digit.

    Raw reconstructions are preserved.  The *_listen files apply only a shared
    peak gain per example so quiet synthesis remains audible during inspection.
    """
    torch,_,_=require_torch();import torchaudio
    random.seed(seed);checkpoint=Path(checkpoint);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False);config=HumanSpeechConfig(**payload["model_config"]);model=create_human_speech_model(config);model.load_state_dict(payload["model"]);model.eval()
    dataset=ShortMemoryEpisodeDataset(manifest,split,config.sample_rate,config.tick_samples,max_records=None);indices={}
    for index,row in enumerate(dataset.rows):indices.setdefault(int(row["label"]),index)
    missing=[digit for digit in range(10) if digit not in indices]
    if missing:raise ValueError(f"missing digits in {split}: {missing}")
    silence=torch.zeros(round(config.sample_rate*separator_ms/1000));real_parts=[];generated_parts=[];records=[]
    with torch.no_grad():
        for digit in range(10):
            ticks,_,metadata=dataset[indices[digit]];generated=model(ticks.unsqueeze(0))[0][0].flatten().cpu();real=ticks.flatten().cpu();start=metadata["audio_start_tick"]*config.tick_samples;end=(metadata["audio_end_tick"]+1)*config.tick_samples;real=real[start:end];generated=generated[start:end]
            real_path=output_dir/f"digit_{digit}_input.wav";raw_path=output_dir/f"digit_{digit}_output_raw.wav";listen_path=output_dir/f"digit_{digit}_output_listen.wav";gain=min(.9/generated.abs().max().clamp_min(1e-5).item(),20.0);audible=(generated*gain).clamp(-1,1)
            torchaudio.save(str(real_path),real.unsqueeze(0),config.sample_rate);torchaudio.save(str(raw_path),generated.unsqueeze(0),config.sample_rate);torchaudio.save(str(listen_path),audible.unsqueeze(0),config.sample_rate);real_parts.extend([real,silence]);generated_parts.extend([audible,silence]);records.append({"digit":digit,"id":metadata["id"],"input_path":str(real_path),"output_raw_path":str(raw_path),"output_listen_path":str(listen_path),"audition_gain":gain})
    input_sequence=output_dir/"digits_0_to_9_input.wav";output_sequence=output_dir/"digits_0_to_9_output_listen.wav";torchaudio.save(str(input_sequence),torch.cat(real_parts).unsqueeze(0),config.sample_rate);torchaudio.save(str(output_sequence),torch.cat(generated_parts).unsqueeze(0),config.sample_rate)
    report={"checkpoint":str(checkpoint),"split":split,"sample_rate":config.sample_rate,"separator_ms":separator_ms,"input_sequence":str(input_sequence),"output_sequence":str(output_sequence),"records":records};(output_dir/"digit_audition.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");return report
