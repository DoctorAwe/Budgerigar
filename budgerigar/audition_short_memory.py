from __future__ import annotations
from pathlib import Path
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset
from .streaming_short_memory import ShortMemoryConfig,create_short_memory_model


def _load_checkpoint(torch,path,device):
    try:return torch.load(path,map_location=device,weights_only=False)
    except TypeError:return torch.load(path,map_location=device)


def render_short_memory_audition(manifest,checkpoint_path,output_dir,index=0,split="validation"):
    """Render a stereo, full-timeline audit: input left and emitted token tones right.

    The right channel is deliberately a sonification of token output, not synthesized
    speech. A silent right channel means that the model emitted no non-blank token.
    """
    torch,_,_=require_torch();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");checkpoint=_load_checkpoint(torch,checkpoint_path,device)
    config=ShortMemoryConfig(**checkpoint["model_config"]);model=create_short_memory_model(config).to(device);model.load_state_dict(checkpoint["model"]);model.eval()
    dataset=ShortMemoryEpisodeDataset(manifest,split,config.sample_rate,config.tick_samples,preload=True);samples,_,metadata=dataset[index%len(dataset)]
    with torch.no_grad():logits,_,diagnostics=model(samples.unsqueeze(0).to(device))
    chosen=logits[0].argmax(-1).cpu();emission=diagnostics["emission_probability"][0].cpu();content=diagnostics["content_logits"][0].softmax(-1).cpu();emitted_ticks=torch.nonzero(chosen.gt(0)).flatten();runs=[]
    for tick in emitted_ticks.tolist():
        token=int(chosen[tick])-1
        if not runs or runs[-1]["digit"]!=token or runs[-1]["end_tick"]+1!=tick:runs.append({"digit":token,"start_tick":tick,"end_tick":tick})
        else:runs[-1]["end_tick"]=tick
    input_audio=samples.flatten().cpu();minimum_tone=round(.12*config.sample_rate);last_tone=max([run["start_tick"]*config.tick_samples+max(minimum_tone,(run["end_tick"]-run["start_tick"]+1)*config.tick_samples) for run in runs],default=0);total=max(len(input_audio),last_tone);input_audio=torch.nn.functional.pad(input_audio,(0,total-len(input_audio)));output_audio=torch.zeros_like(input_audio);sample_positions=torch.arange(total,dtype=input_audio.dtype)
    for run in runs:
        start=run["start_tick"]*config.tick_samples;duration=max(minimum_tone,(run["end_tick"]-run["start_tick"]+1)*config.tick_samples);end=min(start+duration,total);frequency=300.0+70.0*run["digit"]
        output_audio[start:end]=.18*torch.sin(2*torch.pi*frequency*sample_positions[start:end]/config.sample_rate)
    stereo=torch.stack([input_audio,output_audio]);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);path=output_dir/f"{metadata['id']}.input_left_token_right.wav"
    import torchaudio
    torchaudio.save(str(path),stereo,config.sample_rate)
    end_tick=metadata["audio_end_tick"];delay_tick=metadata["window_end_tick"]
    return {"path":str(path),"id":metadata["id"],"target_digit":metadata["label"],"predicted_at_audio_end":int(content[end_tick].argmax()),"predicted_after_delay":int(content[delay_tick].argmax()),"audio_end_ms":(end_tick+1)*config.tick_samples*1000/config.sample_rate,"allowed_window_ms":[metadata["window_start_tick"]*config.tick_samples*1000/config.sample_rate,(metadata["window_end_tick"]+1)*config.tick_samples*1000/config.sample_rate],"peak_emission_probability":float(emission.max()),"emitted_runs":runs,"sonification_minimum_duration_ms":120,"channel_layout":"left=input waveform; right=model token sonification (digit frequencies 300+70*d Hz)"}
