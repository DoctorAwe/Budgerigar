from __future__ import annotations
import random
from .neural_echo import require_torch
from .short_memory_data import ShortMemoryEpisodeDataset


class DelayedRepeatDataset(ShortMemoryEpisodeDataset):
    """Continuous input/output timeline with no action or phase labels."""
    def __getitem__(self,index):
        torch,_,_=require_torch();row=self.rows[index];waveform=self.cache.get(row["id"])
        if waveform is None:waveform=self._load(row)
        leading=random.randrange(0,self.tick_samples*4);audio=torch.nn.functional.pad(waveform,(leading,0));source_ticks=(len(audio)+self.tick_samples-1)//self.tick_samples;audio=torch.nn.functional.pad(audio,(0,source_ticks*self.tick_samples-len(audio))).view(source_ticks,self.tick_samples)
        tick_ms=1000*self.tick_samples/self.sample_rate;low=round(self.thinking_ms[0]/tick_ms);high=round(self.thinking_ms[1]/tick_ms);thinking=random.randint(low,high);repeat_start=source_ticks+thinking;total=repeat_start+source_ticks
        inputs=torch.zeros(total,self.tick_samples);targets=torch.zeros_like(inputs);inputs[:source_ticks]=audio;targets[repeat_start:]=audio
        return inputs,targets,{"id":row["id"],"label":int(row["label"]),"source_ticks":source_ticks,"thinking_ticks":thinking,"repeat_start":repeat_start,"total_ticks":total}


def collate_delayed_repeat(batch):
    torch,_,_=require_torch();length=max(len(item[0]) for item in batch);inputs=torch.zeros(len(batch),length,batch[0][0].shape[-1]);targets=torch.zeros_like(inputs);valid=torch.zeros(len(batch),length,dtype=torch.bool)
    for index,(source,target,_) in enumerate(batch):inputs[index,:len(source)]=source;targets[index,:len(target)]=target;valid[index,:len(source)]=True
    return inputs,targets,valid,[item[2] for item in batch]
