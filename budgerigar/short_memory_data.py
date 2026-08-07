from __future__ import annotations
import hashlib,json,random
from pathlib import Path
from .neural_echo import require_torch

def _balanced_limit(rows,limit):
    if limit is None or len(rows)<=limit:return rows
    buckets={label:[] for label in range(10)}
    for row in rows:buckets[int(row["label"])].append(row)
    selected=[]
    for label in range(10):selected.extend(buckets[label][:limit//10])
    return sorted(selected,key=lambda row:row["id"])

class ShortMemoryEpisodeDataset:
    """A spoken digit followed by silence and one delayed output-token event."""
    def __init__(self,manifest,split,sample_rate=16000,tick_samples=80,thinking_ms=(120,280),max_records=None,preload=True):
        self.manifest=Path(manifest);self.sample_rate=sample_rate;self.tick_samples=tick_samples;self.thinking_ms=thinking_ms
        rows=[json.loads(x) for x in self.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.rows=_balanced_limit([row for row in rows if row["split"]==split],max_records);self.cache={}
        if not self.rows:raise ValueError(f"no FSDD rows for {split!r}")
        if preload:
            for index,row in enumerate(self.rows,1):
                self.cache[row["id"]]=self._load(row)
                if index==1 or index%250==0 or index==len(self.rows):print(f"[short preload] {index}/{len(self.rows)}",flush=True)
    def _load(self,row):
        import torchaudio
        waveform,rate=torchaudio.load(row["audio"]);waveform=waveform.mean(0)
        if rate!=self.sample_rate:waveform=torchaudio.functional.resample(waveform,rate,self.sample_rate)
        return waveform.float()/waveform.abs().max().clamp_min(1e-5)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        torch,_,_=require_torch();row=self.rows[index];waveform=self.cache.get(row["id"])
        if waveform is None:waveform=self._load(row)
        # Random leading silence changes external chunk alignment without revealing an end marker.
        leading=random.randrange(0,self.tick_samples*4)
        audio=torch.nn.functional.pad(waveform,(leading,0));audio_ticks=(len(audio)+self.tick_samples-1)//self.tick_samples
        audio=torch.nn.functional.pad(audio,(0,audio_ticks*self.tick_samples-len(audio)))
        digest=int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4],"big")
        low=self.thinking_ms[0]//5;high=self.thinking_ms[1]//5;thinking=low+digest%(high-low+1)
        emission_tick=audio_ticks+thinking;total_ticks=emission_tick+4
        samples=torch.nn.functional.pad(audio,(0,(total_ticks-audio_ticks)*self.tick_samples)).view(total_ticks,self.tick_samples)
        targets=torch.zeros(total_ticks,dtype=torch.long);targets[emission_tick]=int(row["label"])+1
        return samples,targets,{"id":row["id"],"label":int(row["label"]),"audio_end_tick":audio_ticks-1,"emission_tick":emission_tick,"thinking_ticks":thinking}

def collate_short_memory(batch):
    torch,_,_=require_torch();ticks=max(len(x[0]) for x in batch);samples=torch.zeros(len(batch),ticks,batch[0][0].shape[-1]);targets=torch.zeros(len(batch),ticks,dtype=torch.long);valid=torch.zeros(len(batch),ticks,dtype=torch.bool)
    for index,(x,y,_) in enumerate(batch):samples[index,:len(x)]=x;targets[index,:len(y)]=y;valid[index,:len(x)]=True
    return samples,targets,valid,[x[2] for x in batch]
