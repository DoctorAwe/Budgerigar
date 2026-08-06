from __future__ import annotations
import json,random
from pathlib import Path
from .neural_echo import require_torch

class WaveformChunkDataset:
    def __init__(self,manifest,split,sample_rate=24000,chunk_samples=24000,max_records=None,preload=True,random_crop=True):
        self.manifest=Path(manifest);self.sample_rate=sample_rate;self.chunk_samples=chunk_samples;self.random_crop=random_crop
        self.rows=[json.loads(x) for x in self.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.rows=[x for x in self.rows if x["split"]==split]
        if max_records is not None:self.rows=self.rows[:max_records]
        if not self.rows:raise ValueError(f"no waveform records for split {split!r}")
        self.cache={}
        if preload:
            for i,row in enumerate(self.rows,1):
                self.cache[row["id"]]=self._load(row)
                if i==1 or i%100==0 or i==len(self.rows):print(f"[wave preload] {i}/{len(self.rows)}",flush=True)
    def _load(self,row):
        import torchaudio
        path=Path(row["audio"])
        if not path.is_absolute():path=(self.manifest.parent/path).resolve()
        waveform,rate=torchaudio.load(str(path));waveform=waveform.mean(0)
        if rate!=self.sample_rate:waveform=torchaudio.functional.resample(waveform,rate,self.sample_rate)
        return (waveform/waveform.abs().max().clamp_min(1.0)).float()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        torch,_,_=require_torch();row=self.rows[index];waveform=self.cache.get(row["id"])
        if waveform is None:waveform=self._load(row)
        if len(waveform)>=self.chunk_samples:
            maximum=len(waveform)-self.chunk_samples;start=random.randint(0,maximum) if self.random_crop and maximum else maximum//2
            chunk=waveform[start:start+self.chunk_samples]
        else:chunk=torch.nn.functional.pad(waveform,(0,self.chunk_samples-len(waveform)));start=0
        return chunk,row["id"],start

def collate_waveform_chunks(batch):
    torch,_,_=require_torch();return torch.stack([x[0] for x in batch]),[x[1] for x in batch],[x[2] for x in batch]
