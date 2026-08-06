from __future__ import annotations

import hashlib, json
from pathlib import Path

from .neural_echo import require_torch


def normalize_audio_codes(codes):
    """Return EnCodec payload codes as [frames, codebooks]."""
    while codes.ndim > 2 and codes.shape[0] == 1: codes = codes.squeeze(0)
    if codes.ndim != 2: raise ValueError(f"unsupported audio_codes shape {tuple(codes.shape)}")
    # Codebook count is small; time is normally the longest dimension.
    return codes.transpose(0, 1).contiguous() if codes.shape[0] <= codes.shape[1] else codes.contiguous()


class CodecCopyEpisodeDataset:
    """Listen, retain a continuous pause, then reproduce the same codec tokens."""
    def __init__(self, codec_manifest, split, thinking_ms=(160,280), max_records=None, preload=True):
        self.rows=[json.loads(line) for line in Path(codec_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows=[row for row in self.rows if row["split"]==split]
        if max_records is not None: self.rows=self.rows[:max_records]
        if not self.rows: raise ValueError(f"no codec records for split {split!r}")
        self.thinking_ms=thinking_ms; self.cache={}
        if preload:
            torch,_,_=require_torch()
            for index,row in enumerate(self.rows,1):
                self.cache[row["codec_path"]]=torch.load(row["codec_path"],map_location="cpu",weights_only=True)
                if index==1 or index%100==0 or index==len(self.rows): print(f"[codec preload] {index}/{len(self.rows)}",flush=True)

    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        torch,_,_=require_torch(); row=self.rows[index]
        payload=self.cache.get(row["codec_path"])
        if payload is None: payload=torch.load(row["codec_path"],map_location="cpu",weights_only=True)
        codes=normalize_audio_codes(payload["audio_codes"]).long(); frames=len(codes)
        seconds=payload["audio_length"]/payload["sample_rate"]; frame_rate=frames/max(seconds,1e-6)
        low=round(self.thinking_ms[0]*frame_rate/1000); high=round(self.thinking_ms[1]*frame_rate/1000)
        offset=int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4],"big")%(high-low+1)
        thinking=low+offset; repeat_start=frames+thinking; total=repeat_start+frames
        inputs=torch.full((total,codes.shape[1]),-1,dtype=torch.long); inputs[:frames]=codes
        targets=torch.full_like(inputs,-100); targets[repeat_start:]=codes
        voice=torch.zeros(total); voice[repeat_start:]=1
        metadata={"id":row["id"],"source_frames":frames,"thinking_frames":thinking,"repeat_start":repeat_start,
                  "total_frames":total,"codebooks":codes.shape[1],"frame_rate":frame_rate}
        return inputs,targets,voice,metadata


def audit_codec_copy_manifest(codec_manifest, sample_payloads=64):
    torch,_,_=require_torch(); rows=[json.loads(x) for x in Path(codec_manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest_fingerprints={row.get("codec_fingerprint") for row in rows}
    required=("id","codec_path","codec_frames","codec_fingerprint","split")
    missing=sum(any(key not in row for key in required) for row in rows)
    count=min(sample_payloads,len(rows)); indices=sorted({round(i*(len(rows)-1)/max(count-1,1)) for i in range(count)})
    shapes=set(); rates=[]; fingerprints=set(); code_min=None; code_max=None
    for progress,index in enumerate(indices,1):
        row=rows[index]
        payload=torch.load(row["codec_path"],map_location="cpu",weights_only=True); codes=normalize_audio_codes(payload["audio_codes"])
        shapes.add(codes.shape[1]); rates.append(len(codes)/(payload["audio_length"]/payload["sample_rate"])); fingerprints.add(payload["codec_fingerprint"])
        value_min=int(codes.min()); value_max=int(codes.max()); code_min=value_min if code_min is None else min(code_min,value_min); code_max=value_max if code_max is None else max(code_max,value_max)
        if progress==1 or progress%16==0 or progress==len(indices): print(f"[codec audit] {progress}/{len(indices)} sampled payloads",flush=True)
    return {"records":len(rows),"manifest_missing_required":missing,"manifest_fingerprints":sorted(str(value) for value in manifest_fingerprints),
            "sampled_payloads":len(indices),"codebooks":sorted(shapes),"frame_rate_min":min(rates),"frame_rate_max":max(rates),
            "token_min":code_min,"token_max":code_max,"payload_fingerprints":sorted(fingerprints)}
