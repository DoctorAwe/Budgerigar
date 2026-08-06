from __future__ import annotations

import hashlib, json
from pathlib import Path


def codec_fingerprint(model_id="facebook/encodec_24khz", bandwidth=6.0, revision="main"):
    payload=json.dumps({"model_id":model_id,"bandwidth":bandwidth,"revision":revision},sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def extract_encodec_manifest(manifest, output_root, output_manifest, model_id="facebook/encodec_24khz",
                            bandwidth=6.0, revision="main", limit=None):
    import torch, torchaudio
    from transformers import AutoProcessor, EncodecModel
    rows=[json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None: rows=rows[:limit]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor=AutoProcessor.from_pretrained(model_id,revision=revision)
    model=EncodecModel.from_pretrained(model_id,revision=revision).to(device).eval()
    fingerprint=codec_fingerprint(model_id,bandwidth,revision); root=Path(output_root)/fingerprint; root.mkdir(parents=True,exist_ok=True)
    written=[]
    with torch.no_grad():
        for index,row in enumerate(rows,1):
            waveform,sample_rate=torchaudio.load(row["audio_path"]); waveform=waveform.mean(0)
            if sample_rate != 24000: waveform=torchaudio.functional.resample(waveform,sample_rate,24000)
            prepared=processor(raw_audio=waveform.numpy(),sampling_rate=24000,return_tensors="pt")
            values=prepared["input_values"].to(device); mask=prepared.get("padding_mask")
            if mask is not None: mask=mask.to(device)
            encoded=model.encode(values,padding_mask=mask,bandwidth=bandwidth)
            path=root/f"{row['id'].replace('/','_')}.pt"
            torch.save({"audio_codes":encoded.audio_codes.cpu(),"audio_scales":[s.cpu() if s is not None else None for s in encoded.audio_scales],
                        "audio_length":len(waveform),"sample_rate":24000,"model_id":model_id,"bandwidth":bandwidth,
                        "revision":revision,"codec_fingerprint":fingerprint},path)
            item=dict(row); item.update({"codec_path":str(path),"codec_fingerprint":fingerprint,
                                         "codec_frames":int(encoded.audio_codes.shape[-1]),"codec_bandwidth":bandwidth})
            written.append(item)
            if index==1 or index%50==0 or index==len(rows): print(f"[codec] {index}/{len(rows)}",flush=True)
    output_manifest=Path(output_manifest); output_manifest.parent.mkdir(parents=True,exist_ok=True)
    output_manifest.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in written)+"\n",encoding="utf-8")
    return {"records":len(written),"codec_fingerprint":fingerprint,"manifest":str(output_manifest)}


def decode_encodec_payload(payload_path, model_id="facebook/encodec_24khz", revision="main"):
    import torch
    from transformers import EncodecModel
    payload=torch.load(payload_path,map_location="cpu",weights_only=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=EncodecModel.from_pretrained(model_id,revision=revision).to(device).eval()
    scales=[s.to(device) if s is not None else None for s in payload["audio_scales"]]
    with torch.no_grad(): decoded=model.decode(payload["audio_codes"].to(device),scales).audio_values[0,0]
    return decoded[:payload["audio_length"]].cpu(),payload["sample_rate"]
