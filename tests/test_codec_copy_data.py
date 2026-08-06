import json
import pytest
torch=pytest.importorskip("torch")
from budgerigar.codec_copy_data import CodecCopyEpisodeDataset,normalize_audio_codes


def test_normalize_and_self_copy_episode(tmp_path):
    payload=tmp_path/"a.pt"; torch.save({"audio_codes":torch.arange(40).reshape(1,1,4,10),"audio_length":3200,"sample_rate":24000,"codec_fingerprint":"x"},payload)
    manifest=tmp_path/"m.jsonl"; manifest.write_text(json.dumps({"id":"a","split":"train","codec_path":str(payload)})+"\n")
    dataset=CodecCopyEpisodeDataset(manifest,"train",thinking_ms=(160,160),preload=False)
    inputs,targets,voice,meta=dataset[0]
    assert normalize_audio_codes(torch.arange(40).reshape(1,1,4,10)).shape==(10,4)
    assert torch.equal(inputs[:meta["source_frames"]],targets[meta["repeat_start"]:])
    assert not voice[:meta["repeat_start"]].any() and voice[meta["repeat_start"]:].all()
