import json
from pathlib import Path

import torch

from budgerigar.streaming_dataset import EMIT, START, STOP, StreamingEpisodeDataset


def test_episode_has_independent_input_and_output_timelines(tmp_path: Path):
    rows = []
    for index in range(3):
        source = tmp_path / f"source_{index}.pt"
        target = tmp_path / f"target_{index}.pt"
        torch.save(torch.randn(8 + index, 80), source)
        torch.save(torch.randn(6 + index, 80), target)
        rows.append({
            "source_mel_path": str(source), "raw_target_mel_path": str(target),
            "target_speaker": "slt",
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    dataset = StreamingEpisodeDataset(
        manifest, episodes=1, utterances_per_episode=2,
        gap_frames=(1, 1), reaction_frames=(1, 1),
    )
    inputs, actions, targets, emission, endpoints, speaker = dataset[0]
    assert speaker == "slt"
    assert (actions == START).sum() == 2
    assert (actions == STOP).sum() == 2
    assert (actions == EMIT).any()
    assert emission.sum() == 13 or emission.sum() == 14 or emission.sum() == 15
    assert endpoints.sum() == 2
    assert inputs.shape == targets.shape
