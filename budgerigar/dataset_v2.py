from __future__ import annotations

import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset


class VariableParallelSpeechDataset(Dataset):
    """Whole-utterance cached features retaining the target speaker's duration."""

    def __init__(self, manifest: str | Path, max_frames: int = 900):
        self.max_frames = max_frames
        with Path(manifest).open(encoding="utf-8") as handle:
            self.items = [json.loads(line) for line in handle if line.strip()]
        self.items = [item for item in self.items if "raw_target_mel_path" in item]
        if not self.items:
            raise ValueError("V2 requires a cached manifest with raw_target_mel_path")
        self.speaker_names = tuple(sorted({item["target_speaker"] for item in self.items}))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        source = torch.load(item["source_mel_path"], map_location="cpu", weights_only=True)
        target = torch.load(item["raw_target_mel_path"], map_location="cpu", weights_only=True)
        if len(source) > self.max_frames or len(target) > self.max_frames:
            scale = min(self.max_frames / len(source), self.max_frames / len(target))
            source_frames = max(2, round(len(source) * scale))
            target_frames = max(2, round(len(target) * scale))
            source = torch.nn.functional.interpolate(
                source.T[None], size=source_frames, mode="linear", align_corners=False
            )[0].T
            target = torch.nn.functional.interpolate(
                target.T[None], size=target_frames, mode="linear", align_corners=False
            )[0].T
        return source, target, item["target_speaker"]


def collate_variable(batch):
    source_lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    mel_dim = batch[0][0].shape[1]
    sources = torch.zeros(len(batch), int(source_lengths.max()), mel_dim)
    targets = torch.zeros(len(batch), int(target_lengths.max()), mel_dim)
    for index, (source, target, _) in enumerate(batch):
        sources[index, :len(source)] = source
        targets[index, :len(target)] = target
    return sources, source_lengths, targets, target_lengths, [item[2] for item in batch]

