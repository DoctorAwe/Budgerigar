from __future__ import annotations

import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .audio import AudioConfig, align_target, load_wave, log_mel


class ParallelSpeechDataset(Dataset):
    def __init__(
        self, manifest: str | Path, audio: AudioConfig = AudioConfig(), segment_frames: int | None = 320
    ):
        self.audio = audio
        self.segment_frames = segment_frames
        with Path(manifest).open(encoding="utf-8") as handle:
            self.items = [json.loads(line) for line in handle if line.strip()]
        if not self.items:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        if "source_mel_path" in item and "target_mel_path" in item:
            source = torch.load(item["source_mel_path"], map_location="cpu", weights_only=True)
            target = torch.load(item["target_mel_path"], map_location="cpu", weights_only=True)
        else:
            source = log_mel(load_wave(item["source_path"], self.audio.sample_rate), self.audio)
            target = log_mel(load_wave(item["target_path"], self.audio.sample_rate), self.audio)
            target = align_target(target, len(source))
        if self.segment_frames and len(source) > self.segment_frames:
            start = random.randint(0, len(source) - self.segment_frames)
            source = source[start:start + self.segment_frames]
            target = target[start:start + self.segment_frames]
        return source, target, item["source_speaker"], item["utterance_id"]


def collate_parallel(batch):
    frames = min(item[0].shape[0] for item in batch)
    source = torch.stack([item[0][:frames] for item in batch])
    target = torch.stack([item[1][:frames] for item in batch])
    return source, target
