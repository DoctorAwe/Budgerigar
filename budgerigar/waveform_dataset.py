from __future__ import annotations

import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .audio import load_wave


class WaveformStreamDataset(Dataset):
    """Continuous input/output timelines; behavior is represented only by waveform."""

    def __init__(
        self, manifest: str | Path, episodes: int = 20_000, utterances_per_episode: int = 2,
        sample_rate: int = 16_000, hop_samples: int = 160,
        gap_ms: tuple[int, int] = (0, 800), seed: int = 29,
    ):
        manifest = Path(manifest)
        if not manifest.exists():
            raise FileNotFoundError(
                f"Multi-target manifest not found: {manifest}. Generate it with: "
                "python -m budgerigar.prepare_arctic --root /content/data/cmu_arctic "
                "--output data/arctic_multi --all-targets"
            )
        self.rows = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.by_target: dict[str, list[dict]] = {}
        for row in self.rows:
            self.by_target.setdefault(row["target_speaker"], []).append(row)
        self.speaker_names = tuple(sorted(self.by_target))
        self.episodes = episodes
        self.utterances_per_episode = utterances_per_episode
        self.sample_rate = sample_rate
        self.hop_samples = hop_samples
        self.gap_ms = gap_ms
        self.seed = seed

    def __len__(self) -> int:
        return self.episodes

    def _align(self, samples: int) -> int:
        return ((samples + self.hop_samples - 1) // self.hop_samples) * self.hop_samples

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        target_speaker = rng.choice(self.speaker_names)
        candidates = self.by_target[target_speaker]
        rows = rng.sample(candidates, k=min(self.utterances_per_episode, len(candidates)))
        schedule = []
        input_cursor = output_cursor = 0
        for row in rows:
            source = load_wave(row["source_path"], self.sample_rate).float()
            target = load_wave(row["target_path"], self.sample_rate).float()
            source = source / source.abs().max().clamp_min(1.0)
            target = target / target.abs().max().clamp_min(1.0)
            input_start = self._align(input_cursor)
            input_end = input_start + len(source)
            # Output starts at the first model hop after the complete sentence.
            output_start = max(self._align(input_end), self._align(output_cursor))
            output_end = output_start + len(target)
            schedule.append((source, target, input_start, input_end, output_start, output_end))
            gap = rng.randint(*self.gap_ms) * self.sample_rate // 1000
            input_cursor = input_end + gap
            output_cursor = output_end
        total = self._align(max(input_cursor, output_cursor))
        input_waveform = torch.zeros(total)
        output_waveform = torch.zeros(total)
        speech_mask = torch.zeros(total, dtype=torch.bool)
        completion_samples = []
        output_start_samples = []
        for source, target, input_start, input_end, output_start, output_end in schedule:
            input_waveform[input_start:input_end] = source
            output_waveform[output_start:output_end] = target
            speech_mask[output_start:output_end] = True
            completion_samples.append(input_end)
            output_start_samples.append(output_start)
        return {
            "input": input_waveform,
            "target": output_waveform,
            "speech_mask": speech_mask,
            "target_speaker": target_speaker,
            "completion_samples": torch.tensor(completion_samples),
            "output_start_samples": torch.tensor(output_start_samples),
        }


def collate_waveform(batch):
    lengths = torch.tensor([len(item["input"]) for item in batch], dtype=torch.long)
    samples = int(lengths.max())
    inputs = torch.zeros(len(batch), samples)
    targets = torch.zeros(len(batch), samples)
    speech_mask = torch.zeros(len(batch), samples, dtype=torch.bool)
    for index, item in enumerate(batch):
        size = len(item["input"])
        inputs[index, :size] = item["input"]
        targets[index, :size] = item["target"]
        speech_mask[index, :size] = item["speech_mask"]
    return {
        "input": inputs, "target": targets, "speech_mask": speech_mask,
        "lengths": lengths, "target_speakers": [item["target_speaker"] for item in batch],
        "completion_samples": [item["completion_samples"] for item in batch],
        "output_start_samples": [item["output_start_samples"] for item in batch],
    }
