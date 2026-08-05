from __future__ import annotations

import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset


WAIT, START, EMIT, STOP = 0, 1, 2, 3
ACTION_NAMES = ("WAIT", "START", "EMIT", "STOP")


class StreamingEpisodeDataset(Dataset):
    """Synthetic duplex timelines for learning when to speak on continuous input."""

    def __init__(
        self, manifest: str | Path, episodes: int = 10_000, utterances_per_episode: int = 3,
        gap_frames: tuple[int, int] = (0, 120), reaction_frames: tuple[int, int] = (0, 0),
        seed: int = 17,
    ):
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if "source_mel_path" in row and "raw_target_mel_path" in row]
        if not rows:
            raise ValueError("Streaming episodes require a V2 cached manifest")
        self.by_target: dict[str, list[dict]] = {}
        for row in rows:
            self.by_target.setdefault(row["target_speaker"], []).append(row)
        self.targets = tuple(sorted(self.by_target))
        self.episodes = episodes
        self.utterances_per_episode = utterances_per_episode
        self.gap_frames = gap_frames
        self.reaction_frames = reaction_frames
        self.seed = seed

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        target_speaker = rng.choice(self.targets)
        rows = rng.sample(
            self.by_target[target_speaker],
            k=min(self.utterances_per_episode, len(self.by_target[target_speaker])),
        )
        scheduled = []
        input_cursor = 0
        output_cursor = 0
        mel_dim = 80
        for row in rows:
            source = torch.load(row["source_mel_path"], map_location="cpu", weights_only=True)
            target = torch.load(row["raw_target_mel_path"], map_location="cpu", weights_only=True)
            mel_dim = source.shape[1]
            input_start = input_cursor
            input_end = input_start + len(source)
            # The desired behavior is immediate start once a complete sentence
            # has been recognized. A non-zero range is only augmentation.
            reaction = rng.randint(*self.reaction_frames)
            output_start = max(input_end + reaction, output_cursor)
            output_end = output_start + len(target)
            scheduled.append((source, target, input_start, input_end, output_start, output_end))
            # The next person may speak while the parrot is still producing output.
            input_cursor = input_end + rng.randint(*self.gap_frames)
            output_cursor = output_end + 1
        total = max(input_cursor, output_cursor + 1)
        silence = -11.512925
        input_stream = torch.full((total, mel_dim), silence)
        output_target = torch.zeros(total, mel_dim)
        actions = torch.full((total,), WAIT, dtype=torch.long)
        emission_mask = torch.zeros(total, dtype=torch.bool)
        utterance_end = torch.zeros(total, dtype=torch.bool)
        for source, target, input_start, input_end, output_start, output_end in scheduled:
            input_stream[input_start:input_end] = source
            utterance_end[input_end - 1] = True
            output_target[output_start:output_end] = target
            emission_mask[output_start:output_end] = True
            actions[output_start] = START
            if output_end - output_start > 1:
                actions[output_start + 1:output_end] = EMIT
            actions[output_end] = STOP
        return input_stream, actions, output_target, emission_mask, utterance_end, target_speaker


def collate_streaming(batch):
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    frames = int(lengths.max())
    mel_dim = batch[0][0].shape[1]
    inputs = torch.full((len(batch), frames, mel_dim), -11.512925)
    targets = torch.zeros(len(batch), frames, mel_dim)
    actions = torch.full((len(batch), frames), WAIT, dtype=torch.long)
    emission = torch.zeros(len(batch), frames, dtype=torch.bool)
    endpoints = torch.zeros(len(batch), frames, dtype=torch.bool)
    for index, item in enumerate(batch):
        size = len(item[0])
        inputs[index, :size] = item[0]
        actions[index, :size] = item[1]
        targets[index, :size] = item[2]
        emission[index, :size] = item[3]
        endpoints[index, :size] = item[4]
    return inputs, lengths, actions, targets, emission, endpoints, [item[5] for item in batch]
