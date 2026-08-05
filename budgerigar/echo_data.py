from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .neural_echo import require_torch


@dataclass(frozen=True)
class EchoPair:
    source_id: str
    source_speaker: str
    source_feature: Path
    target_id: str
    target_speaker: str
    target_feature: Path
    parallel_id: str
    split: str


def load_pairs(feature_manifest: str | Path, target_speaker: str) -> list[EchoPair]:
    rows = [json.loads(line) for line in Path(feature_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    speakers = sorted({row["speaker"] for row in rows})
    if target_speaker not in speakers:
        raise ValueError(f"target speaker {target_speaker!r} not found; choose from {speakers}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["parallel_id"], row["split"], row["language"], row["emotion"])].append(row)
    pairs = []
    for (parallel_id, split, _, _), group in sorted(groups.items()):
        targets = [row for row in group if row["speaker"] == target_speaker]
        if len(targets) > 1:
            raise ValueError(f"multiple target recordings in {parallel_id}")
        if not targets:
            continue
        target = targets[0]
        for source in group:
            if source["speaker"] != target_speaker:
                pairs.append(EchoPair(
                    source["id"], source["speaker"], Path(source["feature_path"]),
                    target["id"], target_speaker, Path(target["feature_path"]), parallel_id, split,
                ))
    if not pairs:
        raise ValueError("no fixed-voice parallel pairs were found")
    return pairs


def pair_report(pairs):
    pairs = list(pairs)
    return {
        "pairs": len(pairs), "target_speaker": sorted({pair.target_speaker for pair in pairs}),
        "source_speakers": sorted({pair.source_speaker for pair in pairs}),
        "by_split": {split: sum(pair.split == split for pair in pairs) for split in ("train", "validation", "test")},
    }


def feature_stats(pairs, verbose=True):
    torch, _, _ = require_torch()
    paths = sorted({pair.source_feature for pair in pairs} | {pair.target_feature for pair in pairs})
    total = square = None; frames = 0; fingerprint = None
    for index, path in enumerate(paths, 1):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mel = payload["log_mel"].double()
        if total is None:
            total = torch.zeros(mel.shape[1], dtype=torch.float64); square = torch.zeros_like(total)
        total += mel.sum(0); square += mel.square().sum(0); frames += len(mel)
        fingerprint = fingerprint or payload["config_fingerprint"]
        if fingerprint != payload["config_fingerprint"]:
            raise ValueError("mixed feature fingerprints")
        if verbose and (index == 1 or index % 100 == 0 or index == len(paths)):
            print(f"[stats] {index}/{len(paths)} feature files", flush=True)
    mean = total / frames; variance = square / frames - mean.square()
    return {"mean": mean.float(), "std": variance.clamp_min(1e-6).sqrt().float(), "frames": frames, "feature_fingerprint": fingerprint}


class EchoEpisodeDataset:
    """Input/output timelines; there are no END/START/READ labels."""

    def __init__(self, pairs, stats, thinking_frames=(16, 28), preload=True, verbose=True):
        if thinking_frames[0] < 0 or thinking_frames[1] < thinking_frames[0]:
            raise ValueError("thinking_frames must be a non-negative (minimum, maximum) pair")
        self.pairs = list(pairs); self.stats = stats; self.thinking_frames = thinking_frames; self.cache = {}
        if preload:
            torch, _, _ = require_torch()
            paths = sorted({p.source_feature for p in self.pairs} | {p.target_feature for p in self.pairs})
            for index, path in enumerate(paths, 1):
                self.cache[path] = torch.load(path, map_location="cpu", weights_only=True)
                if verbose and (index == 1 or index % 100 == 0 or index == len(paths)):
                    print(f"[preload] {index}/{len(paths)} feature files", flush=True)

    def __len__(self): return len(self.pairs)

    def _load(self, path):
        torch, _, _ = require_torch()
        if path not in self.cache:
            return torch.load(path, map_location="cpu", weights_only=True)
        return self.cache[path]

    def __getitem__(self, index):
        torch, _, _ = require_torch()
        pair = self.pairs[index]; source_payload = self._load(pair.source_feature); target_payload = self._load(pair.target_feature)
        mean, std = self.stats["mean"], self.stats["std"]
        source = (source_payload["log_mel"] - mean) / std; target = (target_payload["log_mel"] - mean) / std
        silence = (torch.full((source.shape[1],), -11.512925) - mean) / std
        span = self.thinking_frames[1] - self.thinking_frames[0] + 1
        offset = int.from_bytes(hashlib.sha256(pair.parallel_id.encode("utf-8")).digest()[:4], "big") % span
        thinking_frames = self.thinking_frames[0] + offset
        repeat_start = len(source) + thinking_frames; total_frames = repeat_start + len(target)
        input_mel = silence.repeat(total_frames, 1); input_mel[:len(source)] = source
        energy = torch.full((total_frames,), -80.0); energy[:len(source)] = source_payload["energy_db"][:len(source)]
        input_vad = torch.zeros(total_frames); input_vad[:len(source)] = source_payload["vad"][:len(source)].float()
        inputs = torch.cat([input_mel, (energy / 80.0).unsqueeze(-1), input_vad.unsqueeze(-1)], -1)
        outputs = silence.repeat(total_frames, 1); outputs[repeat_start:] = target
        voice = torch.zeros(total_frames); voice[repeat_start:] = target_payload["vad"][:len(target)].float()
        return inputs, outputs, voice, pair.source_id


def collate_episodes(batch):
    torch, _, _ = require_torch(); lengths = [len(item[0]) for item in batch]; frames = max(lengths)
    inputs = torch.zeros(len(batch), frames, batch[0][0].shape[1]); outputs = torch.zeros(len(batch), frames, batch[0][1].shape[1])
    voice = torch.zeros(len(batch), frames); valid = torch.zeros(len(batch), frames, dtype=torch.bool)
    for i, (x, y, v, _) in enumerate(batch):
        inputs[i, :len(x)] = x; outputs[i, :len(y)] = y; voice[i, :len(v)] = v; valid[i, :len(x)] = True
    return inputs, outputs, voice, valid, [item[3] for item in batch]
