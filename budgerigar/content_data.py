from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .neural_echo import require_torch


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class CharacterVocabulary:
    symbols: tuple[str, ...] = ("<blank>", "<unk>", " ", "'", *tuple("abcdefghijklmnopqrstuvwxyz"))

    @property
    def lookup(self): return {symbol: index for index, symbol in enumerate(self.symbols)}

    def encode(self, text):
        lookup = self.lookup
        return [lookup.get(character, 1) for character in normalize_text(text)]

    def decode(self, values): return "".join(self.symbols[value] if value < len(self.symbols) else "<unk>" for value in values)


class ContentFeatureDataset:
    def __init__(self, manifest, split, stats, vocabulary=CharacterVocabulary(), max_records=None, preload=True, update_stride=4, token_slots=128):
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = []
        seen_parallel = set()
        for row in rows:
            encoded = vocabulary.encode(row.get("text", ""))
            ctc_required = len(encoded) + sum(left == right for left, right in zip(encoded, encoded[1:]))
            updates = (int(row["feature_frames"]) + update_stride - 1) // update_stride
            if row["split"] == split and row["parallel_id"] not in seen_parallel and encoded and ctc_required <= min(updates, token_slots):
                self.rows.append((row, encoded))
                seen_parallel.add(row["parallel_id"])
        if max_records is not None: self.rows = self.rows[:max_records]
        if not self.rows: raise ValueError(f"no CTC-compatible records for split {split!r}")
        self.stats = stats; self.vocabulary = vocabulary; self.cache = {}
        if preload:
            torch, _, _ = require_torch()
            for index, (row, _) in enumerate(self.rows, 1):
                self.cache[row["feature_path"]] = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
                if index == 1 or index % 100 == 0 or index == len(self.rows): print(f"[content preload] {index}/{len(self.rows)}", flush=True)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        torch, _, _ = require_torch(); row, text = self.rows[index]
        payload = self.cache.get(row["feature_path"])
        if payload is None: payload = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        mel = (payload["log_mel"] - self.stats["mean"]) / self.stats["std"]
        energy = (payload["energy_db"] / 80.0).unsqueeze(-1); vad = payload["vad"].float().unsqueeze(-1)
        return torch.cat([mel, energy, vad], -1), torch.tensor(text, dtype=torch.long), row["id"], normalize_text(row["text"])


def collate_content(batch):
    torch, _, _ = require_torch(); frame_lengths = torch.tensor([len(item[0]) for item in batch]); text_lengths = torch.tensor([len(item[1]) for item in batch])
    inputs = torch.zeros(len(batch), int(frame_lengths.max()), batch[0][0].shape[1]); texts = torch.zeros(len(batch), int(text_lengths.max()), dtype=torch.long)
    for index, item in enumerate(batch): inputs[index, :len(item[0])] = item[0]; texts[index, :len(item[1])] = item[1]
    return inputs, frame_lengths, texts, text_lengths, [item[2] for item in batch], [item[3] for item in batch]
