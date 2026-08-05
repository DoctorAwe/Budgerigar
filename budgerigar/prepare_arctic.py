from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def find_speakers(root: Path) -> dict[str, dict[str, Path]]:
    speakers: dict[str, dict[str, Path]] = {}
    for wav_dir in root.rglob("wav"):
        files = {path.stem: path.resolve() for path in wav_dir.glob("arctic_*.wav")}
        if files:
            name = wav_dir.parent.name.removeprefix("cmu_us_").removesuffix("_arctic")
            speakers[name] = files
    return speakers


def build_pairs(root: Path, target_speaker: str | None = None) -> list[dict[str, str]]:
    speakers = find_speakers(root)
    if len(speakers) < 2:
        raise ValueError("Expected at least two extracted CMU ARCTIC speaker directories")
    target_speaker = target_speaker or ("slt" if "slt" in speakers else sorted(speakers)[-1])
    if target_speaker not in speakers:
        raise ValueError(f"Unknown target speaker {target_speaker!r}; found {sorted(speakers)}")
    pairs = []
    for source_speaker, utterances in sorted(speakers.items()):
        if source_speaker == target_speaker:
            continue
        for utterance_id in sorted(utterances.keys() & speakers[target_speaker].keys()):
            pairs.append({
                "utterance_id": utterance_id,
                "source_speaker": source_speaker,
                "target_speaker": target_speaker,
                "source_path": str(utterances[utterance_id]),
                "target_path": str(speakers[target_speaker][utterance_id]),
            })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create same-utterance CMU ARCTIC manifests")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-speaker")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    pairs = build_pairs(args.root, args.target_speaker)
    utterances = sorted({item["utterance_id"] for item in pairs})
    random.Random(args.seed).shuffle(utterances)
    validation_ids = set(utterances[: max(1, round(len(utterances) * 0.1))])
    args.output.mkdir(parents=True, exist_ok=True)
    for split, predicate in (("train", lambda value: value not in validation_ids), ("validation", validation_ids.__contains__)):
        selected = [item for item in pairs if predicate(item["utterance_id"])]
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in selected:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{split}: {len(selected)} pairs")


if __name__ == "__main__":
    main()

