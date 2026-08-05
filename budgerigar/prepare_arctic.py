from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re


def find_speakers(root: Path) -> dict[str, dict[str, Path]]:
    speakers: dict[str, dict[str, Path]] = {}
    for wav_dir in root.rglob("wav"):
        files = {path.stem: path.resolve() for path in wav_dir.glob("arctic_*.wav")}
        if files:
            name = wav_dir.parent.name.removeprefix("cmu_us_").removesuffix("_arctic")
            speakers[name] = files
    return speakers


def find_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    pattern = re.compile(r'^\(\s*(arctic_[ab]\d+)\s+"(.*)"\s*\)$')
    for path in root.rglob("txt.done.data"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line.strip())
            if match:
                transcripts.setdefault(match.group(1), match.group(2))
    return transcripts


def build_pairs(
    root: Path, target_speaker: str | None = None, all_targets: bool = False
) -> list[dict[str, str]]:
    speakers = find_speakers(root)
    transcripts = find_transcripts(root)
    if len(speakers) < 2:
        raise ValueError("Expected at least two extracted CMU ARCTIC speaker directories")
    if all_targets and target_speaker is not None:
        raise ValueError("--all-targets and --target-speaker are mutually exclusive")
    target_speaker = target_speaker or ("slt" if "slt" in speakers else sorted(speakers)[-1])
    if not all_targets and target_speaker not in speakers:
        raise ValueError(f"Unknown target speaker {target_speaker!r}; found {sorted(speakers)}")
    pairs = []
    targets = sorted(speakers) if all_targets else [target_speaker]
    for selected_target in targets:
        for source_speaker, utterances in sorted(speakers.items()):
            if source_speaker == selected_target:
                continue
            for utterance_id in sorted(utterances.keys() & speakers[selected_target].keys()):
                pairs.append({
                    "utterance_id": utterance_id,
                    "source_speaker": source_speaker,
                    "target_speaker": selected_target,
                    "source_path": str(utterances[utterance_id]),
                    "target_path": str(speakers[selected_target][utterance_id]),
                    "transcript": transcripts.get(utterance_id, ""),
                })
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create same-utterance CMU ARCTIC manifests")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-speaker")
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    pairs = build_pairs(args.root, args.target_speaker, args.all_targets)
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
    speakers = sorted({item["source_speaker"] for item in pairs} | {item["target_speaker"] for item in pairs})
    print(f"speakers ({len(speakers)}): {', '.join(speakers)}")


if __name__ == "__main__":
    main()
