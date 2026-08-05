from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from .alignment import dtw_align_target
from .audio import AudioConfig, load_wave, log_mel


def process_item(task: tuple[dict, str, float]) -> dict:
    item, output_string, band_radius = task
    output = Path(output_string)
    speaker = item["source_speaker"]
    utterance = item["utterance_id"]
    source_path = output / f"{speaker}_{utterance}_source.pt"
    target_path = output / f"{speaker}_{utterance}_target_dtw.pt"
    if not source_path.exists() or not target_path.exists():
        audio = AudioConfig()
        source = log_mel(load_wave(item["source_path"], audio.sample_rate), audio)
        target = log_mel(load_wave(item["target_path"], audio.sample_rate), audio)
        aligned_target = dtw_align_target(source, target, band_radius)
        torch.save(source.contiguous(), source_path)
        torch.save(aligned_target.contiguous(), target_path)
    cached = dict(item)
    cached["source_mel_path"] = str(source_path.resolve())
    cached["target_mel_path"] = str(target_path.resolve())
    cached["alignment"] = "dtw"
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Mel features with cross-speaker DTW alignment")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--band-radius", type=float, default=0.25)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {args.manifest}")
    feature_dir = args.output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(row, str(feature_dir), args.band_radius) for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(process_item, tasks), 1):
            results.append(result)
            if index % 50 == 0 or index == len(tasks):
                print(f"aligned {index}/{len(tasks)}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / args.manifest.name
    with manifest.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved cached manifest to {manifest}")


if __name__ == "__main__":
    main()

