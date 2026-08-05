from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from .echo_data import EchoEpisodeDataset, load_pairs
from .checkpoint_model import forward_echo, restore_echo_model
from .neural_echo import require_torch


def _resize(sequence, frames, functional):
    if len(sequence) == frames:
        return sequence
    return functional.interpolate(
        sequence.transpose(0, 1).unsqueeze(0), size=frames,
        mode="linear", align_corners=False,
    )[0].transpose(0, 1)


def _candidate_indices(index: int, count: int, candidates: int, key: str) -> list[int]:
    others = [value for value in range(count) if value != index]
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    # Deterministic local shuffle without depending on Python's hash seed.
    others.sort(key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())
    return [index, *others[:max(0, candidates - 1)]]


def evaluate_content(
    checkpoint_path: str | Path,
    feature_manifest: str | Path,
    output_dir: str | Path,
    split: str = "validation",
    max_pairs: int = 64,
    candidates: int = 16,
) -> dict:
    torch, _, functional = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_config = checkpoint["training_config"]
    thinking_frames = (
        training_config.get("thinking_frames_min", 16),
        training_config.get("thinking_frames_max", 28),
    )
    pairs = [
        pair for pair in load_pairs(feature_manifest, training_config["target_speaker"])
        if pair.split == split
    ][:max_pairs]
    if len(pairs) < 2:
        raise ValueError("content evaluation needs at least two parallel pairs")
    dataset = EchoEpisodeDataset(pairs, checkpoint["stats"], thinking_frames, preload=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, architecture = restore_echo_model(checkpoint, device); model.eval()

    episodes = [dataset[index] for index in range(len(dataset))]
    target_sequences = [item[1][item[3]["repeat_start"]:] for item in episodes]
    all_target_frames = torch.cat(target_sequences)
    mean_frame = all_target_frames.mean(0)
    correct_distances = []; shuffled_distances = []; mean_template_distances = []
    silence_distances = []; ablated_distances = []; output_changes = []
    pairwise_wins = 0; retrieval_hits = 0; ranks = []; examples = []
    with torch.no_grad():
        for index, (inputs, target_timeline, _, metadata) in enumerate(episodes):
            repeat_start = metadata["repeat_start"]
            predicted, _, _, _ = forward_echo(model, architecture, inputs.unsqueeze(0).to(device))
            predicted = predicted[0, repeat_start:].cpu()
            correct = target_sequences[index]
            correct_distance = float((predicted - correct).abs().mean())
            correct_distances.append(correct_distance)

            candidate_indices = _candidate_indices(
                index, len(episodes), min(candidates, len(episodes)), metadata["source_id"],
            )
            distances = []
            for candidate_index in candidate_indices:
                candidate = _resize(target_sequences[candidate_index], len(predicted), functional)
                distances.append(float((predicted - candidate).abs().mean()))
            correct_rank = sorted(range(len(distances)), key=distances.__getitem__).index(0) + 1
            ranks.append(correct_rank); retrieval_hits += correct_rank == 1
            wrong_distance = mean(distances[1:])
            shuffled_distances.append(wrong_distance); pairwise_wins += correct_distance < wrong_distance

            template = mean_frame.repeat(len(predicted), 1)
            mean_template_distances.append(float((predicted - template).abs().mean()))
            silence = target_timeline[0].repeat(len(predicted), 1)
            silence_distances.append(float((predicted - silence).abs().mean()))

            ablated = inputs.clone()
            # Replace the source evidence with the same continuous silence seen later.
            silence_input = inputs[repeat_start].clone()
            ablated[:metadata["source_frames"]] = silence_input
            predicted_ablated, _, _, _ = forward_echo(
                model, architecture, ablated.unsqueeze(0).to(device),
            )
            predicted_ablated = predicted_ablated[0, repeat_start:].cpu()
            ablated_distances.append(float((predicted_ablated - correct).abs().mean()))
            output_changes.append(float((predicted_ablated - predicted).abs().mean()))

            if len(examples) < 8:
                examples.append({
                    "source_id": metadata["source_id"], "correct_l1": correct_distance,
                    "shuffled_l1": wrong_distance, "rank": correct_rank,
                    "ablated_l1": ablated_distances[-1], "output_change": output_changes[-1],
                })

    candidate_count = min(candidates, len(episodes))
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "architecture": architecture,
        "split": split,
        "pairs": len(episodes), "candidates": candidate_count,
        "distance_l1": {
            "correct_target": mean(correct_distances),
            "shuffled_targets": mean(shuffled_distances),
            "mean_target_template": mean(mean_template_distances),
            "silence_template": mean(silence_distances),
            "input_ablated": mean(ablated_distances),
        },
        "correct_vs_shuffled_margin": mean(shuffled_distances) - mean(correct_distances),
        "correct_vs_shuffled_win_rate": pairwise_wins / len(episodes),
        "retrieval_top1": retrieval_hits / len(episodes),
        "retrieval_chance": 1.0 / candidate_count,
        "mean_retrieval_rank": mean(ranks),
        "input_ablation_degradation": mean(ablated_distances) - mean(correct_distances),
        "input_ablation_output_change": mean(output_changes),
    }
    report["content_pass"] = bool(
        report["correct_vs_shuffled_margin"] > 0.02
        and report["correct_vs_shuffled_win_rate"] > 0.75
        and report["retrieval_top1"] > report["retrieval_chance"] * 3
        and report["input_ablation_degradation"] > 0.02
    )
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "content_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "content_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Test whether NeuralEcho remembers sentence content")
    parser.add_argument("checkpoint", type=Path); parser.add_argument("feature_manifest", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-pairs", type=int, default=64); parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args(); report = evaluate_content(args.checkpoint, args.feature_manifest, args.output_dir, max_pairs=args.max_pairs, candidates=args.candidates)
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
