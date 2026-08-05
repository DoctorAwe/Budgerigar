from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, median

from .echo_data import EchoEpisodeDataset, load_pairs
from .checkpoint_model import forward_echo, restore_echo_model
from .neural_echo import require_torch


def _first_true(values):
    indices = values.nonzero(as_tuple=False)
    return int(indices[0]) if len(indices) else None


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    feature_manifest: str | Path,
    output_dir: str | Path,
    split: str = "validation",
    max_pairs: int = 64,
    threshold: float = 0.5,
) -> dict:
    torch, _, _ = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_config = checkpoint["training_config"]
    target_speaker = training_config["target_speaker"]
    thinking_frames = (
        training_config.get("thinking_frames_min", 16),
        training_config.get("thinking_frames_max", 28),
    )
    pairs = [pair for pair in load_pairs(feature_manifest, target_speaker) if pair.split == split][:max_pairs]
    if not pairs:
        raise ValueError(f"no pairs found for split {split!r}")
    dataset = EchoEpisodeDataset(pairs, checkpoint["stats"], thinking_frames, preload=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, architecture = restore_echo_model(checkpoint, device); model.eval()
    examples = []
    listen_probs = []; thinking_probs = []; repeat_probs = []
    listen_false = []; thinking_false = []; repeat_recalls = []; onset_errors = []
    listen_mel = []; thinking_mel = []; repeat_mel = []; silent_outputs = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, target_mel, target_voice, metadata = dataset[index]
            predicted_mel, voice_logits, _, _ = forward_echo(
                model, architecture, inputs.unsqueeze(0).to(device),
            )
            predicted_mel = predicted_mel[0].cpu(); probability = voice_logits[0].sigmoid().cpu()
            source_end = metadata["source_frames"]; repeat_start = metadata["repeat_start"]; total = metadata["total_frames"]
            listen_slice = slice(0, source_end); thinking_slice = slice(source_end, repeat_start); repeat_slice = slice(repeat_start, total)
            listen_probs.append(float(probability[listen_slice].mean()))
            thinking_probs.append(float(probability[thinking_slice].mean()))
            repeat_probs.append(float(probability[repeat_slice].mean()))
            listen_false.append(float((probability[listen_slice] >= threshold).float().mean()))
            thinking_false.append(float((probability[thinking_slice] >= threshold).float().mean()))
            target_voiced = target_voice[repeat_slice] >= 0.5
            repeat_recalls.append(float((probability[repeat_slice][target_voiced] >= threshold).float().mean()) if target_voiced.any() else 0.0)
            predicted_onset = _first_true(probability >= threshold)
            target_onset_relative = _first_true(target_voice[repeat_slice] >= 0.5)
            target_onset = repeat_start + (target_onset_relative or 0)
            if predicted_onset is None:
                silent_outputs += 1
            else:
                onset_errors.append(predicted_onset - target_onset)
            frame_l1 = (predicted_mel - target_mel).abs().mean(-1)
            listen_mel.append(float(frame_l1[listen_slice].mean()))
            thinking_mel.append(float(frame_l1[thinking_slice].mean()))
            repeat_mel.append(float(frame_l1[repeat_slice].mean()))
            if len(examples) < 8:
                examples.append({
                    "metadata": metadata, "probability": probability,
                    "target_voice": target_voice, "frame_l1": frame_l1,
                })
    def average(values): return mean(values) if values else None
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "architecture": architecture,
        "split": split, "pairs": len(pairs),
        "threshold": threshold, "hop_ms": 10,
        "voice_probability": {"listen": average(listen_probs), "thinking": average(thinking_probs), "repeat": average(repeat_probs)},
        "false_voice_rate": {"listen": average(listen_false), "thinking": average(thinking_false)},
        "repeat_voiced_recall": average(repeat_recalls),
        "silent_output_rate": silent_outputs / len(pairs),
        "onset_error_frames": {
            "mean": average(onset_errors), "median": median(onset_errors) if onset_errors else None,
            "mean_ms": average(onset_errors) * 10 if onset_errors else None,
        },
        "mel_l1": {"listen": average(listen_mel), "thinking": average(thinking_mel), "repeat": average(repeat_mel)},
        "behavior_pass": bool(
            average(listen_false) < 0.05 and average(thinking_false) < 0.05
            and average(repeat_recalls) > 0.5 and silent_outputs / len(pairs) < 0.1
        ),
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "behavior_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(examples, output_dir / "behavior_examples.pt")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate learned listen-think-repeat timing")
    parser.add_argument("checkpoint", type=Path); parser.add_argument("feature_manifest", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-pairs", type=int, default=64); parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(); report = evaluate_checkpoint(args.checkpoint, args.feature_manifest, args.output_dir, max_pairs=args.max_pairs, threshold=args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
