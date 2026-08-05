from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

from .echo_data import EchoEpisodeDataset, collate_episodes, feature_stats, load_pairs, pair_report
from .neural_echo import NeuralEchoConfig, create_neural_echo, require_torch


@dataclass(frozen=True)
class EchoTrainingConfig:
    target_speaker: str = "arctic_slt"
    batch_size: int = 6
    learning_rate: float = 3e-4
    epochs: int = 20
    max_steps: int | None = 300
    seed: int = 17
    thinking_frames_min: int = 16
    thinking_frames_max: int = 28
    gradient_clip: float = 1.0
    max_train_pairs: int | None = 256
    max_validation_pairs: int | None = 64


def train_neural_echo(
    feature_manifest, output_dir, training=EchoTrainingConfig(),
    model_config=NeuralEchoConfig(), stats=None,
):
    torch, _, functional = require_torch(); torch.manual_seed(training.seed); random.seed(training.seed)
    pairs = load_pairs(feature_manifest, training.target_speaker)
    train_pairs = [p for p in pairs if p.split == "train"]; validation_pairs = [p for p in pairs if p.split == "validation"]
    if training.max_train_pairs is not None:
        train_pairs = train_pairs[:training.max_train_pairs]
    if training.max_validation_pairs is not None:
        validation_pairs = validation_pairs[:training.max_validation_pairs]
    if not train_pairs or not validation_pairs: raise ValueError("non-empty train and validation pairs are required")
    print("[prepare] all pairs:", pair_report(pairs), flush=True)
    print(f"[prepare] smoke subset: train={len(train_pairs)}, validation={len(validation_pairs)}", flush=True)
    if stats is None:
        print("[prepare] computing feature statistics", flush=True)
        stats = feature_stats(train_pairs)
    else:
        print("[prepare] reusing feature statistics from the notebook", flush=True)
    thinking_frames = (training.thinking_frames_min, training.thinking_frames_max)
    print("[prepare] preloading train features from Drive", flush=True)
    train_set = EchoEpisodeDataset(train_pairs, stats, thinking_frames, preload=True)
    print("[prepare] preloading validation features from Drive", flush=True)
    validation_set = EchoEpisodeDataset(validation_pairs, stats, thinking_frames, preload=True)
    loader = torch.utils.data.DataLoader
    train_loader = loader(train_set, batch_size=training.batch_size, shuffle=True, num_workers=0, collate_fn=collate_episodes)
    validation_loader = loader(validation_set, batch_size=training.batch_size, shuffle=False, num_workers=0, collate_fn=collate_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = create_neural_echo(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=1e-4)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history = []; best = float("inf"); step = 0
    print(f"[train] device={device}, max_steps={training.max_steps}", flush=True)
    for epoch in range(training.epochs):
        model.train(); total = count = 0
        for inputs, targets, voice, valid, _ in train_loader:
            inputs, targets, voice, valid = inputs.to(device), targets.to(device), voice.to(device), valid.to(device)
            predicted, voice_logits, _ = model(inputs)
            acoustic = ((predicted - targets).abs().mean(-1) * (1.0 + 3.0 * voice))[valid].mean()
            confidence = functional.binary_cross_entropy_with_logits(voice_logits[valid], voice[valid])
            loss = acoustic + 0.25 * confidence
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip); optimizer.step()
            frames = int(valid.sum()); total += float(loss) * frames; count += frames; step += 1
            if step == 1 or step % 25 == 0:
                print(f"[train] step={step} loss={float(loss):.5f}", flush=True)
            if training.max_steps and step >= training.max_steps: break
        model.eval(); validation_total = validation_count = 0
        with torch.no_grad():
            for inputs, targets, voice, valid, _ in validation_loader:
                inputs, targets, voice, valid = inputs.to(device), targets.to(device), voice.to(device), valid.to(device)
                predicted, voice_logits, _ = model(inputs)
                loss = ((predicted - targets).abs().mean(-1) * (1.0 + 3.0 * voice))[valid].mean()
                loss += 0.25 * functional.binary_cross_entropy_with_logits(voice_logits[valid], voice[valid])
                frames = int(valid.sum()); validation_total += float(loss) * frames; validation_count += frames
        metrics = {"epoch": epoch + 1, "step": step, "train_loss": total / count, "validation_loss": validation_total / validation_count}
        history.append(metrics); print(json.dumps(metrics))
        subset = {"train": len(train_pairs), "validation": len(validation_pairs)}
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "stats": stats, "model_config": asdict(model_config), "training_config": asdict(training), "pairs": pair_report(pairs), "training_subset": subset, "history": history}
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["validation_loss"] < best: best = metrics["validation_loss"]; torch.save(checkpoint, output_dir / "best.pt")
        if training.max_steps and step >= training.max_steps: break
    report = {"behavior": "continuous_neural_listen_then_repeat", "device": str(device), "steps": step, "best_validation_loss": best, "pairs": pair_report(pairs), "training_subset": {"train": len(train_pairs), "validation": len(validation_pairs)}, "history": history}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("feature_manifest", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-speaker", default="arctic_slt"); parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(); train_neural_echo(args.feature_manifest, args.output_dir, EchoTrainingConfig(target_speaker=args.target_speaker, max_steps=args.max_steps)); return 0


if __name__ == "__main__": raise SystemExit(main())
