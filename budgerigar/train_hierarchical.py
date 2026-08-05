from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

from .echo_data import EchoEpisodeDataset, collate_episodes, feature_stats, load_pairs, pair_report
from .hierarchical_echo import HierarchicalEchoConfig, create_hierarchical_echo
from .neural_echo import require_torch


@dataclass(frozen=True)
class HierarchicalTrainingConfig:
    target_speaker: str = "arctic_slt"
    batch_size: int = 4
    learning_rate: float = 2e-4
    epochs: int = 20
    max_steps: int = 200
    seed: int = 23
    thinking_frames_min: int = 16
    thinking_frames_max: int = 28
    max_train_pairs: int = 256
    max_validation_pairs: int = 64
    contrastive_margin: float = 0.15
    contrastive_weight: float = 0.5
    gradient_clip: float = 1.0


def _resize(sequence, frames, functional):
    if len(sequence) == frames: return sequence
    return functional.interpolate(sequence.T.unsqueeze(0), size=frames, mode="linear", align_corners=False)[0].T


def sequence_contrastive_loss(predicted, targets, voice, metadata, functional, margin=0.15):
    """Prefer the paired sentence over another target sentence in the batch."""
    if len(predicted) < 2:
        return predicted.new_zeros(())
    losses = []
    for index, item in enumerate(metadata):
        start = item["repeat_start"]; end = item["total_frames"]
        prediction = predicted[index, start:end]
        correct = targets[index, start:end]
        correct_mask = voice[index, start:end] >= 0.5
        correct_distance = (prediction[correct_mask] - correct[correct_mask]).abs().mean() if correct_mask.any() else (prediction - correct).abs().mean()
        wrong_index = (index + 1) % len(predicted)
        wrong_item = metadata[wrong_index]
        wrong = targets[wrong_index, wrong_item["repeat_start"]:wrong_item["total_frames"]]
        wrong = _resize(wrong, len(prediction), functional)
        wrong_distance = (prediction - wrong).abs().mean()
        losses.append(functional.relu(margin + correct_distance - wrong_distance))
    return sum(losses) / len(losses)


def train_hierarchical_echo(feature_manifest, output_dir, training=HierarchicalTrainingConfig(), model_config=HierarchicalEchoConfig(), stats=None):
    torch, _, functional = require_torch(); torch.manual_seed(training.seed); random.seed(training.seed)
    all_pairs = load_pairs(feature_manifest, training.target_speaker)
    train_pairs = [p for p in all_pairs if p.split == "train"][:training.max_train_pairs]
    validation_pairs = [p for p in all_pairs if p.split == "validation"][:training.max_validation_pairs]
    if not train_pairs or not validation_pairs: raise ValueError("non-empty train and validation pairs required")
    print("[prepare]", pair_report(all_pairs), flush=True)
    stats = stats or feature_stats(train_pairs)
    thinking = (training.thinking_frames_min, training.thinking_frames_max)
    print("[prepare] preloading hierarchical train subset", flush=True)
    train_set = EchoEpisodeDataset(train_pairs, stats, thinking, preload=True)
    print("[prepare] preloading hierarchical validation subset", flush=True)
    validation_set = EchoEpisodeDataset(validation_pairs, stats, thinking, preload=True)
    loader = torch.utils.data.DataLoader
    train_loader = loader(train_set, batch_size=training.batch_size, shuffle=True, num_workers=0, collate_fn=collate_episodes)
    validation_loader = loader(validation_set, batch_size=training.batch_size, shuffle=False, num_workers=0, collate_fn=collate_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_hierarchical_echo(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=1e-4)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history = []; best = float("inf"); step = 0
    for epoch in range(training.epochs):
        model.train(); total = count = 0
        for inputs, targets, voice, valid, metadata in train_loader:
            inputs, targets, voice, valid = inputs.to(device), targets.to(device), voice.to(device), valid.to(device)
            predicted, voice_logits, _, diagnostics = model(inputs)
            acoustic = ((predicted - targets).abs().mean(-1) * (1 + 3 * voice))[valid].mean()
            confidence = functional.binary_cross_entropy_with_logits(voice_logits[valid], voice[valid])
            contrastive = sequence_contrastive_loss(
                predicted, targets, voice, metadata, functional, training.contrastive_margin,
            )
            # Encourage memory stability when input evidence is absent, without hard-closing a gate.
            silence = inputs[..., -1] < 0.5
            stability = diagnostics["write_strength"][silence & valid].mean()
            loss = acoustic + 0.25 * confidence + training.contrastive_weight * contrastive + 0.02 * stability
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip); optimizer.step()
            step += 1; total += float(loss); count += 1
            if step == 1 or step % 10 == 0:
                print(f"[train] step={step} total={float(loss):.4f} acoustic={float(acoustic):.4f} contrastive={float(contrastive):.4f} write={float(stability):.4f}", flush=True)
            if step >= training.max_steps: break
        model.eval(); validation_total = validation_count = 0
        with torch.no_grad():
            for inputs, targets, voice, valid, metadata in validation_loader:
                inputs, targets, voice, valid = inputs.to(device), targets.to(device), voice.to(device), valid.to(device)
                predicted, voice_logits, _, _ = model(inputs)
                acoustic = ((predicted - targets).abs().mean(-1) * (1 + 3 * voice))[valid].mean()
                contrastive = sequence_contrastive_loss(
                    predicted, targets, voice, metadata, functional, training.contrastive_margin,
                )
                validation_total += float(acoustic + training.contrastive_weight * contrastive); validation_count += 1
        metrics = {"epoch": epoch + 1, "step": step, "train_loss": total / count, "validation_content_loss": validation_total / validation_count}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        checkpoint = {"architecture": "hierarchical_token_echo", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "stats": stats, "model_config": asdict(model_config), "training_config": asdict(training), "history": history}
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["validation_content_loss"] < best: best = metrics["validation_content_loss"]; torch.save(checkpoint, output_dir / "best.pt")
        if step >= training.max_steps: break
    report = {"architecture": "hierarchical_token_echo", "steps": step, "best_validation_content_loss": best, "history": history}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("feature_manifest", type=Path); parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); train_hierarchical_echo(args.feature_manifest, args.output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
