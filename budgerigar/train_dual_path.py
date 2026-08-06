from __future__ import annotations

from dataclasses import asdict, dataclass
import json, random
from pathlib import Path

from .dual_path_echo import DualPathEchoConfig, create_dual_path_echo
from .echo_data import EchoEpisodeDataset, collate_episodes, feature_stats, load_pairs
from .neural_echo import require_torch
from .train_hierarchical import sequence_contrastive_loss


@dataclass(frozen=True)
class DualPathTrainingConfig:
    target_speaker: str = "arctic_slt"
    batch_size: int = 3
    learning_rate: float = 2e-4
    epochs: int = 20
    max_steps: int = 300
    max_train_pairs: int = 256
    max_validation_pairs: int = 64
    thinking_frames_min: int = 16
    thinking_frames_max: int = 28
    contrastive_weight: float = 0.5
    gradient_clip: float = 1.0
    clock_weight: float = 0.0
    teacher_forcing_ratio: float = 0.0
    seed: int = 41
    initialization_checkpoint: str | None = None


def train_dual_path_echo(feature_manifest, output_dir, training=DualPathTrainingConfig(),
                         model_config=DualPathEchoConfig(), stats=None, model_factory=create_dual_path_echo,
                         architecture="dual_path_neural_echo", ablation_names=("local", "abstract")):
    torch, _, functional = require_torch(); torch.manual_seed(training.seed); random.seed(training.seed)
    pairs = load_pairs(feature_manifest, training.target_speaker)
    train_pairs = [p for p in pairs if p.split == "train"][:training.max_train_pairs]
    validation_pairs = [p for p in pairs if p.split == "validation"][:training.max_validation_pairs]
    stats = stats or feature_stats(train_pairs); thinking = (training.thinking_frames_min, training.thinking_frames_max)
    train_set = EchoEpisodeDataset(train_pairs, stats, thinking, preload=True)
    validation_set = EchoEpisodeDataset(validation_pairs, stats, thinking, preload=True)
    loader = torch.utils.data.DataLoader
    train_loader = loader(train_set, batch_size=training.batch_size, shuffle=True, num_workers=0, collate_fn=collate_episodes)
    validation_loader = loader(validation_set, batch_size=training.batch_size, shuffle=False, num_workers=0, collate_fn=collate_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(model_config).to(device)
    if training.initialization_checkpoint:
        source = torch.load(training.initialization_checkpoint, map_location="cpu", weights_only=True)
        current = model.state_dict(); transferred = {k: v for k, v in source["model"].items() if k in current and current[k].shape == v.shape}
        model.load_state_dict(transferred, strict=False)
        print(f"[transfer] tensors={len(transferred)} source={training.initialization_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=1e-4)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history = []; best = float("inf"); step = 0
    for epoch in range(training.epochs):
        model.train(); total = count = 0
        for inputs, targets, voice, valid, metadata in train_loader:
            inputs, targets, voice, valid = inputs.to(device), targets.to(device), voice.to(device), valid.to(device)
            if getattr(model_config, "output_feedback", False):
                predicted, voice_logits, _, diagnostics = model(inputs, teacher_mel=targets, teacher_forcing_ratio=training.teacher_forcing_ratio)
            else:
                predicted, voice_logits, _, diagnostics = model(inputs)
            frame_l1 = (predicted - targets).abs().mean(-1)
            acoustic = (frame_l1 * (1 + 3 * voice))[valid].mean()
            confidence = functional.binary_cross_entropy_with_logits(voice_logits[valid], voice[valid])
            contrastive = sequence_contrastive_loss(predicted, targets, voice, metadata, functional)
            clock = predicted.new_zeros(())
            if "write_phase" in diagnostics:
                slots = float(getattr(model_config, "event_slots", 1))
                write_target = inputs[..., -1].cumsum(1) / max(float(getattr(model_config, "update_stride", 1)), 1.0)
                write_target = write_target.clamp(max=slots - 1)
                final_write = diagnostics["write_phase"][:, -1].detach().unsqueeze(1)
                voiced_progress = voice.cumsum(1) / voice.sum(1, keepdim=True).clamp_min(1.0)
                read_target = voiced_progress * final_write
                clock = (functional.smooth_l1_loss(diagnostics["write_phase"][valid] / slots, write_target[valid] / slots)
                         + functional.smooth_l1_loss(diagnostics["read_phase"][valid] / slots, read_target[valid] / slots))
            loss = acoustic + 0.25 * confidence + training.contrastive_weight * contrastive + training.clock_weight * clock
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip); optimizer.step()
            step += 1; total += float(loss); count += 1
            if step == 1 or step % 10 == 0: print(f"[dual train] step={step} loss={float(loss):.4f} acoustic={float(acoustic):.4f} contrastive={float(contrastive):.4f} clock={float(clock):.4f}", flush=True)
            if step >= training.max_steps: break
        model.eval(); full_l1 = local_l1 = abstract_l1 = batches = 0
        with torch.no_grad():
            for inputs, targets, voice, valid, _ in validation_loader:
                inputs, targets, voice, valid = inputs.to(device), targets.to(device), voice.to(device), valid.to(device)
                mask = (voice > .5) & valid
                full = model(inputs)[0]
                no_local = model(inputs, **{f"ablate_{ablation_names[0]}": True})[0]
                no_abstract = model(inputs, **{f"ablate_{ablation_names[1]}": True})[0]
                full_l1 += float((full[mask] - targets[mask]).abs().mean())
                local_l1 += float((no_local[mask] - targets[mask]).abs().mean())
                abstract_l1 += float((no_abstract[mask] - targets[mask]).abs().mean()); batches += 1
        metrics = {"epoch": epoch + 1, "step": step, "train_loss": total / count,
                   "validation_repeat_l1": full_l1 / batches,
                   "no_local_degradation": (local_l1 - full_l1) / batches,
                   "no_abstract_degradation": (abstract_l1 - full_l1) / batches}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        checkpoint = {"architecture":architecture, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                      "stats":stats, "model_config":asdict(model_config), "training_config":asdict(training), "history":history}
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["validation_repeat_l1"] < best: best = metrics["validation_repeat_l1"]; torch.save(checkpoint, output_dir / "best.pt")
        if step >= training.max_steps: break
    report = {"architecture":architecture, "steps":step, "best_validation_repeat_l1":best, "history":history}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
