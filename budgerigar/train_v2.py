from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset_v2 import VariableParallelSpeechDataset, collate_variable
from .model_v2 import BudgerigarV2Config, BudgerigarV2Model, length_mask
from .train import choose_device


def masked_acoustic_loss(prediction, target, lengths):
    mask = ~length_mask(lengths, target.shape[1])
    reconstruction = (prediction - target).abs()[mask].mean()
    delta_mask = mask[:, 1:] & mask[:, :-1]
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    delta = (prediction_delta - target_delta).abs()[delta_mask].mean()
    return reconstruction + 0.25 * delta, reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Budgerigar V2 listen-then-repeat model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/budgerigar_v2.pt"))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    device = choose_device(args.device)
    dataset = VariableParallelSpeechDataset(args.manifest)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    config = BudgerigarV2Config(
        hidden_dim=args.hidden_dim, encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers, speaker_names=dataset.speaker_names,
    )
    model = BudgerigarV2Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = 0
    log_started = time.perf_counter()
    while step < args.steps:
        for source, source_lengths, target, target_lengths, target_names in loader:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            source_lengths = source_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            target_ids = torch.tensor(
                [config.speaker_names.index(name) for name in target_names], device=device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction, log_ratio, _, speaker_logits = model(
                    source, source_lengths, target_ids, target_lengths,
                )
                acoustic_loss, mel_mae = masked_acoustic_loss(prediction, target, target_lengths)
                duration_target = torch.log(target_lengths.float() / source_lengths.float())
                duration_loss = nn.functional.smooth_l1_loss(log_ratio, duration_target)
                speaker_loss = nn.functional.cross_entropy(speaker_logits, target_ids)
                loss = acoustic_loss + 0.3 * duration_loss + 0.2 * speaker_loss
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                elapsed = time.perf_counter() - log_started
                accuracy = (speaker_logits.argmax(-1) == target_ids).float().mean().item()
                print(
                    f"step={step} loss={loss.item():.4f} mel={mel_mae.item():.4f} "
                    f"duration={duration_loss.item():.4f} speaker_acc={accuracy:.1%} "
                    f"speed={args.log_every / elapsed:.2f} steps/s"
                )
                log_started = time.perf_counter()
            if step % args.save_every == 0 or step == args.steps:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "architecture": "budgerigar_v2", "model": model.state_dict(),
                    "config": asdict(config), "step": step,
                }, args.checkpoint)
                print(f"saved {args.checkpoint}")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()

