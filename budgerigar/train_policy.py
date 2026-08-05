from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .realtime_policy import RealtimePolicyConfig, RealtimePolicyModel
from .streaming_dataset import StreamingEpisodeDataset, collate_streaming
from .train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the real-time endpoint/output policy")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/realtime_policy.pt"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    dataset = StreamingEpisodeDataset(args.manifest, episodes=args.episodes)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_streaming,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    config = RealtimePolicyConfig(speaker_names=dataset.targets)
    model = RealtimePolicyModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    action_weights = torch.tensor([0.15, 3.0, 0.35, 3.0], device=device)
    step = 0
    started = time.perf_counter()
    while step < args.steps:
        for audio, lengths, actions, _, _, endpoints, target_names in loader:
            audio, actions, endpoints = audio.to(device), actions.to(device), endpoints.to(device)
            lengths = lengths.to(device)
            speakers = torch.tensor([config.speaker_names.index(name) for name in target_names], device=device)
            mask = torch.arange(audio.shape[1], device=device)[None] < lengths[:, None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                endpoint_logits, action_logits, _ = model.forward_sequence(audio, speakers, actions=actions)
                action_loss = nn.functional.cross_entropy(
                    action_logits[mask], actions[mask], weight=action_weights,
                )
                endpoint_loss = nn.functional.binary_cross_entropy_with_logits(
                    endpoint_logits[mask], endpoints[mask].float(), pos_weight=torch.tensor(20.0, device=device),
                )
                loss = action_loss + 0.5 * endpoint_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % 20 == 0:
                elapsed = time.perf_counter() - started
                action_accuracy = (action_logits[mask].argmax(-1) == actions[mask]).float().mean().item()
                endpoint_prediction = endpoint_logits[mask].sigmoid() > 0.5
                endpoint_recall = (
                    (endpoint_prediction & endpoints[mask]).sum().float()
                    / endpoints[mask].sum().clamp_min(1)
                ).item()
                print(
                    f"step={step} loss={loss.item():.4f} action_acc={action_accuracy:.1%} "
                    f"endpoint_recall={endpoint_recall:.1%} speed={20 / elapsed:.2f} steps/s"
                )
                started = time.perf_counter()
            if step % 1000 == 0 or step == args.steps:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "architecture": "realtime_policy", "model": model.state_dict(),
                    "config": asdict(config), "step": step,
                }, args.checkpoint)
                print(f"saved {args.checkpoint}")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
