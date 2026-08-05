from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ParallelSpeechDataset, collate_parallel
from .model import BudgerigarConfig, BudgerigarModel


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstruction = nn.functional.l1_loss(prediction, target)
    delta_prediction = prediction[:, 1:] - prediction[:, :-1]
    delta_target = target[:, 1:] - target[:, :-1]
    return reconstruction + 0.25 * nn.functional.l1_loss(delta_prediction, delta_target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the causal Budgerigar baseline")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/budgerigar.pt"))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-frames", type=int, default=320)
    parser.add_argument("--chunk-min", type=int, default=16)
    parser.add_argument("--chunk-max", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    device = choose_device(args.device)
    dataset = ParallelSpeechDataset(args.manifest, segment_frames=args.segment_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_parallel, drop_last=True)
    config = BudgerigarConfig(token_dim=args.token_dim, layers=args.layers)
    model = BudgerigarModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # torch.cuda.amp keeps compatibility with the minimum supported PyTorch 2.2.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    step = 0
    model.train()
    while step < args.steps:
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            state = None
            predictions = []
            cursor = 0
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                while cursor < source.shape[1]:
                    width = random.randint(args.chunk_min, args.chunk_max)
                    output, state = model.forward_chunk(source[:, cursor:cursor + width], state)
                    predictions.append(output)
                    cursor += width
                prediction = torch.cat(predictions, dim=1)
                loss = spectral_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % args.log_every == 0:
                print(f"step={step} loss={loss.item():.5f}")
            if step % args.save_every == 0 or step == args.steps:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "config": asdict(config), "step": step}, args.checkpoint)
                print(f"saved {args.checkpoint}")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
