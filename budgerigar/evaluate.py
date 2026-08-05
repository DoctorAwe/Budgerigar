from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ParallelSpeechDataset, collate_parallel
from .model import BudgerigarConfig, BudgerigarModel
from .train import choose_device


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction and chunk equivalence")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BudgerigarModel(BudgerigarConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    loader = DataLoader(ParallelSpeechDataset(args.manifest), batch_size=args.batch_size, collate_fn=collate_parallel)
    absolute_error = equivalence_error = frames = 0.0
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        whole = model(source)
        state, pieces = None, []
        for start in range(0, source.shape[1], args.chunk_frames):
            output, state = model.forward_chunk(source[:, start:start + args.chunk_frames], state)
            pieces.append(output)
        chunked = torch.cat(pieces, 1)
        absolute_error += (whole - target).abs().sum().item()
        equivalence_error = max(equivalence_error, (whole - chunked).abs().max().item())
        frames += target.numel()
    print(json.dumps({"mel_mae": absolute_error / frames, "chunk_max_error": equivalence_error}, indent=2))


if __name__ == "__main__":
    main()

