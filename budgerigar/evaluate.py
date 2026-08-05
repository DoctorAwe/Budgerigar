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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BudgerigarModel(BudgerigarConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    loader = DataLoader(
        ParallelSpeechDataset(args.manifest, segment_frames=None),
        batch_size=args.batch_size, collate_fn=collate_parallel,
    )
    absolute_error = copy_error = equivalence_error = elements = 0.0
    output_sum = output_square_sum = target_sum = target_square_sum = 0.0
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        whole = model(source)
        state, pieces = None, []
        for start in range(0, source.shape[1], args.chunk_frames):
            output, state = model.forward_chunk(source[:, start:start + args.chunk_frames], state)
            pieces.append(output)
        chunked = torch.cat(pieces, 1)
        absolute_error += (whole - target).abs().sum().item()
        copy_error += (source - target).abs().sum().item()
        equivalence_error = max(equivalence_error, (whole - chunked).abs().max().item())
        elements += target.numel()
        output_sum += whole.double().sum().item()
        output_square_sum += whole.double().square().sum().item()
        target_sum += target.double().sum().item()
        target_square_sum += target.double().square().sum().item()
    model_mae = absolute_error / elements
    copy_mae = copy_error / elements
    output_variance = max(0.0, output_square_sum / elements - (output_sum / elements) ** 2)
    target_variance = max(0.0, target_square_sum / elements - (target_sum / elements) ** 2)
    print(json.dumps({
        "mel_mae": model_mae,
        "copy_source_mel_mae": copy_mae,
        "relative_improvement_over_copy": (copy_mae - model_mae) / max(copy_mae, 1e-12),
        "output_std": output_variance ** 0.5,
        "target_std": target_variance ** 0.5,
        "dynamic_range_ratio": (output_variance / max(target_variance, 1e-12)) ** 0.5,
        "chunk_max_error": equivalence_error,
    }, indent=2))


if __name__ == "__main__":
    main()
