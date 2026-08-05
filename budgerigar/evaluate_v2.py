from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset_v2 import VariableParallelSpeechDataset, collate_variable
from .model_v2 import BudgerigarV2Config, BudgerigarV2Model, length_mask
from .train import choose_device


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Budgerigar V2")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BudgerigarV2Model(BudgerigarV2Config(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    dataset = VariableParallelSpeechDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_variable)
    error = elements = duration_error = speaker_correct = samples = 0.0
    for source, source_lengths, target, target_lengths, target_names in loader:
        source, target = source.to(device), target.to(device)
        source_lengths, target_lengths = source_lengths.to(device), target_lengths.to(device)
        target_ids = torch.tensor([model.speaker_index(target_names[0])], device=device)
        prediction, log_ratio, _, logits = model(source, source_lengths, target_ids, target_lengths)
        mask = ~length_mask(target_lengths, target.shape[1])
        error += (prediction - target).abs()[mask].sum().item()
        elements += int(mask.sum()) * target.shape[-1]
        true_ratio = torch.log(target_lengths.float() / source_lengths.float())
        duration_error += (log_ratio - true_ratio).abs().sum().item()
        speaker_correct += (logits.argmax(-1) == target_ids).sum().item()
        samples += 1
    print(json.dumps({
        "mel_mae_teacher_duration": error / elements,
        "log_duration_mae": duration_error / samples,
        "speaker_classification_accuracy": speaker_correct / samples,
        "samples": int(samples),
    }, indent=2))


if __name__ == "__main__":
    main()
