from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .train import choose_device
from .waveform_dataset import WaveformStreamDataset, collate_waveform
from .waveform_model import StreamingWaveformProcessor, WaveformProcessorConfig


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unified waveform processor")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = WaveformProcessorConfig(**saved["config"])
    model = StreamingWaveformProcessor(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    dataset = WaveformStreamDataset(args.manifest, episodes=args.episodes, utterances_per_episode=1, seed=9_000_000)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_waveform)
    mae = silence_energy = speech_energy = target_speech_energy = chunk_error = 0.0
    for batch in loader:
        source = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["speech_mask"].to(device)
        speaker = torch.tensor([model.speaker_index(batch["target_speakers"][0])], device=device)
        whole = model(source, speaker)
        state, pieces = None, []
        for start in range(0, source.shape[1], config.hop_samples * 16):
            output, state = model.forward_chunk(source[:, start:start + config.hop_samples * 16], speaker, state)
            pieces.append(output)
        chunked = torch.cat(pieces, 1)
        mae += (whole - target).abs().mean().item()
        silence_energy += whole[~mask].square().mean().sqrt().item()
        speech_energy += whole[mask].square().mean().sqrt().item()
        target_speech_energy += target[mask].square().mean().sqrt().item()
        chunk_error = max(chunk_error, (whole - chunked).abs().max().item())
    count = len(dataset)
    print(json.dumps({
        "waveform_mae": mae / count,
        "output_silence_rms": silence_energy / count,
        "output_speech_rms": speech_energy / count,
        "target_speech_rms": target_speech_energy / count,
        "silence_to_speech_ratio": (silence_energy / count) / max(speech_energy / count, 1e-12),
        "chunk_max_error": chunk_error,
    }, indent=2))


if __name__ == "__main__":
    main()
