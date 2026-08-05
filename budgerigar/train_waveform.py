from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .train import choose_device
from .waveform_dataset import WaveformStreamDataset, collate_waveform
from .waveform_model import StreamingWaveformProcessor, WaveformProcessorConfig


def multi_resolution_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for n_fft, hop, window_size in ((256, 64, 256), (512, 128, 512), (1024, 256, 1024)):
        window = torch.hann_window(window_size, device=prediction.device, dtype=prediction.dtype)
        predicted = torch.stft(
            prediction, n_fft, hop, window_size, window, return_complex=True,
        ).abs().clamp_min(1e-5)
        expected = torch.stft(
            target, n_fft, hop, window_size, window, return_complex=True,
        ).abs().clamp_min(1e-5)
        spectral_convergence = (predicted - expected).norm() / expected.norm().clamp_min(1e-5)
        log_magnitude = (predicted.log() - expected.log()).abs().mean()
        losses.append(spectral_convergence + log_magnitude)
    return torch.stack(losses).mean()


def waveform_losses(prediction, target, speech_mask, hop_samples):
    speech = speech_mask
    silence = ~speech_mask
    speech_waveform = (prediction[speech] - target[speech]).abs().mean()
    silence_rms = prediction[silence].float().square().mean().sqrt()
    predicted_energy = F.avg_pool1d(prediction.abs().unsqueeze(1), hop_samples, hop_samples)[:, 0]
    target_energy = F.avg_pool1d(target.abs().unsqueeze(1), hop_samples, hop_samples)[:, 0]
    envelope = F.l1_loss(predicted_energy, target_energy)
    speech_float = speech_mask.to(prediction.dtype)
    spectral = multi_resolution_stft_loss(
        (prediction * speech_float).float(), (target * speech_float).float(),
    )
    loss = 2.0 * speech_waveform + spectral + 5.0 * silence_rms + 3.0 * envelope
    return loss, speech_waveform, spectral, envelope, silence_rms


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified streaming waveform processor")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/budgerigar_waveform.pt"))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=30_000)
    parser.add_argument("--utterances-per-episode", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    dataset = WaveformStreamDataset(
        args.manifest, episodes=args.episodes,
        utterances_per_episode=args.utterances_per_episode,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_waveform,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    config = WaveformProcessorConfig(
        hidden_dim=args.hidden_dim, speaker_names=dataset.speaker_names,
    )
    model = StreamingWaveformProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = 0
    started = time.perf_counter()
    chunk_samples = args.chunk_frames * config.hop_samples
    while step < args.steps:
        for batch in loader:
            input_wave = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            speech_mask = batch["speech_mask"].to(device, non_blocking=True)
            speakers = torch.tensor(
                [config.speaker_names.index(name) for name in batch["target_speakers"]],
                device=device,
            )
            state = None
            pieces = []
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                for start in range(0, input_wave.shape[1], chunk_samples):
                    output, state = model.forward_chunk(
                        input_wave[:, start:start + chunk_samples], speakers, state,
                    )
                    pieces.append(output)
                prediction = torch.cat(pieces, dim=1)
                loss, waveform_loss, spectral_loss, envelope_loss, silence_loss = waveform_losses(
                    prediction, target, speech_mask, config.hop_samples,
                )
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
            if step % 20 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step={step} loss={loss.item():.4f} wave={waveform_loss.item():.4f} "
                    f"stft={spectral_loss.item():.4f} envelope={envelope_loss.item():.4f} "
                    f"silence_rms={silence_loss.item():.4f} "
                    f"speed={20 / elapsed:.2f} steps/s"
                )
                started = time.perf_counter()
            if step % 1000 == 0 or step == args.steps:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "architecture": "streaming_waveform", "model": model.state_dict(),
                    "config": asdict(config), "step": step,
                }, args.checkpoint)
                print(f"saved {args.checkpoint}")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
