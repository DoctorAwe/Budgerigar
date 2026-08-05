from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .audio import AudioConfig, load_wave, log_mel, mel_filterbank
from .model import BudgerigarConfig, BudgerigarModel
from .train import choose_device


def approximate_waveform(
    log_mel_features: torch.Tensor,
    audio: AudioConfig,
    phase_reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diagnostic inversion; source phase is preferable to random Griffin-Lim phase."""
    try:
        import torchaudio
    except ImportError as error:
        raise RuntimeError("torchaudio is required to synthesize the demo WAV") from error
    # InverseMelScale uses an exact least-squares solve which can fail when
    # adjacent low-frequency Mel bands map to the same FFT bin. A truncated
    # Moore-Penrose inverse is stable for this intentionally lossy operation.
    mel_power = log_mel_features.float().clamp(-11.5, 12.0).transpose(0, 1).exp().cpu()
    filters = mel_filterbank(audio, device="cpu", dtype=torch.float32)
    inverse_filters = torch.linalg.pinv(filters, rtol=1e-4)
    spectrum = (inverse_filters @ mel_power).clamp_min(1e-8).sqrt()
    if phase_reference is not None:
        reference = phase_reference.detach().float().cpu()
        window = torch.hann_window(audio.win_length)
        reference_spectrum = torch.stft(
            reference, audio.n_fft, audio.hop_length, audio.win_length,
            window, return_complex=True, center=True,
        )
        frames = min(spectrum.shape[-1], reference_spectrum.shape[-1])
        magnitude = spectrum[:, :frames]
        phase = torch.angle(reference_spectrum[:, :frames])
        complex_spectrum = torch.polar(magnitude, phase)
        waveform = torch.istft(
            complex_spectrum, audio.n_fft, audio.hop_length, audio.win_length,
            window, center=True, length=reference.numel(),
        )
        peak = waveform.abs().max().clamp_min(1e-6)
        return waveform * (0.95 / peak).clamp_max(1.0)
    return torchaudio.transforms.GriffinLim(
        n_fft=audio.n_fft, hop_length=audio.hop_length, win_length=audio.win_length,
    )(spectrum)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunked listen-and-repeat inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-speaker", help="Required for a multi-speaker checkpoint")
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BudgerigarModel(BudgerigarConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    target_ids = None
    if model.config.speaker_names:
        if not args.target_speaker:
            raise ValueError(f"--target-speaker is required; choose from {model.config.speaker_names}")
        target_ids = torch.tensor([model.speaker_index(args.target_speaker)], device=device)
    audio = AudioConfig()
    features = log_mel(load_wave(args.input, audio.sample_rate), audio).to(device)
    chunk_frames = max(1, round(args.chunk_ms / 1000 * audio.sample_rate / audio.hop_length))
    state, output = None, []
    for start in range(0, len(features), chunk_frames):
        prediction, state = model.forward_chunk(
            features[None, start:start + chunk_frames], state, target_speaker=target_ids,
        )
        output.append(prediction[0].cpu())
    waveform = approximate_waveform(torch.cat(output), audio, phase_reference=load_wave(args.input, audio.sample_rate))
    import torchaudio
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), waveform[None], audio.sample_rate)
    print(f"saved diagnostic audio to {args.output}")


if __name__ == "__main__":
    main()
