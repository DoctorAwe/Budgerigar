from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .audio import AudioConfig, load_wave, log_mel
from .model import BudgerigarConfig, BudgerigarModel
from .train import choose_device


def approximate_waveform(log_mel_features: torch.Tensor, audio: AudioConfig) -> torch.Tensor:
    """Diagnostic inversion only; replace with a trained neural vocoder for quality."""
    try:
        import torchaudio
    except ImportError as error:
        raise RuntimeError("torchaudio is required to synthesize the demo WAV") from error
    mel = log_mel_features.transpose(0, 1).exp().cpu()
    inverse = torchaudio.transforms.InverseMelScale(
        n_stft=audio.n_fft // 2 + 1, n_mels=audio.n_mels,
        sample_rate=audio.sample_rate, f_min=audio.f_min, f_max=audio.f_max,
    )
    spectrum = inverse(mel).clamp_min(1e-8).sqrt()
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
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BudgerigarModel(BudgerigarConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    audio = AudioConfig()
    features = log_mel(load_wave(args.input, audio.sample_rate), audio).to(device)
    chunk_frames = max(1, round(args.chunk_ms / 1000 * audio.sample_rate / audio.hop_length))
    state, output = None, []
    for start in range(0, len(features), chunk_frames):
        prediction, state = model.forward_chunk(features[None, start:start + chunk_frames], state)
        output.append(prediction[0].cpu())
    waveform = approximate_waveform(torch.cat(output), audio)
    import torchaudio
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), waveform[None], audio.sample_rate)
    print(f"saved diagnostic audio to {args.output}")


if __name__ == "__main__":
    main()

