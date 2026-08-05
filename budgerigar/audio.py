from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import wave

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 40.0
    f_max: float = 7_600.0


def load_wave(path: str | Path, sample_rate: int = 16_000) -> Tensor:
    """Load mono audio. torchaudio is used when available; PCM WAV has a fallback."""
    try:
        import torchaudio
        waveform, source_rate = torchaudio.load(str(path))
        waveform = waveform.mean(0)
        if source_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
        return waveform
    except (ImportError, OSError):
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise ValueError("Fallback loader supports only 16-bit PCM WAV")
            source_rate = handle.getframerate()
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
        waveform = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
        waveform = waveform.reshape(-1, channels).mean(1)
        if source_rate != sample_rate:
            size = round(len(waveform) * sample_rate / source_rate)
            waveform = F.interpolate(waveform[None, None], size=size, mode="linear", align_corners=False)[0, 0]
        return waveform


def _hz_to_mel(value: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def mel_filterbank(config: AudioConfig, device=None, dtype=torch.float32) -> Tensor:
    low = _hz_to_mel(torch.tensor(config.f_min))
    high = _hz_to_mel(torch.tensor(config.f_max))
    points = _mel_to_hz(torch.linspace(low, high, config.n_mels + 2))
    bins = torch.floor((config.n_fft + 1) * points / config.sample_rate).long()
    filters = torch.zeros(config.n_mels, config.n_fft // 2 + 1)
    for index in range(config.n_mels):
        left, center, right = bins[index:index + 3].tolist()
        center, right = max(center, left + 1), max(right, center + 1)
        for frequency in range(left, min(center, filters.shape[1])):
            filters[index, frequency] = (frequency - left) / (center - left)
        for frequency in range(center, min(right, filters.shape[1])):
            filters[index, frequency] = (right - frequency) / (right - center)
    return filters.to(device=device, dtype=dtype)


def log_mel(waveform: Tensor, config: AudioConfig = AudioConfig()) -> Tensor:
    """Return time-major log-Mel features [..., time, mel]."""
    window = torch.hann_window(config.win_length, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform, config.n_fft, config.hop_length, config.win_length, window,
        return_complex=True, center=True,
    ).abs().square()
    filters = mel_filterbank(config, waveform.device, waveform.dtype)
    mel = torch.matmul(filters, spectrum)
    return torch.log(mel.clamp_min(1e-5)).transpose(-1, -2)


def align_target(target: Tensor, frames: int) -> Tensor:
    """Length-normalize a parallel target; replace with phoneme alignment later."""
    if target.shape[-2] == frames:
        return target
    leading = target.shape[:-2]
    mel = target.shape[-1]
    flat = target.reshape(-1, target.shape[-2], mel).transpose(1, 2)
    result = F.interpolate(flat, size=frames, mode="linear", align_corners=False)
    return result.transpose(1, 2).reshape(*leading, frames, mel)

