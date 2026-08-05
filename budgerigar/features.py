from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 24_000
    n_fft: int = 1_024
    win_length: int = 600
    hop_length: int = 240
    n_mels: int = 100
    f_min: float = 40.0
    f_max: float = 11_000.0
    target_rms_dbfs: float = -24.0
    max_gain_db: float = 18.0
    vad_relative_db: float = -38.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("sample rate and frame sizes must be positive")
        if self.n_fft < self.win_length:
            raise ValueError("n_fft must be at least win_length")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("Mel frequency range must fit the Nyquist frequency")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _torch_modules():
    try:
        import torch
        import torchaudio
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "feature extraction requires matching torch and torchaudio; install the 'train' extra in Colab"
        ) from error
    return torch, torchaudio


def load_and_normalize(path: str | Path, config: FeatureConfig):
    torch, torchaudio = _torch_modules()
    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform.float().mean(dim=0)
    if source_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, config.sample_rate)
    peak_before = float(waveform.abs().max()) if waveform.numel() else 0.0
    rms_before = float(waveform.square().mean().sqrt()) if waveform.numel() else 0.0
    gain_db = 0.0
    if rms_before > 1e-7:
        current_db = 20.0 * torch.log10(torch.tensor(rms_before)).item()
        gain_db = min(config.max_gain_db, config.target_rms_dbfs - current_db)
        waveform = waveform * (10.0 ** (gain_db / 20.0))
    waveform = waveform.clamp(-1.0, 1.0)
    stats = {
        "source_sample_rate": int(source_rate), "samples": int(waveform.numel()),
        "peak_before": peak_before, "rms_before": rms_before, "gain_db": gain_db,
        "clipped_fraction_after": float((waveform.abs() >= 1.0).float().mean()) if waveform.numel() else 0.0,
    }
    return waveform, stats


def extract_features(path: str | Path, config: FeatureConfig = FeatureConfig()) -> dict:
    torch, torchaudio = _torch_modules()
    waveform, stats = load_and_normalize(path, config)
    if waveform.numel() < config.win_length:
        waveform = torch.nn.functional.pad(waveform, (0, config.win_length - waveform.numel()))
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate, n_fft=config.n_fft, win_length=config.win_length,
        hop_length=config.hop_length, f_min=config.f_min, f_max=config.f_max,
        n_mels=config.n_mels, power=2.0, center=False, norm="slaney", mel_scale="slaney",
    )
    mel_power = transform(waveform)
    log_mel = torch.log(mel_power.clamp_min(1e-5)).transpose(0, 1).contiguous()

    frames = waveform.unfold(0, config.win_length, config.hop_length)
    energy = frames.square().mean(dim=-1).sqrt().clamp_min(1e-7)
    energy_db = 20.0 * torch.log10(energy)
    threshold = max(-80.0, float(energy_db.max()) + config.vad_relative_db)
    vad = energy_db >= threshold
    frame_count = min(log_mel.shape[0], energy_db.shape[0])
    return {
        "log_mel": log_mel[:frame_count].cpu(),
        "energy_db": energy_db[:frame_count].cpu(),
        "vad": vad[:frame_count].cpu(),
        "stats": {**stats, "vad_threshold_db": threshold},
        "config": asdict(config), "config_fingerprint": config.fingerprint,
    }


def cache_manifest_features(
    manifest: str | Path,
    output_dir: str | Path,
    output_manifest: str | Path,
    config: FeatureConfig = FeatureConfig(),
) -> tuple[Path, dict]:
    torch, _ = _torch_modules()
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    total_frames = voiced_frames = cache_hits = 0
    clipped_files = 0
    gain_values: list[float] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        feature_path = output_dir / f"{row['id']}.{config.fingerprint}.pt"
        payload = None
        if feature_path.is_file():
            candidate = torch.load(feature_path, map_location="cpu", weights_only=True)
            if candidate.get("config_fingerprint") == config.fingerprint:
                payload = candidate
                cache_hits += 1
        if payload is None:
            try:
                payload = extract_features(row["audio"], config)
            except Exception as error:
                raise RuntimeError(f"feature extraction failed at {manifest}:{line_number} ({row.get('id')})") from error
            temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(feature_path)
        frames = int(payload["log_mel"].shape[0])
        voiced = int(payload["vad"].sum())
        total_frames += frames
        voiced_frames += voiced
        gain_values.append(float(payload["stats"]["gain_db"]))
        clipped_files += payload["stats"]["clipped_fraction_after"] > 0.001
        output_rows.append({
            **row, "feature_path": str(feature_path.resolve()), "feature_frames": frames,
            "feature_fingerprint": config.fingerprint,
        })
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n", encoding="utf-8",
    )
    report = {
        "records": len(output_rows), "total_frames": total_frames,
        "voiced_fraction": voiced_frames / max(total_frames, 1), "cache_hits": cache_hits,
        "clipped_files": clipped_files,
        "gain_db_min": min(gain_values, default=0.0), "gain_db_max": max(gain_values, default=0.0),
        "feature_config": asdict(config), "feature_fingerprint": config.fingerprint,
    }
    return output_manifest, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Budgerigar log-Mel/VAD features in Colab")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output, report = cache_manifest_features(args.manifest, args.output_dir, args.output_manifest)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

