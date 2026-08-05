from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _content_normalize(features: Tensor) -> np.ndarray:
    values = features.detach().float().cpu().numpy()
    values = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-5)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-6)


def dtw_align_target(source: Tensor, target: Tensor, band_radius: float = 0.25) -> Tensor:
    """Warp target frames onto the source time axis using cosine-cost DTW."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be [time, mel] with equal Mel dimensions")
    if not 0.0 < band_radius <= 1.0:
        raise ValueError("band_radius must be in (0, 1]")
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError("DTW preprocessing requires: pip install -r requirements-data.txt") from error

    source_content = _content_normalize(source)
    target_content = _content_normalize(target)
    cost = np.clip(1.0 - source_content @ target_content.T, 0.0, 2.0)
    _, path = librosa.sequence.dtw(
        C=cost, backtrack=True, global_constraints=True, band_rad=band_radius,
    )
    path = path[::-1]
    target_np = target.detach().float().cpu().numpy()
    aligned = np.zeros((source.shape[0], target.shape[1]), dtype=np.float32)
    counts = np.zeros(source.shape[0], dtype=np.int32)
    for source_index, target_index in path:
        aligned[source_index] += target_np[target_index]
        counts[source_index] += 1
    known = np.flatnonzero(counts)
    if known.size == 0:
        raise RuntimeError("DTW returned an empty alignment path")
    aligned[known] /= counts[known, None]
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        for mel_index in range(target.shape[1]):
            aligned[missing, mel_index] = np.interp(missing, known, aligned[known, mel_index])
    return torch.from_numpy(aligned).to(dtype=target.dtype)

