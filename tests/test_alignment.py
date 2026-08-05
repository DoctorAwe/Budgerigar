import pytest
import torch

from budgerigar.alignment import dtw_align_target


def test_dtw_alignment_recovers_repeated_timing():
    pytest.importorskip("librosa")
    source = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = dtw_align_target(source, target, band_radius=1.0)
    assert aligned.shape == source.shape
    assert torch.isfinite(aligned).all()
    assert aligned[:2, 0].mean() > aligned[:2, 1].mean()
    assert aligned[2:, 1].mean() > aligned[2:, 0].mean()
