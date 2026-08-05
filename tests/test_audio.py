import torch

from budgerigar.audio import AudioConfig, align_target, log_mel, mel_filterbank


def test_log_mel_shape_and_finiteness():
    config = AudioConfig(sample_rate=16_000, n_fft=400, n_mels=24)
    result = log_mel(torch.randn(3_200), config)
    assert result.shape[1] == 24
    assert torch.isfinite(result).all()


def test_alignment_changes_only_time_axis():
    target = torch.randn(12, 8)
    aligned = align_target(target, 19)
    assert aligned.shape == (19, 8)


def test_filterbank_shape():
    config = AudioConfig(n_fft=400, n_mels=32)
    assert mel_filterbank(config).shape == (32, 201)

