import torch

from budgerigar.waveform_model import StreamingWaveformProcessor, WaveformProcessorConfig


def tiny_waveform_model():
    torch.manual_seed(5)
    return StreamingWaveformProcessor(WaveformProcessorConfig(
        hidden_dim=16, recurrent_layers=1, attention_layers=2,
        attention_heads=4, memory_frames=8, dropout=0.0,
        speaker_names=("bdl", "slt"),
    )).eval()


def test_waveform_output_is_equal_length():
    model = tiny_waveform_model()
    waveform = torch.randn(2, 640)
    output = model(waveform, torch.tensor([0, 1]))
    assert output.shape == waveform.shape


def test_waveform_chunk_equivalence():
    model = tiny_waveform_model()
    waveform = torch.randn(1, 640)
    speaker = torch.tensor([1])
    whole = model(waveform, speaker)
    state = None
    pieces = []
    for start in range(0, 640, 160):
        output, state = model.forward_chunk(waveform[:, start:start + 160], speaker, state)
        pieces.append(output)
    torch.testing.assert_close(whole, torch.cat(pieces, 1), rtol=1e-4, atol=1e-5)


def test_future_input_cannot_change_past_waveform():
    model = tiny_waveform_model()
    prefix = torch.randn(1, 320)
    speaker = torch.tensor([0])
    first = model(torch.cat([prefix, torch.randn(1, 320)], 1), speaker)[:, :320]
    second = model(torch.cat([prefix, torch.randn(1, 160)], 1), speaker)[:, :320]
    torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-5)
