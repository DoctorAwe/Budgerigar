import pytest


torch = pytest.importorskip("torch")

from budgerigar.content_memory import ContentMemoryConfig, create_content_memory


def test_dual_ctc_shapes_and_causal_prefix():
    config = ContentMemoryConfig(n_mels=4, hidden_dim=16, token_slots=12, update_stride=2,
                                 vocabulary_size=9, sequence_contrastive=True,
                                 local_encoder_layers=2, acoustic_ctc=True)
    model = create_content_memory(config).eval()
    inputs = torch.randn(2, 18, 6)
    lengths = torch.tensor([18, 15])
    texts = torch.tensor([[2, 3, 4], [4, 5, 0]])
    text_lengths = torch.tensor([3, 2])
    with torch.no_grad():
        logits, memory_lengths, audio, text, diagnostics = model(inputs, lengths, texts, text_lengths)
    assert logits.shape == (2, 12, 9)
    assert memory_lengths.tolist() == [9, 8]
    assert diagnostics["acoustic_logits"].shape == (2, 9, 9)
    assert diagnostics["acoustic_lengths"].tolist() == [9, 8]
    assert audio.shape == text.shape == (2, config.contrastive_dim)


def test_parameter_expansion_is_bounded():
    small = create_content_memory(ContentMemoryConfig(hidden_dim=128, local_encoder_layers=0))
    expanded = create_content_memory(ContentMemoryConfig(hidden_dim=192, local_encoder_layers=4, acoustic_ctc=True))
    small_count = sum(parameter.numel() for parameter in small.parameters())
    expanded_count = sum(parameter.numel() for parameter in expanded.parameters())
    assert expanded_count > small_count
    assert expanded_count < 10_000_000
