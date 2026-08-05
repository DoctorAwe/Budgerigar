import torch

from budgerigar.model import BudgerigarConfig, BudgerigarModel


def tiny_model():
    torch.manual_seed(3)
    return BudgerigarModel(BudgerigarConfig(n_mels=8, token_dim=16, layers=4))


def test_output_shape():
    model = tiny_model()
    inputs = torch.randn(2, 11, 8)
    assert model(inputs).shape == inputs.shape


def test_chunked_equals_whole():
    model = tiny_model().eval()
    inputs = torch.randn(2, 13, 8)
    whole = model(inputs)
    state = None
    pieces = []
    for start, end in ((0, 2), (2, 7), (7, 8), (8, 13)):
        output, state = model.forward_chunk(inputs[:, start:end], state)
        pieces.append(output)
    torch.testing.assert_close(whole, torch.cat(pieces, dim=1), rtol=0, atol=0)


def test_future_does_not_change_past():
    model = tiny_model().eval()
    prefix = torch.randn(1, 7, 8)
    one = model(torch.cat([prefix, torch.randn(1, 3, 8)], dim=1))[:, :7]
    two = model(torch.cat([prefix, torch.randn(1, 5, 8)], dim=1))[:, :7]
    torch.testing.assert_close(one, two, rtol=0, atol=0)


def test_empty_chunk_keeps_state():
    model = tiny_model()
    state = model.init_state(1, torch.device("cpu"), torch.float32)
    output, next_state = model.forward_chunk(torch.empty(1, 0, 8), state)
    assert output.shape == (1, 0, 8)
    assert next_state.steps == state.steps

