import pytest

torch = pytest.importorskip("torch")
from budgerigar.dual_path_echo import DualPathEchoConfig, create_dual_path_echo


def test_dual_path_forward_and_ablations():
    config = DualPathEchoConfig(n_mels=4, hidden_dim=16, local_slots=8, abstract_slots=8,
                                update_stride=2, local_encoder_layers=2)
    model = create_dual_path_echo(config).eval(); inputs = torch.randn(2, 12, 6)
    with torch.no_grad():
        full = model(inputs); no_local = model(inputs, ablate_local=True); no_abstract = model(inputs, ablate_abstract=True)
    assert full[0].shape == (2, 12, 4) and full[1].shape == (2, 12)
    assert full[3]["local_write"].shape == (2, 6)
    assert not torch.equal(full[0], no_local[0])
    assert not torch.equal(full[0], no_abstract[0])
