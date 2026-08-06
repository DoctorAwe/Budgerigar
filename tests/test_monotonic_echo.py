import pytest
torch = pytest.importorskip("torch")
from budgerigar.monotonic_echo import MonotonicEchoConfig, create_monotonic_echo


def test_phases_are_monotonic_and_layers_are_holistic_states():
    model=create_monotonic_echo(MonotonicEchoConfig(n_mels=4,hidden_dim=16,event_slots=12,understanding_layers=3,update_stride=2,local_encoder_layers=1,output_feedback=True,recursive_layer_attention=True,attention_heads=4)).eval()
    inputs=torch.randn(2,12,6)
    with torch.no_grad(): mel,voice,_,diagnostics=model(inputs)
    assert mel.shape==(2,12,4) and voice.shape==(2,12)
    assert diagnostics["understanding"].shape==(2,3,16)
    assert len(model.layer_attention)==3
    assert torch.all(diagnostics["write_phase"][:,1:]>=diagnostics["write_phase"][:,:-1])
    assert torch.all(diagnostics["read_phase"][:,1:]>=diagnostics["read_phase"][:,:-1])
    with torch.no_grad(): taught=model(inputs,teacher_mel=torch.randn(2,12,4),teacher_forcing_ratio=.5)[0]
    assert taught.shape==mel.shape
