from __future__ import annotations

from .hierarchical_echo import HierarchicalEchoConfig, create_hierarchical_echo
from .neural_echo import NeuralEchoConfig, create_neural_echo


def restore_echo_model(checkpoint, device):
    architecture = checkpoint.get("architecture", "neural_echo")
    if architecture == "hierarchical_token_echo":
        model = create_hierarchical_echo(HierarchicalEchoConfig(**checkpoint["model_config"]))
    elif architecture in {"neural_echo", "continuous_neural_listen_then_repeat"}:
        model = create_neural_echo(NeuralEchoConfig(**checkpoint["model_config"]))
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device), architecture


def forward_echo(model, architecture, inputs):
    result = model(inputs)
    if architecture == "hierarchical_token_echo":
        mel, voice_logits, memory, diagnostics = result
        return mel, voice_logits, memory, diagnostics
    mel, voice_logits, memory = result
    return mel, voice_logits, memory, {}

