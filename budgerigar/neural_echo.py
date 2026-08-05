from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NeuralEchoConfig:
    n_mels: int = 100
    hidden_dim: int = 256
    layers: int = 4
    kernel_size: int = 5
    dropout: float = 0.1


def require_torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except (ImportError, OSError) as error:
        raise RuntimeError("NeuralEcho requires the 'train' dependencies in Colab") from error
    return torch, nn, functional


def create_neural_echo(config: NeuralEchoConfig = NeuralEchoConfig()):
    torch, nn, functional = require_torch()

    class NeuralEcho(nn.Module):
        """Continuous neural listen-then-repeat model without discrete states."""

        def __init__(self):
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(config.n_mels + 2)
            self.input_projection = nn.Linear(config.n_mels + 2, config.hidden_dim)
            self.causal_conv = nn.Conv1d(config.hidden_dim, config.hidden_dim, config.kernel_size)
            self.memory = nn.GRU(
                config.hidden_dim, config.hidden_dim, config.layers, batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
            self.personality = nn.Parameter(torch.zeros(config.hidden_dim))
            self.acoustic_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim * 2, config.n_mels),
            )
            # Continuous neural confidence, never interpreted as a finite-state action in the model.
            self.voice_strength_head = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1))

        def forward(self, input_stream, memory=None):
            value = self.input_projection(self.input_norm(input_stream))
            convolved = functional.pad(value.transpose(1, 2), (config.kernel_size - 1, 0))
            value = value + functional.silu(self.causal_conv(convolved).transpose(1, 2))
            value, memory = self.memory(value, memory)
            value = value + self.personality.view(1, 1, -1)
            return self.acoustic_head(value), self.voice_strength_head(value).squeeze(-1), memory

        def export_config(self):
            return asdict(config)

    return NeuralEcho()

