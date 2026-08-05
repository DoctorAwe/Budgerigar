from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BudgerigarConfig:
    n_mels: int = 80
    token_dim: int = 192
    layers: int = 16
    dropout: float = 0.0


@dataclass
class BudgerigarState:
    layers: tuple[Tensor, ...]
    steps: int = 0

    def detach(self) -> "BudgerigarState":
        return BudgerigarState(tuple(value.detach() for value in self.layers), self.steps)


class CausalTokenLayer(nn.Module):
    """One stateful stage. Incoming values are always from the previous time step."""

    def __init__(self, dim: int, depth: int, count: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.candidate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.gate = nn.Linear(dim * 2, dim)
        retention = 0.25 + 0.70 * depth / max(count - 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, torch.logit(torch.tensor(retention)).item())

    def forward(self, state: Tensor, incoming: Tensor) -> Tensor:
        joined = self.norm(torch.cat([state, incoming], dim=-1))
        proposal = torch.tanh(self.candidate(joined))
        retain = torch.sigmoid(self.gate(joined))
        return retain * state + (1.0 - retain) * proposal


class BudgerigarModel(nn.Module):
    """Causal fixed-target voice conversion baseline operating on log-Mel frames."""

    def __init__(self, config: BudgerigarConfig = BudgerigarConfig()):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.n_mels),
            nn.Linear(config.n_mels, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.token_dim),
        )
        self.pipeline = nn.ModuleList([
            CausalTokenLayer(config.token_dim, index, config.layers, config.dropout)
            for index in range(config.layers)
        ])
        self.readout = nn.Sequential(
            nn.LayerNorm(config.token_dim * 2),
            nn.Linear(config.token_dim * 2, config.token_dim * 2),
            nn.GELU(),
            nn.Linear(config.token_dim * 2, config.n_mels),
        )

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> BudgerigarState:
        empty = tuple(torch.zeros(batch, self.config.token_dim, device=device, dtype=dtype) for _ in self.pipeline)
        return BudgerigarState(empty)

    def forward_chunk(self, mel: Tensor, state: BudgerigarState | None = None) -> tuple[Tensor, BudgerigarState]:
        if mel.ndim != 3 or mel.shape[-1] != self.config.n_mels:
            raise ValueError(f"Expected [batch,time,{self.config.n_mels}], got {tuple(mel.shape)}")
        if state is None:
            state = self.init_state(mel.shape[0], mel.device, mel.dtype)
        if len(state.layers) != len(self.pipeline):
            raise ValueError("State layer count differs from model configuration")
        encoded = self.encoder(mel)
        outputs = []
        running = state
        for time in range(mel.shape[1]):
            perception = encoded[:, time]
            previous = running.layers
            updated = []
            for index, layer in enumerate(self.pipeline):
                incoming = perception if index == 0 else previous[index - 1]
                updated.append(layer(previous[index], incoming))
            outputs.append(self.readout(torch.cat([perception, updated[-1]], dim=-1)))
            running = BudgerigarState(tuple(updated), running.steps + 1)
        if not outputs:
            return mel.new_empty(mel.shape), running
        return torch.stack(outputs, dim=1), running

    def forward(self, mel: Tensor) -> Tensor:
        return self.forward_chunk(mel)[0]

