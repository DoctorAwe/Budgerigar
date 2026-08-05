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
    speaker_names: tuple[str, ...] = ()


@dataclass
class BudgerigarState:
    layers: tuple[Tensor, ...]
    steps: int = 0

    def detach(self) -> "BudgerigarState":
        return BudgerigarState(tuple(value.detach() for value in self.layers), self.steps)


class CausalTokenLayer(nn.Module):
    """One fused recurrent stage for an entire chunk."""

    def __init__(self, dim: int, depth: int, count: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.recurrent = nn.GRU(dim, dim, batch_first=True)
        # Deeper stages start with a stronger update-gate retention prior.
        retention = 0.25 + 0.70 * depth / max(count - 1, 1)
        with torch.no_grad():
            self.recurrent.bias_ih_l0[dim:2 * dim].fill_(torch.logit(torch.tensor(retention)))

    def forward_sequence(self, incoming: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        output, final = self.recurrent(self.input_norm(incoming), state.unsqueeze(0))
        return output, final[0]


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
        self.speaker_embedding = (
            nn.Embedding(len(config.speaker_names), config.token_dim)
            if config.speaker_names else None
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

    def speaker_index(self, name: str) -> int:
        if name not in self.config.speaker_names:
            raise ValueError(f"Unknown target speaker {name!r}; expected {self.config.speaker_names}")
        return self.config.speaker_names.index(name)

    def forward_chunk(
        self, mel: Tensor, state: BudgerigarState | None = None,
        target_speaker: Tensor | None = None,
    ) -> tuple[Tensor, BudgerigarState]:
        if mel.ndim != 3 or mel.shape[-1] != self.config.n_mels:
            raise ValueError(f"Expected [batch,time,{self.config.n_mels}], got {tuple(mel.shape)}")
        if state is None:
            state = self.init_state(mel.shape[0], mel.device, mel.dtype)
        if len(state.layers) != len(self.pipeline):
            raise ValueError("State layer count differs from model configuration")
        if mel.shape[1] == 0:
            return mel.new_empty(mel.shape), state
        encoded = self.encoder(mel)
        if self.speaker_embedding is not None:
            if target_speaker is None:
                raise ValueError("target_speaker IDs are required by this multi-speaker model")
            if target_speaker.shape != (mel.shape[0],):
                raise ValueError(f"target_speaker must be [batch], got {tuple(target_speaker.shape)}")
            encoded = encoded + self.speaker_embedding(target_speaker).unsqueeze(1)
        previous = state.layers
        source = encoded
        updated = []
        for index, layer in enumerate(self.pipeline):
            # Stage zero receives the current acoustic frame. Every deeper
            # stage receives the preceding stage delayed by exactly one frame.
            if index:
                source = torch.cat([previous[index - 1].unsqueeze(1), source[:, :-1]], dim=1)
            source, final = layer.forward_sequence(source, previous[index])
            updated.append(final)
        output = self.readout(torch.cat([encoded, source], dim=-1))
        return output, BudgerigarState(tuple(updated), state.steps + mel.shape[1])

    def forward(self, mel: Tensor, target_speaker: Tensor | None = None) -> Tensor:
        return self.forward_chunk(mel, target_speaker=target_speaker)[0]
