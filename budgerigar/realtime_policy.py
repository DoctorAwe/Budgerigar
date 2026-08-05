from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .streaming_dataset import WAIT


@dataclass(frozen=True)
class RealtimePolicyConfig:
    n_mels: int = 80
    hidden_dim: int = 256
    action_dim: int = 32
    layers: int = 3
    speaker_names: tuple[str, ...] = ()


@dataclass
class RealtimePolicyState:
    hidden: Tensor
    previous_action: Tensor


class RealtimePolicyModel(nn.Module):
    """Causal endpoint and output-clock policy for a continuous audio stream."""

    def __init__(self, config: RealtimePolicyConfig):
        super().__init__()
        if not config.speaker_names:
            raise ValueError("Realtime policy requires target speakers")
        self.config = config
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(config.n_mels), nn.Linear(config.n_mels, config.hidden_dim), nn.SiLU(),
        )
        self.action_embedding = nn.Embedding(4, config.action_dim)
        self.speaker_embedding = nn.Embedding(len(config.speaker_names), config.hidden_dim)
        self.recurrent = nn.GRU(
            config.hidden_dim + config.action_dim, config.hidden_dim,
            num_layers=config.layers, batch_first=True,
        )
        self.endpoint_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 1),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 4),
        )

    def speaker_index(self, name: str) -> int:
        return self.config.speaker_names.index(name)

    def init_state(self, batch: int, device: torch.device) -> RealtimePolicyState:
        hidden = torch.zeros(self.config.layers, batch, self.config.hidden_dim, device=device)
        action = torch.full((batch,), WAIT, dtype=torch.long, device=device)
        return RealtimePolicyState(hidden, action)

    def forward_sequence(
        self, audio: Tensor, target_speaker: Tensor, actions: Tensor | None = None,
        state: RealtimePolicyState | None = None,
    ) -> tuple[Tensor, Tensor, RealtimePolicyState]:
        batch, frames, _ = audio.shape
        if state is None:
            state = self.init_state(batch, audio.device)
        if actions is None:
            # Autoregressive policy inference is intentionally frame-wise.
            action_logits = []
            endpoint_logits = []
            running = state
            for frame in range(frames):
                endpoint, policy, running = self.forward_sequence(
                    audio[:, frame:frame + 1], target_speaker,
                    actions=running.previous_action[:, None], state=running,
                )
                running.previous_action = policy[:, 0].argmax(-1)
                endpoint_logits.append(endpoint)
                action_logits.append(policy)
            return torch.cat(endpoint_logits, 1), torch.cat(action_logits, 1), running
        previous = torch.cat([state.previous_action[:, None], actions[:, :-1]], dim=1)
        encoded = self.audio_projection(audio) + self.speaker_embedding(target_speaker).unsqueeze(1)
        inputs = torch.cat([encoded, self.action_embedding(previous)], dim=-1)
        output, hidden = self.recurrent(inputs, state.hidden)
        endpoint = self.endpoint_head(output).squeeze(-1)
        policy = self.action_head(output)
        next_state = RealtimePolicyState(hidden, actions[:, -1])
        return endpoint, policy, next_state

