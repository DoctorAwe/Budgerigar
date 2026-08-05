from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .neural_echo import require_torch


@dataclass(frozen=True)
class HierarchicalEchoConfig:
    n_mels: int = 100
    hidden_dim: int = 192
    token_slots: int = 16
    dropout: float = 0.1


def create_hierarchical_echo(config: HierarchicalEchoConfig = HierarchicalEchoConfig()):
    torch, nn, functional = require_torch()

    class HierarchicalEcho(nn.Module):
        """AutoMachine-style continuous token memory without discrete behavior states."""

        def __init__(self):
            super().__init__()
            dim = config.hidden_dim
            self.config = config
            self.input_norm = nn.LayerNorm(config.n_mels + 2)
            self.input_projection = nn.Linear(config.n_mels + 2, dim)
            self.level_embedding = nn.Parameter(torch.randn(config.token_slots, dim) * 0.02)
            # One optimizer is shared across slots; level embeddings allow slot-specific behavior.
            self.token_optimizer = nn.GRUCell(dim, dim)
            self.write_gate = nn.Sequential(
                nn.Linear(dim * 3 + 1, dim), nn.SiLU(), nn.Linear(dim, 1),
            )
            self.query_state = nn.GRUCell(dim * 2, dim)
            self.query_projection = nn.Linear(dim, dim)
            self.key_projection = nn.Linear(dim, dim)
            self.value_projection = nn.Linear(dim, dim)
            self.personality = nn.Parameter(torch.zeros(dim))
            self.acoustic_head = nn.Sequential(
                nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim * 2), nn.SiLU(),
                nn.Dropout(config.dropout), nn.Linear(dim * 2, config.n_mels),
            )
            self.voice_strength_head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 1))

        def initial_memory(self, batch, device, dtype):
            tokens = torch.zeros(batch, config.token_slots, config.hidden_dim, device=device, dtype=dtype)
            query = torch.zeros(batch, config.hidden_dim, device=device, dtype=dtype)
            return tokens, query

        def _tick(self, encoded, vad, tokens, query):
            batch, slots, dim = tokens.shape
            incoming = torch.cat([encoded.unsqueeze(1), tokens[:, :-1]], dim=1)
            optimized = self.token_optimizer(
                (incoming + self.level_embedding.unsqueeze(0)).reshape(batch * slots, dim),
                tokens.reshape(batch * slots, dim),
            ).reshape(batch, slots, dim)
            current = encoded.unsqueeze(1).expand(-1, slots, -1)
            gate_input = torch.cat([
                tokens, incoming, current, vad.view(batch, 1, 1).expand(-1, slots, 1),
            ], dim=-1)
            write = self.write_gate(gate_input).sigmoid()
            tokens = tokens + write * (optimized - tokens)

            keys = self.key_projection(tokens)
            scores = torch.einsum("bd,bsd->bs", self.query_projection(query), keys) / math.sqrt(dim)
            attention = scores.softmax(dim=-1)
            context = torch.einsum("bs,bsd->bd", attention, self.value_projection(tokens))
            query = self.query_state(torch.cat([encoded, context], dim=-1), query)
            decoded = torch.cat([query + self.personality, context], dim=-1)
            return tokens, query, decoded, write.mean(dim=1).squeeze(-1), attention

        def forward(self, input_stream, memory=None):
            encoded_stream = self.input_projection(self.input_norm(input_stream))
            if memory is None:
                tokens, query = self.initial_memory(
                    input_stream.shape[0], input_stream.device, input_stream.dtype,
                )
            else:
                tokens, query = memory
            acoustic = []; strength = []; writes = []
            for tick in range(input_stream.shape[1]):
                tokens, query, decoded, write, _ = self._tick(
                    encoded_stream[:, tick], input_stream[:, tick, -1], tokens, query,
                )
                acoustic.append(self.acoustic_head(decoded))
                strength.append(self.voice_strength_head(decoded).squeeze(-1))
                writes.append(write)
            return (
                torch.stack(acoustic, dim=1), torch.stack(strength, dim=1),
                (tokens, query), {"write_strength": torch.stack(writes, dim=1)},
            )

        def export_config(self):
            return asdict(config)

    return HierarchicalEcho()

