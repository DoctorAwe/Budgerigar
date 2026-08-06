from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .neural_echo import require_torch


@dataclass(frozen=True)
class DualPathEchoConfig:
    n_mels: int = 100
    hidden_dim: int = 192
    local_slots: int = 160
    abstract_slots: int = 160
    update_stride: int = 4
    local_encoder_layers: int = 4
    local_kernel_size: int = 5
    dropout: float = 0.1


def create_dual_path_echo(config=DualPathEchoConfig()):
    torch, nn, functional = require_torch()

    class DualPathEcho(nn.Module):
        """Continuous local replay memory plus optimizer-refined abstract memory."""

        def __init__(self):
            super().__init__(); dim = config.hidden_dim; self.config = config
            self.input_norm = nn.LayerNorm(config.n_mels + 2)
            self.input_projection = nn.Linear(config.n_mels + 2, dim)
            self.local_encoder = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim),
            ) for _ in range(config.local_encoder_layers)])
            self.local_convolution = nn.ModuleList([
                nn.Conv1d(dim, dim, config.local_kernel_size, groups=dim)
                for _ in range(config.local_encoder_layers)
            ])
            self.local_write = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, 1))
            self.level_embedding = nn.Parameter(torch.randn(config.abstract_slots, dim) * 0.02)
            self.optimizer = nn.GRUCell(dim, dim)
            self.write_gate = nn.Sequential(nn.Linear(dim * 3 + 1, dim), nn.SiLU(), nn.Linear(dim, 1))
            self.query = nn.GRUCell(dim * 3, dim)
            self.local_key = nn.Linear(dim, dim); self.local_value = nn.Linear(dim, dim)
            self.abstract_key = nn.Linear(dim, dim); self.abstract_value = nn.Linear(dim, dim)
            self.query_projection = nn.Linear(dim, dim)
            self.personality = nn.Parameter(torch.zeros(dim))
            self.acoustic_head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2),
                                               nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(dim * 2, config.n_mels))
            self.voice_head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, 1))

        def _attend(self, query, memory, key, value):
            score = torch.einsum("bd,bsd->bs", self.query_projection(query), key(memory)) / math.sqrt(config.hidden_dim)
            return torch.einsum("bs,bsd->bd", score.softmax(-1), value(memory))

        def forward(self, inputs, ablate_local=False, ablate_abstract=False):
            encoded = self.input_projection(self.input_norm(inputs))
            for convolution, feed_forward in zip(self.local_convolution, self.local_encoder):
                causal = functional.pad(encoded.transpose(1, 2), (config.local_kernel_size - 1, 0))
                encoded = encoded + convolution(causal).transpose(1, 2)
                encoded = encoded + feed_forward(encoded)
            batch = len(inputs); dim = config.hidden_dim
            local = inputs.new_zeros(batch, config.local_slots, dim)
            abstract = inputs.new_zeros(batch, config.abstract_slots, dim)
            query = inputs.new_zeros(batch, dim)
            acoustic = []; voices = []; local_writes = []; abstract_writes = []
            for tick in range(inputs.shape[1]):
                if tick % config.update_stride == config.update_stride - 1:
                    chunk = encoded[:, tick + 1 - config.update_stride:tick + 1].mean(1)
                    vad = inputs[:, tick + 1 - config.update_stride:tick + 1, -1].mean(1, keepdim=True)
                    local_gate = self.local_write(torch.cat([chunk, vad], -1)).sigmoid().view(batch, 1, 1)
                    shifted = torch.cat([chunk.unsqueeze(1), local[:, :-1]], 1)
                    local = local + local_gate * (shifted - local)
                    incoming = torch.cat([chunk.unsqueeze(1), abstract[:, :-1]], 1)
                    optimized = self.optimizer(
                        (incoming + self.level_embedding.unsqueeze(0)).reshape(-1, dim), abstract.reshape(-1, dim),
                    ).reshape_as(abstract)
                    current = chunk.unsqueeze(1).expand_as(abstract)
                    gate = self.write_gate(torch.cat([abstract, incoming, current, vad.unsqueeze(1).expand(-1, config.abstract_slots, -1)], -1)).sigmoid()
                    abstract = abstract + gate * (optimized - abstract)
                    local_writes.append(local_gate.flatten()); abstract_writes.append(gate.mean((1, 2)))
                local_context = self._attend(query, local, self.local_key, self.local_value)
                abstract_context = self._attend(query, abstract, self.abstract_key, self.abstract_value)
                if ablate_local: local_context = local_context * 0
                if ablate_abstract: abstract_context = abstract_context * 0
                query = self.query(torch.cat([encoded[:, tick], local_context, abstract_context], -1), query)
                decoded = torch.cat([query + self.personality, local_context, abstract_context], -1)
                acoustic.append(self.acoustic_head(decoded)); voices.append(self.voice_head(decoded).squeeze(-1))
            diagnostics = {
                "local_write": torch.stack(local_writes, 1),
                "abstract_write": torch.stack(abstract_writes, 1),
            }
            return torch.stack(acoustic, 1), torch.stack(voices, 1), (local, abstract, query), diagnostics

        def export_config(self): return asdict(config)

    return DualPathEcho()
