from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .neural_echo import require_torch


@dataclass(frozen=True)
class ContentMemoryConfig:
    n_mels: int = 100
    hidden_dim: int = 128
    token_slots: int = 128
    update_stride: int = 4
    vocabulary_size: int = 32
    contrastive_dim: int = 128


def create_content_memory(config: ContentMemoryConfig):
    torch, nn, functional = require_torch()

    class ContentMemory(nn.Module):
        """Multi-rate AutoMachine token bank with explicit content probes."""

        def __init__(self):
            super().__init__()
            dim = config.hidden_dim
            self.config = config
            self.input_norm = nn.LayerNorm(config.n_mels + 2)
            self.input_projection = nn.Linear(config.n_mels + 2, dim)
            self.level_embedding = nn.Parameter(torch.randn(config.token_slots, dim) * 0.02)
            self.optimizer = nn.GRUCell(dim, dim)
            self.write_gate = nn.Sequential(nn.Linear(dim * 3 + 1, dim), nn.SiLU(), nn.Linear(dim, 1))
            self.ctc_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.vocabulary_size))
            self.audio_projection = nn.Linear(dim, config.contrastive_dim)
            self.text_embedding = nn.Embedding(config.vocabulary_size, dim, padding_idx=0)
            self.text_projection = nn.Linear(dim, config.contrastive_dim)
            self.log_temperature = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        def _update(self, encoded, vad, tokens, active):
            batch, slots, dim = tokens.shape
            incoming = torch.cat([encoded.unsqueeze(1), tokens[:, :-1]], dim=1)
            optimized = self.optimizer(
                (incoming + self.level_embedding.unsqueeze(0)).reshape(batch * slots, dim),
                tokens.reshape(batch * slots, dim),
            ).reshape(batch, slots, dim)
            current = encoded.unsqueeze(1).expand(-1, slots, -1)
            gate = self.write_gate(torch.cat([
                tokens, incoming, current, vad.view(batch, 1, 1).expand(-1, slots, 1),
            ], dim=-1)).sigmoid()
            proposed = tokens + gate * (optimized - tokens)
            return torch.where(active.view(batch, 1, 1), proposed, tokens), gate.mean((1, 2))

        def forward(self, inputs, frame_lengths, text_tokens=None, text_lengths=None):
            batch = len(inputs); stride = config.update_stride
            updates = (frame_lengths + stride - 1) // stride
            memory_lengths = updates.clamp(max=config.token_slots)
            tokens = torch.zeros(batch, config.token_slots, config.hidden_dim, device=inputs.device, dtype=inputs.dtype)
            write_history = []
            for update in range(int(updates.max())):
                start = update * stride; end = min(start + stride, inputs.shape[1])
                encoded = self.input_projection(self.input_norm(inputs[:, start:end])).mean(1)
                vad = inputs[:, start:end, -1].mean(1)
                active = update < updates
                tokens, write = self._update(encoded, vad, tokens, active)
                write_history.append(write)
            ordered = tokens.new_zeros(batch, config.token_slots, config.hidden_dim)
            for index, length in enumerate(memory_lengths.tolist()):
                ordered[index, :length] = tokens[index, :length].flip(0)
            logits = self.ctc_head(ordered)
            positions = torch.arange(config.token_slots, device=inputs.device).unsqueeze(0)
            valid = positions < memory_lengths.unsqueeze(1)
            audio_pool = (ordered * valid.unsqueeze(-1)).sum(1) / memory_lengths.clamp_min(1).unsqueeze(1)
            audio_embedding = functional.normalize(self.audio_projection(audio_pool), dim=-1)
            text_embedding = None
            if text_tokens is not None:
                text_valid = torch.arange(text_tokens.shape[1], device=inputs.device).unsqueeze(0) < text_lengths.unsqueeze(1)
                text_pool = (self.text_embedding(text_tokens) * text_valid.unsqueeze(-1)).sum(1) / text_lengths.clamp_min(1).unsqueeze(1)
                text_embedding = functional.normalize(self.text_projection(text_pool), dim=-1)
            diagnostics = {"write_strength": torch.stack(write_history, 1)}
            return logits, memory_lengths, audio_embedding, text_embedding, diagnostics

        def export_config(self): return asdict(config)

    return ContentMemory()

