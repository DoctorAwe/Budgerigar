from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WaveformProcessorConfig:
    sample_rate: int = 16_000
    hop_samples: int = 160
    hidden_dim: int = 256
    recurrent_layers: int = 3
    attention_layers: int = 4
    attention_heads: int = 4
    memory_frames: int = 1200  # 12 seconds at a 10 ms hop
    dropout: float = 0.1
    speaker_names: tuple[str, ...] = ()


@dataclass
class WaveformProcessorState:
    recurrent: Tensor
    attention_memory: tuple[Tensor | None, ...]

    def detach(self) -> "WaveformProcessorState":
        return WaveformProcessorState(
            self.recurrent.detach(),
            tuple(None if value is None else value.detach() for value in self.attention_memory),
        )


class CausalMemoryBlock(nn.Module):
    def __init__(self, config: WaveformProcessorConfig):
        super().__init__()
        dim = config.hidden_dim
        self.memory_frames = config.memory_frames
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, config.attention_heads, dropout=config.dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(config.dropout),
        )

    def forward(self, current: Tensor, memory: Tensor | None) -> tuple[Tensor, Tensor]:
        cache = current if memory is None else torch.cat([memory, current], dim=1)
        memory_length = cache.shape[1] - current.shape[1]
        query_positions = torch.arange(current.shape[1], device=current.device) + memory_length
        key_positions = torch.arange(cache.shape[1], device=current.device)
        future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        too_old = key_positions.unsqueeze(0) < (query_positions.unsqueeze(1) - self.memory_frames + 1)
        attention_mask = future | too_old
        normalized_cache = self.attention_norm(cache)
        attended, _ = self.attention(
            self.attention_norm(current), normalized_cache, normalized_cache,
            attn_mask=attention_mask, need_weights=False,
        )
        output = current + attended
        output = output + self.ffn(self.ffn_norm(output))
        next_memory = cache[:, -self.memory_frames:]
        return output, next_memory


class CausalConv1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.pad(value, (self.left_padding, 0)))


class WaveformDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(dim, 128, kernel_size=5, stride=5), nn.LeakyReLU(0.1),
            CausalConv1d(128, 128, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4), nn.LeakyReLU(0.1),
            CausalConv1d(64, 64, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=8), nn.LeakyReLU(0.1),
            CausalConv1d(32, 32, 1), nn.LeakyReLU(0.1),
            CausalConv1d(32, 1, 1), nn.Tanh(),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent.transpose(1, 2))[:, 0]


class StreamingWaveformProcessor(nn.Module):
    """One causal processor: fixed-rate waveform in, equally long waveform out."""

    def __init__(self, config: WaveformProcessorConfig):
        super().__init__()
        if len(config.speaker_names) < 2:
            raise ValueError("Waveform processor requires multiple target speakers")
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv1d(1, config.hidden_dim, kernel_size=config.hop_samples, stride=config.hop_samples),
            nn.GELU(), nn.Conv1d(config.hidden_dim, config.hidden_dim, 1),
        )
        self.speaker_embedding = nn.Embedding(len(config.speaker_names), config.hidden_dim)
        self.recurrent = nn.GRU(
            config.hidden_dim, config.hidden_dim, num_layers=config.recurrent_layers,
            batch_first=True, dropout=config.dropout,
        )
        self.memory_blocks = nn.ModuleList([
            CausalMemoryBlock(config) for _ in range(config.attention_layers)
        ])
        self.decoder = WaveformDecoder(config.hidden_dim)

    def speaker_index(self, name: str) -> int:
        return self.config.speaker_names.index(name)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> WaveformProcessorState:
        recurrent = torch.zeros(
            self.config.recurrent_layers, batch, self.config.hidden_dim,
            device=device, dtype=dtype,
        )
        return WaveformProcessorState(recurrent, (None,) * len(self.memory_blocks))

    def forward_chunk(
        self, waveform: Tensor, target_speaker: Tensor,
        state: WaveformProcessorState | None = None,
    ) -> tuple[Tensor, WaveformProcessorState]:
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be [batch,samples], got {tuple(waveform.shape)}")
        if waveform.shape[1] == 0 or waveform.shape[1] % self.config.hop_samples:
            raise ValueError(f"sample count must be a positive multiple of {self.config.hop_samples}")
        if state is None:
            state = self.init_state(waveform.shape[0], waveform.device, waveform.dtype)
        latent = self.encoder(waveform.unsqueeze(1)).transpose(1, 2)
        latent = latent + self.speaker_embedding(target_speaker).unsqueeze(1)
        latent, recurrent = self.recurrent(latent, state.recurrent)
        memories = []
        for block, memory in zip(self.memory_blocks, state.attention_memory):
            latent, next_memory = block(latent, memory)
            memories.append(next_memory)
        output = self.decoder(latent)
        if output.shape[1] != waveform.shape[1]:
            raise RuntimeError("Decoder violated fixed-rate input/output length")
        return output, WaveformProcessorState(recurrent, tuple(memories))

    def forward(self, waveform: Tensor, target_speaker: Tensor) -> Tensor:
        return self.forward_chunk(waveform, target_speaker)[0]
