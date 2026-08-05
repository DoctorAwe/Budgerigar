from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BudgerigarV2Config:
    n_mels: int = 80
    hidden_dim: int = 256
    speaker_dim: int = 128
    encoder_layers: int = 4
    decoder_layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    speaker_names: tuple[str, ...] = ()
    max_output_ratio: float = 2.0


def length_mask(lengths: Tensor, frames: int) -> Tensor:
    return torch.arange(frames, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, 1)
        self.depthwise = nn.Conv1d(dim, dim, 15, padding=7, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.ffn2 = FeedForwardModule(dim, dropout)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, value: Tensor, padding_mask: Tensor) -> Tensor:
        value = value + 0.5 * self.ffn1(value)
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False,
        )
        value = value + self.dropout(attended)
        convolved = self.conv_norm(value).transpose(1, 2)
        convolved = F.glu(self.pointwise_in(convolved), dim=1)
        convolved = F.silu(self.batch_norm(self.depthwise(convolved)))
        value = value + self.dropout(self.pointwise_out(convolved).transpose(1, 2))
        value = value + 0.5 * self.ffn2(value)
        return self.output_norm(value).masked_fill(padding_mask.unsqueeze(-1), 0.0)


class BudgerigarV2Model(nn.Module):
    """Whole-utterance, speaker-conditioned, variable-duration acoustic model."""

    def __init__(self, config: BudgerigarV2Config):
        super().__init__()
        if len(config.speaker_names) < 2:
            raise ValueError("V2 requires at least two target speakers")
        self.config = config
        dim = config.hidden_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.n_mels), nn.Linear(config.n_mels, dim), nn.SiLU(),
        )
        self.encoder = nn.ModuleList([
            ConformerBlock(dim, config.heads, config.dropout) for _ in range(config.encoder_layers)
        ])
        self.speaker_embedding = nn.Embedding(len(config.speaker_names), config.speaker_dim)
        self.speaker_projection = nn.Linear(config.speaker_dim, dim)
        self.duration_predictor = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.SiLU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=config.heads, dim_feedforward=dim * 4,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.n_mels))
        self.speaker_classifier = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, len(config.speaker_names)),
        )

    def speaker_index(self, name: str) -> int:
        return self.config.speaker_names.index(name)

    def encode(self, source: Tensor, source_lengths: Tensor) -> tuple[Tensor, Tensor]:
        mask = length_mask(source_lengths, source.shape[1])
        content = self.input_projection(source).masked_fill(mask.unsqueeze(-1), 0.0)
        for block in self.encoder:
            content = block(content, mask)
        return content, mask

    @staticmethod
    def _pool(content: Tensor, mask: Tensor) -> Tensor:
        valid = (~mask).unsqueeze(-1)
        return (content * valid).sum(1) / valid.sum(1).clamp_min(1)

    def predict_lengths(
        self, content: Tensor, source_lengths: Tensor, source_mask: Tensor,
        speaker_condition: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pooled = self._pool(content, source_mask)
        log_ratio = self.duration_predictor(torch.cat([pooled, speaker_condition], dim=-1)).squeeze(-1)
        max_lengths = (source_lengths.float() * self.config.max_output_ratio).long()
        predicted = (source_lengths.float() * log_ratio.exp()).round().long()
        return log_ratio, predicted.clamp(min=2, max=max_lengths)

    @staticmethod
    def regulate(content: Tensor, source_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        pieces = []
        max_target = int(target_lengths.max())
        for index in range(len(content)):
            sequence = content[index, :source_lengths[index]].T.unsqueeze(0)
            resized = F.interpolate(
                sequence, size=int(target_lengths[index]), mode="linear", align_corners=False,
            )[0].T
            pieces.append(F.pad(resized, (0, 0, 0, max_target - len(resized))))
        return torch.stack(pieces)

    def forward(
        self, source: Tensor, source_lengths: Tensor, target_speaker: Tensor,
        target_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        content, source_mask = self.encode(source, source_lengths)
        speaker = self.speaker_projection(self.speaker_embedding(target_speaker))
        log_ratio, predicted_lengths = self.predict_lengths(
            content, source_lengths, source_mask, speaker,
        )
        output_lengths = predicted_lengths if target_lengths is None else target_lengths
        queries = self.regulate(content, source_lengths, output_lengths) + speaker.unsqueeze(1)
        target_mask = length_mask(output_lengths, queries.shape[1])
        decoded = self.decoder(
            queries, content, tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=source_mask,
        )
        mel = self.output(decoded).masked_fill(target_mask.unsqueeze(-1), 0.0)
        pooled_decoded = self._pool(decoded, target_mask)
        speaker_logits = self.speaker_classifier(pooled_decoded)
        return mel, log_ratio, output_lengths, speaker_logits

