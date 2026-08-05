from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AudioConfig:
    """Canonical audio timing used by preprocessing and streaming inference."""

    sample_rate: int = 24_000
    tick_ms: int = 20
    lookahead_ms: int = 40

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        if self.sample_rate * self.tick_ms % 1000:
            raise ValueError("tick_ms must map to an integer number of samples")
        if self.lookahead_ms < 0:
            raise ValueError("lookahead_ms cannot be negative")

    @property
    def tick_samples(self) -> int:
        return self.sample_rate * self.tick_ms // 1000

    @property
    def algorithmic_buffer_ms(self) -> int:
        return self.tick_ms + self.lookahead_ms


class AudioChunker:
    """Convert irregular device blocks into exact model ticks.

    The chunker owns per-session state. ``flush`` optionally emits a final
    zero-padded tick and returns the number of real samples in that tick.
    """

    def __init__(self, config: AudioConfig = AudioConfig()) -> None:
        self.config = config
        self._buffer: list[float] = []

    @property
    def pending_samples(self) -> int:
        return len(self._buffer)

    def push(self, samples: Iterable[float]) -> list[tuple[float, ...]]:
        self._buffer.extend(float(sample) for sample in samples)
        size = self.config.tick_samples
        ticks: list[tuple[float, ...]] = []
        while len(self._buffer) >= size:
            ticks.append(tuple(self._buffer[:size]))
            del self._buffer[:size]
        return ticks

    def flush(self, pad: bool = True) -> tuple[tuple[float, ...] | None, int]:
        valid = len(self._buffer)
        if valid == 0:
            return None, 0
        values: Sequence[float] = self._buffer
        if pad:
            values = (*values, *(0.0 for _ in range(self.config.tick_samples - valid)))
        result = tuple(values)
        self._buffer.clear()
        return result, valid

    def reset(self) -> None:
        self._buffer.clear()

