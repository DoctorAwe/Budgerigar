from __future__ import annotations

from dataclasses import asdict, dataclass

from .neural_echo import require_torch


@dataclass(frozen=True)
class MonotonicEchoConfig:
    n_mels: int = 100
    hidden_dim: int = 192
    event_slots: int = 160
    understanding_layers: int = 4
    update_stride: int = 4
    local_encoder_layers: int = 4
    local_kernel_size: int = 5
    read_window_sigma: float = 1.5
    dropout: float = 0.1


def create_monotonic_echo(config=MonotonicEchoConfig()):
    torch, nn, functional = require_torch(); dim = config.hidden_dim

    class MonotonicEcho(nn.Module):
        """Ordered event tape with recurrent whole-understanding token layers."""

        def __init__(self):
            super().__init__(); self.config = config
            self.input_norm = nn.LayerNorm(config.n_mels + 2)
            self.input_projection = nn.Linear(config.n_mels + 2, dim)
            self.local_encoder = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim),
            ) for _ in range(config.local_encoder_layers)])
            self.local_convolution = nn.ModuleList([
                nn.Conv1d(dim, dim, config.local_kernel_size, groups=dim)
                for _ in range(config.local_encoder_layers)
            ])
            self.event_write = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, 1))
            self.duration_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Softplus())
            self.layer_embedding = nn.Parameter(torch.randn(config.understanding_layers, dim) * .02)
            self.understanding_optimizers = nn.ModuleList([nn.GRUCell(dim * 3, dim) for _ in range(config.understanding_layers)])
            self.understanding_gates = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, 1)) for _ in range(config.understanding_layers)])
            self.readiness_head = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, 1))
            self.advance_head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, 1), nn.Softplus())
            self.decoder_state = nn.GRUCell(dim * 3, dim)
            self.personality = nn.Parameter(torch.zeros(dim))
            self.acoustic_head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.SiLU(),
                                               nn.Dropout(config.dropout), nn.Linear(dim * 2, config.n_mels))
            self.voice_head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, 1))

        def forward(self, inputs, ablate_events=False, ablate_understanding=False):
            encoded = self.input_projection(self.input_norm(inputs))
            for convolution, feed_forward in zip(self.local_convolution, self.local_encoder):
                causal = functional.pad(encoded.transpose(1, 2), (config.local_kernel_size - 1, 0))
                encoded = encoded + convolution(causal).transpose(1, 2); encoded = encoded + feed_forward(encoded)
            batch = len(inputs); positions = torch.arange(config.event_slots, device=inputs.device, dtype=inputs.dtype)
            events = inputs.new_zeros(batch, config.event_slots, dim); occupancy = inputs.new_zeros(batch, config.event_slots)
            write_phase = inputs.new_zeros(batch); read_phase = inputs.new_zeros(batch)
            layers = [inputs.new_zeros(batch, dim) for _ in range(config.understanding_layers)]
            decoder = inputs.new_zeros(batch, dim)
            acoustic=[]; voices=[]; write_phases=[]; read_phases=[]; readiness_values=[]; advances=[]
            for tick in range(inputs.shape[1]):
                if tick % config.update_stride == config.update_stride - 1:
                    chunk = encoded[:, tick + 1 - config.update_stride:tick + 1].mean(1)
                    vad = inputs[:, tick + 1 - config.update_stride:tick + 1, -1].mean(1, keepdim=True)
                    write = self.event_write(torch.cat([chunk, vad], -1)).sigmoid().squeeze(-1)
                    center = write_phase.clamp(0, config.event_slots - 1).unsqueeze(1)
                    weight = torch.exp(-.5 * ((positions.unsqueeze(0) - center) / .65) ** 2) * write.unsqueeze(1)
                    events = events * (1 - weight.unsqueeze(-1)) + chunk.unsqueeze(1) * weight.unsqueeze(-1)
                    occupancy = torch.maximum(occupancy, weight)
                    write_phase = (write_phase + write * self.duration_head(chunk).squeeze(-1)).clamp(max=config.event_slots - 1)
                    previous = layers
                    updated=[]
                    for level, (optimizer, gate) in enumerate(zip(self.understanding_optimizers, self.understanding_gates)):
                        lower = chunk if level == 0 else updated[level - 1]
                        upper = previous[level + 1] if level + 1 < len(previous) else previous[level]
                        evidence = torch.cat([lower, previous[level] + self.layer_embedding[level], upper], -1)
                        proposal = optimizer(evidence, previous[level]); amount = gate(evidence).sigmoid()
                        updated.append(previous[level] + amount * (proposal - previous[level]))
                    layers = updated
                top = layers[-1]
                readiness = self.readiness_head(torch.cat([top, inputs[:, tick, -1:].contiguous()], -1)).sigmoid().squeeze(-1)
                distance = positions.unsqueeze(0) - read_phase.unsqueeze(1)
                attention = torch.exp(-.5 * (distance / config.read_window_sigma) ** 2) * occupancy
                attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
                event_context = torch.einsum("bs,bsd->bd", attention, events)
                if ablate_events: event_context = event_context * 0
                if ablate_understanding: top = top * 0
                decoder_input = torch.cat([encoded[:, tick], event_context, top], -1)
                decoder = self.decoder_state(decoder_input, decoder)
                advance = self.advance_head(torch.cat([decoder, event_context, top], -1)).squeeze(-1) * readiness
                remaining = (write_phase - read_phase).sigmoid()
                read_phase = (read_phase + advance * remaining).clamp(max=config.event_slots - 1)
                decoded = torch.cat([decoder + self.personality, event_context, top], -1)
                acoustic.append(self.acoustic_head(decoded)); voices.append(self.voice_head(decoded).squeeze(-1))
                write_phases.append(write_phase); read_phases.append(read_phase); readiness_values.append(readiness); advances.append(advance)
            diagnostics={"write_phase":torch.stack(write_phases,1), "read_phase":torch.stack(read_phases,1),
                         "readiness":torch.stack(readiness_values,1), "advance":torch.stack(advances,1),
                         "understanding":torch.stack(layers,1)}
            return torch.stack(acoustic,1), torch.stack(voices,1), (events, layers, decoder), diagnostics

        def export_config(self): return asdict(config)

    return MonotonicEcho()
