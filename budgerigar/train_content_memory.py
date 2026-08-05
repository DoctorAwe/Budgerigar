from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .content_data import CharacterVocabulary, ContentFeatureDataset, collate_content
from .content_memory import ContentMemoryConfig, create_content_memory
from .neural_echo import require_torch


@dataclass(frozen=True)
class ContentTrainingConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    epochs: int = 30
    max_steps: int = 300
    max_train_records: int = 256
    max_validation_records: int = 64
    ctc_weight: float = 1.0
    contrastive_weight: float = 0.25
    gradient_clip: float = 1.0
    seed: int = 31


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, 1):
        current = [i]
        for j, target in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1]


def greedy_decode(logits, lengths, blank=0):
    sequences = []
    for values, length in zip(logits.argmax(-1), lengths.tolist()):
        result = []; previous = None
        for value in values[:length].tolist():
            if value != blank and value != previous: result.append(value)
            previous = value
        sequences.append(result)
    return sequences


def train_content_memory(feature_manifest, stats, output_dir, training=ContentTrainingConfig(), model_config=None):
    torch, _, functional = require_torch(); torch.manual_seed(training.seed)
    vocabulary = CharacterVocabulary()
    model_config = model_config or ContentMemoryConfig(vocabulary_size=len(vocabulary.symbols))
    train_set = ContentFeatureDataset(feature_manifest, "train", stats, vocabulary, training.max_train_records, True, model_config.update_stride, model_config.token_slots)
    validation_set = ContentFeatureDataset(feature_manifest, "validation", stats, vocabulary, training.max_validation_records, True, model_config.update_stride, model_config.token_slots)
    print(f"[content] train={len(train_set)} validation={len(validation_set)} vocab={len(vocabulary.symbols)}", flush=True)
    loader = torch.utils.data.DataLoader
    train_loader = loader(train_set, batch_size=training.batch_size, shuffle=True, num_workers=0, collate_fn=collate_content)
    validation_loader = loader(validation_set, batch_size=training.batch_size, shuffle=False, num_workers=0, collate_fn=collate_content)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_content_memory(model_config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=1e-4)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history = []; best_cer = float("inf"); step = 0
    for epoch in range(training.epochs):
        model.train(); total = count = 0
        for inputs, frame_lengths, texts, text_lengths, _, _ in train_loader:
            inputs, frame_lengths, texts, text_lengths = inputs.to(device), frame_lengths.to(device), texts.to(device), text_lengths.to(device)
            logits, memory_lengths, audio_embedding, text_embedding, _ = model(inputs, frame_lengths, texts, text_lengths)
            ctc = ctc_loss(logits.log_softmax(-1).transpose(0, 1), texts, memory_lengths, text_lengths)
            scale = model.log_temperature.exp().clamp(max=100)
            similarities = scale * audio_embedding @ text_embedding.T
            labels = torch.arange(len(inputs), device=device)
            contrastive = (functional.cross_entropy(similarities, labels) + functional.cross_entropy(similarities.T, labels)) / 2
            loss = training.ctc_weight * ctc + training.contrastive_weight * contrastive
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip); optimizer.step()
            step += 1; total += float(loss); count += 1
            if step == 1 or step % 10 == 0: print(f"[content train] step={step} loss={float(loss):.4f} ctc={float(ctc):.4f} nce={float(contrastive):.4f}", flush=True)
            if step >= training.max_steps: break
        model.eval(); edits = characters = retrieval_hits = samples = 0; validation_ctc = 0; batches = 0
        with torch.no_grad():
            for inputs, frame_lengths, texts, text_lengths, _, _ in validation_loader:
                inputs, frame_lengths, texts, text_lengths = inputs.to(device), frame_lengths.to(device), texts.to(device), text_lengths.to(device)
                logits, memory_lengths, audio_embedding, text_embedding, _ = model(inputs, frame_lengths, texts, text_lengths)
                validation_ctc += float(ctc_loss(logits.log_softmax(-1).transpose(0, 1), texts, memory_lengths, text_lengths)); batches += 1
                decoded = greedy_decode(logits.cpu(), memory_lengths.cpu())
                for index, hypothesis in enumerate(decoded):
                    reference = texts[index, :text_lengths[index]].cpu().tolist(); edits += _edit_distance(reference, hypothesis); characters += len(reference)
                retrieval_hits += int((audio_embedding @ text_embedding.T).argmax(1).eq(torch.arange(len(inputs), device=device)).sum()); samples += len(inputs)
        metrics = {"epoch": epoch + 1, "step": step, "train_loss": total / count, "validation_ctc": validation_ctc / batches, "validation_cer": edits / max(characters, 1), "validation_retrieval_top1": retrieval_hits / samples}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        checkpoint = {"architecture": "content_token_memory", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "stats": stats, "model_config": asdict(model_config), "training_config": asdict(training), "vocabulary": vocabulary.symbols, "history": history}
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["validation_cer"] < best_cer: best_cer = metrics["validation_cer"]; torch.save(checkpoint, output_dir / "best.pt")
        if step >= training.max_steps: break
    report = {"architecture": "content_token_memory", "steps": step, "best_validation_cer": best_cer, "history": history}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("feature_manifest", type=Path); parser.add_argument("stats", type=Path); parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); torch, _, _ = require_torch(); train_content_memory(args.feature_manifest, torch.load(args.stats, map_location="cpu", weights_only=True), args.output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
