from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .content_data import CharacterVocabulary, ContentFeatureDataset, collate_content
from .content_memory import ContentMemoryConfig, create_content_memory
from .neural_echo import require_torch


def _architecture(config):
    if config.acoustic_ctc: return "content_local_dual_ctc_memory"
    return "content_token_memory_sequence" if config.sequence_contrastive else "content_token_memory"


@dataclass(frozen=True)
class ContentTrainingConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    epochs: int = 30
    max_steps: int = 300
    max_train_records: int | None = 256
    max_validation_records: int | None = 64
    ctc_weight: float = 1.0
    acoustic_ctc_weight: float = 0.0
    contrastive_weight: float = 0.25
    gradient_clip: float = 1.0
    seed: int = 31
    initialization_checkpoint: str | None = None


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


def train_content_memory(
    feature_manifest, stats, output_dir, training=ContentTrainingConfig(),
    model_config=None, resume_from=None,
):
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
    model = create_content_memory(model_config).to(device)
    if resume_from is None and training.initialization_checkpoint:
        initialization = torch.load(training.initialization_checkpoint, map_location="cpu", weights_only=True)
        current = model.state_dict(); transferred = {}
        for key, value in initialization["model"].items():
            if key in current and current[key].shape == value.shape:
                transferred[key] = value
        missing, unexpected = model.load_state_dict(transferred, strict=False)
        print(
            f"[transfer] source={training.initialization_checkpoint} tensors={len(transferred)} "
            f"new_or_changed={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=1e-4)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history = []; best_cer = float("inf"); best_score = float("inf"); step = 0
    if resume_from is not None and Path(resume_from).is_file():
        resume = torch.load(resume_from, map_location="cpu", weights_only=True)
        expected_architecture = _architecture(model_config)
        if resume.get("architecture") != expected_architecture:
            raise ValueError(f"resume checkpoint is not a {expected_architecture} model")
        normalized_resume_config = asdict(ContentMemoryConfig(**resume["model_config"]))
        if normalized_resume_config != asdict(model_config):
            raise ValueError("resume model configuration does not match the current configuration")
        model.load_state_dict(resume["model"]); optimizer.load_state_dict(resume["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = training.learning_rate
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value): state[key] = value.to(device)
        history = list(resume.get("history", []))
        step = int(history[-1]["step"]) if history else 0
        best_cer = min((row["validation_cer"] for row in history), default=float("inf"))
        best_score = min((row.get("selection_score", row["validation_cer"]) for row in history), default=float("inf"))
        print(f"[resume] {resume_from} step={step} best_cer={best_cer:.4f}", flush=True)
    for epoch in range(len(history), training.epochs):
        model.train(); total = count = 0
        for inputs, frame_lengths, texts, text_lengths, _, _ in train_loader:
            inputs, frame_lengths, texts, text_lengths = inputs.to(device), frame_lengths.to(device), texts.to(device), text_lengths.to(device)
            logits, memory_lengths, audio_embedding, text_embedding, diagnostics = model(inputs, frame_lengths, texts, text_lengths)
            ctc = ctc_loss(logits.log_softmax(-1).transpose(0, 1), texts, memory_lengths, text_lengths)
            acoustic_logits = diagnostics["acoustic_logits"]
            acoustic_ctc = (
                ctc_loss(acoustic_logits.log_softmax(-1).transpose(0, 1), texts, diagnostics["acoustic_lengths"], text_lengths)
                if acoustic_logits is not None else ctc.new_zeros(())
            )
            scale = model.log_temperature.exp().clamp(max=100)
            similarities = scale * audio_embedding @ text_embedding.T
            labels = torch.arange(len(inputs), device=device)
            contrastive = (functional.cross_entropy(similarities, labels) + functional.cross_entropy(similarities.T, labels)) / 2
            loss = training.ctc_weight * ctc + training.acoustic_ctc_weight * acoustic_ctc + training.contrastive_weight * contrastive
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip); optimizer.step()
            step += 1; total += float(loss); count += 1
            if step == 1 or step % 10 == 0: print(f"[content train] step={step} loss={float(loss):.4f} bank_ctc={float(ctc):.4f} acoustic_ctc={float(acoustic_ctc):.4f} nce={float(contrastive):.4f}", flush=True)
            if step >= training.max_steps: break
        model.eval(); edits = acoustic_edits = characters = retrieval_hits = samples = 0; validation_ctc = validation_acoustic_ctc = 0; batches = 0
        with torch.no_grad():
            for inputs, frame_lengths, texts, text_lengths, _, _ in validation_loader:
                inputs, frame_lengths, texts, text_lengths = inputs.to(device), frame_lengths.to(device), texts.to(device), text_lengths.to(device)
                logits, memory_lengths, audio_embedding, text_embedding, diagnostics = model(inputs, frame_lengths, texts, text_lengths)
                validation_ctc += float(ctc_loss(logits.log_softmax(-1).transpose(0, 1), texts, memory_lengths, text_lengths)); batches += 1
                acoustic_logits = diagnostics["acoustic_logits"]
                if acoustic_logits is not None:
                    validation_acoustic_ctc += float(ctc_loss(acoustic_logits.log_softmax(-1).transpose(0, 1), texts, diagnostics["acoustic_lengths"], text_lengths))
                decoded = greedy_decode(logits.cpu(), memory_lengths.cpu())
                acoustic_decoded = greedy_decode(acoustic_logits.cpu(), diagnostics["acoustic_lengths"].cpu()) if acoustic_logits is not None else decoded
                for index, hypothesis in enumerate(decoded):
                    reference = texts[index, :text_lengths[index]].cpu().tolist(); edits += _edit_distance(reference, hypothesis); acoustic_edits += _edit_distance(reference, acoustic_decoded[index]); characters += len(reference)
                retrieval_hits += int((audio_embedding @ text_embedding.T).argmax(1).eq(torch.arange(len(inputs), device=device)).sum()); samples += len(inputs)
        validation_cer = edits / max(characters, 1); acoustic_cer = acoustic_edits / max(characters, 1); validation_retrieval = retrieval_hits / samples
        metrics = {"epoch": epoch + 1, "step": step, "train_loss": total / count, "validation_ctc": validation_ctc / batches, "validation_acoustic_ctc": validation_acoustic_ctc / batches if model_config.acoustic_ctc else None, "validation_cer": validation_cer, "validation_acoustic_cer": acoustic_cer if model_config.acoustic_ctc else None, "validation_retrieval_top1": validation_retrieval, "selection_score": validation_cer - 0.5 * validation_retrieval}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        architecture = _architecture(model_config)
        checkpoint = {"architecture": architecture, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "stats": stats, "model_config": asdict(model_config), "training_config": asdict(training), "vocabulary": vocabulary.symbols, "history": history}
        torch.save(checkpoint, output_dir / "last.pt")
        best_cer = min(best_cer, metrics["validation_cer"])
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]; torch.save(checkpoint, output_dir / "best.pt")
        if step >= training.max_steps: break
    architecture = _architecture(model_config)
    report = {"architecture": architecture, "steps": step, "best_validation_cer": best_cer, "best_selection_score": best_score, "history": history}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("feature_manifest", type=Path); parser.add_argument("stats", type=Path); parser.add_argument("output_dir", type=Path); parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); torch, _, _ = require_torch(); train_content_memory(args.feature_manifest, torch.load(args.stats, map_location="cpu", weights_only=True), args.output_dir, resume_from=args.resume); return 0


if __name__ == "__main__": raise SystemExit(main())
