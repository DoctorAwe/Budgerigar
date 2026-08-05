from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import wave


ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class ManifestRecord:
    id: str
    audio: Path
    speaker: str
    text: str
    language: str
    emotion: str
    parallel_id: str
    duration: float
    split: str

    @classmethod
    def from_dict(cls, value: dict, base_dir: Path) -> "ManifestRecord":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        audio = Path(value["audio"])
        if not audio.is_absolute():
            audio = (base_dir / audio).resolve()
        return cls(
            id=str(value["id"]), audio=audio, speaker=str(value["speaker"]),
            text=str(value["text"]), language=str(value["language"]),
            emotion=str(value["emotion"]), parallel_id=str(value["parallel_id"]),
            duration=float(value["duration"]), split=str(value["split"]),
        )


@dataclass(frozen=True)
class AuditReport:
    records: int
    speakers: int
    hours: float
    parallel_groups: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest = Path(path)
    records: list[ManifestRecord] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            records.append(ManifestRecord.from_dict(value, manifest.parent))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"{manifest}:{line_number}: {error}") from error
    return records


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def audit_manifest(records: Iterable[ManifestRecord], check_audio: bool = True) -> AuditReport:
    rows = list(records)
    errors: list[str] = []
    warnings: list[str] = []
    ids = Counter(row.id for row in rows)
    for duplicate, count in ids.items():
        if count > 1:
            errors.append(f"duplicate id {duplicate!r} appears {count} times")

    parallel_speakers: dict[str, set[str]] = defaultdict(set)
    parallel_splits: dict[str, set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not row.id or not row.speaker or not row.parallel_id:
            errors.append(f"record {row.id!r} has an empty identifier")
        if row.split not in ALLOWED_SPLITS:
            errors.append(f"record {row.id!r} has invalid split {row.split!r}")
        if row.duration <= 0:
            errors.append(f"record {row.id!r} has non-positive duration")
        parallel_speakers[row.parallel_id].add(row.speaker)
        parallel_splits[row.parallel_id].add(row.split)
        text_splits[" ".join(row.text.casefold().split())].add(row.split)
        if check_audio:
            if not row.audio.is_file():
                errors.append(f"record {row.id!r} audio does not exist: {row.audio}")
            elif row.audio.suffix.casefold() == ".wav":
                try:
                    actual = _wav_duration(row.audio)
                    if abs(actual - row.duration) > max(0.05, actual * 0.02):
                        warnings.append(
                            f"record {row.id!r} duration differs: manifest={row.duration:.3f}s, wav={actual:.3f}s"
                        )
                except (wave.Error, EOFError) as error:
                    errors.append(f"record {row.id!r} is not a readable WAV: {error}")

    for parallel_id, splits in parallel_splits.items():
        if len(splits) > 1:
            errors.append(f"parallel group {parallel_id!r} leaks across splits: {sorted(splits)}")
    for text, splits in text_splits.items():
        if text and len(splits) > 1:
            warnings.append(f"normalized text leaks across splits: {text[:60]!r}")

    paired = sum(len(speakers) >= 2 for speakers in parallel_speakers.values())
    if rows and paired == 0:
        warnings.append("no parallel_id contains at least two speakers")
    return AuditReport(
        records=len(rows), speakers=len({row.speaker for row in rows}),
        hours=sum(row.duration for row in rows) / 3600.0,
        parallel_groups=paired, errors=tuple(errors), warnings=tuple(warnings),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Budgerigar JSONL manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()
    report = audit_manifest(load_manifest(args.manifest), check_audio=not args.skip_audio)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

