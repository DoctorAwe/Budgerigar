from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
from typing import Iterable
from urllib.request import urlopen
import wave


CMU_ARCTIC = {
    "bdl": "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",
    "clb": "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
    "jmk": "3a37c0e1dfc91e734fdbc88b562d9e2ebca621772402cdc693bbc9b09b211d73",
    "ksp": "8029cafce8296f9bed3022c44ef1e7953332b6bf6943c14b929f468122532717",
    "rms": "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
    "slt": "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",
}
CMU_BASE_URL = "https://www.festvox.org/cmu_arctic/packed"


def stable_split(key: str, validation_percent: int = 5, test_percent: int = 5) -> str:
    """Assign a parallel group to a deterministic, machine-independent split."""
    if validation_percent < 0 or test_percent < 0 or validation_percent + test_percent >= 100:
        raise ValueError("validation_percent + test_percent must be between 0 and 99")
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + validation_percent:
        return "validation"
    return "train"


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:bz2") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"archive contains unsafe path: {member.name}")
        bundle.extractall(destination, filter="data")


def download_cmu_arctic(root: str | Path, speakers: Iterable[str]) -> list[Path]:
    """Idempotently download verified CMU ARCTIC archives in Colab."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for speaker in speakers:
        if speaker not in CMU_ARCTIC:
            raise ValueError(f"unsupported speaker {speaker!r}; choose from {sorted(CMU_ARCTIC)}")
        name = f"cmu_us_{speaker}_arctic"
        folder = root / name
        if folder.is_dir():
            extracted.append(folder)
            continue
        archive = root / f"{name}.tar.bz2"
        if not archive.is_file() or _sha256(archive) != CMU_ARCTIC[speaker]:
            archive.unlink(missing_ok=True)
            with urlopen(f"{CMU_BASE_URL}/{archive.name}") as source, archive.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
        digest = _sha256(archive)
        if digest != CMU_ARCTIC[speaker]:
            raise ValueError(f"checksum mismatch for {archive.name}: {digest}")
        _safe_extract(archive, root)
        if not folder.is_dir():
            raise FileNotFoundError(f"archive did not create expected folder {folder}")
        extracted.append(folder)
    return extracted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


_ARCTIC_LINE = re.compile(r'^\(\s*(?P<id>\S+)\s+"(?P<text>.*)"\s*\)$')


def build_cmu_manifest(root: str | Path) -> list[dict]:
    rows: list[dict] = []
    for speaker_root in sorted(Path(root).glob("cmu_us_*_arctic")):
        speaker = speaker_root.name.removeprefix("cmu_us_").removesuffix("_arctic")
        transcript = speaker_root / "etc" / "txt.done.data"
        if not transcript.is_file():
            continue
        for line_number, line in enumerate(transcript.read_text(encoding="utf-8").splitlines(), 1):
            match = _ARCTIC_LINE.match(line.strip())
            if not match:
                raise ValueError(f"{transcript}:{line_number}: unsupported transcript line")
            utterance_id = match.group("id")
            parallel_id = utterance_id.split("_", 1)[-1]
            audio = (speaker_root / "wav" / f"{utterance_id}.wav").resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            rows.append({
                "id": f"arctic_{speaker}_{parallel_id}", "audio": str(audio),
                "speaker": f"arctic_{speaker}", "text": match.group("text"),
                "language": "en", "emotion": "neutral",
                "parallel_id": f"arctic_en_{parallel_id}", "duration": _duration(audio),
                "split": stable_split(f"arctic_en_{parallel_id}"),
            })
    return rows


def _load_esd_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in root.rglob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "|" not in line:
                continue
            key, text = line.split("|", 1)
            transcripts[key.strip()] = text.strip()
    return transcripts


def build_esd_manifest(root: str | Path) -> list[dict]:
    """Build a manifest from an already downloaded official ESD directory.

    ESD is research-only and its official page currently delegates downloading
    to an external link, so this function intentionally does not bypass the
    license/download step. It tolerates an additional enclosing directory.
    """
    root = Path(root)
    transcripts = _load_esd_transcripts(root)
    split_names = {"train": "train", "evaluation": "validation", "validation": "validation", "test": "test"}
    rows: list[dict] = []
    for audio in sorted(root.rglob("*.wav")):
        speaker_match = next((part for part in audio.parts if re.fullmatch(r"\d{4}", part)), None)
        if speaker_match is None:
            continue
        stem = audio.stem
        utterance = stem.split("_", 1)[-1]
        emotion = next((part.casefold() for part in audio.parts if part.casefold() in {"angry", "happy", "neutral", "sad", "surprise"}), "unknown")
        source_split = next((split_names[part.casefold()] for part in audio.parts if part.casefold() in split_names), None)
        language = "zh" if int(speaker_match) <= 10 else "en"
        parallel_id = f"esd_{language}_{emotion}_{utterance}"
        text = transcripts.get(stem, transcripts.get(utterance, ""))
        rows.append({
            "id": f"esd_{speaker_match}_{emotion}_{utterance}", "audio": str(audio.resolve()),
            "speaker": f"esd_{speaker_match}", "text": text, "language": language,
            "emotion": emotion, "parallel_id": parallel_id, "duration": _duration(audio),
            "split": source_split or stable_split(parallel_id),
        })
    if not rows:
        raise ValueError(f"no ESD WAV files found below {root}")
    return rows


def write_manifest(rows: Iterable[dict], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["id"])
    destination.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Colab speech datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cmu = subparsers.add_parser("cmu", help="download and index CMU ARCTIC")
    cmu.add_argument("root", type=Path)
    cmu.add_argument("output", type=Path)
    cmu.add_argument("--speakers", nargs="+", default=["bdl", "slt"])
    esd = subparsers.add_parser("esd", help="index an already downloaded official ESD directory")
    esd.add_argument("root", type=Path)
    esd.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "cmu":
        download_cmu_arctic(args.root, args.speakers)
        rows = build_cmu_manifest(args.root)
    else:
        rows = build_esd_manifest(args.root)
    print(write_manifest(rows, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
