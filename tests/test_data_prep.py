from pathlib import Path
import wave

from budgerigar.data_prep import build_cmu_manifest, stable_split, write_manifest
from budgerigar.manifest import audit_manifest, load_manifest


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 1_600)


def test_stable_split_keeps_parallel_group_together():
    assert stable_split("same") == stable_split("same")
    assert stable_split("same", validation_percent=0, test_percent=0) == "train"


def test_cmu_manifest_pairs_speakers(tmp_path):
    for speaker in ("bdl", "slt"):
        root = tmp_path / f"cmu_us_{speaker}_arctic"
        (root / "etc").mkdir(parents=True)
        utterance = "arctic_a0001"
        (root / "etc" / "txt.done.data").write_text(
            f'( {utterance} "A shared sentence." )\n', encoding="utf-8",
        )
        _write_wav(root / "wav" / f"{utterance}.wav")
    rows = build_cmu_manifest(tmp_path)
    assert len(rows) == 2
    assert rows[0]["parallel_id"] == rows[1]["parallel_id"]
    destination = write_manifest(rows, tmp_path / "manifest.jsonl")
    report = audit_manifest(load_manifest(destination))
    assert report.ok
    assert report.parallel_groups == 1


def test_split_percentages_are_valid():
    try:
        stable_split("x", validation_percent=50, test_percent=50)
    except ValueError as error:
        assert "between 0 and 99" in str(error)
    else:
        raise AssertionError("expected invalid split percentages to fail")
