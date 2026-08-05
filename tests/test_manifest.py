import json
from pathlib import Path
import wave

from budgerigar.manifest import audit_manifest, load_manifest


def _write_wav(path: Path, frames: int = 1_600, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * frames)


def test_manifest_load_and_parallel_audit(tmp_path):
    _write_wav(tmp_path / "a.wav")
    _write_wav(tmp_path / "b.wav")
    rows = [
        {"id": "a", "audio": "a.wav", "speaker": "s1", "text": "hello", "language": "en", "emotion": "neutral", "parallel_id": "p1", "duration": 0.1, "split": "train"},
        {"id": "b", "audio": "b.wav", "speaker": "s2", "text": "hello", "language": "en", "emotion": "neutral", "parallel_id": "p1", "duration": 0.1, "split": "train"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = audit_manifest(load_manifest(manifest))
    assert report.ok
    assert report.records == 2
    assert report.speakers == 2
    assert report.parallel_groups == 1


def test_audit_detects_parallel_split_leakage(tmp_path):
    base = {"audio": "missing.wav", "text": "same", "language": "en", "emotion": "neutral", "parallel_id": "p1", "duration": 1.0}
    rows = [
        {**base, "id": "a", "speaker": "s1", "split": "train"},
        {**base, "id": "b", "speaker": "s2", "split": "test"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = audit_manifest(load_manifest(manifest), check_audio=False)
    assert not report.ok
    assert any("leaks across splits" in error for error in report.errors)


def test_manifest_error_has_line_number(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"id": "incomplete"}', encoding="utf-8")
    try:
        load_manifest(manifest)
    except ValueError as error:
        assert ":1:" in str(error)
        assert "missing fields" in str(error)
    else:
        raise AssertionError("expected invalid manifest to fail")

