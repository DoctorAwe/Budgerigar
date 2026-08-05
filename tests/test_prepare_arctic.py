from pathlib import Path

from budgerigar.prepare_arctic import build_pairs


def test_pairs_only_matching_utterances(tmp_path: Path):
    for speaker in ("bdl", "slt"):
        wav = tmp_path / f"cmu_us_{speaker}_arctic" / "wav"
        wav.mkdir(parents=True)
        (wav / "arctic_a0001.wav").touch()
    extra = tmp_path / "cmu_us_bdl_arctic" / "wav" / "arctic_a0002.wav"
    extra.touch()
    pairs = build_pairs(tmp_path, "slt")
    assert len(pairs) == 1
    assert pairs[0]["utterance_id"] == "arctic_a0001"

