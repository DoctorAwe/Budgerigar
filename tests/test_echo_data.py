import json

from budgerigar.echo_data import load_pairs, pair_report


def test_echo_pairs_have_one_fixed_target(tmp_path):
    rows = []
    for split, suffix in (("train", "1"), ("validation", "2")):
        for speaker in ("arctic_bdl", "arctic_slt"):
            rows.append({
                "id": f"{speaker}_{suffix}", "speaker": speaker,
                "parallel_id": f"p{suffix}", "split": split, "language": "en",
                "emotion": "neutral", "feature_path": f"/{speaker}_{suffix}.pt",
            })
    manifest = tmp_path / "features.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    pairs = load_pairs(manifest, "arctic_slt")
    report = pair_report(pairs)
    assert report["source_speakers"] == ["arctic_bdl"]
    assert report["target_speaker"] == ["arctic_slt"]
    assert report["by_split"]["train"] == 1
    assert report["by_split"]["validation"] == 1

