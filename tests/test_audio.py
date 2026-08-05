from budgerigar.audio import AudioChunker, AudioConfig


def test_irregular_blocks_become_exact_ticks():
    config = AudioConfig(sample_rate=1_000, tick_ms=20, lookahead_ms=0)
    chunker = AudioChunker(config)
    assert chunker.push(range(7)) == []
    ticks = chunker.push(range(7, 47))
    assert len(ticks) == 2
    assert ticks[0] == tuple(float(value) for value in range(20))
    assert chunker.pending_samples == 7


def test_flush_reports_valid_samples_and_reset_clears_state():
    chunker = AudioChunker(AudioConfig(sample_rate=1_000, tick_ms=10))
    chunker.push([1.0, 2.0, 3.0])
    tick, valid = chunker.flush()
    assert valid == 3
    assert tick == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert chunker.flush() == (None, 0)


def test_audio_config_rejects_fractional_tick():
    try:
        AudioConfig(sample_rate=22_050, tick_ms=17)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("expected invalid tick configuration to fail")

