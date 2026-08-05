from budgerigar.features import FeatureConfig


def test_feature_fingerprint_is_stable_and_sensitive():
    baseline = FeatureConfig()
    assert baseline.fingerprint == FeatureConfig().fingerprint
    assert baseline.fingerprint != FeatureConfig(n_mels=80).fingerprint


def test_feature_frequency_range_must_fit_nyquist():
    try:
        FeatureConfig(sample_rate=16_000, f_max=11_000.0)
    except ValueError as error:
        assert "Nyquist" in str(error)
    else:
        raise AssertionError("expected invalid Mel frequency range to fail")

