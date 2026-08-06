from budgerigar.codec_features import codec_fingerprint


def test_codec_fingerprint_is_stable_and_configuration_sensitive():
    first=codec_fingerprint("facebook/encodec_24khz",6.0,"main")
    assert first==codec_fingerprint("facebook/encodec_24khz",6.0,"main")
    assert first!=codec_fingerprint("facebook/encodec_24khz",3.0,"main")
    assert len(first)==16
