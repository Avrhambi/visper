"""The shipped config.yaml is packaged and read through the single loader."""
from visper._config import CONFIG_PATH, load_config, reload_config


def test_config_is_packaged_beside_the_module():
    assert CONFIG_PATH.exists()
    assert CONFIG_PATH.parent.name == "visper"


def test_load_config_returns_dict_and_is_cached():
    a = load_config()
    b = load_config()
    assert isinstance(a, dict) and a
    assert a is b  # lru_cache
    assert reload_config() == a


def test_dead_keys_are_gone():
    cfg = load_config()
    assert "model_id" not in cfg
    assert "print_to_stdout" not in cfg


def test_bucket_overrides_shipped_empty():
    # A blanket "accurate" here disables the auto tier ladder + RTF safety demotion.
    assert load_config().get("bucket_accuracy_overrides") == {}


def test_audio_flags_present():
    cfg = load_config()
    for k in ("audio_denoise", "audio_highpass", "audio_normalize"):
        assert k in cfg
