import dataclasses

import pytest
import yaml

from cardiac_twin.config import DEFAULT_CONFIG_PATH, Config, config_to_dict, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.data.fs == 360
    assert cfg.signal.bandpass_hz == (0.5, 40.0)
    assert cfg.profile.tuning_grid.k == (1.0, 1.5, 2.0, 2.5, 3.0)
    assert cfg.data.aami_map["V"] == ("V", "E")
    assert cfg.data.val_records is None
    assert cfg.engine.feedback_enabled is False


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.profile.k = 3.0


def test_override_file_merges_over_default(tmp_path):
    override = tmp_path / "exp.yaml"
    override.write_text(yaml.safe_dump({"seed": 7, "profile": {"k": 2.5}}))
    cfg = load_config(override)
    assert cfg.seed == 7
    assert cfg.profile.k == 2.5
    assert cfg.profile.alpha == load_config().profile.alpha


def test_unknown_key_is_rejected(tmp_path):
    override = tmp_path / "typo.yaml"
    override.write_text(yaml.safe_dump({"profile": {"kk": 2.5}}))
    with pytest.raises(ValueError, match="kk"):
        load_config(override)


def test_round_trip_matches_yaml():
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    assert config_to_dict(load_config()) == raw
