"""Load `configs/default.yaml` into frozen dataclasses."""

from __future__ import annotations

import argparse
import dataclasses
import typing
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


@dataclasses.dataclass(frozen=True)
class DataConfig:
    raw_dir: str
    processed_dir: str
    physionet_db: str
    download_workers: int
    lead: str
    fs: int
    paced_records: tuple[str, ...]
    ds1: tuple[str, ...]
    ds2: tuple[str, ...]
    val_records: tuple[str, ...] | None
    n_val_records: int
    val_min_beats: dict[str, int]
    val_max_fraction: dict[str, float]
    aami_map: dict[str, tuple[str, ...]]
    classes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SignalConfig:
    bandpass_hz: tuple[float, float]
    filter_order: int
    peak_match_tol_ms: float


@dataclasses.dataclass(frozen=True)
class RPeaksConfig:
    mwi_ms: float
    refractory_ms: float
    learning_s: float
    threshold_frac: float
    peak_ema: float
    rr_avg_beats: int
    searchback_rr_factor: float
    searchback_threshold_scale: float
    searchback_peak_ema: float
    twave_ms: float
    twave_slope_frac: float


@dataclasses.dataclass(frozen=True)
class SegmentConfig:
    pre_samples: int
    post_samples: int
    r_refine_ms: float
    normalize: str


@dataclasses.dataclass(frozen=True)
class FeaturesConfig:
    local_rr_beats: int
    long_rr_beats: int
    qrs_search_ms: float
    qrs_slope_frac: float
    qrs_valid_range_ms: tuple[float, float]


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    context_beats: int
    future_beats: int
    rr_features: tuple[str, ...]
    augment_noise_std: float
    augment_baseline_std: float
    cnn_filters: tuple[int, ...]
    cnn_kernels: tuple[int, ...]
    lstm_units: int
    attention_units: int
    dense_units: int
    dropout: float
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
    loss: str
    focal_gamma: float
    class_weight_cap: float


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    macro_classes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TuningGridConfig:
    k: tuple[float, ...]
    window_seconds: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ProfileConfig:
    calibration_seconds: int
    features: tuple[str, ...]
    n_bins: int
    range_percentiles: tuple[float, float]
    eps: float
    window_seconds: int
    step_seconds: int
    k: float
    alpha: float
    tuning_grid: TuningGridConfig
    max_sensitivity_drop_pp: float


@dataclasses.dataclass(frozen=True)
class DecisionConfig:
    vt_consecutive_v: int
    episode_gap_s: float


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    chunk_seconds: float
    rpeak_source: str
    use_profile: bool
    feedback_enabled: bool


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    db_path: str
    model_run: str | None
    cors_origins: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    runs_dir: str


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    signal: SignalConfig
    rpeaks: RPeaksConfig
    segment: SegmentConfig
    features: FeaturesConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    profile: ProfileConfig
    decision: DecisionConfig
    engine: EngineConfig
    server: ServerConfig
    paths: PathsConfig


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    return value


def _build(cls: type, raw: dict[str, Any], prefix: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(prefix + k for k in unknown)}")
    missing = names - set(raw)
    if missing:
        raise ValueError(f"missing config keys: {sorted(prefix + k for k in missing)}")
    kwargs = {}
    for name in names:
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, raw[name], f"{prefix}{name}.")
        else:
            kwargs[name] = _freeze(raw[name])
    return cls(**kwargs)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> Config:
    """Load the default config, then merge the optional override file over it."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        raw = _merge(raw, yaml.safe_load(Path(path).read_text()) or {})
    return _build(Config, raw)


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, dict):
        return {k: _thaw(v) for k, v in value.items()}
    return value


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Return plain YAML-serializable data, used to save the effective config of a run."""
    return _thaw(dataclasses.asdict(cfg))


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML file merged over configs/default.yaml"
    )
