import numpy as np
import pytest

from cardiac_twin.config import load_config
from cardiac_twin.signal.features import (
    RR_FEATURE_NAMES,
    qrs_width_ms,
    r_amplitude,
    rr_columns,
    rr_features,
)
from cardiac_twin.signal.segmentation import (
    context_indices,
    extract_beats,
    normalize,
    template_difference,
    valid_windows,
    zscore,
)

CFG = load_config()
FS = CFG.data.fs
PRE, POST = CFG.segment.pre_samples, CFG.segment.post_samples


def test_window_has_256_samples_with_r_at_pre():
    x = np.arange(2000, dtype=float)
    beats = extract_beats(x, np.array([500, 1000]), PRE, POST)
    assert beats.shape == (2, 256)
    assert beats[0, PRE] == 500 and beats[1, PRE] == 1000


def test_edge_beats_are_discarded():
    mask = valid_windows(np.array([PRE - 1, PRE, 1000, 2000 - POST, 2000 - POST + 1]), 2000, PRE, POST)
    assert mask.tolist() == [False, True, True, True, False]
    with pytest.raises(ValueError):
        extract_beats(np.zeros(2000), np.array([10]), PRE, POST)


def test_zscore_per_beat():
    rng = np.random.default_rng(0)
    beats = rng.normal(3.0, 2.0, size=(4, 256))
    z = normalize(beats, CFG.segment.normalize)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)
    assert np.all(zscore(np.ones((1, 256))) == 0)


def test_context_indices_one_future_beat():
    centers, seq = context_indices(8, context=5, future=1)
    assert centers.tolist() == [3, 4, 5, 6]
    assert seq[0].tolist() == [0, 1, 2, 3, 4]
    assert seq[-1].tolist() == [3, 4, 5, 6, 7]


def test_rr_features():
    r = np.array([0, 360, 720, 1260, 1620])  # RR = 1.0, 1.0, 1.5, 1.0 s
    f = rr_features(r, FS, local_beats=2, long_beats=3)
    assert f.shape == (5, len(RR_FEATURE_NAMES))
    assert np.isnan(f[0, 0]) and np.isnan(f[-1, 1])
    long = 3.5 / 3
    np.testing.assert_allclose(f[3], [1.5, 1.0, 1.25, 1.2, 0.8, 1.5 / long, 1.0 / long, 1.25 / long])
    np.testing.assert_allclose(f[1, 2], 1.0)
    np.testing.assert_allclose(f[4, 2], 1.25)


def test_long_ratio_sees_a_fast_run_that_the_local_ratio_misses():
    rr = np.r_[np.full(300, 1.0), np.full(20, 0.5)]
    r = np.round(np.r_[0, np.cumsum(rr)] * FS).astype(int)
    f = rr_features(r, FS, local_beats=10, long_beats=300)
    last = f[-2]
    assert last[3] == pytest.approx(1.0, abs=0.01)
    assert last[5] < 0.6


def test_rr_features_are_causal():
    r = np.array([0, 360, 720, 1080, 1440])
    a = rr_features(r, FS, local_beats=10, long_beats=300)
    r2 = r.copy()
    r2[-1] = 1800
    b = rr_features(r2, FS, local_beats=10, long_beats=300)
    pre_based = [0, 2, 3, 5, 7]
    np.testing.assert_array_equal(a[:4, pre_based], b[:4, pre_based])


def test_rr_columns():
    assert rr_columns(["rr_pre", "rr_local_long_ratio"]) == [0, 7]
    with pytest.raises(ValueError):
        rr_columns(["rr_typo"])


def _triangle_beats(width_ms: float, n: int = 5, rr: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Isosceles triangle QRS (base = width) on a flat baseline, R at the apex."""
    half = int(round(width_ms * FS / 1000 / 2))
    x = np.zeros(rr * (n + 1))
    r = rr * np.arange(1, n + 1)
    for c in r:
        x[c - half : c + 1] = np.linspace(0, 1, half + 1)
        x[c : c + half + 1] = np.linspace(1, 0, half + 1)
    return x, r


@pytest.mark.parametrize("width", [60, 100, 160])
def test_qrs_width_on_triangles(width):
    x, r = _triangle_beats(width)
    f = CFG.features
    w = qrs_width_ms(x, r, FS, f.qrs_search_ms, f.qrs_slope_frac)
    np.testing.assert_allclose(w, width, atol=1000 / FS * 1.5)


def test_qrs_width_nan_at_edges():
    x, _ = _triangle_beats(80)
    w = qrs_width_ms(x, np.array([5, len(x) - 5]), FS, CFG.features.qrs_search_ms, 0.2)
    assert np.isnan(w).all()


def test_r_amplitude_relative_to_baseline():
    x, r = _triangle_beats(80)
    amp = r_amplitude(x + 0.7, r, PRE, POST)
    np.testing.assert_allclose(amp, 1.0)


def test_template_difference_is_causal_and_ignores_isolated_ectopics():
    normal = np.sin(np.linspace(0, np.pi, 256))
    beats = np.tile(normal, (8, 1))
    beats[5] = -normal
    diff = template_difference(beats, window=4)
    np.testing.assert_array_equal(diff[0], 0)
    np.testing.assert_allclose(diff[4], 0, atol=1e-12)
    np.testing.assert_allclose(diff[5], -2 * normal)
    np.testing.assert_allclose(diff[6], 0, atol=1e-12)
    later = beats.copy()
    later[7] = 5.0
    np.testing.assert_array_equal(template_difference(later, 4)[:7], diff[:7])
