import numpy as np
import pytest

from cardiac_twin.config import load_config
from cardiac_twin.signal.features import qrs_width_ms, r_amplitude, rr_features
from cardiac_twin.signal.segmentation import (
    context_indices,
    extract_beats,
    normalize,
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
    f = rr_features(r, FS, local_beats=2)
    assert np.isnan(f[0, 0]) and np.isnan(f[-1, 1])
    np.testing.assert_allclose(f[3], [1.5, 1.0, 1.25, 1.2, 0.8])
    np.testing.assert_allclose(f[1, 2], 1.0)  # fewer than local_beats intervals available
    np.testing.assert_allclose(f[4, 2], 1.25)  # mean of the last 2 intervals (1.5, 1.0)


def test_rr_features_are_causal_in_local_mean():
    r = np.array([0, 360, 720, 1080, 1440])
    a = rr_features(r, FS, local_beats=10)
    r2 = r.copy()
    r2[-1] = 1800  # changes only rr_post of beat 3 and rr_pre of beat 4
    b = rr_features(r2, FS, local_beats=10)
    np.testing.assert_array_equal(a[:4, [0, 2, 3]], b[:4, [0, 2, 3]])


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
