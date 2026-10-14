import numpy as np
import pytest

from cardiac_twin.config import load_config
from cardiac_twin.signal.filters import CausalBandpass

CFG = load_config()
FS = CFG.data.fs


def _signal(n: int = 20 * FS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    # baseline offset + wander + in-band "ECG" + mains + noise
    return (
        1.5
        + 0.4 * np.sin(2 * np.pi * 0.2 * t)
        + np.sin(2 * np.pi * 8 * t)
        + 0.2 * np.sin(2 * np.pi * 60 * t)
        + 0.05 * rng.standard_normal(n)
    )


@pytest.mark.parametrize("block", [FS, 1, 97, 1000])
def test_blocks_equal_full_signal(block):
    x = _signal()
    full = CausalBandpass.from_config(CFG).process_full(x)
    f = CausalBandpass.from_config(CFG)
    chunks = [f.process(x[i : i + block]) for i in range(0, len(x), block)]
    np.testing.assert_allclose(np.concatenate(chunks), full, rtol=0, atol=1e-9)


def test_random_block_sizes_equal_full_signal():
    x = _signal(seed=1)
    rng = np.random.default_rng(2)
    cuts = np.sort(rng.choice(np.arange(1, len(x)), size=40, replace=False))
    f = CausalBandpass.from_config(CFG)
    out = np.concatenate([f.process(c) for c in np.split(x, cuts)])
    np.testing.assert_allclose(out, f.process_full(x), rtol=0, atol=1e-9)


def test_is_causal():
    x = _signal()
    f = CausalBandpass.from_config(CFG)
    y = f.process_full(x)
    x2 = x.copy()
    x2[5000:] += 10.0
    np.testing.assert_array_equal(f.process_full(x2)[:5000], y[:5000])


def test_removes_offset_and_out_of_band_components():
    x = _signal()
    y = CausalBandpass.from_config(CFG).process_full(x)
    settled = y[5 * FS :]
    assert abs(settled.mean()) < 0.01  # DC offset of 1.5 removed
    spectrum = np.abs(np.fft.rfft(settled)) / len(settled)
    freqs = np.fft.rfftfreq(len(settled), 1 / FS)
    amp = {f: spectrum[np.argmin(np.abs(freqs - f))] for f in (8, 60)}
    assert amp[8] > 0.45  # in-band sine: amplitude 1 → 0.5 in the one-sided spectrum
    assert amp[60] < 0.025  # mains: 0.1 at the input; 4th-order Butterworth gives ~14 dB at 60 Hz


def test_no_startup_transient_from_offset():
    f = CausalBandpass.from_config(CFG)
    y = f.process_full(np.full(2 * FS, 3.0))
    assert np.max(np.abs(y)) < 1e-9


def test_reset_restarts_state():
    x = _signal()
    f = CausalBandpass.from_config(CFG)
    a = f.process(x[:FS])
    f.process(x[FS : 2 * FS])
    f.reset()
    np.testing.assert_array_equal(f.process(x[:FS]), a)
