import numpy as np
import pytest

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.data.labels import beat_annotations
from cardiac_twin.data.mitbih import load_record
from cardiac_twin.signal.filters import CausalBandpass
from cardiac_twin.signal.rpeaks import StreamingRPeakDetector, detect_r_peaks, match_peaks
from cardiac_twin.signal.segmentation import refine_r

CFG = load_config()
FS = CFG.data.fs
TOL = round(CFG.signal.peak_match_tol_ms * FS / 1000)


def _synthetic_ecg(seconds: float = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """QRS-like Gaussian spikes with RR jitter, a T wave, baseline wander and noise; filtered."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n)
    r = []
    pos = int(0.5 * FS)
    while pos < n - FS:
        r.append(pos)
        pos += int(FS * rng.uniform(0.65, 1.1))
    x = 0.3 * np.sin(2 * np.pi * 0.25 * t / FS) + 0.03 * rng.standard_normal(n)
    for c in r:
        x += np.exp(-0.5 * ((t - c) / (0.012 * FS)) ** 2)
        x += 0.25 * np.exp(-0.5 * ((t - c - 0.3 * FS) / (0.05 * FS)) ** 2)
    return CausalBandpass.from_config(CFG).process_full(x), np.array(r)


def test_detects_synthetic_beats():
    x, r = _synthetic_ecg()
    m = match_peaks(detect_r_peaks(x, FS, CFG.rpeaks), r, TOL)
    assert m.sensitivity == 1.0 and m.ppv == 1.0


@pytest.mark.parametrize("block", [FS, 1, 250, 7 * FS])
def test_blocks_equal_whole_signal(block):
    x, _ = _synthetic_ecg(seed=3)
    whole = detect_r_peaks(x, FS, CFG.rpeaks)
    det = StreamingRPeakDetector(FS, CFG.rpeaks)
    chunked = [p for i in range(0, len(x), block) for p in det.process(x[i : i + block])]
    assert chunked == whole.tolist()
    assert np.all(np.diff(whole) > 0)


def test_nothing_before_learning_phase():
    x, _ = _synthetic_ecg()
    det = StreamingRPeakDetector(FS, CFG.rpeaks)
    assert det.process(x[: det.n_learn - 1]) == []


def test_match_peaks_counts():
    ann = np.array([100, 500, 900, 1300])
    det = np.array([95, 520, 700, 1310, 1320])
    m = match_peaks(det, ann, tol=30)
    assert m.ann_index.tolist() == [0, 1, 3]
    assert m.det_index.tolist() == [0, 1, 3]  # 1310 is closer to 1300 than 1320
    assert m.false_negative.tolist() == [2]
    assert m.false_positive.tolist() == [2, 4]
    assert m.sensitivity == pytest.approx(0.75) and m.ppv == pytest.approx(0.6)


def test_refine_r_moves_to_local_extremum():
    x = np.zeros(100)
    x[40], x[60] = 1.0, -2.0
    assert refine_r(x, np.array([45, 55, 2]), half=10).tolist() == [40, 60, 0]


# Clean DS1 records. The PLANO example (100, 103, 113) is DS2 and must not guide the detector (RNF02).
@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("record", ["112", "115", "122"])
def test_detector_on_clean_records(record):
    d = CFG.data
    rec = load_record(record, ROOT / d.raw_dir, d.lead)
    r, _ = beat_annotations(rec.ann_samples, rec.ann_symbols, d.aami_map, d.classes)
    x = CausalBandpass.from_config(CFG).process_full(rec.signal)
    m = match_peaks(detect_r_peaks(x, FS, CFG.rpeaks), r, TOL)
    assert m.sensitivity > 0.99
    assert m.ppv > 0.99


@pytest.mark.data
@pytest.mark.slow
def test_detector_streaming_equals_offline_on_record():
    d = CFG.data
    x = CausalBandpass.from_config(CFG).process_full(load_record("119", ROOT / d.raw_dir, d.lead).signal)
    x = x[: 5 * 60 * FS]
    det = StreamingRPeakDetector(FS, CFG.rpeaks)
    chunks = [p for i in range(0, len(x), FS) for p in det.process(x[i : i + FS])]
    assert chunks == detect_r_peaks(x, FS, CFG.rpeaks).tolist()
