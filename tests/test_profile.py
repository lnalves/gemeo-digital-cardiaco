import dataclasses

import numpy as np
import pytest

from cardiac_twin.config import load_config
from cardiac_twin.profile.histogram import bin_index, calibration_edges, histogram
from cardiac_twin.profile.kl import kl, kl_sum
from cardiac_twin.profile.monitor import ProfileMonitor
from cardiac_twin.profile.profile import PatientProfile, in_window, window_ends

CFG = load_config().profile
EPS = CFG.eps


def _beats(seconds: float, rr: float = 0.8, seed: int = 0, qrs: float = 90.0):
    rng = np.random.default_rng(seed)
    t = np.arange(rr, seconds, rr)
    n = len(t)
    feats = {
        "rr_pre": rng.normal(rr, 0.03, n),
        "qrs_width_ms": rng.normal(qrs, 5.0, n),
        "r_amplitude": rng.normal(1.2, 0.1, n),
    }
    return t, feats


def test_kl_is_non_negative_and_zero_for_equal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(7)), rng.dirichlet(np.ones(7))
        assert kl(p, q) >= 0
        assert kl(p, p) == pytest.approx(0.0, abs=1e-15)


def test_kl_is_finite_with_empty_bins_after_smoothing():
    edges = np.linspace(0, 1, 5)
    p = histogram(np.array([0.1, 0.1, 0.2]), edges, EPS)
    q = histogram(np.array([0.9, 0.95]), edges, EPS)
    assert np.isfinite(kl(p, q)) and kl(p, q) > 1
    with pytest.raises(ValueError):
        kl(np.array([1.0, 0.0]), np.array([0.5, 0.5]))


def test_kl_sum_adds_features():
    p = {"a": np.array([0.5, 0.5]), "b": np.array([0.9, 0.1])}
    q = {"a": np.array([0.4, 0.6]), "b": np.array([0.5, 0.5])}
    assert kl_sum(p, q) == pytest.approx(kl(p["a"], q["a"]) + kl(p["b"], q["b"]))


def test_histogram_overflow_bins_and_smoothing():
    edges = calibration_edges(np.linspace(0, 10, 1001), n_bins=4, percentiles=(1, 99))
    np.testing.assert_allclose(edges, np.linspace(0.1, 9.9, 5))
    assert bin_index(np.array([-5.0, 0.1, 9.9, 50.0]), edges).tolist() == [0, 1, 4, 5]
    h = histogram(np.array([-5.0, 50.0, np.nan]), edges, EPS)
    assert h.shape == (6,) and h.sum() == pytest.approx(1.0)
    assert h[0] == pytest.approx(h[-1]) and h[0] > 0.49
    assert np.all(h > 0)


def test_constant_calibration_feature_still_has_edges():
    edges = calibration_edges(np.full(50, 3.0), 4, (1, 99))
    assert np.all(np.diff(edges) >= 0) and edges[-1] > edges[0]


def test_calibration_subwindows_match_monitoring_windows():
    ends = window_ends(CFG)
    assert ends[0] == CFG.window_seconds and ends[-1] == CFG.calibration_seconds
    assert len(ends) == (CFG.calibration_seconds - CFG.window_seconds) // CFG.step_seconds + 1
    assert in_window(np.array([0.0, 0.5, 60.0, 60.1]), 60.0, 60.0).tolist() == [False, True, True, False]


def test_build_threshold_and_score():
    t, f = _beats(CFG.calibration_seconds)
    prof = PatientProfile.build(t, f, CFG)
    assert len(prof.calibration_scores) == len(window_ends(CFG))
    assert prof.sigma > 0 and prof.threshold(2.0) == pytest.approx(prof.mu + 2 * prof.sigma)
    same = {k: v[-75:] for k, v in f.items()}
    _, other = _beats(60, qrs=140.0, seed=5)
    assert prof.score(other) > prof.threshold(3.0) > prof.score(same)


def test_update_sums_to_one_and_lies_between_q_and_p():
    t, f = _beats(CFG.calibration_seconds)
    prof = PatientProfile.build(t, f, CFG)
    _, other = _beats(60, qrs=140.0, seed=5)
    p = prof.distributions(other)
    new = prof.update(p, CFG.alpha)
    assert new.version == prof.version + 1 and new.mu == prof.mu
    for feat in prof.features:
        assert new.q[feat].sum() == pytest.approx(1.0)
        lo, hi = np.minimum(prof.q[feat], p[feat]), np.maximum(prof.q[feat], p[feat])
        assert np.all(new.q[feat] >= lo - 1e-15) and np.all(new.q[feat] <= hi + 1e-15)
    assert new.score(other) < prof.score(other)


def test_json_round_trip():
    t, f = _beats(CFG.calibration_seconds)
    prof = PatientProfile.build(t, f, CFG)
    back = PatientProfile.from_dict(prof.to_dict())
    assert back.version == prof.version and back.mu == prof.mu
    for feat in prof.features:
        np.testing.assert_array_equal(back.q[feat], prof.q[feat])


def _feed(monitor: ProfileMonitor, t: np.ndarray, f: dict, step: float):
    updates, i = [], 0
    for now in np.arange(step, t[-1] + step, step):
        while i < len(t) and t[i] <= now:
            monitor.add_beat(t[i], {k: v[i] for k, v in f.items()})
            i += 1
        updates.append(monitor.advance(now))
    return updates


def test_monitor_calibrates_then_scores_every_step():
    t, f = _beats(CFG.calibration_seconds + 120)
    m = ProfileMonitor(CFG)
    ups = _feed(m, t, f, step=1.0)
    built = [u for u in ups if u.profile_built]
    scores = [s for u in ups for s in u.scores]
    assert len(built) == 1
    assert [s.t_end for s in scores] == [
        float(CFG.calibration_seconds + i * CFG.step_seconds) for i in range(13)
    ]
    assert all(s.threshold == m.threshold for s in scores)


def test_monitor_does_not_depend_on_chunk_size():
    t, f = _beats(CFG.calibration_seconds + 90)
    a = [s.kl for u in _feed(ProfileMonitor(CFG), t, f, 1.0) for s in u.scores]
    b = [s.kl for u in _feed(ProfileMonitor(CFG), t, f, 7.0) for s in u.scores]
    assert a == b


def test_monitor_feedback_updates_version():
    t, f = _beats(CFG.calibration_seconds + 30)
    m = ProfileMonitor(CFG)
    _feed(m, t, f, 1.0)
    assert m.apply_feedback(m.latest.p, CFG.alpha) == 2
    with pytest.raises(RuntimeError):
        ProfileMonitor(CFG).apply_feedback(m.latest.p, CFG.alpha)


def test_monitor_threshold_uses_k_from_config():
    t, f = _beats(CFG.calibration_seconds + 10)
    m1, m3 = ProfileMonitor(CFG), ProfileMonitor(dataclasses.replace(CFG, k=3.0))
    _feed(m1, t, f, 1.0)
    _feed(m3, t, f, 1.0)
    assert m3.threshold > m1.threshold
