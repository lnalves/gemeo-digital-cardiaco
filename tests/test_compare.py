import numpy as np
import pytest

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.decision.rules import ALERT, ATTENTION, CRITICAL, NORMAL
from cardiac_twin.evaluation.compare import (
    check_b_subset_of_a,
    chunk_ms,
    example_records,
    per_record_row,
    plot_fp_per_record,
    plot_kl_threshold,
    plot_latency,
    wilcoxon_fp,
)
from cardiac_twin.evaluation.replay import NO_LABEL, ReplayResult, replay_record
from cardiac_twin.evaluation.system import aggregate, system_metrics

CFG = load_config()
CLASSES = CFG.data.classes


def _replay(levels, true=None, alert_t=(), alert_episode=(), rid="900"):
    n = len(levels)
    t = np.arange(n, dtype=float) * 60
    return ReplayResult(
        record_id=rid,
        duration_s=float(n * 60),
        beat_sample=(t * CFG.data.fs).astype(np.int64),
        beat_t=t,
        beat_cls=np.zeros(n, dtype=np.int64),
        beat_level=np.array(levels, dtype=object),
        beat_kl=np.ones(n),
        beat_threshold=np.full(n, 2.0),
        beat_true=np.zeros(n, dtype=np.int64) if true is None else np.array(true, dtype=np.int64),
        beat_latency_s=np.full(n, 0.002),
        beat_delay_s=np.full(n, 2.0),
        ann_t=t,
        ann_label=np.zeros(n, dtype=np.int64),
        ann_level=np.array(levels, dtype=object),
        window_t=t,
        window_kl=np.ones(n),
        window_threshold=np.full(n, 2.0),
        window_level=np.array([NORMAL] * n, dtype=object),
        alert_t=np.array(alert_t, dtype=float),
        alert_episode=np.array(alert_episode, dtype=np.int64),
        push_seconds=np.full(n, 0.002),
    )


def test_wilcoxon_one_sided_when_b_is_lower():
    fp_a = [10, 20, 30, 40, 50, 60, 70, 80]
    fp_b = [1, 2, 3, 4, 5, 6, 7, 8]
    w = wilcoxon_fp(fp_a, fp_b, 0.05)
    assert w["w"] == 0
    assert w["p_value"] == pytest.approx(1 / 2**8)
    assert w["significant"]
    assert w["median_diff_b_minus_a"] == pytest.approx(-40.5)


def test_wilcoxon_drops_zero_differences():
    w = wilcoxon_fp([5, 5, 9, 7], [5, 5, 1, 2], 0.05)
    assert w["n_pairs"] == 4
    assert w["n_nonzero"] == 2
    assert not w["significant"]


def test_wilcoxon_without_differences_is_not_significant():
    w = wilcoxon_fp([3, 4], [3, 4], 0.05)
    assert w["p_value"] is None
    assert not w["significant"]


def test_b_alarms_must_be_a_subset_of_a():
    a = _replay([ALERT, NORMAL, CRITICAL])
    check_b_subset_of_a(a, _replay([ATTENTION, NORMAL, CRITICAL]))
    with pytest.raises(ValueError, match="not in A"):
        check_b_subset_of_a(a, _replay([ALERT, ALERT, CRITICAL]))


def test_b_must_see_the_same_beats_as_a():
    a = _replay([ALERT, NORMAL])
    b = _replay([ALERT, NORMAL, NORMAL])
    with pytest.raises(ValueError, match="different beats"):
        check_b_subset_of_a(a, b)


def test_episodes_after_calibration():
    r = _replay([NORMAL] * 10, alert_t=[100, 400, 410, 900], alert_episode=[1, 2, 2, 3])
    m = system_metrics(r, CLASSES, 300)
    assert m["episodes"] == 2
    agg = aggregate([m, m])
    assert agg["episodes"] == 4
    assert agg["episodes_per_hour"] == pytest.approx(4 / agg["hours"])


def test_per_record_row_pairs_a_and_b():
    n = CLASSES.index("N")
    a = system_metrics(_replay([NORMAL, ALERT, ALERT, ALERT], true=[n, n, n, NO_LABEL]), CLASSES, 0)
    b = system_metrics(_replay([NORMAL, ATTENTION, ALERT, ATTENTION], true=[n, n, n, NO_LABEL]), CLASSES, 0)
    row = per_record_row(a, b)
    assert (row["fp_a"], row["fp_b"]) == (2, 1)
    assert (row["fp_spurious_a"], row["fp_spurious_b"]) == (1, 0)
    assert row["fp_per_hour_a"] == pytest.approx(2 / row["hours"])


def test_example_records_are_the_largest_gain_and_the_largest_cost():
    rows = [
        {"record_id": "1", "fp_a": 50, "fp_b": 5, "alarmed_sv_a": 10, "alarmed_sv_b": 9},
        {"record_id": "2", "fp_a": 10, "fp_b": 9, "alarmed_sv_a": 90, "alarmed_sv_b": 20},
        {"record_id": "3", "fp_a": 20, "fp_b": 20, "alarmed_sv_a": 5, "alarmed_sv_b": 5},
    ]
    assert example_records(rows) == ["1", "2"]


def test_figures_are_written(tmp_path):
    a = _replay([ALERT, ALERT, NORMAL, CRITICAL])
    b = _replay([ATTENTION, ALERT, NORMAL, CRITICAL])
    row = per_record_row(system_metrics(a, CLASSES, 0), system_metrics(b, CLASSES, 0))
    plot_fp_per_record([row], tmp_path / "fp.png")
    plot_kl_threshold(a, b, CLASSES.index("N"), 60, tmp_path / "kl.png")
    plot_latency(chunk_ms({"900": (a, b)}), CFG.evaluation.latency_budget_ms, tmp_path / "lat.png")
    assert all((tmp_path / f).stat().st_size > 0 for f in ("fp.png", "kl.png", "lat.png"))


@pytest.mark.data
@pytest.mark.slow
def test_processing_time_p95_is_below_the_budget():
    model_run = ROOT / CFG.server.model_run
    if not (model_run / "model.keras").exists():
        pytest.skip("modelo final ausente")
    from cardiac_twin.model.inference import BeatClassifier

    clf = BeatClassifier.load(model_run, CLASSES, CFG.model.batch_size)
    r = replay_record(CFG, clf, "205", ROOT / CFG.data.raw_dir, use_profile=True, rpeak_source="detector")
    assert np.percentile(r.push_seconds * 1000, 95) < CFG.evaluation.latency_budget_ms
    assert np.percentile(r.beat_latency_s * 1000, 95) < CFG.evaluation.latency_budget_ms
