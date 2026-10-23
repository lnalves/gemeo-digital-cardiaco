import numpy as np
import pytest

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.data.builder import process_record
from cardiac_twin.data.labels import beat_annotations
from cardiac_twin.data.mitbih import load_record
from cardiac_twin.decision.rules import CRITICAL
from cardiac_twin.engine.events import Alert, BeatEvent, KLUpdate, StateChange
from cardiac_twin.engine.pipeline import StreamingPipeline
from cardiac_twin.signal.features import rr_columns

CFG = load_config()
FS = CFG.data.fs
CLASSES = CFG.data.classes

pytestmark = [pytest.mark.data, pytest.mark.slow]


class SpyClassifier:
    """Deterministic stand-in for BeatClassifier that records its inputs.

    Most beats are N; a beat whose RR before is short against the long mean is V or S, so the
    test streams contain alerts, V runs and windows with non-N beats.
    """

    def __init__(self) -> None:
        self.classes = CLASSES
        self.inputs: list[tuple[np.ndarray, np.ndarray]] = []
        self._ratio = CFG.model.rr_features.index("rr_pre_long_ratio")

    def predict(self, beats, rr):
        self.inputs.append((beats.copy(), rr.copy()))
        center = CFG.model.context_beats - 1 - CFG.model.future_beats
        ratio = rr[:, center, self._ratio]
        pred = np.where(ratio < 0.8, CLASSES.index("V"), np.where(ratio < 0.9, CLASSES.index("S"), 0))
        probs = np.eye(len(CLASSES))[pred]
        attn = np.full((len(beats), beats.shape[1]), 1 / beats.shape[1])
        return pred, probs, attn


def _record(rid: str, minutes: float):
    rec = load_record(rid, ROOT / CFG.data.raw_dir, CFG.data.lead)
    return rec, rec.signal[: int(minutes * 60 * FS)]


def _run(signal, chunk, use_profile=True, rpeak_source="detector", annotations=None):
    spy = SpyClassifier()
    p = StreamingPipeline(
        CFG,
        spy,
        "test",
        use_profile=use_profile,
        feedback_enabled=False,
        rpeak_source=rpeak_source,
        annotation_samples=annotations,
    )
    events = []
    for start in range(0, len(signal), chunk):
        events += p.push(signal[start : start + chunk], start, 0.0)
    return events, spy, p


def _key(events):
    """Content of each event stream; the interleaving of kinds depends on the chunk size."""
    return {
        "beat": [(e.sample, e.cls, e.level, e.kl) for e in events if isinstance(e, BeatEvent)],
        "kl": [(e.t_end, e.kl, e.level) for e in events if isinstance(e, KLUpdate)],
        "alert": [(e.alert_id, e.episode_id, e.level) for e in events if isinstance(e, Alert)],
    }


def test_chunked_equals_whole_signal():
    _, x = _record("119", 10)
    whole, _, _ = _run(x, len(x))
    for chunk in (FS, 7 * FS + 13):
        events, _, _ = _run(x, chunk)
        assert _key(events) == _key(whole)
    assert any(isinstance(e, Alert) for e in whole)
    assert any(isinstance(e, KLUpdate) for e in whole)


def test_model_inputs_equal_the_training_dataset():
    rec, x = _record("119", 6)
    r_ann, _ = beat_annotations(rec.ann_samples, rec.ann_symbols, CFG.data.aami_map, CFG.data.classes)
    events, spy, _ = _run(x, FS, rpeak_source="annotation", annotations=r_ann[r_ann < len(x)])
    offline, _ = process_record(rec, CFG)
    by_sample = {int(s): i for i, s in enumerate(offline["r_sample"])}
    center = CFG.model.context_beats - 1 - CFG.model.future_beats
    beats = np.concatenate([b[:, center, :, 0] for b, _ in spy.inputs])
    rr = np.concatenate([r[:, center] for _, r in spy.inputs])
    samples = [e.sample for e in events if isinstance(e, BeatEvent)]
    idx = [by_sample[s] for s in samples]
    np.testing.assert_allclose(beats, offline["beats"][idx], atol=1e-5)
    np.testing.assert_allclose(rr, offline["rr_feats"][idx][:, rr_columns(CFG.model.rr_features)], rtol=1e-6)


def test_only_critical_alerts_during_calibration():
    _, x = _record("119", 7)
    events, _, _ = _run(x, FS)
    calib = CFG.profile.calibration_seconds
    assert all(e.level == CRITICAL for e in events if isinstance(e, Alert) and e.t <= calib)
    states = [e.state for e in events if isinstance(e, StateChange)]
    assert states == ["monitoring"]


def test_profile_alerts_are_a_subset_of_classifier_alone():
    _, x = _record("119", 10)
    a, _, _ = _run(x, FS, use_profile=False)
    b, _, _ = _run(x, FS, use_profile=True)
    alerts_a = {e.sample for e in a if isinstance(e, Alert)}
    alerts_b = {e.sample for e in b if isinstance(e, Alert)}
    assert alerts_b <= alerts_a
    assert len(alerts_a) > len(alerts_b)


def test_discard_updates_profile_only_with_feedback():
    _, x = _record("119", 7)
    for feedback, expected in ((False, 1), (True, 2)):
        spy = SpyClassifier()
        p = StreamingPipeline(
            CFG, spy, "t", use_profile=False, feedback_enabled=feedback, rpeak_source="detector"
        )
        alerts = [
            e for s in range(0, len(x), FS) for e in p.push(x[s : s + FS], s, 0.0) if isinstance(e, Alert)
        ]
        late = [a for a in alerts if a.kl is not None]
        p.apply_decision(late[0].alert_id, "discard")
        assert p.profile.version == expected


def test_alert_segment_covers_the_context_beats():
    _, x = _record("119", 7)
    events, _, _ = _run(x, FS, use_profile=False)
    signal = np.concatenate([e.samples for e in events if e.kind == "signal"])
    alerts = [e for e in events if isinstance(e, Alert)]
    assert alerts
    for a in alerts:
        assert len(a.context_samples) == CFG.model.context_beats
        assert a.context_samples[-1 - CFG.model.future_beats] == a.sample
        assert a.segment_t0 <= a.context_samples[0] - CFG.segment.pre_samples or a.segment_t0 == 0
        np.testing.assert_allclose(a.segment, signal[a.segment_t0 : a.segment_t0 + len(a.segment)])
        assert a.segment_t0 + len(a.segment) >= a.context_samples[-1] + CFG.segment.post_samples
