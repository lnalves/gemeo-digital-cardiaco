"""StreamingPipeline: one session = one patient. Same code for offline replay and the server (rule 2)."""

from __future__ import annotations

import bisect
import dataclasses
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from cardiac_twin.config import Config
from cardiac_twin.decision.rules import (
    ALERT,
    CRITICAL,
    NORMAL_CLASS,
    VRunCounter,
    decide_beat,
    decide_window,
)
from cardiac_twin.engine.events import (
    Alert,
    BeatEvent,
    Event,
    KLUpdate,
    ProfileUpdated,
    SignalChunk,
    StateChange,
)
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.profile.monitor import ProfileMonitor, WindowScore
from cardiac_twin.signal.features import qrs_width_ms, r_amplitude, rr_columns, rr_features
from cardiac_twin.signal.filters import CausalBandpass
from cardiac_twin.signal.rpeaks import StreamingRPeakDetector
from cardiac_twin.signal.segmentation import normalize, refine_r

State = Literal["calibrating", "monitoring", "paused", "finished"]


@dataclasses.dataclass
class _Beat:
    index: int
    sample: int
    t: float
    beat: np.ndarray | None = None
    qrs_width_ms: float = float("nan")
    r_amplitude: float = float("nan")
    rr: np.ndarray | None = None
    valid: bool = True


class StreamingPipeline:
    """Push raw signal chunks; get beat classifications, KL scores and decisions as events.

    Beat i is classified when beats i-3 … i+1 are complete; a beat is complete when its window
    and the next R peak are known (RR after), so the algorithmic delay is about two RR intervals
    (D-25). The result does not depend on the chunk size.
    """

    def __init__(
        self,
        cfg: Config,
        classifier: BeatClassifier,
        record_id: str,
        *,
        use_profile: bool,
        feedback_enabled: bool,
        rpeak_source: str,
        annotation_samples: Sequence[int] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.cfg = cfg
        self.classifier = classifier
        self.record_id = record_id
        self.use_profile = use_profile
        self.feedback_enabled = feedback_enabled
        self.clock = clock
        fs = cfg.data.fs
        self.fs = fs
        self._filter = CausalBandpass.from_config(cfg)
        if rpeak_source == "detector":
            self._detector: StreamingRPeakDetector | None = StreamingRPeakDetector(fs, cfg.rpeaks)
            self._annotations = np.empty(0, dtype=np.int64)
        elif rpeak_source == "annotation":
            if annotation_samples is None:
                raise ValueError("annotation mode needs the annotated R samples")
            self._detector = None
            self._annotations = np.sort(np.asarray(annotation_samples, dtype=np.int64))
        else:
            raise ValueError(f"unknown rpeak_source {rpeak_source!r}")
        self._next_annotation = 0

        s, f, m = cfg.segment, cfg.features, cfg.model
        self._pre, self._post = s.pre_samples, s.post_samples
        self._refine_half = round(s.r_refine_ms * fs / 1000)
        self._qrs_half = round(f.qrs_search_ms * fs / 1000)
        self._rr_cols = rr_columns(m.rr_features)
        self._channels = m.beat_channels
        self._past = m.context_beats - 1 - m.future_beats
        self._future = m.future_beats
        self._keep_samples = round(cfg.engine.signal_buffer_seconds * fs)

        self._x = np.empty(0)
        self._offset = 0
        self._n = 0
        self._beats: list[_Beat] = []
        self._beat_count = 0
        self._prev: _Beat | None = None
        self._r_tail: deque[int] = deque(maxlen=f.long_rr_beats + 2)
        self._templates: deque[np.ndarray] = deque(maxlen=s.template_beats)
        self._stream: list[_Beat] = []
        self._next_center = 0
        self._classified: deque[tuple[float, str]] = deque()
        self._scores: list[WindowScore] = []
        self._pending_windows: deque[WindowScore] = deque()
        self._monitor = ProfileMonitor(cfg.profile)
        self._vrun = VRunCounter()
        self._alerts: dict[str, WindowScore | None] = {}
        self._episode = 0
        self._last_alert_t: float | None = None
        self.state: State = "calibrating"

    @property
    def profile(self):
        return self._monitor.profile

    @property
    def n_samples(self) -> int:
        """Samples received so far: the `t0_sample` of the next chunk."""
        return self._n

    def push(self, chunk: np.ndarray, t0_sample: int, recv_ts: float) -> list[Event]:
        if t0_sample != self._n:
            raise ValueError(f"expected chunk starting at {self._n}, got {t0_sample}")
        raw = np.asarray(chunk, dtype=np.float64)
        x = self._filter.process(raw)
        self._x = np.concatenate([self._x, x])
        self._n += len(x)
        ctx = (recv_ts, self._n)
        events: list[Event] = [SignalChunk(*self._stamp(ctx), t0_sample=t0_sample, samples=x.tolist())]

        for r in self._new_r_peaks(x):
            self._add_r_peak(r)
        for beat in self._beats:
            self._fill_signal_features(beat)
        events += self._complete_beats(ctx)
        events += self._classify(ctx)
        events += self._decide_windows(ctx)
        self._trim()
        return events

    def apply_decision(self, alert_id: str, decision: Literal["confirm", "discard"]) -> list[Event]:
        """Expert decision on an alert. A discard updates the profile only if feedback is on (rule 5)."""
        if alert_id not in self._alerts:
            raise KeyError(alert_id)
        window = self._alerts.pop(alert_id)
        if decision == "discard" and self.feedback_enabled and window is not None:
            version = self._monitor.apply_feedback(window.p, self.cfg.profile.alpha)
            return [ProfileUpdated(*self._stamp((self.clock(), self._n)), version=version)]
        return []

    def finish(self) -> list[Event]:
        self.state = "finished"
        return [StateChange(*self._stamp((self.clock(), self._n)), state=self.state)]

    def _stamp(self, ctx: tuple[float, int]) -> tuple[float, float, int]:
        return ctx[0], self.clock(), ctx[1]

    def _new_r_peaks(self, x: np.ndarray) -> list[int]:
        if self._detector is not None:
            raw = self._detector.process(x)
        else:
            stop = int(np.searchsorted(self._annotations, self._n - self._refine_half, side="left"))
            raw = self._annotations[self._next_annotation : stop].tolist()
            self._next_annotation = stop
        if not raw:
            return []
        local = np.asarray(raw) - self._offset
        refined = refine_r(self._x, local, self._refine_half) + self._offset
        return refined.tolist()

    def _add_r_peak(self, r: int) -> None:
        """Register a new R peak; it gives the RR after the previous beat, which completes its RR row."""
        beat = _Beat(index=self._beat_count, sample=r, t=r / self.fs)
        self._beat_count += 1
        self._r_tail.append(r)
        if self._prev is not None:
            f = self.cfg.features
            rows = rr_features(np.asarray(self._r_tail), self.fs, f.local_rr_beats, f.long_rr_beats)
            self._prev.rr = rows[-2]
        self._beats.append(beat)
        self._prev = beat

    def _fill_signal_features(self, beat: _Beat) -> None:
        if beat.beat is not None or not beat.valid:
            return
        r = beat.sample
        if r - self._pre < 0:
            beat.valid = False
            return
        need = max(r + self._post, r + self._qrs_half + 1)
        if need > self._n:
            return
        local = r - self._offset
        x = self._x
        beat.qrs_width_ms = float(
            qrs_width_ms(
                x,
                np.array([local]),
                self.fs,
                self.cfg.features.qrs_search_ms,
                self.cfg.features.qrs_slope_frac,
            )[0]
        )
        beat.r_amplitude = float(r_amplitude(x, np.array([local]), self._pre, self._post)[0])
        window = normalize(x[local - self._pre : local + self._post][None, :], self.cfg.segment.normalize)[0]
        beat.beat = window

    def _complete_beats(self, ctx: tuple[float, int]) -> list[Event]:
        events: list[Event] = []
        while self._beats:
            beat = self._beats[0]
            if beat.valid and (beat.beat is None or beat.rr is None):
                break
            self._beats.pop(0)
            if not beat.valid or np.isnan(beat.rr[0]):
                continue
            channels = {"beats": beat.beat}
            if "beat_diff" in self._channels:
                if self._templates:
                    channels["beat_diff"] = beat.beat - np.median(np.stack(self._templates), axis=0)
                else:
                    channels["beat_diff"] = np.zeros_like(beat.beat)
            self._templates.append(beat.beat)
            beat.beat = np.stack([channels[c] for c in self._channels], axis=-1)
            self._stream.append(beat)
            features = {
                "rr_pre": beat.rr[0],
                "qrs_width_ms": beat.qrs_width_ms,
                "r_amplitude": beat.r_amplitude,
            }
            self._monitor.add_beat(beat.t, features)
            update = self._monitor.advance(beat.t)
            if update.profile_built:
                self.state = "monitoring"
                events.append(StateChange(*self._stamp(ctx), state=self.state))
            self._scores.extend(update.scores)
            self._pending_windows.extend(update.scores)
        return events

    def _classify(self, ctx: tuple[float, int]) -> list[Event]:
        first = max(self._next_center, self._past)
        last = len(self._stream) - 1 - self._future
        if last < first:
            return []
        centers = list(range(first, last + 1))
        seqs = [self._stream[c - self._past : c + self._future + 1] for c in centers]
        beats = np.stack([np.stack([b.beat for b in s]) for s in seqs]).astype(np.float32)
        rr = np.stack([np.stack([b.rr[self._rr_cols] for b in s]) for s in seqs]).astype(np.float32)
        pred, probs, attention = self.classifier.predict(beats, rr)
        self._next_center = last + 1

        events: list[Event] = []
        classes = self.classifier.classes
        for c, seq, k, p, a in zip(centers, seqs, pred, probs, attention, strict=True):
            beat = self._stream[c]
            cls = classes[int(k)]
            score = self._score_at(beat.t)
            kl = None if score is None else score.kl
            threshold = None if score is None else score.threshold
            level = decide_beat(
                cls,
                self._vrun.update(cls),
                kl,
                threshold,
                use_profile=self.use_profile,
                calibrating=score is None,
                vt_consecutive_v=self.cfg.decision.vt_consecutive_v,
            )
            stamp = self._stamp(ctx)
            common = dict(sample=beat.sample, t=beat.t, cls=cls, prob=p.tolist(), attention=a.tolist())
            events.append(BeatEvent(*stamp, **common, level=level, kl=kl, threshold=threshold))
            if level in (ALERT, CRITICAL):
                events.append(self._alert(stamp, common, level, score, [b.sample for b in seq]))
            self._classified.append((beat.t, cls))
        self._stream = self._stream[self._next_center - self._past :]
        self._next_center = self._past
        return events

    def _alert(self, stamp, common: dict, level: str, score: WindowScore | None, context: list[int]) -> Alert:
        """The alert carries the filtered signal of its K context beats, for the alert detail screen."""
        t = common["t"]
        if self._last_alert_t is None or t - self._last_alert_t > self.cfg.decision.episode_gap_s:
            self._episode += 1
        self._last_alert_t = t
        alert_id = f"{self.record_id}-{common['sample']}"
        self._alerts[alert_id] = score
        kl = None if score is None else score.kl
        threshold = None if score is None else score.threshold
        start = max(context[0] - self._pre, self._offset)
        stop = min(context[-1] + self._post, self._n)
        segment = self._x[start - self._offset : stop - self._offset]
        return Alert(
            *stamp,
            alert_id=alert_id,
            episode_id=self._episode,
            level=level,
            **common,
            kl=kl,
            threshold=threshold,
            context_samples=context,
            segment_t0=start,
            segment=segment.tolist(),
        )

    def _score_at(self, t: float) -> WindowScore | None:
        """Latest window that ends at or before t ("current KL" of the beat, D-25)."""
        ends = [s.t_end for s in self._scores]
        i = bisect.bisect_right(ends, t)
        return self._scores[i - 1] if i else None

    def _decide_windows(self, ctx: tuple[float, int]) -> list[Event]:
        events: list[Event] = []
        last_t = self._classified[-1][0] if self._classified else -np.inf
        while self._pending_windows and self._pending_windows[0].t_end < last_t:
            w = self._pending_windows.popleft()
            start = w.t_end - self.cfg.profile.window_seconds
            has_non_n = any(start < t <= w.t_end and cls != NORMAL_CLASS for t, cls in self._classified)
            level = decide_window(has_non_n, w.kl, w.threshold)
            events.append(
                KLUpdate(
                    *self._stamp(ctx),
                    t_end=w.t_end,
                    kl=w.kl,
                    threshold=w.threshold,
                    n_beats=w.n_beats,
                    level=level,
                )
            )
        return events

    def _trim(self) -> None:
        horizon = (self._classified[-1][0] if self._classified else 0.0) - 2 * self.cfg.profile.window_seconds
        while self._classified and self._classified[0][0] < horizon:
            self._classified.popleft()
        if len(self._scores) > 2:
            keep_from = max(0, len(self._scores) - 2 - len(self._pending_windows))
            self._scores = self._scores[keep_from:]
        cut = self._n - self._keep_samples
        if self._beats:
            cut = min(cut, self._beats[0].sample - self._pre - self._qrs_half)
        cut -= self._offset
        if cut > 0:
            self._x = self._x[cut:]
            self._offset += cut
