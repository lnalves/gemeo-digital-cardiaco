"""Offline replay: feed a record to StreamingPipeline in chunks without sleeping; keep the ground truth."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from cardiac_twin.config import Config
from cardiac_twin.data.labels import beat_annotations
from cardiac_twin.data.mitbih import load_record
from cardiac_twin.engine.events import Alert, BeatEvent, KLUpdate
from cardiac_twin.engine.pipeline import StreamingPipeline
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.signal.rpeaks import match_peaks

NO_LABEL = -1


@dataclasses.dataclass(frozen=True)
class ReplayResult:
    record_id: str
    duration_s: float
    beat_sample: np.ndarray
    beat_t: np.ndarray
    beat_cls: np.ndarray
    beat_level: np.ndarray
    beat_kl: np.ndarray
    beat_threshold: np.ndarray
    beat_true: np.ndarray
    beat_latency_s: np.ndarray
    beat_delay_s: np.ndarray
    ann_t: np.ndarray
    ann_label: np.ndarray
    ann_level: np.ndarray
    window_t: np.ndarray
    window_kl: np.ndarray
    window_threshold: np.ndarray
    window_level: np.ndarray
    n_alerts: int
    push_seconds: np.ndarray


def replay_record(
    cfg: Config,
    classifier: BeatClassifier,
    record_id: str,
    raw_dir: Path,
    *,
    use_profile: bool,
    rpeak_source: str,
) -> ReplayResult:
    """Run one record through the pipeline in `engine.chunk_seconds` chunks. Feedback is off (rule 5)."""
    d = cfg.data
    rec = load_record(record_id, raw_dir, d.lead)
    r_ann, labels = beat_annotations(rec.ann_samples, rec.ann_symbols, d.aami_map, d.classes)
    pipeline = StreamingPipeline(
        cfg,
        classifier,
        record_id,
        use_profile=use_profile,
        feedback_enabled=False,
        rpeak_source=rpeak_source,
        annotation_samples=r_ann,
    )
    chunk = round(cfg.engine.chunk_seconds * d.fs)
    beats: list[BeatEvent] = []
    windows: list[KLUpdate] = []
    n_alerts = 0
    push_seconds = []
    for start in range(0, len(rec.signal), chunk):
        recv = time.perf_counter()
        events = pipeline.push(rec.signal[start : start + chunk], start, recv)
        push_seconds.append(time.perf_counter() - recv)
        for e in events:
            if isinstance(e, BeatEvent):
                beats.append(e)
            elif isinstance(e, KLUpdate):
                windows.append(e)
            elif isinstance(e, Alert):
                n_alerts += 1
    pipeline.finish()

    sample = np.array([b.sample for b in beats], dtype=np.int64)
    tol = round(cfg.signal.peak_match_tol_ms * d.fs / 1000)
    match = match_peaks(sample, r_ann, tol)
    true = np.full(len(beats), NO_LABEL, dtype=np.int64)
    true[match.det_index] = labels[match.ann_index]
    level = np.array([b.level for b in beats], dtype=object)
    ann_level = np.full(len(r_ann), "", dtype=object)
    ann_level[match.ann_index] = level[match.det_index]

    def floats(values):
        return np.array([np.nan if v is None else v for v in values], dtype=np.float64)

    return ReplayResult(
        record_id=record_id,
        duration_s=len(rec.signal) / d.fs,
        beat_sample=sample,
        beat_t=sample / d.fs,
        beat_cls=np.array([d.classes.index(b.cls) for b in beats], dtype=np.int64),
        beat_level=level,
        beat_kl=floats(b.kl for b in beats),
        beat_threshold=floats(b.threshold for b in beats),
        beat_true=true,
        beat_latency_s=np.array([b.emit_ts - b.recv_ts for b in beats]),
        beat_delay_s=np.array([(b.chunk_end_sample - b.sample) / d.fs for b in beats]),
        ann_t=r_ann / d.fs,
        ann_label=labels.astype(np.int64),
        ann_level=ann_level,
        window_t=np.array([w.t_end for w in windows]),
        window_kl=np.array([w.kl for w in windows]),
        window_threshold=np.array([w.threshold for w in windows]),
        window_level=np.array([w.level for w in windows], dtype=object),
        n_alerts=n_alerts,
        push_seconds=np.array(push_seconds),
    )
