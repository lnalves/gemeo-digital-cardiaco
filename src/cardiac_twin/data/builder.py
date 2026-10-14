"""Offline beat dataset: the same filter, R refinement, segmentation and features used in streaming."""

from __future__ import annotations

import collections
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cardiac_twin.config import Config
from cardiac_twin.data.labels import beat_annotations
from cardiac_twin.data.mitbih import Record, load_record
from cardiac_twin.signal.features import beat_features
from cardiac_twin.signal.filters import CausalBandpass
from cardiac_twin.signal.segmentation import extract_beats, normalize, refine_r, valid_windows


def process_record(rec: Record, cfg: Config) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the beats of one record and its statistics.

    R positions come from the annotations (D-03), refined on the filtered signal (D-16). Beats
    whose window leaves the signal or that lack RR pre/post (first and last beat) are dropped.
    """
    d, s = cfg.data, cfg.segment
    x = CausalBandpass.from_config(cfg).process_full(rec.signal)
    ann, labels = beat_annotations(rec.ann_samples, rec.ann_symbols, d.aami_map, d.classes)
    r = refine_r(x, ann, round(s.r_refine_ms * rec.fs / 1000))
    feats = beat_features(x, r, cfg)

    in_signal = valid_windows(r, len(x), s.pre_samples, s.post_samples)
    has_rr = ~np.isnan(feats["rr"][:, :2]).any(axis=1)
    keep = in_signal & has_rr
    beats = normalize(extract_beats(x, r[keep], s.pre_samples, s.post_samples), s.normalize)

    out = {
        "beats": beats.astype(np.float32),
        "rr_feats": feats["rr"][keep].astype(np.float32),
        "qrs_width_ms": feats["qrs_width_ms"][keep].astype(np.float32),
        "r_amplitude": feats["r_amplitude"][keep].astype(np.float32),
        "labels": labels[keep],
        "record_id": np.full(int(keep.sum()), rec.record_id),
        "r_sample": r[keep],
        "ann_sample": ann[keep],
    }
    qrs = out["qrs_width_ms"]
    lo, hi = cfg.features.qrs_valid_range_ms
    stats = {
        "counts": _class_counts(out["labels"], d.classes),
        "dropped": {"edge": int((~in_signal).sum()), "no_rr": int((in_signal & ~has_rr).sum())},
        "qrs_width_ms_median": float(np.nanmedian(qrs)),
        "qrs_width_out_of_range": float(np.mean((qrs < lo) | (qrs > hi))),
        "r_amplitude_median": float(np.nanmedian(out["r_amplitude"])),
    }
    return out, stats


def build_split(
    records: Sequence[str], raw_dir: Path, cfg: Config
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Concatenate `process_record` over the records of one split."""
    parts, per_record = [], {}
    for rid in records:
        arrays, stats = process_record(load_record(rid, raw_dir, cfg.data.lead), cfg)
        parts.append(arrays)
        per_record[rid] = stats
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    merged["classes"] = np.array(cfg.data.classes)
    totals = _class_counts(merged["labels"], cfg.data.classes)
    summary = {"n_beats": int(len(merged["labels"])), "counts": totals, "records": per_record}
    return merged, summary


def _class_counts(labels: np.ndarray, classes: Sequence[str]) -> dict[str, int]:
    c = collections.Counter(labels.tolist())
    return {cls: int(c[i]) for i, cls in enumerate(classes)}
