"""System metrics of a replay: false alarms, alert sensitivity, rates and latency (ARQUITETURA §11.2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cardiac_twin.decision.rules import ALERT, CRITICAL
from cardiac_twin.evaluation.replay import NO_LABEL, ReplayResult

ALARM_LEVELS = (ALERT, CRITICAL)


def system_metrics(r: ReplayResult, classes: Sequence[str], skip_seconds: float) -> dict[str, Any]:
    """Counts after the first `skip_seconds` (rule 6).

    FP = alarms on beats annotated N; spurious = alarms on detections with no annotation. The alert
    sensitivity of a class is the share of its annotated beats that got an alarm; missed detections
    count as not alarmed.
    """
    n_idx = classes.index("N")
    beat = r.beat_t > skip_seconds
    ann = r.ann_t > skip_seconds
    alarm = np.isin(r.beat_level, ALARM_LEVELS) & beat
    ann_alarm = np.isin(r.ann_level, ALARM_LEVELS) & ann
    hours = max(r.duration_s - skip_seconds, 0.0) / 3600

    def sens(labels: Sequence[str]) -> tuple[int, int]:
        mask = ann & np.isin(r.ann_label, [classes.index(c) for c in labels])
        return int((ann_alarm & mask).sum()), int(mask.sum())

    out: dict[str, Any] = {
        "record_id": r.record_id,
        "hours": hours,
        "alarms": int(alarm.sum()),
        "fp": int((alarm & (r.beat_true == n_idx)).sum()),
        "fp_spurious": int((alarm & (r.beat_true == NO_LABEL)).sum()),
        "n_beats_n": int((beat & (r.beat_true == n_idx)).sum()),
        "episodes": len(np.unique(r.alert_episode[r.alert_t > skip_seconds])),
    }
    for name, labels in (("s", ["S"]), ("v", ["V"]), ("sv", ["S", "V"])):
        hit, total = sens(labels)
        out[f"alarmed_{name}"], out[f"total_{name}"] = hit, total
    out["latency_s"] = r.beat_latency_s[beat].tolist()
    out["delay_s"] = r.beat_delay_s[beat].tolist()
    return out


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Totals over records: FP, FP/h, alert sensitivity (S, V, S+V), alarms/h, episodes/h, latency, delay."""
    total = {
        k: sum(r[k] for r in rows) for k in ("hours", "alarms", "episodes", "fp", "fp_spurious", "n_beats_n")
    }
    out: dict[str, float] = dict(total)
    out["fp_per_hour"] = total["fp"] / total["hours"] if total["hours"] else float("nan")
    out["alarms_per_hour"] = total["alarms"] / total["hours"] if total["hours"] else float("nan")
    out["episodes_per_hour"] = total["episodes"] / total["hours"] if total["hours"] else float("nan")
    for name in ("s", "v", "sv"):
        hit = sum(r[f"alarmed_{name}"] for r in rows)
        n = sum(r[f"total_{name}"] for r in rows)
        out[f"sensitivity_{name}"] = hit / n if n else float("nan")
        out[f"total_{name}"] = n
    lat = np.concatenate([np.asarray(r["latency_s"]) for r in rows]) if rows else np.empty(0)
    delay = np.concatenate([np.asarray(r["delay_s"]) for r in rows]) if rows else np.empty(0)
    if lat.size:
        out["latency_p50_ms"] = float(np.percentile(lat, 50) * 1000)
        out["latency_p95_ms"] = float(np.percentile(lat, 95) * 1000)
        out["latency_max_ms"] = float(lat.max() * 1000)
        out["delay_mean_s"] = float(delay.mean())
    return out
