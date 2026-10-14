"""Choice of k and window on DS1-val (D-09): fewest false alarms with at most a small loss of S+V alerts."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cardiac_twin.config import Config
from cardiac_twin.evaluation.replay import replay_record
from cardiac_twin.evaluation.system import aggregate, system_metrics
from cardiac_twin.model.inference import BeatClassifier


def with_profile(cfg: Config, k: float, window_seconds: int) -> Config:
    return dataclasses.replace(
        cfg, profile=dataclasses.replace(cfg.profile, k=k, window_seconds=window_seconds)
    )


def tune_profile(
    cfg: Config, classifier: BeatClassifier, records: Sequence[str], raw_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay every record with the classifier alone (A) and with the profile (B) for each grid point."""
    skip = cfg.profile.calibration_seconds
    classes = cfg.data.classes
    source = cfg.engine.rpeak_source

    def run(c: Config, use_profile: bool) -> dict[str, Any]:
        rows = [
            system_metrics(
                replay_record(c, classifier, r, raw_dir, use_profile=use_profile, rpeak_source=source),
                classes,
                skip,
            )
            for r in records
        ]
        return aggregate(rows)

    a = run(cfg, use_profile=False)
    grid = cfg.profile.tuning_grid
    table = []
    for window, k in itertools.product(grid.window_seconds, grid.k):
        b = run(with_profile(cfg, k, window), use_profile=True)
        drop = 100 * (a["sensitivity_sv"] - b["sensitivity_sv"])
        table.append(
            {
                "k": k,
                "window_seconds": window,
                "fp_a": a["fp"],
                "fp_b": b["fp"],
                "fp_reduction": 1 - b["fp"] / a["fp"] if a["fp"] else float("nan"),
                "sensitivity_sv_a": a["sensitivity_sv"],
                "sensitivity_sv_b": b["sensitivity_sv"],
                "sensitivity_drop_pp": drop,
                "feasible": drop <= cfg.profile.max_sensitivity_drop_pp,
            }
        )
    feasible = [row for row in table if row["feasible"]]
    best = min(feasible, key=lambda row: (row["fp_b"], -row["sensitivity_sv_b"])) if feasible else None
    return {"choice": best, "classifier_alone": a}, table
