"""Decision rules of the TC Quadro 5, applied per classified beat and per window update (ARQUITETURA §6)."""

from __future__ import annotations

from typing import Literal

Level = Literal["normal", "attention", "alert", "critical"]
NORMAL: Level = "normal"
ATTENTION: Level = "attention"
ALERT: Level = "alert"
CRITICAL: Level = "critical"

NORMAL_CLASS = "N"
VENTRICULAR_CLASS = "V"


class VRunCounter:
    """Number of consecutive beats classified as V, the current one included."""

    def __init__(self) -> None:
        self.run = 0

    def update(self, cls: str) -> int:
        self.run = self.run + 1 if cls == VENTRICULAR_CLASS else 0
        return self.run


def decide_beat(
    cls: str,
    v_run: int,
    kl: float | None,
    threshold: float | None,
    *,
    use_profile: bool,
    calibrating: bool,
    vt_consecutive_v: int,
) -> Level:
    """Level of one classified beat.

    A run of `vt_consecutive_v` or more V beats is critical in every case. Otherwise N is normal, and
    during calibration only critical alerts exist. Without the profile (configuration A) every other
    class is an alert. With the profile (B) it is an alert only if the latest KL is above L.
    """
    if cls == VENTRICULAR_CLASS and v_run >= vt_consecutive_v:
        return CRITICAL
    if cls == NORMAL_CLASS or calibrating:
        return NORMAL
    if not use_profile:
        return ALERT
    if kl is None or threshold is None:
        return ATTENTION
    return ALERT if kl > threshold else ATTENTION


def decide_window(has_non_n: bool, kl: float, threshold: float) -> Level:
    """Level of a window update: a deviation with only N beats is a call for attention."""
    return ATTENTION if not has_non_n and kl > threshold else NORMAL
