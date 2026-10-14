"""ProfileMonitor: calibration, then one KL score per step over the last window (ARQUITETURA §5)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from cardiac_twin.config import ProfileConfig
from cardiac_twin.profile.profile import PatientProfile, in_window


@dataclasses.dataclass(frozen=True)
class WindowScore:
    t_end: float
    kl: float
    threshold: float
    n_beats: int
    p: dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class MonitorUpdate:
    profile_built: bool
    scores: list[WindowScore]


class ProfileMonitor:
    """Feed beats with `add_beat` and the time up to which all beats are known with `advance`.

    The profile is built when `advance` passes the end of calibration. After that, one window is
    scored every `step_seconds`, the first one ending at the end of calibration. Scores only depend
    on beat times, so the result does not depend on how the signal is split into chunks.
    """

    def __init__(self, cfg: ProfileConfig) -> None:
        self.cfg = cfg
        self.profile: PatientProfile | None = None
        self.latest: WindowScore | None = None
        self._times: list[float] = []
        self._values: dict[str, list[float]] = {f: [] for f in cfg.features}
        self._next_end = float(cfg.calibration_seconds)

    @property
    def calibrating(self) -> bool:
        return self.profile is None

    @property
    def threshold(self) -> float | None:
        return None if self.profile is None else self.profile.threshold(self.cfg.k)

    def add_beat(self, t: float, features: Mapping[str, float]) -> None:
        if self._times and t < self._times[-1]:
            raise ValueError("beats must arrive in time order")
        self._times.append(float(t))
        for f in self.cfg.features:
            self._values[f].append(float(features[f]))

    def advance(self, t_known: float) -> MonitorUpdate:
        """Build the profile and score the windows that end at or before `t_known`."""
        built = False
        scores: list[WindowScore] = []
        while self._next_end <= t_known:
            times = np.asarray(self._times)
            values = {f: np.asarray(v) for f, v in self._values.items()}
            if self.profile is None:
                self.profile = PatientProfile.build(times, values, self.cfg)
                built = True
            mask = in_window(times, self._next_end, self.cfg.window_seconds)
            window = {f: v[mask] for f, v in values.items()}
            p = self.profile.distributions(window)
            score = WindowScore(
                t_end=self._next_end,
                kl=self.profile.score(window),
                threshold=self.profile.threshold(self.cfg.k),
                n_beats=int(mask.sum()),
                p=p,
            )
            scores.append(score)
            self.latest = score
            self._next_end += self.cfg.step_seconds
            self._trim()
        return MonitorUpdate(built, scores)

    def apply_feedback(self, p: Mapping[str, np.ndarray], alpha: float) -> int:
        """Update Q with the window of a discarded alert (Eq. 10); return the new profile version."""
        if self.profile is None:
            raise RuntimeError("no profile during calibration")
        self.profile = self.profile.update(p, alpha)
        return self.profile.version

    def _trim(self) -> None:
        cut = self._next_end - self.cfg.window_seconds
        start = int(np.searchsorted(self._times, cut, side="right"))
        if start:
            self._times = self._times[start:]
            self._values = {f: v[start:] for f, v in self._values.items()}
