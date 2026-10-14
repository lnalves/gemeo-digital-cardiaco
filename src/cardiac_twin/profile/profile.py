"""PatientProfile: reference distributions Q_f, threshold L = μ + kσ (Eq. 11) and update (Eq. 10)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from cardiac_twin.config import ProfileConfig
from cardiac_twin.profile.histogram import calibration_edges, histogram
from cardiac_twin.profile.kl import kl_sum


def in_window(times: np.ndarray, end: float, length: float) -> np.ndarray:
    """Mask of the beats with time in (end - length, end]. Used for calibration and monitoring alike."""
    t = np.asarray(times, dtype=np.float64)
    return (t > end - length) & (t <= end)


def window_ends(cfg: ProfileConfig) -> list[float]:
    """Ends of the calibration subwindows: same length and step as the monitoring windows (D-07)."""
    first = cfg.window_seconds
    return [float(e) for e in range(first, cfg.calibration_seconds + 1, cfg.step_seconds)]


@dataclasses.dataclass(frozen=True)
class PatientProfile:
    edges: dict[str, np.ndarray]
    q: dict[str, np.ndarray]
    mu: float
    sigma: float
    eps: float
    calibration_scores: tuple[float, ...]
    version: int = 1

    @classmethod
    def build(
        cls, times: np.ndarray, features: Mapping[str, np.ndarray], cfg: ProfileConfig
    ) -> PatientProfile:
        """Build Q_f from all calibration beats, and μ, σ from the D of its subwindows (D-23).

        Each subwindow is scored against Q built from the calibration without that subwindow
        (leave-one-out). Against the full Q its own beats always have mass, so D is too low and L
        would sit below the D of new, unseen windows.
        """
        values = {f: np.asarray(features[f]) for f in cfg.features}
        calib = in_window(times, cfg.calibration_seconds, cfg.calibration_seconds)
        edges = {f: calibration_edges(v[calib], cfg.n_bins, cfg.range_percentiles) for f, v in values.items()}
        q = {f: histogram(v[calib], edges[f], cfg.eps) for f, v in values.items()}
        base = cls(edges, q, mu=0.0, sigma=0.0, eps=cfg.eps, calibration_scores=())
        scores = []
        for end in window_ends(cfg):
            sub = in_window(times, end, cfg.window_seconds)
            q_out = {f: histogram(v[calib & ~sub], edges[f], cfg.eps) for f, v in values.items()}
            p_sub = base.distributions({f: v[sub] for f, v in values.items()})
            scores.append(kl_sum(p_sub, q_out))
        s = np.array(scores)
        sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        return dataclasses.replace(base, mu=float(s.mean()), sigma=sigma, calibration_scores=tuple(scores))

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.q)

    def threshold(self, k: float) -> float:
        return self.mu + k * self.sigma

    def distributions(self, window: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """P_f of a window, on the bins of the profile."""
        return {f: histogram(np.asarray(window[f]), self.edges[f], self.eps) for f in self.features}

    def score(self, window: Mapping[str, np.ndarray]) -> float:
        """D = Σ_f D_KL(P_f ‖ Q_f) of a window."""
        return kl_sum(self.distributions(window), self.q)

    def update(self, p: Mapping[str, np.ndarray], alpha: float) -> PatientProfile:
        """Eq. 10: Q_f ← (1 − α) Q_f + α P_f, for a discarded alert. The threshold is kept."""
        q = {f: (1 - alpha) * self.q[f] + alpha * np.asarray(p[f]) for f in self.features}
        return dataclasses.replace(self, q=q, version=self.version + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": {f: e.tolist() for f, e in self.edges.items()},
            "q": {f: v.tolist() for f, v in self.q.items()},
            "mu": self.mu,
            "sigma": self.sigma,
            "eps": self.eps,
            "calibration_scores": list(self.calibration_scores),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PatientProfile:
        return cls(
            edges={f: np.asarray(e) for f, e in d["edges"].items()},
            q={f: np.asarray(v) for f, v in d["q"].items()},
            mu=float(d["mu"]),
            sigma=float(d["sigma"]),
            eps=float(d["eps"]),
            calibration_scores=tuple(d["calibration_scores"]),
            version=int(d["version"]),
        )
