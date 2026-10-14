"""Streaming R-peak detector (simplified Pan-Tompkins) and EC57 beat matching (D-03).

Input is the band-passed signal (`CausalBandpass`). Pipeline: 5-point derivative → square → moving-window
integration (MWI), all causal with state. A sample is an MWI candidate peak when it is the largest value
within ±refractory; it is decided once `refractory` later samples have arrived, so the output never depends
on how the signal was split into chunks.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from cardiac_twin.config import RPeaksConfig

# Pan-Tompkins 5-point derivative: y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8 (delay of 2 samples)
_DERIVATIVE = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / 8.0
_DERIVATIVE_DELAY = 2


@dataclasses.dataclass
class _Candidate:
    mwi_index: int
    value: float
    r_sample: int
    slope: float


class StreamingRPeakDetector:
    """`process(chunk)` returns the R samples (global indices, increasing) confirmed by that chunk."""

    def __init__(self, fs: float, cfg: RPeaksConfig) -> None:
        self.cfg = cfg
        self.n_mwi = max(1, round(cfg.mwi_ms * fs / 1000))
        self.refractory = round(cfg.refractory_ms * fs / 1000)
        self.n_learn = round(cfg.learning_s * fs)
        self.twave = round(cfg.twave_ms * fs / 1000)
        self._mwi_kernel = np.full(self.n_mwi, 1.0 / self.n_mwi)
        self._zi_der = np.zeros(len(_DERIVATIVE) - 1)
        self._zi_mwi = np.zeros(self.n_mwi - 1)
        # buffers hold global samples [_offset, _n)
        self._x = np.empty(0)
        self._der = np.empty(0)
        self._mwi = np.empty(0)
        self._offset = 0
        self._n = 0
        self._next = 0  # first MWI index not examined yet
        self._initialized = False
        self.spk = 0.0
        self.npk = 0.0
        self._rr: deque[int] = deque(maxlen=cfg.rr_avg_beats)
        self._last: _Candidate | None = None
        self._noise: list[_Candidate] = []

    @property
    def threshold(self) -> float:
        return self.npk + self.cfg.threshold_frac * (self.spk - self.npk)

    def process(self, chunk: np.ndarray) -> list[int]:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.size == 0:
            return []
        der, self._zi_der = lfilter(_DERIVATIVE, 1.0, chunk, zi=self._zi_der)
        mwi, self._zi_mwi = lfilter(self._mwi_kernel, 1.0, der**2, zi=self._zi_mwi)
        self._x = np.concatenate([self._x, chunk])
        self._der = np.concatenate([self._der, der])
        self._mwi = np.concatenate([self._mwi, mwi])
        self._n += chunk.size

        if not self._initialized:
            if self._n < self.n_learn:
                return []
            learn = self._mwi[: self.n_learn]  # buffer is never trimmed before this point
            self.spk, self.npk = float(learn.max()), float(learn.mean())
            self._initialized = True

        out: list[int] = []
        last = self._n - 1 - self.refractory  # last index with `refractory` samples after it
        if last >= self._next:
            for p in self._candidates(self._next, last):
                out.extend(self._decide(p))
            self._next = last + 1
        self._trim()
        return out

    def _candidates(self, first: int, last: int) -> list[int]:
        """MWI indices in [first, last] above the `refractory` previous samples and ≥ the next ones."""
        r = self.refractory
        lo = first - r
        seg = self._mwi[max(lo, self._offset) - self._offset : last + r + 1 - self._offset]
        if lo < self._offset:
            seg = np.concatenate([np.full(self._offset - lo, -np.inf), seg])
        if r == 0:
            return list(range(first, last + 1))
        n = last - first + 1
        win = sliding_window_view(seg, r).max(axis=1)
        center = seg[r : r + n]
        is_peak = (center > win[:n]) & (center >= win[r + 1 : r + 1 + n]) & (center > 0)
        return (first + np.flatnonzero(is_peak)).tolist()

    def _candidate(self, p: int) -> _Candidate:
        lo = max(p - self.n_mwi - _DERIVATIVE_DELAY, self._offset) - self._offset
        hi = p + 1 - self._offset
        r_sample = self._offset + lo + int(np.argmax(np.abs(self._x[lo:hi])))
        slope = float(np.max(np.abs(self._der[lo:hi])))
        return _Candidate(p, float(self._mwi[p - self._offset]), r_sample, slope)

    def _decide(self, p: int) -> list[int]:
        cand = self._candidate(p)
        out = self._searchback(p)
        if cand.value > self.threshold and not self._is_refractory_or_twave(cand):
            self._accept(cand, self.cfg.peak_ema)
            out.append(cand.r_sample)
        else:
            self.npk += self.cfg.peak_ema * (cand.value - self.npk)
            if cand.value <= self.threshold:
                self._noise.append(cand)
        return out

    def _is_refractory_or_twave(self, cand: _Candidate) -> bool:
        if self._last is None:
            return False
        gap = cand.mwi_index - self._last.mwi_index
        if gap < self.refractory:
            return True
        return gap < self.twave and cand.slope < self.cfg.twave_slope_frac * self._last.slope

    def _searchback(self, p: int) -> list[int]:
        """While no QRS for `searchback_rr_factor` × mean RR, accept the largest past noise peak
        above `searchback_threshold_scale` × threshold."""
        out: list[int] = []
        while self._last is not None and self._rr:
            if p - self._last.mwi_index <= self.cfg.searchback_rr_factor * np.mean(self._rr):
                break
            limit = self.cfg.searchback_threshold_scale * self.threshold
            pool = [
                c
                for c in self._noise
                if c.value > limit and c.mwi_index - self._last.mwi_index >= self.refractory
            ]
            if not pool:
                break
            best = max(pool, key=lambda c: c.value)
            self._accept(best, self.cfg.searchback_peak_ema)
            out.append(best.r_sample)
        return out

    def _accept(self, cand: _Candidate, ema: float) -> None:
        self.spk += ema * (cand.value - self.spk)
        if self._last is not None:
            self._rr.append(cand.mwi_index - self._last.mwi_index)
        self._last = cand
        self._noise = [c for c in self._noise if c.mwi_index > cand.mwi_index]

    def _trim(self) -> None:
        if not self._initialized:
            return
        keep_from = max(self._offset, self._next - self.refractory - self.n_mwi - _DERIVATIVE_DELAY)
        cut = keep_from - self._offset
        if cut > 0:
            self._x, self._der, self._mwi = self._x[cut:], self._der[cut:], self._mwi[cut:]
            self._offset = keep_from


def detect_r_peaks(x: np.ndarray, fs: float, cfg: RPeaksConfig) -> np.ndarray:
    """Run a fresh detector over a whole (band-passed) signal."""
    return np.asarray(StreamingRPeakDetector(fs, cfg).process(x), dtype=np.int64)


@dataclasses.dataclass(frozen=True)
class PeakMatch:
    det_index: np.ndarray  # matched pairs (indices into det / ann)
    ann_index: np.ndarray
    false_positive: np.ndarray  # detections without an annotation
    false_negative: np.ndarray  # annotations without a detection

    @property
    def sensitivity(self) -> float:
        tp = len(self.ann_index)
        return tp / (tp + len(self.false_negative)) if tp + len(self.false_negative) else float("nan")

    @property
    def ppv(self) -> float:
        tp = len(self.det_index)
        return tp / (tp + len(self.false_positive)) if tp + len(self.false_positive) else float("nan")


def match_peaks(det: np.ndarray, ann: np.ndarray, tol: int) -> PeakMatch:
    """One-to-one matching within ±`tol` samples, each annotation taking the closest free detection (EC57)."""
    det = np.asarray(det, dtype=np.int64)
    ann = np.asarray(ann, dtype=np.int64)
    used = np.zeros(len(det), dtype=bool)
    pairs: list[tuple[int, int]] = []
    for j, a in enumerate(ann):
        lo = np.searchsorted(det, a - tol, side="left")
        hi = np.searchsorted(det, a + tol, side="right")
        free = [i for i in range(lo, hi) if not used[i]]
        if free:
            i = min(free, key=lambda i: abs(det[i] - a))
            used[i] = True
            pairs.append((i, j))
    det_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    ann_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    return PeakMatch(
        det_index=det_idx,
        ann_index=ann_idx,
        false_positive=np.flatnonzero(~used),
        false_negative=np.setdiff1d(np.arange(len(ann)), ann_idx),
    )
