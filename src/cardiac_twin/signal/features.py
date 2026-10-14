"""Per-beat features: RR intervals, QRS width and R amplitude (ARQUITETURA §3)."""

from __future__ import annotations

import numpy as np

from cardiac_twin.config import Config

RR_FEATURE_NAMES = ("rr_pre", "rr_post", "rr_local", "rr_pre_ratio", "rr_post_ratio")


def beat_features(x: np.ndarray, r_samples: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """All per-beat features, from the filtered signal and the refined R samples.

    Single entry point for offline (data.builder) and streaming (engine) use.
    `rr` feeds the classifier; `rr_pre`, `qrs_width_ms` and `r_amplitude` feed the profile.
    """
    fs, f, s = cfg.data.fs, cfg.features, cfg.segment
    rr = rr_features(r_samples, fs, f.local_rr_beats)
    return {
        "rr": rr,
        "rr_pre": rr[:, 0],
        "qrs_width_ms": qrs_width_ms(x, r_samples, fs, f.qrs_search_ms, f.qrs_slope_frac),
        "r_amplitude": r_amplitude(x, r_samples, s.pre_samples, s.post_samples),
    }


def rr_features(r_samples: np.ndarray, fs: float, local_beats: int) -> np.ndarray:
    """Return (n, 5) RR features in seconds: pre, post, local mean, pre/local, post/local.

    `rr_local` is the mean of the last `local_beats` intervals up to and including `rr_pre`
    (causal; fewer at the start of the record). Undefined values (first/last beat) are NaN.
    """
    r = np.asarray(r_samples, dtype=np.float64)
    n = len(r)
    out = np.full((n, len(RR_FEATURE_NAMES)), np.nan)
    if n < 2:
        return out
    d = np.diff(r) / fs  # d[j] = interval between beats j and j+1
    out[1:, 0] = d
    out[:-1, 1] = d
    csum = np.concatenate([[0.0], np.cumsum(d)])
    i = np.arange(1, n)  # beats with rr_pre = d[i-1]
    start = np.maximum(0, i - local_beats)
    out[1:, 2] = (csum[i] - csum[start]) / (i - start)
    out[:, 3] = out[:, 0] / out[:, 2]
    out[:, 4] = out[:, 1] / out[:, 2]
    return out


def qrs_width_ms(
    x: np.ndarray, r_samples: np.ndarray, fs: float, search_ms: float, slope_frac: float
) -> np.ndarray:
    """Estimate the QRS width of each beat from the slope of the filtered signal (D-04).

    In [R - search, R + search], the threshold is `slope_frac` × the largest |first difference|;
    onset/offset are the first/last samples whose slope reaches it. NaN if the window leaves the signal.
    """
    x = np.asarray(x, dtype=np.float64)
    half = int(round(search_ms * fs / 1000))
    out = np.full(len(r_samples), np.nan)
    for k, r in enumerate(np.asarray(r_samples, dtype=np.int64)):
        lo, hi = r - half, r + half
        if lo < 0 or hi >= len(x):
            continue
        slope = np.abs(np.diff(x[lo : hi + 1]))
        if slope.max() == 0:
            continue
        steep = np.flatnonzero(slope >= slope_frac * slope.max())
        out[k] = (steep[-1] + 1 - steep[0]) * 1000 / fs
    return out


def r_amplitude(x: np.ndarray, r_samples: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Filtered signal at R minus the median of the beat window [R - pre, R + post) (local baseline)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(r_samples), np.nan)
    for k, r in enumerate(np.asarray(r_samples, dtype=np.int64)):
        if r - pre >= 0 and r + post <= len(x):
            out[k] = x[r] - np.median(x[r - pre : r + post])
    return out
