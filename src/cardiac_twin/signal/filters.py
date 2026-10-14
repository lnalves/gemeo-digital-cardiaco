"""Causal Butterworth band-pass, identical in training and streaming (D-02)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from cardiac_twin.config import Config


class CausalBandpass:
    """Band-pass with persistent state: `process` chunk by chunk ≡ `process_full` on the whole signal.

    The state starts at the steady state for the first sample, which removes the start-up transient
    of the baseline offset. Never replace by `filtfilt`: zero-phase filtering cannot run in streaming.
    """

    def __init__(self, fs: float, band_hz: Sequence[float], order: int) -> None:
        self.sos = butter(order, band_hz, btype="bandpass", fs=fs, output="sos")
        self._zi: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> CausalBandpass:
        return cls(cfg.data.fs, cfg.signal.bandpass_hz, cfg.signal.filter_order)

    def reset(self) -> None:
        self._zi = None

    def _initial_state(self, first_sample: float) -> np.ndarray:
        return sosfilt_zi(self.sos) * first_sample

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter the next chunk, continuing from the state left by the previous one."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.size == 0:
            return chunk.copy()
        if self._zi is None:
            self._zi = self._initial_state(chunk[0])
        y, self._zi = sosfilt(self.sos, chunk, zi=self._zi)
        return y

    def process_full(self, x: np.ndarray) -> np.ndarray:
        """Filter a whole signal from a fresh state; does not touch the streaming state."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x.copy()
        y, _ = sosfilt(self.sos, x, zi=self._initial_state(x[0]))
        return y
