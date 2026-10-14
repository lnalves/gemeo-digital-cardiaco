"""Beat arrays (data/processed/*.npz) → K-beat sequences → tf.data."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf

from cardiac_twin.signal.features import rr_columns
from cardiac_twin.signal.segmentation import context_indices


@dataclasses.dataclass(frozen=True)
class BeatArrays:
    beats: np.ndarray  # (n, 256, channels) float32
    rr: np.ndarray  # (n, n_rr) float32
    labels: np.ndarray  # (n,) int8
    record_id: np.ndarray  # (n,) str, contiguous per record, beats in time order

    def select(self, records: Sequence[str]) -> BeatArrays:
        mask = np.isin(self.record_id, list(records))
        return BeatArrays(self.beats[mask], self.rr[mask], self.labels[mask], self.record_id[mask])


def load_beats(path: Path, rr_features: Sequence[str], channels: Sequence[str]) -> BeatArrays:
    """Load a split with the beat channels and RR feature columns the model uses."""
    with np.load(path) as z:
        beats = np.stack([z[c] for c in channels], axis=-1)
        rr = z["rr_feats"][:, rr_columns(rr_features)]
        return BeatArrays(beats, rr, z["labels"], z["record_id"])


def sequence_index(record_id: np.ndarray, context: int, future: int) -> tuple[np.ndarray, np.ndarray]:
    """(centers, sequences) of global beat indices; a sequence never crosses two records."""
    centers, seqs = [], []
    boundaries = np.flatnonzero(record_id[1:] != record_id[:-1]) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(record_id)], strict=True):
        c, s = context_indices(stop - start, context, future)
        centers.append(c + start)
        seqs.append(s + start)
    return np.concatenate(centers), np.concatenate(seqs)


def gather_sequences(arrays: BeatArrays, seqs: np.ndarray) -> dict[str, np.ndarray]:
    """Materialize model inputs for the given sequences (use for evaluation; training gathers lazily)."""
    return {"beats": arrays.beats[seqs], "rr": arrays.rr[seqs]}


def make_dataset(
    arrays: BeatArrays,
    centers: np.ndarray,
    seqs: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    noise_std: float = 0.0,
    baseline_std: float = 0.0,
) -> tf.data.Dataset:
    """Yield ({beats (B, K, 256, channels), rr (B, K, n_rr)}, label of the center beat).

    Shuffle and augmentation are for training only. Augmentation adds, to each beat, Gaussian
    noise and a random offset plus linear ramp (baseline drift).
    """
    beats = tf.constant(arrays.beats)
    rr = tf.constant(arrays.rr)
    labels = arrays.labels[centers].astype(np.int32)
    ramp = tf.linspace(-1.0, 1.0, arrays.beats.shape[1])[:, None]

    def to_inputs(s, y):
        x = tf.gather(beats, s)
        if noise_std > 0:
            x += tf.random.normal(tf.shape(x), stddev=noise_std)
        if baseline_std > 0:
            shape = tf.concat([tf.shape(x)[:2], [1, 1]], axis=0)
            offset = tf.random.normal(shape, stddev=baseline_std)
            slope = tf.random.normal(shape, stddev=baseline_std)
            x += offset + slope * ramp
        return {"beats": x, "rr": tf.gather(rr, s)}, y

    ds = tf.data.Dataset.from_tensor_slices((seqs.astype(np.int32), labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(to_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
