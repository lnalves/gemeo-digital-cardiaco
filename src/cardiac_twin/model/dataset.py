"""Beat arrays (data/processed/*.npz) → K-beat sequences → tf.data."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf

from cardiac_twin.signal.segmentation import context_indices


@dataclasses.dataclass(frozen=True)
class BeatArrays:
    beats: np.ndarray  # (n, 256) float32
    rr: np.ndarray  # (n, 5) float32
    labels: np.ndarray  # (n,) int8
    record_id: np.ndarray  # (n,) str, contiguous per record, beats in time order

    def select(self, records: Sequence[str]) -> BeatArrays:
        mask = np.isin(self.record_id, list(records))
        return BeatArrays(self.beats[mask], self.rr[mask], self.labels[mask], self.record_id[mask])


def load_beats(path: Path) -> BeatArrays:
    with np.load(path) as z:
        return BeatArrays(z["beats"], z["rr_feats"], z["labels"], z["record_id"])


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
    return {"beats": arrays.beats[seqs][..., None], "rr": arrays.rr[seqs]}


def make_dataset(
    arrays: BeatArrays,
    centers: np.ndarray,
    seqs: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Yield ({beats (B, K, 256, 1), rr (B, K, 5)}, label of the center beat). Shuffle only for training."""
    beats = tf.constant(arrays.beats)
    rr = tf.constant(arrays.rr)
    labels = arrays.labels[centers].astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((seqs.astype(np.int32), labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda s, y: ({"beats": tf.gather(beats, s)[..., None], "rr": tf.gather(rr, s)}, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
