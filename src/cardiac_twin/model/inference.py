"""BeatClassifier: the only entry point to the model for the rest of the system."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from cardiac_twin.model.architecture import build_inference_model

MODEL_FILE = "model.keras"


class BeatClassifier:
    """Classify K-beat sequences: class of the center beat, its probabilities and the attention."""

    def __init__(self, model: keras.Model, classes: tuple[str, ...], batch_size: int) -> None:
        self._model = build_inference_model(model)
        self.classes = classes
        self.batch_size = batch_size
        k, n, channels = model.input["beats"].shape[1:]
        n_rr = model.input["rr"].shape[-1]
        self._call = tf.function(
            lambda beats, rr: self._model({"beats": beats, "rr": rr}, training=False),
            input_signature=[
                tf.TensorSpec((None, k, n, channels), tf.float32),
                tf.TensorSpec((None, k, n_rr), tf.float32),
            ],
        )

    @classmethod
    def load(cls, run_dir: Path, classes: tuple[str, ...], batch_size: int) -> BeatClassifier:
        return cls(keras.models.load_model(run_dir / MODEL_FILE, compile=False), classes, batch_size)

    def predict(self, beats: np.ndarray, rr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """beats (m, K, 256[, channels]), rr (m, K, n_rr) → (class (m,), probs (m, C), attention (m, K)).

        The model runs as a compiled graph: one call per chunk is cheap in streaming.
        """
        beats = np.asarray(beats, dtype=np.float32)
        if beats.ndim == 3:
            beats = beats[..., None]
        rr = np.asarray(rr, dtype=np.float32)
        probs, attn = [], []
        for i in range(0, len(beats), self.batch_size):
            p, a = self._call(beats[i : i + self.batch_size], rr[i : i + self.batch_size])
            probs.append(np.asarray(p))
            attn.append(np.asarray(a))
        if not probs:
            k = beats.shape[1]
            return np.empty(0, np.int64), np.empty((0, len(self.classes))), np.empty((0, k))
        p = np.concatenate(probs)
        return p.argmax(axis=1), p, np.concatenate(attn)
