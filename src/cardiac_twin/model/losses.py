"""Class weights and the class-weighted focal loss (γ = 0 gives the weighted cross-entropy)."""

from __future__ import annotations

import keras
import numpy as np
from keras import ops


def class_weights(labels: np.ndarray, n_classes: int, cap: float) -> np.ndarray:
    """Inverse-frequency weights n / (n_classes · n_c), capped at `cap` (absent classes get the cap)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    with np.errstate(divide="ignore"):
        w = len(labels) / (n_classes * counts)
    return np.minimum(np.where(counts > 0, w, cap), cap).astype(np.float32)


@keras.saving.register_keras_serializable(package="cardiac_twin")
class WeightedFocalLoss(keras.losses.Loss):
    """-w_y · (1 - p_y)^γ · log p_y for sparse integer labels (Lin et al., 2017, with α = class weights)."""

    def __init__(self, weights: list[float], gamma: float, name: str = "weighted_focal", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.weights = [float(w) for w in weights]
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y = ops.cast(ops.reshape(y_true, (-1,)), "int32")
        eps = keras.config.epsilon()
        p = ops.clip(y_pred, eps, 1.0 - eps)
        p_y = ops.take_along_axis(p, ops.expand_dims(y, -1), axis=-1)[:, 0]
        w = ops.take(ops.convert_to_tensor(self.weights, dtype=p.dtype), y)
        return -w * ops.power(1.0 - p_y, self.gamma) * ops.log(p_y)

    def get_config(self) -> dict:
        return {**super().get_config(), "weights": self.weights, "gamma": self.gamma}


def make_loss(name: str, weights: np.ndarray, focal_gamma: float) -> WeightedFocalLoss:
    """`weighted_ce` or `focal`; both use the same capped class weights, so only γ differs (D-18)."""
    if name == "weighted_ce":
        return WeightedFocalLoss(weights.tolist(), gamma=0.0, name="weighted_ce")
    if name == "focal":
        return WeightedFocalLoss(weights.tolist(), gamma=focal_gamma, name="focal")
    raise ValueError(f"unknown loss {name!r}")
