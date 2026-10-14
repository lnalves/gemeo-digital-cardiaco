"""CNN 1D per beat + BiLSTM over K beats + additive attention (ARQUITETURA §4, D-08)."""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from cardiac_twin.config import Config

ATTENTION_LAYER = "attention"


@keras.saving.register_keras_serializable(package="cardiac_twin")
class AdditiveAttention(layers.Layer):
    """Bahdanau-style attention pooling: e_t = vᵀ tanh(W h_t + b), α = softmax(e), context = Σ α_t h_t.

    Returns (context, α); α has one weight per beat of the sequence and sums to 1.
    """

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape) -> None:
        dim = input_shape[-1]
        self.w = self.add_weight(name="w", shape=(dim, self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.units, 1), initializer="glorot_uniform")

    def call(self, h):
        scores = ops.matmul(ops.tanh(ops.matmul(h, self.w) + self.b), self.v)  # (batch, K, 1)
        alpha = ops.softmax(scores, axis=1)
        context = ops.sum(alpha * h, axis=1)
        return context, ops.squeeze(alpha, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])

    def get_config(self) -> dict:
        return {**super().get_config(), "units": self.units}


def _beat_encoder(cfg: Config) -> keras.Sequential:
    m = cfg.model
    n = cfg.segment.pre_samples + cfg.segment.post_samples
    blocks: list[layers.Layer] = [layers.Input((n, 1))]
    for i, (filters, kernel) in enumerate(zip(m.cnn_filters, m.cnn_kernels, strict=True)):
        blocks += [
            layers.Conv1D(filters, kernel, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]
        last = i == len(m.cnn_filters) - 1
        blocks.append(layers.GlobalAveragePooling1D() if last else layers.MaxPooling1D(2))
    return keras.Sequential(blocks, name="cnn_encoder")


def build_model(cfg: Config, rr_mean: np.ndarray, rr_variance: np.ndarray) -> keras.Model:
    """Training model: inputs {beats (K, 256, 1), rr (K, 5)} → class probabilities.

    `rr_mean`/`rr_variance` come from DS1-train and are stored in the model (no external state).
    """
    m = cfg.model
    k = m.context_beats
    n = cfg.segment.pre_samples + cfg.segment.post_samples
    beats = layers.Input((k, n, 1), name="beats")
    rr = layers.Input((k, len(m.rr_features)), name="rr")

    z = layers.TimeDistributed(_beat_encoder(cfg), name="beat_encoder")(beats)
    r = layers.Normalization(mean=rr_mean, variance=rr_variance, name="rr_norm")(rr)
    h = layers.Concatenate(name="beat_and_rr")([z, r])
    h = layers.Bidirectional(layers.LSTM(m.lstm_units, return_sequences=True), name="bilstm")(h)
    context, _ = AdditiveAttention(m.attention_units, name=ATTENTION_LAYER)(h)
    x = layers.Dense(m.dense_units, activation="relu", name="dense")(context)
    x = layers.Dropout(m.dropout, name="dropout")(x)
    probs = layers.Dense(len(cfg.data.classes), activation="softmax", name="probs")(x)
    return keras.Model({"beats": beats, "rr": rr}, probs, name="cardiac_twin_classifier")


def build_inference_model(model: keras.Model) -> keras.Model:
    """Same weights, also exposing the attention weights over the K beats (RNF03)."""
    attention = model.get_layer(ATTENTION_LAYER).output[1]
    return keras.Model(model.input, [model.output, attention], name="cardiac_twin_inference")
