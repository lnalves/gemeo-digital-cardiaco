"""Training on DS1-train with early stopping by macro-F1 on DS1-val. Never reads the test split (RNF02)."""

from __future__ import annotations

import dataclasses
import subprocess
import time
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from cardiac_twin.config import Config
from cardiac_twin.data.splits import get_splits
from cardiac_twin.evaluation.metrics import classification_report, macro_f1
from cardiac_twin.model.architecture import build_model
from cardiac_twin.model.dataset import (
    BeatArrays,
    gather_sequences,
    load_beats,
    make_dataset,
    sequence_index,
)
from cardiac_twin.model.inference import MODEL_FILE
from cardiac_twin.model.losses import class_weights, make_loss
from cardiac_twin.runs import write_json

TRAIN_FILE = "ds1.npz"


class MacroF1EarlyStopping(keras.callbacks.Callback):
    """Stop when macro-F1 on DS1-val has not improved for `patience` epochs; restore the best weights."""

    def __init__(
        self,
        val_inputs: dict[str, np.ndarray],
        val_labels: np.ndarray,
        classes: tuple[str, ...],
        subset: tuple[str, ...],
        patience: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.val_inputs, self.val_labels = val_inputs, val_labels
        self.classes, self.subset = classes, subset
        self.patience, self.batch_size = patience, batch_size
        self.best = -np.inf
        self.best_epoch = -1
        self._best_weights: list[np.ndarray] | None = None
        self._epoch_start = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        probs = self.model.predict(self.val_inputs, batch_size=self.batch_size, verbose=0)
        f1 = macro_f1(self.val_labels, probs.argmax(axis=1), self.classes, self.subset)
        if logs is not None:
            logs["val_macro_f1"] = f1
            logs["epoch_seconds"] = time.perf_counter() - self._epoch_start
        if f1 > self.best:
            self.best, self.best_epoch = f1, epoch
            self._best_weights = self.model.get_weights()
        elif epoch - self.best_epoch >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs: dict | None = None) -> None:
        if self._best_weights is not None:
            self.model.set_weights(self._best_weights)


def environment() -> dict[str, Any]:
    """TensorFlow build and GPU, for the run record (D-10)."""
    build = tf.sysconfig.get_build_info()
    gpus = []
    for gpu in tf.config.list_physical_devices("GPU"):
        details = tf.config.experimental.get_device_details(gpu)
        cc = details.get("compute_capability")
        gpus.append({"name": details.get("device_name"), "compute_capability": ".".join(map(str, cc or ()))})
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return {
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "cuda": build.get("cuda_version"),
        "cudnn": build.get("cudnn_version"),
        "gpus": gpus,
        "nvidia_driver": driver,
    }


def _sequences(arrays: BeatArrays, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    return sequence_index(arrays.record_id, cfg.model.context_beats, cfg.model.future_beats)


def train(cfg: Config, processed_dir: Path, run_dir: Path, verbose: int = 2) -> dict[str, Any]:
    """Train with `cfg.model.loss`, save model.keras, history.json and metrics.json (DS1-val) in `run_dir`."""
    keras.utils.set_random_seed(cfg.seed)
    m, classes = cfg.model, cfg.data.classes
    splits = get_splits(cfg.data)
    ds1 = load_beats(processed_dir / TRAIN_FILE, m.rr_features, m.beat_channels)
    train_a, val_a = ds1.select(splits.train), ds1.select(splits.val)
    train_c, train_s = _sequences(train_a, cfg)
    val_c, val_s = _sequences(val_a, cfg)
    val_inputs, val_labels = gather_sequences(val_a, val_s), val_a.labels[val_c]

    weights = class_weights(train_a.labels[train_c], len(classes), m.class_weight_cap)
    model = build_model(cfg, train_a.rr.mean(axis=0), train_a.rr.var(axis=0))
    model.compile(optimizer=keras.optimizers.Adam(m.lr), loss=make_loss(m.loss, weights, m.focal_gamma))

    stopper = MacroF1EarlyStopping(
        val_inputs, val_labels, classes, cfg.evaluation.macro_classes, m.patience, m.batch_size
    )
    start = time.perf_counter()
    history = model.fit(
        make_dataset(
            train_a,
            train_c,
            train_s,
            m.batch_size,
            shuffle=True,
            seed=cfg.seed,
            noise_std=m.augment_noise_std,
            baseline_std=m.augment_baseline_std,
        ),
        epochs=m.max_epochs,
        callbacks=[stopper],
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - start
    model.save(run_dir / MODEL_FILE)

    probs = model.predict(val_inputs, batch_size=m.batch_size, verbose=0)
    report = classification_report(
        val_labels, probs.argmax(axis=1), probs, classes, cfg.evaluation.macro_classes
    )
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    write_json(run_dir / "history.json", hist)
    summary = {
        "loss": m.loss,
        "best_epoch": stopper.best_epoch + 1,
        "epochs_run": len(hist["loss"]),
        "best_val_macro_f1": stopper.best,
        "train_seconds": train_seconds,
        "n_train_sequences": int(len(train_c)),
        "n_val_sequences": int(len(val_c)),
        "train_records": list(splits.train),
        "val_records": list(splits.val or ()),
        "class_weights": dict(zip(classes, weights.tolist(), strict=True)),
        "n_params": int(model.count_params()),
        "ds1_val": report,
    }
    write_json(run_dir / "metrics.json", summary)
    return summary


def with_loss(cfg: Config, loss: str) -> Config:
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, loss=loss))
