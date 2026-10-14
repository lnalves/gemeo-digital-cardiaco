"""Per-beat evaluation of a trained classifier on DS2 (annotated R peaks, ARQUITETURA §11.1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from cardiac_twin.config import Config
from cardiac_twin.evaluation.metrics import classification_report, save_confusion_matrix
from cardiac_twin.model.dataset import gather_sequences, load_beats, sequence_index
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.runs import write_json

TEST_FILE = "ds2.npz"


def evaluate_classifier(cfg: Config, model_run: Path, processed_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Classify every DS2 beat with a full K-beat context; save metrics, confusion matrix and predictions."""
    with np.load(processed_dir / TEST_FILE) as z:
        r_sample = z["r_sample"]
    arrays = load_beats(processed_dir / TEST_FILE, cfg.model.rr_features, cfg.model.beat_channels)
    centers, seqs = sequence_index(arrays.record_id, cfg.model.context_beats, cfg.model.future_beats)
    inputs = gather_sequences(arrays, seqs)
    clf = BeatClassifier.load(model_run, cfg.data.classes, cfg.model.batch_size)

    start = time.perf_counter()
    pred, probs, attention = clf.predict(inputs["beats"], inputs["rr"])
    seconds = time.perf_counter() - start
    y = arrays.labels[centers]

    classes = cfg.data.classes
    report = classification_report(y, pred, probs, classes, cfg.evaluation.macro_classes)
    save_confusion_matrix(
        np.array(report["confusion_matrix"]),
        classes,
        out_dir / "confusion_matrix.csv",
        out_dir / "confusion_matrix.png",
    )
    np.savez_compressed(
        out_dir / "predictions.npz",
        record_id=arrays.record_id[centers],
        r_sample=r_sample[centers],
        label=y,
        pred=pred,
        probs=probs.astype(np.float32),
        attention=attention.astype(np.float32),
        classes=np.array(classes),
    )
    summary = {
        "model_run": model_run.name,
        "split": "ds2",
        "n_beats_in_split": int(len(arrays.labels)),
        "n_beats_without_context": int(len(arrays.labels) - len(centers)),
        "inference_seconds": seconds,
        "inference_ms_per_beat": 1000 * seconds / max(len(centers), 1),
        "ds2": report,
    }
    write_json(out_dir / "metrics.json", summary)
    return summary
