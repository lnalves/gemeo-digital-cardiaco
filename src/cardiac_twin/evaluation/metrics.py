"""Per-beat classification metrics (TC Eq. 4–8), macro averages, AUC one-vs-rest, confusion matrix."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def per_class_metrics(cm: np.ndarray, classes: Sequence[str]) -> dict[str, dict[str, float]]:
    """One-vs-rest accuracy, precision, sensitivity, specificity and F1 for each class.

    A class without true beats gets NaN sensitivity/F1 (left out of macro averages); a class
    never predicted gets precision 0.
    """
    total = cm.sum()
    out: dict[str, dict[str, float]] = {}
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        sensitivity = tp / support if support else float("nan")
        if not support:
            f1 = float("nan")
        else:
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
        out[cls] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": (tp + tn) / total if total else float("nan"),
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": f1,
        }
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], subset: Sequence[str]) -> float:
    """Mean F1 over `subset` (classes without true beats are ignored)."""
    per = per_class_metrics(confusion_matrix(y_true, y_pred, len(classes)), classes)
    return float(np.nanmean([per[c]["f1"] for c in subset]))


def auc_ovr(y_true: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    """Area under the ROC curve of each class against the rest (NaN if the class is absent)."""
    y_true = np.asarray(y_true)
    out = {}
    for i, cls in enumerate(classes):
        positive = y_true == i
        out[cls] = (
            float(roc_auc_score(positive, probs[:, i])) if 0 < positive.sum() < len(y_true) else float("nan")
        )
    return out


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str],
    macro_classes: Sequence[str],
) -> dict[str, Any]:
    """Everything reported for the classifier: per class, macro over all classes and over `macro_classes`."""
    cm = confusion_matrix(y_true, y_pred, len(classes))
    per = per_class_metrics(cm, classes)
    auc = auc_ovr(y_true, probs, classes)
    for cls in classes:
        per[cls]["auc"] = auc[cls]
    keys = ("precision", "sensitivity", "specificity", "f1", "auc")

    def macro(subset: Sequence[str]) -> dict[str, float]:
        present = [c for c in subset if per[c]["support"] > 0]
        return {k: float(np.nanmean([per[c][k] for c in present])) for k in keys}

    return {
        "n_beats": int(cm.sum()),
        "accuracy": float(np.trace(cm) / cm.sum()),
        "per_class": per,
        "macro_all": macro(classes),
        "macro_selected": {"classes": list(macro_classes), **macro(macro_classes)},
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm: np.ndarray, classes: Sequence[str], csv_path: Path, png_path: Path) -> None:
    """CSV with counts and PNG with counts and row-normalized colors (sensitivity on the diagonal)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    header = "true\\pred," + ",".join(classes)
    rows = [f"{c}," + ",".join(str(v) for v in cm[i]) for i, c in enumerate(classes)]
    csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
