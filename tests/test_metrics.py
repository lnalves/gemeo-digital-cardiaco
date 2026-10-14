import numpy as np
import pytest

from cardiac_twin.evaluation.metrics import (
    auc_ovr,
    classification_report,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    save_confusion_matrix,
)

CLASSES = ("N", "S", "V", "F", "Q")


def test_confusion_matrix_rows_true_cols_pred():
    cm = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), 5)
    assert cm[0].tolist() == [1, 1, 0, 0, 0]
    assert cm[2, 0] == 1 and cm.sum() == 4


def test_per_class_metrics_by_hand():
    # N: 8 true (7 ok, 1 → S); S: 2 true (1 ok, 1 → N)
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.array([0] * 7 + [1] + [1, 0])
    m = per_class_metrics(confusion_matrix(y_true, y_pred, 5), CLASSES)
    n, s = m["N"], m["S"]
    assert (n["tp"], n["fp"], n["fn"], n["tn"]) == (7, 1, 1, 1)
    assert n["sensitivity"] == pytest.approx(7 / 8) and n["precision"] == pytest.approx(7 / 8)
    assert n["specificity"] == pytest.approx(1 / 2)
    assert s["sensitivity"] == pytest.approx(0.5) and s["precision"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(0.5) and s["accuracy"] == pytest.approx(0.8)
    assert np.isnan(m["V"]["sensitivity"]) and np.isnan(m["V"]["f1"])  # no true V


def test_macro_f1_ignores_absent_classes():
    y = np.array([0, 0, 1, 1])
    assert macro_f1(y, y, CLASSES, ["N", "S", "V", "F"]) == pytest.approx(1.0)
    assert macro_f1(y, np.zeros(4), CLASSES, ["N", "S"]) == pytest.approx((2 / 3 + 0) / 2)


def test_auc_ovr():
    y = np.array([0, 0, 1, 1])
    probs = np.zeros((4, 5))
    probs[:, 1] = [0.1, 0.2, 0.8, 0.9]
    probs[:, 0] = 1 - probs[:, 1]
    auc = auc_ovr(y, probs, CLASSES)
    assert auc["S"] == pytest.approx(1.0) and auc["N"] == pytest.approx(1.0)
    assert np.isnan(auc["V"])


def test_report_and_files(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    probs = rng.dirichlet(np.ones(5), 200)
    pred = probs.argmax(1)
    rep = classification_report(y, pred, probs, CLASSES, ["N", "S", "V", "F"])
    assert rep["n_beats"] == 200
    assert rep["accuracy"] == pytest.approx(np.mean(y == pred))
    assert rep["macro_selected"]["classes"] == ["N", "S", "V", "F"]
    save_confusion_matrix(
        np.array(rep["confusion_matrix"]), CLASSES, tmp_path / "cm.csv", tmp_path / "cm.png"
    )
    assert (tmp_path / "cm.csv").read_text().splitlines()[0] == "true\\pred,N,S,V,F,Q"
    assert (tmp_path / "cm.png").stat().st_size > 0
