import dataclasses

import keras
import numpy as np
import pytest

from cardiac_twin.config import load_config
from cardiac_twin.model.architecture import build_inference_model, build_model
from cardiac_twin.model.dataset import BeatArrays, gather_sequences, make_dataset, sequence_index
from cardiac_twin.model.inference import MODEL_FILE, BeatClassifier
from cardiac_twin.model.losses import WeightedFocalLoss, class_weights, make_loss

BASE = load_config()
CFG = dataclasses.replace(
    BASE,
    model=dataclasses.replace(
        BASE.model, cnn_filters=(4, 8, 8), lstm_units=4, attention_units=4, dense_units=8, dropout=0.0
    ),
)
K = CFG.model.context_beats
L = CFG.segment.pre_samples + CFG.segment.post_samples
C = len(CFG.data.classes)


def _batch(n: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = {
        "beats": rng.standard_normal((n, K, L, 1)).astype(np.float32),
        "rr": rng.uniform(0.5, 1.5, (n, K, 5)).astype(np.float32),
    }
    return x, rng.integers(0, C, n)


def _model():
    keras.utils.set_random_seed(0)
    return build_model(CFG, np.ones(5), np.full(5, 0.1))


def test_output_shapes_and_attention_sums_to_one():
    x, _ = _batch()
    probs, attn = build_inference_model(_model())(x)
    assert probs.shape == (16, C) and attn.shape == (16, K)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(np.sum(attn, axis=1), 1, atol=1e-5)


def test_training_steps_reduce_loss():
    x, y = _batch(seed=1)
    model = _model()
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=make_loss("weighted_ce", np.ones(C), 2.0))
    losses = [float(model.train_on_batch(x, y)) for _ in range(5)]
    assert losses[-1] < losses[0]


def test_class_weights_inverse_frequency_with_cap():
    w = class_weights(np.array([0] * 90 + [1] * 9 + [2]), 5, cap=20.0)
    np.testing.assert_allclose(w[:2], [100 / (5 * 90), 100 / (5 * 9)], rtol=1e-6)
    assert w[2] == pytest.approx(20.0)  # 100 / 5 = 20, at the cap
    assert w[3] == w[4] == pytest.approx(20.0)  # absent classes


def test_focal_with_gamma_zero_is_weighted_ce_and_focal_is_smaller():
    y = np.array([0, 1, 2])
    p = np.array(
        [[0.7, 0.1, 0.1, 0.05, 0.05], [0.3, 0.4, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]], np.float32
    )
    w = [1.0, 2.0, 3.0, 1.0, 1.0]
    ce = WeightedFocalLoss(w, gamma=0.0).call(y, p).numpy()
    expected = -np.array(w)[y] * np.log(p[np.arange(3), y])
    np.testing.assert_allclose(ce, expected, rtol=1e-5)
    assert np.all(WeightedFocalLoss(w, gamma=2.0).call(y, p).numpy() < ce)


def test_sequences_never_cross_records():
    rid = np.array(["a"] * 6 + ["b"] * 5)
    centers, seqs = sequence_index(rid, K, CFG.model.future_beats)
    assert centers.tolist() == [3, 4, 9]  # a: 3..4; b (6..10): 9
    assert all(len(set(rid[s])) == 1 for s in seqs)


def test_dataset_yields_center_labels():
    n = 12
    rng = np.random.default_rng(0)
    arrays = BeatArrays(
        beats=rng.standard_normal((n, L)).astype(np.float32),
        rr=rng.standard_normal((n, 5)).astype(np.float32),
        labels=np.arange(n, dtype=np.int8) % C,
        record_id=np.array(["a"] * n),
    )
    centers, seqs = sequence_index(arrays.record_id, K, 1)
    x, y = next(iter(make_dataset(arrays, centers, seqs, batch_size=64, shuffle=False, seed=0)))
    assert x["beats"].shape == (len(centers), K, L, 1) and x["rr"].shape == (len(centers), K, 5)
    assert y.numpy().tolist() == (arrays.labels[centers]).tolist()
    np.testing.assert_array_equal(x["beats"].numpy(), gather_sequences(arrays, seqs)["beats"])


def test_beat_classifier_is_deterministic_after_save_and_load(tmp_path):
    model = _model()
    model.save(tmp_path / MODEL_FILE)
    clf = BeatClassifier.load(tmp_path, CFG.data.classes, batch_size=5)
    x, _ = _batch(n=12, seed=2)
    cls1, p1, a1 = clf.predict(x["beats"][..., 0], x["rr"])
    cls2, p2, a2 = clf.predict(x["beats"], x["rr"])
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(cls1, p1.argmax(1))
    np.testing.assert_allclose(p1, model(x, training=False).numpy(), atol=1e-6)
