"""Beat dataset against the literature (de Chazal et al., 2004, inter-patient DS1/DS2)."""

import numpy as np
import pytest

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.data.builder import build_split, process_record
from cardiac_twin.data.mitbih import load_record
from cardiac_twin.data.splits import get_splits
from cardiac_twin.signal.features import RR_FEATURE_NAMES

CFG = load_config()
RAW = ROOT / CFG.data.raw_dir
SPLITS = get_splits(CFG.data)

LITERATURE = {
    "ds1": {"N": 45866, "S": 944, "V": 3788, "F": 415, "Q": 8},
    "ds2": {"N": 44259, "S": 1837, "V": 3221, "F": 388, "Q": 7},
}

pytestmark = [pytest.mark.data, pytest.mark.slow]


@pytest.fixture(scope="module")
def built():
    return {name: build_split(getattr(SPLITS, name), RAW, CFG) for name in ("ds1", "ds2")}


@pytest.mark.parametrize("split", ["ds1", "ds2"])
def test_counts_match_literature(built, split):
    _, summary = built[split]
    lit = LITERATURE[split]
    n_records = len(getattr(SPLITS, split))
    # only the first/last beat of a record (no RR pre/post) or a window past the signal edge is dropped
    assert 0 <= sum(lit.values()) - summary["n_beats"] <= 2 * n_records
    for cls, n in lit.items():
        assert abs(summary["counts"][cls] - n) <= max(2 * n_records, 0.01 * n), cls


def test_qrs_width_plausible_on_ds1(built):
    _, summary = built["ds1"]
    medians = np.array([r["qrs_width_ms_median"] for r in summary["records"].values()])
    assert np.mean((medians >= 60) & (medians <= 200)) >= 0.9


def test_arrays_are_consistent(built):
    arrays, summary = built["ds1"]
    n = summary["n_beats"]
    seg = CFG.segment
    assert arrays["beats"].shape == (n, seg.pre_samples + seg.post_samples)
    assert arrays["beats"].dtype == np.float32
    assert arrays["beat_diff"].shape == arrays["beats"].shape
    assert arrays["rr_feats"].shape == (n, len(RR_FEATURE_NAMES))
    assert not np.isnan(arrays["rr_feats"]).any()
    assert set(np.unique(arrays["labels"])) <= set(range(len(CFG.data.classes)))
    assert set(arrays["record_id"]) == set(SPLITS.ds1)
    np.testing.assert_allclose(arrays["beats"].mean(axis=1), 0, atol=1e-5)


def test_refined_r_stays_close_to_annotation():
    arrays, _ = process_record(load_record("119", RAW, CFG.data.lead), CFG)
    shift_ms = np.abs(arrays["r_sample"] - arrays["ann_sample"]) * 1000 / CFG.data.fs
    assert shift_ms.max() <= CFG.segment.r_refine_ms
    for _, rows in _by_record(arrays):
        assert np.all(np.diff(arrays["r_sample"][rows]) > 0)


def _by_record(arrays):
    for rid in np.unique(arrays["record_id"]):
        yield rid, np.flatnonzero(arrays["record_id"] == rid)
