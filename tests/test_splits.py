"""Inter-patient protocol (RNF02). Do not remove: CLAUDE.md, rule 1."""

import dataclasses
import re

import pytest

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.data.splits import Splits, check_interpatient, get_splits, select_val_records

CFG = load_config()


def test_ds1_ds2_disjoint_and_paced_excluded():
    s = get_splits(CFG.data)
    assert len(s.ds1) == len(s.ds2) == 22
    assert not set(s.ds1) & set(s.ds2)
    assert not set(s.paced) & (set(s.ds1) | set(s.ds2))
    assert len(set(s.ds1) | set(s.ds2) | set(s.paced)) == 48


def test_val_records_are_fixed_and_inside_ds1():
    s = get_splits(CFG.data)
    assert s.val is not None, "data.val_records must be fixed in configs/default.yaml"
    assert len(s.val) == CFG.data.n_val_records
    assert set(s.val) <= set(s.ds1)
    assert not set(s.val) & set(s.train)
    assert set(s.val) | set(s.train) == set(s.ds1)


@pytest.mark.parametrize(
    "change",
    [{"ds2": ("101",) + CFG.data.ds2[1:]}, {"ds1": ("102",) + CFG.data.ds1[1:]}, {"val": ("100",)}],
)
def test_leaks_are_rejected(change):
    s = dataclasses.replace(get_splits(CFG.data), **change)
    with pytest.raises(ValueError):
        check_interpatient(s)


COUNTS = {
    "a": {"S": 50, "V": 0, "F": 0},
    "b": {"S": 50, "V": 10, "F": 90},
    "c": {"S": 0, "V": 10, "F": 0},
    "d": {"S": 0, "V": 0, "F": 10},
    "e": {"S": 0, "V": 0, "F": 0},
}


def test_select_val_respects_constraints_and_is_deterministic():
    kw = dict(n_records=2, min_beats={"S": 40, "V": 5}, max_fraction={"F": 0.5}, seed=1)
    val = select_val_records(COUNTS, list(COUNTS), **kw)
    assert val == ("a", "c")  # the only pair with S and V that leaves "b" (most F) in training
    assert select_val_records(COUNTS, list(COUNTS), **kw) == val


def test_select_val_fails_when_impossible():
    with pytest.raises(ValueError):
        select_val_records(COUNTS, list(COUNTS), 1, {"S": 200}, {}, seed=0)


# Files allowed to name DS2. Anything else (training, tuning, profile, decision...) must not see it.
DS2_ALLOWED = [
    r"src/cardiac_twin/config\.py",
    r"src/cardiac_twin/data/(__init__|splits|mitbih|builder)\.py",
    r"src/cardiac_twin/evaluation/.*\.py",
    r"scripts/(build_dataset|evaluate_[a-z_]+)\.py",
]


def test_ds2_is_only_referenced_by_allowed_files():
    pattern = re.compile(r"\bds2\b", re.IGNORECASE)
    offenders = []
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]:
        rel = path.relative_to(ROOT).as_posix()
        if any(re.fullmatch(a, rel) for a in DS2_ALLOWED):
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert offenders == [], f"DS2 referenced outside evaluation: {offenders}"


def test_splits_dataclass_train_excludes_val():
    s = Splits(ds1=("1", "2", "3"), ds2=("4",), paced=(), val=("2",))
    assert s.train == ("1", "3")
