import numpy as np
import pytest

from cardiac_twin.config import load_config
from cardiac_twin.data.labels import NON_BEAT_SYMBOLS, aami_class, beat_annotations, symbol_map

CFG = load_config().data
SYM = symbol_map(CFG.aami_map)


@pytest.mark.parametrize(
    "symbol, cls",
    [("N", "N"), ("L", "N"), ("R", "N"), ("e", "N"), ("j", "N"),
     ("A", "S"), ("a", "S"), ("J", "S"), ("S", "S"),
     ("V", "V"), ("E", "V"), ("F", "F"), ("/", "Q"), ("f", "Q"), ("Q", "Q")],
)  # fmt: skip
def test_aami_map(symbol, cls):
    assert aami_class(symbol, SYM) == cls


@pytest.mark.parametrize("symbol", list('+~|"x[]!'))
def test_non_beat_symbols_are_ignored(symbol):
    assert symbol in NON_BEAT_SYMBOLS
    assert aami_class(symbol, SYM) is None


def test_unknown_symbol_raises():
    with pytest.raises(ValueError, match="unknown"):
        aami_class("B", SYM)


def test_beat_and_non_beat_symbols_are_disjoint():
    assert not set(SYM) & NON_BEAT_SYMBOLS


def test_beat_annotations_filters_and_indexes():
    samples = np.array([10, 20, 30, 40, 50])
    r, y = beat_annotations(samples, ["+", "N", "V", "~", "A"], CFG.aami_map, CFG.classes)
    assert r.tolist() == [20, 30, 50]
    assert [CFG.classes[i] for i in y] == ["N", "V", "S"]
