"""MIT-BIH annotation symbols → AAMI EC57 classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# PhysioNet annotation codes that do not mark a beat (rhythm, noise, waves, comments...).
NON_BEAT_SYMBOLS = frozenset("+~|\"x[]!@()ptu`'^=sT*D{}")


def symbol_map(aami_map: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Invert `data.aami_map` into symbol → class."""
    out: dict[str, str] = {}
    for cls, symbols in aami_map.items():
        for s in symbols:
            if s in out:
                raise ValueError(f"symbol {s!r} mapped to both {out[s]} and {cls}")
            out[s] = cls
    return out


def aami_class(symbol: str, sym_map: Mapping[str, str]) -> str | None:
    """Return the AAMI class of a beat symbol, None for a non-beat annotation.

    An unknown symbol raises instead of being dropped silently.
    """
    if symbol in sym_map:
        return sym_map[symbol]
    if symbol in NON_BEAT_SYMBOLS:
        return None
    raise ValueError(f"unknown annotation symbol {symbol!r}")


def beat_annotations(
    samples: np.ndarray,
    symbols: Sequence[str],
    aami_map: Mapping[str, Sequence[str]],
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only beat annotations. Return (r_samples, label) with label = index in `classes`."""
    sym_map = symbol_map(aami_map)
    index = {c: i for i, c in enumerate(classes)}
    keep, labels = [], []
    for i, s in enumerate(symbols):
        cls = aami_class(s, sym_map)
        if cls is not None:
            keep.append(i)
            labels.append(index[cls])
    return np.asarray(samples)[keep].astype(np.int64), np.asarray(labels, dtype=np.int8)
