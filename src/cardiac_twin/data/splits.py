"""Inter-patient split (de Chazal et al., 2004): DS1 for training, DS2 for testing, paced records excluded."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from cardiac_twin.config import DataConfig


@dataclasses.dataclass(frozen=True)
class Splits:
    ds1: tuple[str, ...]
    ds2: tuple[str, ...]
    paced: tuple[str, ...]
    val: tuple[str, ...] | None

    @property
    def train(self) -> tuple[str, ...]:
        """DS1 minus DS1-val: the only records whose beats may update the model weights."""
        val = set(self.val or ())
        return tuple(r for r in self.ds1 if r not in val)


def check_interpatient(splits: Splits) -> None:
    """Raise if any record would leak between DS1, DS2 and the paced records (RNF02)."""
    ds1, ds2, paced = set(splits.ds1), set(splits.ds2), set(splits.paced)
    if ds1 & ds2:
        raise ValueError(f"records in both DS1 and DS2: {sorted(ds1 & ds2)}")
    if paced & (ds1 | ds2):
        raise ValueError(f"paced records in DS1/DS2: {sorted(paced & (ds1 | ds2))}")
    if splits.val is not None and not set(splits.val) <= ds1:
        raise ValueError(f"DS1-val records outside DS1: {sorted(set(splits.val) - ds1)}")
    for name, records in (("DS1", splits.ds1), ("DS2", splits.ds2)):
        if len(set(records)) != len(records):
            raise ValueError(f"duplicate records in {name}")


def get_splits(cfg: DataConfig) -> Splits:
    splits = Splits(
        ds1=tuple(cfg.ds1), ds2=tuple(cfg.ds2), paced=tuple(cfg.paced_records), val=cfg.val_records
    )
    check_interpatient(splits)
    return splits


def select_val_records(
    counts: Mapping[str, Mapping[str, int]],
    ds1: Sequence[str],
    n_records: int,
    min_beats: Mapping[str, int],
    max_fraction: Mapping[str, float],
    seed: int,
) -> tuple[str, ...]:
    """Draw DS1-val among the `n_records`-combinations of DS1 that satisfy both constraints.

    `counts[record][cls]` are beat counts per DS1 record. A valid set holds at least `min_beats[cls]`
    beats of each listed class and at most `max_fraction[cls]` of that class's DS1 beats, so the
    training records keep most of every rare class. The draw is deterministic for a given seed.
    """
    totals = {c: sum(counts[r].get(c, 0) for r in ds1) for c in {*min_beats, *max_fraction}}

    def valid(combo: tuple[str, ...]) -> bool:
        got = {c: sum(counts[r].get(c, 0) for r in combo) for c in totals}
        return all(got[c] >= m for c, m in min_beats.items()) and all(
            got[c] <= f * totals[c] for c, f in max_fraction.items()
        )

    candidates = [combo for combo in itertools.combinations(sorted(ds1), n_records) if valid(combo)]
    if not candidates:
        raise ValueError("no DS1-val combination satisfies the constraints")
    rng = np.random.default_rng(seed)
    return candidates[rng.integers(len(candidates))]
