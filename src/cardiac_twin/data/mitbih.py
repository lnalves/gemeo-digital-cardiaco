"""MIT-BIH Arrhythmia Database access."""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import wfdb

from cardiac_twin.config import Config
from cardiac_twin.data.splits import get_splits

RECORD_EXTENSIONS = (".hea", ".dat", ".atr")
TRAINING_SPLIT = "DS1"


@dataclasses.dataclass(frozen=True)
class Record:
    record_id: str
    signal: np.ndarray  # one lead, physical units (mV), float64
    fs: int
    ann_samples: np.ndarray  # every annotation, beats and non-beats; see data.labels.beat_annotations
    ann_symbols: tuple[str, ...]


def load_record(record_id: str, raw_dir: Path, lead: str) -> Record:
    """Read one lead and the `atr` annotations.

    The lead is looked up by name: record 114 has MLII on the second channel.
    """
    path = str(raw_dir / record_id)
    rec = wfdb.rdrecord(path)
    if lead not in rec.sig_name:
        raise ValueError(f"record {record_id} has no {lead} lead (leads: {rec.sig_name})")
    ann = wfdb.rdann(path, "atr")
    return Record(
        record_id=record_id,
        signal=rec.p_signal[:, rec.sig_name.index(lead)].astype(np.float64),
        fs=int(rec.fs),
        ann_samples=np.asarray(ann.sample, dtype=np.int64),
        ann_symbols=tuple(ann.symbol),
    )


def record_duration_s(record_id: str, raw_dir: Path) -> float | None:
    """Duration from the header only; None if the record is not downloaded."""
    if missing_records(raw_dir, [record_id]):
        return None
    header = wfdb.rdheader(str(raw_dir / record_id))
    return header.sig_len / header.fs


def record_catalog(cfg: Config, raw_dir: Path) -> list[tuple[str, str, float | None, int]]:
    """Records the server offers: DS2 for the demo, DS1 marked as training (ARQUITETURA §9)."""
    splits = get_splits(cfg.data)
    named = [(r, "DS2") for r in splits.ds2] + [(r, TRAINING_SPLIT) for r in splits.ds1]
    return [(r, split, record_duration_s(r, raw_dir), cfg.data.fs) for r, split in named]


def all_records(cfg: Config) -> list[str]:
    return sorted({*cfg.data.ds1, *cfg.data.ds2, *cfg.data.paced_records})


def missing_records(raw_dir: Path, records: Sequence[str]) -> list[str]:
    """Return the records that do not have all of their header, signal and annotation files."""
    return [r for r in records if not all((raw_dir / f"{r}{ext}").is_file() for ext in RECORD_EXTENSIONS)]


def _download_record(db: str, record: str, raw_dir: Path) -> None:
    with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
        wfdb.dl_database(db, tmp, records=[record], annotators=["atr"])
        for ext in RECORD_EXTENSIONS:
            shutil.move(Path(tmp) / f"{record}{ext}", raw_dir / f"{record}{ext}")


def download_mitdb(cfg: Config, raw_dir: Path) -> list[str]:
    """Download only the missing records. Return the records that were downloaded.

    Each record goes to a temporary directory first, so an interrupted download never leaves
    a truncated file that `missing_records` would accept.
    """
    missing = missing_records(raw_dir, all_records(cfg))
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=cfg.data.download_workers) as pool:
        list(pool.map(lambda r: _download_record(cfg.data.physionet_db, r, raw_dir), missing))
    still_missing = missing_records(raw_dir, missing)
    if still_missing:
        raise RuntimeError(f"download incomplete: {still_missing}")
    return missing
