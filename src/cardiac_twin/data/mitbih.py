"""MIT-BIH Arrhythmia Database access."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wfdb

from cardiac_twin.config import Config

RECORD_EXTENSIONS = (".hea", ".dat", ".atr")


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
