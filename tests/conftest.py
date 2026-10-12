"""Fixtures compartilhadas. Testes marcados com `data` exigem data/raw/mitdb baixado."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MITDB = ROOT / "data" / "raw" / "mitdb"


def pytest_collection_modifyitems(config, items):
    if MITDB.exists():
        return
    skip = pytest.mark.skip(reason="MIT-BIH não baixada (rode scripts/download_data.py)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
