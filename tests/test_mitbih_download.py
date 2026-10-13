import pytest

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.data.mitbih import RECORD_EXTENSIONS, all_records, missing_records


def test_all_records_has_48():
    assert len(all_records(load_config())) == 48


def test_missing_records_requires_every_file(tmp_path):
    for ext in RECORD_EXTENSIONS:
        (tmp_path / f"100{ext}").touch()
    (tmp_path / "101.hea").touch()
    assert missing_records(tmp_path, ["100", "101", "103"]) == ["101", "103"]


@pytest.mark.data
def test_mitdb_is_complete():
    cfg = load_config()
    assert missing_records(ROOT / cfg.data.raw_dir, all_records(cfg)) == []
