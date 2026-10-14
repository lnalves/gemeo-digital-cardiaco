"""Run directories: artifacts/runs/<YYYYMMDD-HHMMSS>_<name>/, never overwritten (CLAUDE.md, rule 7)."""

from __future__ import annotations

import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cardiac_twin.config import ROOT, Config, config_to_dict
from cardiac_twin.provenance import git_hash

CONFIG_FILE = "config.yaml"
RUN_FILE = "run.json"


def create_run(cfg: Config, name: str, extra: dict[str, Any] | None = None) -> Path:
    """Create the run directory and record the effective config, git hash, seed and environment."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / cfg.paths.runs_dir / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / CONFIG_FILE).write_text(
        yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "git_hash").write_text(git_hash() + "\n", encoding="utf-8")
    info = {
        "name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        **(extra or {}),
    }
    write_json(run_dir / RUN_FILE, info)
    return run_dir


def update_run_info(run_dir: Path, **fields: Any) -> None:
    path = run_dir / RUN_FILE
    write_json(path, {**json.loads(path.read_text(encoding="utf-8")), **fields})


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_to_builtin), encoding="utf-8")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")
