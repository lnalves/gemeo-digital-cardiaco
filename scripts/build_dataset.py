"""Build data/processed/{ds1,ds2}.npz and stats.json from the MIT-BIH records."""

import argparse
import datetime
import json

import numpy as np

from cardiac_twin.config import ROOT, add_config_argument, config_to_dict, load_config
from cardiac_twin.data.builder import build_split
from cardiac_twin.data.splits import get_splits
from cardiac_twin.provenance import git_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits = get_splits(cfg.data)
    raw_dir, out_dir = ROOT / cfg.data.raw_dir, ROOT / cfg.data.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full = config_to_dict(cfg)
    stats = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "config": {k: full[k] for k in ("data", "signal", "segment", "features")},
        "splits": {},
    }
    for name, records in (("ds1", splits.ds1), ("ds2", splits.ds2)):
        arrays, summary = build_split(records, raw_dir, cfg)
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        stats["splits"][name] = summary
        counts = "  ".join(f"{c}={n}" for c, n in summary["counts"].items())
        print(f"{name}: {len(records)} records, {summary['n_beats']} beats  {counts}")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
