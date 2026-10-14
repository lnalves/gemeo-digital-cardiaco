"""Table of the training runs: DS1-val macro-F1 and per-class F1/Se, grouped by experiment tag and loss."""

import argparse
import collections
import json

import numpy as np

from cardiac_twin.config import ROOT, add_config_argument, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    groups = collections.defaultdict(list)
    for run in sorted((ROOT / cfg.paths.runs_dir).glob("*_train_*")):
        metrics, info = run / "metrics.json", run / "run.json"
        if not metrics.exists():
            continue
        m = json.loads(metrics.read_text(encoding="utf-8"))
        tag = json.loads(info.read_text(encoding="utf-8")).get("tag") or "e0_base"
        per = m["ds1_val"]["per_class"]
        cells = "  ".join(f"{c} F1={per[c]['f1']:.2f} Se={per[c]['sensitivity']:.2f}" for c in "SVF")
        print(f"{run.name:45s} macro-F1={m['best_val_macro_f1']:.4f}  ep={m['best_epoch']:2d}  {cells}")
        groups[(tag, m["loss"])].append(m["best_val_macro_f1"])

    print("\nmean ± sd of DS1-val macro-F1 (N,S,V,F):")
    for (tag, loss), values in sorted(groups.items()):
        v = np.array(values)
        print(f"  {tag:18s} {loss:12s} n={len(v)}  {v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}")


if __name__ == "__main__":
    main()
