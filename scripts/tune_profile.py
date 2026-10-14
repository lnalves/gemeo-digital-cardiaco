"""Tune k and the KL window on DS1-val with the streaming pipeline and the final classifier (D-09)."""

import argparse
import csv
from pathlib import Path

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.splits import get_splits
from cardiac_twin.evaluation.tuning import tune_profile
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.runs import create_run, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--run", type=Path, default=None, help="classifier run (default: server.model_run)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_run = (args.run or ROOT / cfg.server.model_run).resolve()
    records = get_splits(cfg.data).val

    out = create_run(cfg, "tune_profile", extra={"model_run": model_run.name, "records": list(records)})
    classifier = BeatClassifier.load(model_run, cfg.data.classes, cfg.model.batch_size)
    summary, table = tune_profile(cfg, classifier, records, ROOT / cfg.data.raw_dir)
    with open(out / "tuning.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(out / "metrics.json", {"model_run": model_run.name, "records": list(records), **summary})

    a = summary["classifier_alone"]
    print(f"run: {out}")
    print(f"A (classificador isolado): FP={a['fp']}  Se S+V={a['sensitivity_sv']:.3f}")
    for row in table:
        mark = "*" if row is summary["choice"] else " "
        print(
            f"{mark} k={row['k']:.1f} janela={row['window_seconds']:3d} s  FP={row['fp_b']:5d} "
            f"({row['fp_reduction']:+.0%})  Se S+V={row['sensitivity_sv_b']:.3f}  "
            f"queda={row['sensitivity_drop_pp']:5.1f} p.p.  {'ok' if row['feasible'] else '--'}"
        )


if __name__ == "__main__":
    main()
