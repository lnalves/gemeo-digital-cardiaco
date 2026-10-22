"""Compare the classifier alone (A) with classifier + profile (B) on the 22 DS2 records (Seção 4.6)."""

import argparse
import csv
from pathlib import Path

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.splits import get_splits
from cardiac_twin.evaluation.compare import (
    chunk_ms,
    compare_system,
    example_records,
    plot_fp_per_record,
    plot_kl_threshold,
    plot_latency,
)
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.model.train import environment
from cardiac_twin.runs import create_run, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--run", type=Path, default=None, help="classifier run (default: server.model_run)")
    parser.add_argument("--examples", nargs=2, default=None, help="records of the KL × L figure")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.engine.feedback_enabled:
        raise SystemExit("engine.feedback_enabled must be false in the quantitative evaluation (rule 5)")
    model_run = (args.run or ROOT / cfg.server.model_run).resolve()
    records = get_splits(cfg.data).ds2

    out = create_run(
        cfg,
        "eval_system",
        extra={"model_run": model_run.name, "records": list(records), "environment": environment()},
    )
    classifier = BeatClassifier.load(model_run, cfg.data.classes, cfg.model.batch_size)
    result = compare_system(cfg, classifier, records, ROOT / cfg.data.raw_dir)

    with open(out / "per_record.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
    examples = list(args.examples or example_records(result.rows))
    write_json(out / "summary.json", {"model_run": model_run.name, "examples": examples, **result.summary})

    skip = cfg.profile.calibration_seconds
    n_index = cfg.data.classes.index("N")
    plot_fp_per_record(result.rows, out / "fp_per_record.png")
    for rid in examples:
        a, b = result.replays[rid]
        plot_kl_threshold(a, b, n_index, skip, out / f"kl_threshold_{rid}.png")
    plot_latency(chunk_ms(result.replays), cfg.evaluation.latency_budget_ms, out / "latency_boxplot.png")

    s = result.summary
    w = s["wilcoxon_fp"]
    print(f"run: {out}")
    for name in ("a", "b"):
        m = s[name]
        print(
            f"{name.upper()}: FP={m['fp']:5d}  FP/h={m['fp_per_hour']:6.1f}  "
            f"Se S={m['sensitivity_s']:.3f}  Se V={m['sensitivity_v']:.3f}  "
            f"Se S+V={m['sensitivity_sv']:.3f}  alertas/h={m['alarms_per_hour']:6.1f}  "
            f"episódios/h={m['episodes_per_hour']:5.1f}"
        )
    below = s["records_fp_b_below_a"]
    print(f"redução de FP: {s['fp_reduction']:.1%}  registros com FP(B) < FP(A): {below}/{len(records)}")
    print(f"Wilcoxon: W={w['w']}  p={w['p_value']}  mediana(B − A)={w['median_diff_b_minus_a']}")
    lat = s["chunk_ms"]["b"]
    print(f"tempo por bloco (B): p50={lat['p50']:.1f} ms  p95={lat['p95']:.1f} ms  máx={lat['max']:.1f} ms")


if __name__ == "__main__":
    main()
