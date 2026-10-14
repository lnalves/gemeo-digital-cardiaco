"""Evaluate a training run on DS2, beat by beat; results go to a new run in artifacts/runs/."""

import argparse
from pathlib import Path

from cardiac_twin.config import ROOT, load_config
from cardiac_twin.evaluation.classifier import evaluate_classifier
from cardiac_twin.model.train import environment
from cardiac_twin.runs import CONFIG_FILE, create_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="training run (artifacts/runs/<run>)")
    args = parser.parse_args()
    model_run = args.run.resolve()
    cfg = load_config(model_run / CONFIG_FILE)  # same preprocessing/context as in training

    out = create_run(
        cfg,
        f"eval_classifier_{cfg.model.loss}",
        extra={"model_run": model_run.name, "environment": environment()},
    )
    s = evaluate_classifier(cfg, model_run, ROOT / cfg.data.processed_dir, out)
    r = s["ds2"]
    print(f"run: {out}")
    print(f"DS2: {r['n_beats']} beats, accuracy {r['accuracy']:.4f}")
    for cls, m in r["per_class"].items():
        print(
            f"  {cls}: n={m['support']:5d}  Se={m['sensitivity']:.3f}  +P={m['precision']:.3f}  "
            f"Sp={m['specificity']:.3f}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}"
        )
    sel = r["macro_selected"]
    print(f"  macro ({','.join(sel['classes'])}): F1={sel['f1']:.4f}  Se={sel['sensitivity']:.4f}")
    print(f"  macro (todas): F1={r['macro_all']['f1']:.4f}")


if __name__ == "__main__":
    main()
