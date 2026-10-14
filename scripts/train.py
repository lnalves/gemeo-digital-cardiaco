"""Train the beat classifier on DS1-train (early stopping on DS1-val) and save a run in artifacts/runs/."""

import argparse

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.model.train import environment, train, with_loss
from cardiac_twin.runs import create_run, update_run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default=None, help="default: model.loss")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.loss:
        cfg = with_loss(cfg, args.loss)

    run_dir = create_run(cfg, f"train_{cfg.model.loss}", extra={"environment": environment()})
    print(f"run: {run_dir}")
    summary = train(cfg, ROOT / cfg.data.processed_dir, run_dir)
    update_run_info(run_dir, train_seconds=summary["train_seconds"], best_epoch=summary["best_epoch"])
    print(
        f"best epoch {summary['best_epoch']}/{summary['epochs_run']}, "
        f"DS1-val macro-F1 (N,S,V,F) = {summary['best_val_macro_f1']:.4f}, "
        f"{summary['train_seconds'] / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
