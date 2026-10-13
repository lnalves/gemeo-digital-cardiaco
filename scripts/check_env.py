"""Check the F0 acceptance criteria on this machine: Python, TensorFlow, GPU and MIT-BIH records."""

import argparse
import platform
import sys

import tensorflow as tf

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.mitbih import all_records, missing_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--require-gpu", action="store_true", help="exit with error if no GPU is visible")
    args = parser.parse_args()
    cfg = load_config(args.config)

    gpus = tf.config.list_physical_devices("GPU")
    build = tf.sysconfig.get_build_info()
    records = all_records(cfg)
    missing = missing_records(ROOT / cfg.data.raw_dir, records)

    print(f"python      {sys.version.split()[0]} ({platform.platform()})")
    cuda, cudnn = build.get("cuda_version", "-"), build.get("cudnn_version", "-")
    print(f"tensorflow  {tf.__version__} (cuda {cuda}, cudnn {cudnn})")
    print(f"gpus        {[g.name for g in gpus] or 'none'}")
    print(f"mitdb       {len(records) - len(missing)}/{len(records)} records")

    if missing:
        sys.exit(f"missing records: {missing}")
    if args.require_gpu and not gpus:
        sys.exit("no GPU visible to TensorFlow")


if __name__ == "__main__":
    main()
