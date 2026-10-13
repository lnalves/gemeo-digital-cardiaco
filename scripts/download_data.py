"""Download the MIT-BIH Arrhythmia Database to data/raw/mitdb (idempotent)."""

import argparse

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.mitbih import all_records, download_mitdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = ROOT / cfg.data.raw_dir
    downloaded = download_mitdb(cfg, raw_dir)
    print(f"{len(downloaded)} downloaded, {len(all_records(cfg))} present in {raw_dir}")


if __name__ == "__main__":
    main()
