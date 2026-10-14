"""Draw DS1-val from the DS1 annotations (reads DS1 only); print the value to fix in configs/default.yaml."""

import argparse
import collections
import json

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.labels import beat_annotations
from cardiac_twin.data.mitbih import load_record
from cardiac_twin.data.splits import get_splits, select_val_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    d = cfg.data
    ds1 = get_splits(d).ds1

    counts = {}
    for r in ds1:
        rec = load_record(r, ROOT / d.raw_dir, d.lead)
        _, y = beat_annotations(rec.ann_samples, rec.ann_symbols, d.aami_map, d.classes)
        counts[r] = collections.Counter(d.classes[i] for i in y)

    val = select_val_records(counts, ds1, d.n_val_records, d.val_min_beats, d.val_max_fraction, cfg.seed)
    for name, records in (("DS1-val", val), ("DS1-train", [r for r in ds1 if r not in val])):
        total = collections.Counter()
        for r in records:
            total.update(counts[r])
        print(f"{name:9s} {len(records):2d} records  " + "  ".join(f"{c}={total[c]}" for c in d.classes))
    print(f"\nval_records: {json.dumps(list(val))}")


if __name__ == "__main__":
    main()
