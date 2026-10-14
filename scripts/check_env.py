"""Check the F0 acceptance criteria on this machine: Python, TensorFlow, GPU and MIT-BIH records."""

import argparse
import platform
import sys

import tensorflow as tf

from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.mitbih import all_records, missing_records


def gpu_train_step() -> str | None:
    """Run one Conv1D + LSTM gradient step on the GPU and return the error, if any.

    A listed GPU is not enough: a cuDNN without kernels for the card only fails here (D-14).
    """
    try:
        with tf.device("/GPU:0"):
            conv, lstm = tf.keras.layers.Conv1D(4, 3), tf.keras.layers.LSTM(4)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(lstm(conv(tf.random.normal((8, 5, 64)))))
            tape.gradient(loss, conv.trainable_weights + lstm.trainable_weights)
    except tf.errors.OpError as err:
        return err.message.splitlines()[0]
    return None


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
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        cc = ".".join(map(str, details.get("compute_capability", ())))
        print(f"gpu         {gpu.name}: {details.get('device_name', '?')} (compute capability {cc})")
    if not gpus:
        print("gpu         none")
    print(f"mitdb       {len(records) - len(missing)}/{len(records)} records")

    if missing:
        sys.exit(f"missing records: {missing}")
    if args.require_gpu:
        if not gpus:
            sys.exit("no GPU visible to TensorFlow")
        if error := gpu_train_step():
            sys.exit(f"GPU visible but training step failed: {error} (cuDNN version? see D-14)")
        print("gpu step    ok (Conv1D + LSTM)")


if __name__ == "__main__":
    main()
