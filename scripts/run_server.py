"""Start the FastAPI server: REST, /ws/ingest and /ws/monitor (ARQUITETURA §9)."""

import argparse
import functools

import uvicorn

from cardiac_twin.acquisition.simulator import spawn
from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.server.app import create_app
from cardiac_twin.storage.db import Database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--port", type=int, default=None, help="default: server.port")
    args = parser.parse_args()
    cfg = load_config(args.config)
    host, port = cfg.server.host, args.port or cfg.server.port
    model_run = ROOT / cfg.server.model_run

    classifier = BeatClassifier.load(model_run, cfg.data.classes, cfg.model.batch_size)
    app = create_app(
        cfg,
        classifier,
        Database(ROOT / cfg.server.db_path),
        raw_dir=ROOT / cfg.data.raw_dir,
        launcher=functools.partial(spawn, f"http://{host}:{port}", config=args.config),
        model_run=model_run.name,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
