"""Send a MIT-BIH record to the server by WebSocket, in chunks of engine.chunk_seconds (ARQUITETURA §9)."""

import argparse
import asyncio

from cardiac_twin.acquisition.simulator import create_session, ingest_url, run
from cardiac_twin.config import ROOT, add_config_argument, load_config
from cardiac_twin.data.mitbih import load_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--record", required=True)
    parser.add_argument("--session", default=None, help="default: open a new session")
    parser.add_argument("--server", default=None, help="default: http://server.host:server.port")
    parser.add_argument(
        "--speed", type=float, default=None, help="1 = real time; default: server.default_speed"
    )
    parser.add_argument("--no-profile", action="store_true", help="new session in configuration A")
    parser.add_argument("--feedback", action="store_true", help="new session with feedback on")
    args = parser.parse_args()
    cfg = load_config(args.config)
    server = args.server or f"http://{cfg.server.host}:{cfg.server.port}"
    speed = args.speed or cfg.server.default_speed

    rec = load_record(args.record, ROOT / cfg.data.raw_dir, cfg.data.lead)
    session = args.session or create_session(server, args.record, not args.no_profile, args.feedback)
    print(f"sessão {session}: registro {args.record} a {speed:g}×")
    sent = asyncio.run(run(ingest_url(server, session), rec.signal, rec.fs, cfg.engine.chunk_seconds, speed))
    print(f"{sent} blocos enviados")


if __name__ == "__main__":
    main()
