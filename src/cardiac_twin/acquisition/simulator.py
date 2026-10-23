"""Acquisition simulator: sends a record to /ws/ingest in chunks, in real time or faster (ARQUITETURA §9).

It talks to the server only by WebSocket, so a real device can take its place (RNF05).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import urllib.request
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cardiac_twin.config import ROOT


class Connection(Protocol):
    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...


def chunk_messages(signal: np.ndarray, fs: int, chunk_size: int, start: int = 0) -> Iterator[dict[str, Any]]:
    for t0 in range(start, len(signal), chunk_size):
        yield {"type": "chunk", "t0": t0, "fs": fs, "samples": signal[t0 : t0 + chunk_size].tolist()}


async def stream(
    ws: Connection,
    signal: np.ndarray,
    fs: int,
    chunk_seconds: float,
    speed: float,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> int:
    """Send the chunks from the `next_t0` of the server `hello`; obey pause, resume and stop.

    Chunk i is due at start + i · chunk_seconds / speed. After a pause the schedule restarts from
    the resume time, so the source does not send a burst to catch up. Return the chunks sent.
    """
    clock = clock or asyncio.get_running_loop().time
    hello = json.loads(await ws.recv())
    running = asyncio.Event()
    running.set()
    stopped = False

    async def listen() -> None:
        nonlocal stopped
        while True:
            action = json.loads(await ws.recv()).get("action")
            if action == "pause":
                running.clear()
            elif action == "resume":
                running.set()
            elif action == "stop":
                stopped = True
                running.set()
                return

    listener = asyncio.create_task(listen())
    interval = chunk_seconds / speed
    sent = 0
    due = clock()
    try:
        for message in chunk_messages(signal, fs, round(chunk_seconds * fs), hello["next_t0"]):
            if not running.is_set():
                await running.wait()
                due = max(due, clock())
            if stopped:
                break
            wait = due - clock()
            if wait > 0:
                await sleep(wait)
            if stopped:
                break
            await ws.send(json.dumps(message))
            sent += 1
            due += interval
        if not stopped:
            await ws.send(json.dumps({"type": "end"}))
    finally:
        listener.cancel()
    return sent


async def run(url: str, signal: np.ndarray, fs: int, chunk_seconds: float, speed: float) -> int:
    from websockets.asyncio.client import connect

    async with connect(url, max_size=None) as ws:
        return await stream(ws, signal, fs, chunk_seconds, speed)


def ingest_url(server: str, session_id: str) -> str:
    return f"{server.replace('http', 'ws', 1)}/ws/ingest/{session_id}"


def create_session(server: str, record_id: str, use_profile: bool, feedback_enabled: bool) -> str:
    """Open a session whose source is this process (source "external": the server starts nothing)."""
    body = {
        "record_id": record_id,
        "use_profile": use_profile,
        "feedback_enabled": feedback_enabled,
        "source": "external",
    }
    request = urllib.request.Request(
        f"{server}/api/sessions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["session_id"]


def spawn(server: str, session_id: str, record_id: str, speed: float, config: Path | None = None):
    """Start `scripts/run_simulator.py` as a separate process for a session that the dashboard opened."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_simulator.py"),
        "--server",
        server,
        "--session",
        session_id,
        "--record",
        record_id,
        "--speed",
        str(speed),
    ]
    if config is not None:
        cmd += ["--config", str(config)]
    return subprocess.Popen(cmd)
