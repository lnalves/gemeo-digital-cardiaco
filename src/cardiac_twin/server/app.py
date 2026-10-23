"""FastAPI app: REST, /ws/ingest (source → server) and /ws/monitor (server → dashboards), ARQUITETURA §9."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from cardiac_twin.config import Config
from cardiac_twin.data.mitbih import TRAINING_SPLIT, record_catalog
from cardiac_twin.engine.pipeline import StreamingPipeline
from cardiac_twin.model.inference import BeatClassifier
from cardiac_twin.server.schemas import (
    ChunkIn,
    DecisionIn,
    Hello,
    SessionCreate,
    SessionCreated,
)
from cardiac_twin.server.session import Session
from cardiac_twin.storage.db import Database

Launcher = Callable[[str, str, float], Any]

CLOSE_NOT_FOUND = 4404
CLOSE_BAD_DATA = 4400
CLOSE_BUSY = 4409
CLOSE_SLOW = 1013

ONLINE_RPEAK_SOURCE = "detector"


def create_app(
    cfg: Config,
    classifier: BeatClassifier,
    db: Database,
    *,
    raw_dir: Path,
    launcher: Launcher | None = None,
    model_run: str | None = None,
) -> FastAPI:
    """`launcher(session_id, record_id, speed)` starts the simulator of a session with source "simulator"."""
    sessions: dict[str, Session] = {}
    db.upsert_records(record_catalog(cfg, raw_dir))
    known = {r["record_id"] for r in db.records()}

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for s in sessions.values():
            if s.status != "stopped":
                await s.control("stop")
            if hasattr(s.launched, "terminate"):
                s.launched.terminate()

    app = FastAPI(title="Gêmeo digital cardíaco", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.server.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def live(session_id: str) -> Session:
        if session_id not in sessions:
            raise HTTPException(404, f"sessão {session_id} não está ativa")
        return sessions[session_id]

    def summary(row: dict[str, Any]) -> dict[str, Any]:
        s = sessions.get(row["id"])
        return {**row, "status": s.status if s else "stopped", "state": s.state_message() if s else None}

    async def stop(session: Session) -> None:
        if session.status == "stopped":
            return
        session.publish(session.finish())
        await session.control("stop")

    @app.get("/api/records")
    def records() -> list[dict[str, Any]]:
        return [{**r, "training": r["split"] == TRAINING_SPLIT} for r in db.records()]

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return [summary(r) for r in db.sessions()]

    @app.post("/api/sessions", status_code=201)
    def create_session(body: SessionCreate) -> SessionCreated:
        if body.record_id not in known:
            raise HTTPException(404, f"registro {body.record_id} desconhecido")
        session_id = uuid.uuid4().hex[:12]
        pipeline = StreamingPipeline(
            cfg,
            classifier,
            body.record_id,
            use_profile=body.use_profile,
            feedback_enabled=body.feedback_enabled,
            rpeak_source=ONLINE_RPEAK_SOURCE,
        )
        db.create_session(session_id, body.record_id, body.use_profile, body.feedback_enabled, model_run)
        session = Session(session_id, body.record_id, cfg, pipeline, db)
        sessions[session_id] = session
        if body.source == "simulator" and launcher is not None:
            session.launched = launcher(session_id, body.record_id, body.speed or cfg.server.default_speed)
        return SessionCreated(session_id=session_id)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        row = db.session(session_id)
        if row is None:
            raise HTTPException(404, f"sessão {session_id} não existe")
        return summary(row)

    @app.post("/api/sessions/{session_id}/pause")
    async def pause(session_id: str) -> dict[str, Any]:
        return await set_status(live(session_id), "paused")

    @app.post("/api/sessions/{session_id}/resume")
    async def resume(session_id: str) -> dict[str, Any]:
        return await set_status(live(session_id), "running")

    async def set_status(session: Session, status: str) -> dict[str, Any]:
        if session.status == "stopped":
            raise HTTPException(409, "sessão encerrada")
        session.status = status
        await session.control("pause" if status == "paused" else "resume")
        message = session.state_message()
        session.publish([message])
        return message

    @app.post("/api/sessions/{session_id}/stop")
    async def stop_session(session_id: str) -> dict[str, Any]:
        session = live(session_id)
        await stop(session)
        return session.state_message()

    @app.get("/api/sessions/{session_id}/history")
    def history(session_id: str) -> dict[str, Any]:
        row = db.session(session_id)
        if row is None:
            raise HTTPException(404, f"sessão {session_id} não existe")
        return {
            "session": summary(row),
            "events": db.events(session_id),
            "alerts": db.alerts(session_id),
            "profiles": db.profiles(session_id),
        }

    @app.post("/api/alerts/{alert_id}/decision")
    async def decide(alert_id: str, body: DecisionIn) -> dict[str, Any]:
        alert = db.alert(alert_id)
        if alert is None:
            raise HTTPException(404, f"alerta {alert_id} não existe")
        if not db.decide_alert(alert_id, body.decision):
            raise HTTPException(409, f"alerta {alert_id} já foi decidido")
        session = sessions.get(alert["session_id"])
        profile_version = None
        if session is not None:
            messages = await asyncio.to_thread(session.apply_decision, alert_id, body.decision)
            decided = {"type": "decision", "id": alert_id, "decision": body.decision}
            session.publish([decided, *messages])
            profile_version = next((m["version"] for m in messages if m["type"] == "profile_updated"), None)
        return {**db.alert(alert_id), "profile_version": profile_version}

    @app.websocket("/ws/ingest/{session_id}")
    async def ingest(ws: WebSocket, session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None or session.status == "stopped":
            await ws.close(CLOSE_NOT_FOUND)
            return
        if session.source is not None:
            await ws.close(CLOSE_BUSY)
            return
        await ws.accept()
        session.source = ws
        try:
            await ws.send_json(Hello(next_t0=session.next_t0).model_dump())
            if session.status == "paused":
                await session.control("pause")
            while session.status != "stopped":
                raw = await ws.receive_json()
                recv_ts = time.perf_counter()
                if session.status == "stopped":
                    break
                if raw.get("type") == "end":
                    await stop(session)
                    break
                chunk = ChunkIn.model_validate(raw)
                if chunk.fs != cfg.data.fs:
                    raise ValueError(f"fs {chunk.fs} Hz, esperado {cfg.data.fs} Hz")
                samples = np.asarray(chunk.samples, dtype=np.float64)
                session.publish(await asyncio.to_thread(session.process_chunk, chunk.t0, samples, recv_ts))
        except WebSocketDisconnect:
            pass
        except (ValidationError, ValueError) as exc:
            await ws.close(CLOSE_BAD_DATA, reason=str(exc)[:120])
        finally:
            if session.source is ws:
                session.source = None

    @app.websocket("/ws/monitor/{session_id}")
    async def monitor(ws: WebSocket, session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            await ws.close(CLOSE_NOT_FOUND)
            return
        await ws.accept()
        queue = session.subscribe()

        async def send() -> None:
            await ws.send_json(session.state_message())
            while (message := await queue.get()) is not None:
                await ws.send_json(message)
            await ws.close(CLOSE_SLOW)

        async def receive() -> None:
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            session.unsubscribe(queue)

    return app
