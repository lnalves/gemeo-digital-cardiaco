"""One monitoring session: a StreamingPipeline, its storage and the dashboards that watch it."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any, Literal, Protocol

import numpy as np

from cardiac_twin.config import Config
from cardiac_twin.decision.rules import NORMAL
from cardiac_twin.engine.events import (
    Alert,
    BeatEvent,
    KLUpdate,
    ProfileUpdated,
    SignalChunk,
    StateChange,
)
from cardiac_twin.engine.pipeline import StreamingPipeline
from cardiac_twin.server.schemas import alert_message, event_message
from cardiac_twin.storage.db import Database

Status = Literal["running", "paused", "stopped"]
Message = dict[str, Any]


class Source(Protocol):
    async def send_json(self, data: Any) -> None: ...


class Session:
    """Pipeline calls run in a worker thread under `lock`; `publish` runs in the event loop only."""

    def __init__(
        self,
        session_id: str,
        record_id: str,
        cfg: Config,
        pipeline: StreamingPipeline,
        db: Database,
    ) -> None:
        self.id = session_id
        self.record_id = record_id
        self.cfg = cfg
        self.pipeline = pipeline
        self.db = db
        self.lock = threading.Lock()
        self.status: Status = "running"
        self.source: Source | None = None
        self.launched: Any = None
        self._subscribers: set[asyncio.Queue[Message | None]] = set()
        self._alert_ids: dict[str, str] = {}
        self._rr_peaks: deque[int] = deque(maxlen=cfg.server.hr_beats + 1)

    @property
    def next_t0(self) -> int:
        return self.pipeline.n_samples

    def state_message(self) -> Message:
        state = "paused" if self.status == "paused" else self.pipeline.state
        return {
            "type": "state",
            "state": state,
            "status": self.status,
            "record_id": self.record_id,
            "calibration_seconds": self.cfg.profile.calibration_seconds,
            "t": self.next_t0 / self.cfg.data.fs,
        }

    def process_chunk(self, t0: int, samples: np.ndarray, recv_ts: float) -> list[Message]:
        """Run one chunk; persist beats, decision events, alerts and profiles.

        The `signal` message comes last: it closes the messages of the chunk.
        """
        with self.lock:
            events = self.pipeline.push(samples, t0, recv_ts)
            messages: list[Message] = []
            beats: list[tuple[int, str, float, list[float]]] = []
            event_ids: dict[int, int] = {}
            fs = self.cfg.data.fs
            for e in events:
                if isinstance(e, BeatEvent):
                    beats.append((e.sample, e.cls, max(e.prob), e.attention))
                    self._rr_peaks.append(e.sample)
                    messages.append(event_message(e, fs))
                    if e.level != NORMAL:
                        event_ids[e.sample] = self._event(messages, e, "beat", e.sample, e.cls, e.kl)
                elif isinstance(e, KLUpdate):
                    messages.append(event_message(e, fs))
                    if e.level != NORMAL:
                        self._event(messages, e, "window", round(e.t_end * fs), None, e.kl)
                elif isinstance(e, Alert):
                    alert_id = f"{self.id}-{e.sample}"
                    self._alert_ids[alert_id] = e.alert_id
                    message = alert_message(e, alert_id)
                    self.db.create_alert(alert_id, self.id, event_ids.get(e.sample), e.episode_id, message)
                    messages.append(message)
                elif isinstance(e, StateChange):
                    if e.state == "monitoring":
                        self._save_profile()
                    messages.append(self.state_message())
                elif isinstance(e, SignalChunk):
                    signal = event_message(e, fs)
            self.db.save_beats(self.id, beats)
            if beats and len(self._rr_peaks) > 1:
                rr = np.diff(np.asarray(self._rr_peaks)).mean() / fs
                messages.append({"type": "hr", "bpm": 60.0 / rr})
            messages.append(signal)
            return messages

    def apply_decision(self, alert_id: str, decision: str) -> list[Message]:
        """Pass the decision to the pipeline; a discard with feedback on gives a new profile version."""
        with self.lock:
            pipeline_id = self._alert_ids.pop(alert_id, None)
            if pipeline_id is None or self.status == "stopped":
                return []
            messages = []
            for e in self.pipeline.apply_decision(pipeline_id, decision):
                if isinstance(e, ProfileUpdated):
                    self._save_profile()
                messages.append(event_message(e, self.cfg.data.fs))
            return messages

    def finish(self) -> list[Message]:
        with self.lock:
            self.status = "stopped"
            self.pipeline.finish()
            self.db.end_session(self.id)
            return [self.state_message()]

    def subscribe(self) -> asyncio.Queue[Message | None]:
        queue: asyncio.Queue[Message | None] = asyncio.Queue(maxsize=self.cfg.server.monitor_queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Message | None]) -> None:
        self._subscribers.discard(queue)

    def publish(self, messages: list[Message]) -> None:
        """Send to every dashboard. A dashboard that falls a full queue behind is dropped; it reconnects."""
        for queue in list(self._subscribers):
            try:
                for m in messages:
                    queue.put_nowait(m)
            except asyncio.QueueFull:
                self.unsubscribe(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)

    async def control(self, action: str) -> None:
        if self.source is not None:
            try:
                await self.source.send_json({"type": "control", "action": action})
            except Exception:
                self.source = None  # the source went away; it gets the state again in `hello` on reconnect

    def _event(self, messages: list[Message], e, kind: str, sample: int, cls: str | None, kl) -> int:
        event_id = self.db.save_event(
            self.id, kind, sample, e.level, cls, kl, e.threshold, e.recv_ts, e.emit_ts
        )
        messages.append(
            {
                "type": "event",
                "id": event_id,
                "kind": kind,
                "level": e.level,
                "sample": sample,
                "t": sample / self.cfg.data.fs,
                "cls": cls,
                "kl": kl,
                "L": e.threshold,
            }
        )
        return event_id

    def _save_profile(self) -> None:
        profile = self.pipeline.profile
        k = self.cfg.profile.k
        self.db.save_profile(self.id, profile.to_dict(), k, profile.threshold(k))
