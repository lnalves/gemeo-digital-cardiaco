"""Pydantic models of the REST bodies and of the WebSocket messages (ARQUITETURA §9)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from cardiac_twin.engine.events import (
    Alert,
    BeatEvent,
    Event,
    KLUpdate,
    ProfileUpdated,
    SignalChunk,
    StateChange,
)

Source = Literal["simulator", "external"]
DecisionValue = Literal["confirm", "discard"]


class SessionCreate(BaseModel):
    record_id: str
    use_profile: bool = True
    feedback_enabled: bool = False
    speed: float | None = Field(default=None, gt=0)
    source: Source = "simulator"


class SessionCreated(BaseModel):
    session_id: str


class DecisionIn(BaseModel):
    decision: DecisionValue


class ChunkIn(BaseModel):
    type: Literal["chunk"]
    t0: int = Field(ge=0)
    fs: int
    samples: list[float]


class EndIn(BaseModel):
    type: Literal["end"]


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    next_t0: int


class Control(BaseModel):
    type: Literal["control"] = "control"
    action: Literal["pause", "resume", "stop"]


def event_message(event: Event, fs: int) -> dict[str, Any] | None:
    """WebSocket /monitor message of one pipeline event. ProfileUpdated and StateChange keep their fields."""
    if isinstance(event, SignalChunk):
        return {"type": "signal", "t0": event.t0_sample, "fs": fs, "samples": event.samples}
    if isinstance(event, BeatEvent):
        return {
            "type": "beat",
            "sample": event.sample,
            "t": event.t,
            "cls": event.cls,
            "prob": event.prob,
            "attention": event.attention,
            "level": event.level,
            "kl": event.kl,
            "L": event.threshold,
        }
    if isinstance(event, KLUpdate):
        return {
            "type": "kl",
            "t": event.t_end,
            "value": event.kl,
            "L": event.threshold,
            "n_beats": event.n_beats,
            "level": event.level,
        }
    if isinstance(event, Alert):
        return alert_message(event)
    if isinstance(event, ProfileUpdated):
        return {"type": "profile_updated", "version": event.version}
    if isinstance(event, StateChange):
        return {"type": "state", "state": event.state}
    return None


def alert_message(event: Alert, alert_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "alert",
        "id": alert_id or event.alert_id,
        "episode_id": event.episode_id,
        "level": event.level,
        "sample": event.sample,
        "t": event.t,
        "cls": event.cls,
        "prob": event.prob,
        "kl": event.kl,
        "L": event.threshold,
        "attention": event.attention,
        "context_samples": event.context_samples,
        "segment_t0": event.segment_t0,
        "segment": event.segment,
    }
