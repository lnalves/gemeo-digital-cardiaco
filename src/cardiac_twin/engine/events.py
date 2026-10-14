"""Events emitted by StreamingPipeline.push (ARQUITETURA §7 and §9)."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar


@dataclasses.dataclass(frozen=True)
class Event:
    kind: ClassVar[str] = "event"
    recv_ts: float
    emit_ts: float
    chunk_end_sample: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, **dataclasses.asdict(self)}


@dataclasses.dataclass(frozen=True)
class StateChange(Event):
    kind: ClassVar[str] = "state"
    state: str


@dataclasses.dataclass(frozen=True)
class SignalChunk(Event):
    kind: ClassVar[str] = "signal"
    t0_sample: int
    samples: list[float]


@dataclasses.dataclass(frozen=True)
class BeatEvent(Event):
    kind: ClassVar[str] = "beat"
    sample: int
    t: float
    cls: str
    prob: list[float]
    attention: list[float]
    level: str
    kl: float | None
    threshold: float | None


@dataclasses.dataclass(frozen=True)
class KLUpdate(Event):
    kind: ClassVar[str] = "kl"
    t_end: float
    kl: float
    threshold: float
    n_beats: int
    level: str


@dataclasses.dataclass(frozen=True)
class Alert(Event):
    kind: ClassVar[str] = "alert"
    alert_id: str
    episode_id: int
    level: str
    sample: int
    t: float
    cls: str
    prob: list[float]
    attention: list[float]
    kl: float | None
    threshold: float | None


@dataclasses.dataclass(frozen=True)
class ProfileUpdated(Event):
    kind: ClassVar[str] = "profile_updated"
    version: int
