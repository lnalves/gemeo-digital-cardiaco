"""Server F5: REST, /ws/ingest and /ws/monitor with a fake classifier and a synthetic ECG."""

import asyncio
import dataclasses
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from cardiac_twin.acquisition import simulator
from cardiac_twin.config import load_config
from cardiac_twin.server.app import CLOSE_NOT_FOUND, create_app
from cardiac_twin.storage.db import Database

BASE = load_config()
CFG = dataclasses.replace(
    BASE,
    profile=dataclasses.replace(BASE.profile, calibration_seconds=30, window_seconds=10, step_seconds=5),
)
FS = CFG.data.fs
RECORD = CFG.data.ds2[0]


class FakeClassifier:
    """Every beat gets the same class."""

    def __init__(self, cls: str = "N") -> None:
        self.classes = CFG.data.classes
        self._k = self.classes.index(cls)

    def predict(self, beats, rr):
        m, k = len(beats), beats.shape[1]
        pred = np.full(m, self._k)
        return pred, np.eye(len(self.classes))[pred], np.full((m, k), 1 / k)


def synthetic_ecg(seconds: float, rr_s: float = 0.8, seed: int = 0) -> np.ndarray:
    n = round(seconds * FS)
    t = np.arange(n)
    x = 0.02 * np.random.default_rng(seed).standard_normal(n)
    for r in np.arange(round(0.5 * FS), n, round(rr_s * FS)):
        x += np.exp(-0.5 * ((t - r) / (0.008 * FS)) ** 2)
        x -= 0.2 * np.exp(-0.5 * ((t - r - 0.02 * FS) / (0.01 * FS)) ** 2)
    return x


@pytest.fixture
def make_client(tmp_path):
    def make(cls="N", launcher=None):
        db = Database(tmp_path / "db.sqlite")
        app = create_app(CFG, FakeClassifier(cls), db, raw_dir=tmp_path, launcher=launcher, model_run="fake")
        return TestClient(app), db

    return make


def _session(client, **body) -> str:
    r = client.post("/api/sessions", json={"record_id": RECORD, "source": "external", **body})
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def _send(ingest, x, chunk_seconds=1.0, start=0):
    size = round(chunk_seconds * FS)
    n = 0
    for message in simulator.chunk_messages(x, FS, size, start):
        ingest.send_json(message)
        n += 1
    return n


def _until(monitor, n_signal):
    """Read monitor messages until the `signal` of the last chunk sent arrives."""
    messages = []
    while n_signal:
        m = monitor.receive_json()
        messages.append(m)
        n_signal -= m["type"] == "signal"
    return messages


def test_session_streams_beats_to_the_dashboard(make_client):
    client, db = make_client()
    with client:
        sid = _session(client)
        x = synthetic_ecg(10)
        with client.websocket_connect(f"/ws/monitor/{sid}") as monitor:
            first = monitor.receive_json()
            assert first["type"] == "state" and first["state"] == "calibrating"
            with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
                assert ingest.receive_json() == {"type": "hello", "next_t0": 0}
                messages = _until(monitor, _send(ingest, x))
                ingest.send_json({"type": "end"})
                assert ingest.receive_json() == {"type": "control", "action": "stop"}
            last = monitor.receive_json()
        beats = [m for m in messages if m["type"] == "beat"]
        assert len(beats) >= 5
        assert set(beats[0]) >= {"sample", "cls", "prob", "attention", "level", "kl", "L"}
        assert any(m["type"] == "hr" for m in messages)
        assert last["type"] == "state" and last["state"] == "finished"
        assert db.beat_count(sid) == len(beats)
        assert db.session(sid)["ended_at"] is not None
        assert client.get(f"/api/sessions/{sid}").json()["status"] == "stopped"


def test_simulator_source_is_launched(make_client):
    calls = []
    client, _ = make_client(launcher=lambda *a: calls.append(a))
    with client:
        sid = client.post("/api/sessions", json={"record_id": RECORD, "speed": 5}).json()["session_id"]
        _session(client)
    assert calls == [(sid, RECORD, 5.0)]


def test_unknown_record_and_session(make_client):
    client, _ = make_client()
    with client:
        assert client.post("/api/sessions", json={"record_id": "999"}).status_code == 404
        assert client.post("/api/sessions/nope/pause").status_code == 404
        assert client.post("/api/alerts/nope/decision", json={"decision": "confirm"}).status_code == 404
        with pytest.raises(Exception) as exc, client.websocket_connect("/ws/monitor/nope") as ws:
            ws.receive_json()
        assert getattr(exc.value, "code", None) == CLOSE_NOT_FOUND


def test_records_mark_ds1_as_training(make_client):
    client, _ = make_client()
    with client:
        rows = client.get("/api/records").json()
    assert {r["record_id"] for r in rows} == {*CFG.data.ds1, *CFG.data.ds2}
    assert all(r["training"] == (r["record_id"] in CFG.data.ds1) for r in rows)
    assert rows[0]["split"] == "DS2"


def _alerts_after_calibration(client, sid, seconds=60):
    with client.websocket_connect(f"/ws/monitor/{sid}") as monitor:
        monitor.receive_json()
        with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
            ingest.receive_json()
            messages = _until(monitor, _send(ingest, synthetic_ecg(seconds)))
    alerts = [m for m in messages if m["type"] == "alert" and m["kl"] is not None]
    assert alerts
    assert any(m["type"] == "state" and m["state"] == "monitoring" for m in messages)
    return alerts, messages


@pytest.mark.parametrize("feedback", [True, False])
def test_decision_persists_and_feedback_updates_profile(make_client, feedback):
    client, db = make_client(cls="S")
    with client:
        sid = _session(client, use_profile=False, feedback_enabled=feedback)
        alerts, messages = _alerts_after_calibration(client, sid)
        a = alerts[0]
        assert a["segment"] and len(a["context_samples"]) == CFG.model.context_beats
        events = {m["id"]: m for m in messages if m["type"] == "event"}
        with client.websocket_connect(f"/ws/monitor/{sid}") as monitor:
            monitor.receive_json()
            r = client.post(f"/api/alerts/{a['id']}/decision", json={"decision": "discard"})
            assert r.status_code == 200
            assert monitor.receive_json() == {"type": "decision", "id": a["id"], "decision": "discard"}
            if feedback:
                assert monitor.receive_json() == {"type": "profile_updated", "version": 2}
        body = r.json()
        assert body["status"] == "discarded" and body["profile_version"] == (2 if feedback else None)
        assert client.post(f"/api/alerts/{a['id']}/decision", json={"decision": "confirm"}).status_code == 409
        confirm = client.post(f"/api/alerts/{alerts[1]['id']}/decision", json={"decision": "confirm"})
        assert confirm.json()["status"] == "confirmed"

        history = client.get(f"/api/sessions/{sid}/history").json()
        by_id = {x["id"]: x for x in history["alerts"]}
        assert by_id[a["id"]]["status"] == "discarded"
        assert by_id[alerts[1]["id"]]["status"] == "confirmed"
        assert events[by_id[a["id"]]["event_id"]]["sample"] == a["sample"]
        assert [p["version"] for p in history["profiles"]] == ([1, 2] if feedback else [1])
        assert len(db.alerts(sid)) == sum(m["type"] == "alert" for m in messages)


def test_pause_resume_stop_reach_the_source(make_client):
    client, db = make_client()
    with client:
        sid = _session(client)
        with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
            ingest.receive_json()
            assert client.post(f"/api/sessions/{sid}/pause").json()["state"] == "paused"
            assert ingest.receive_json() == {"type": "control", "action": "pause"}
            assert client.post(f"/api/sessions/{sid}/resume").json()["state"] == "calibrating"
            assert ingest.receive_json() == {"type": "control", "action": "resume"}
            assert client.post(f"/api/sessions/{sid}/stop").json()["state"] == "finished"
            assert ingest.receive_json() == {"type": "control", "action": "stop"}
        assert client.post(f"/api/sessions/{sid}/resume").status_code == 409
        assert db.session(sid)["ended_at"] is not None


def test_source_reconnects_where_it_stopped(make_client):
    client, _ = make_client()
    with client:
        sid = _session(client)
        x = synthetic_ecg(4)
        with client.websocket_connect(f"/ws/monitor/{sid}") as monitor:
            monitor.receive_json()
            with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
                ingest.receive_json()
                _until(monitor, _send(ingest, x[: 2 * FS]))
            with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
                assert ingest.receive_json() == {"type": "hello", "next_t0": 2 * FS}
                _until(monitor, _send(ingest, x, start=2 * FS))


def test_bad_chunk_closes_the_source(make_client):
    client, _ = make_client()
    with client:
        sid = _session(client)
        with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
            ingest.receive_json()
            ingest.send_json({"type": "chunk", "t0": 0, "fs": 250, "samples": [0.0] * 250})
            with pytest.raises(Exception) as exc:
                ingest.receive_json()
        assert getattr(exc.value, "code", None) == 4400


def test_other_source_gives_the_same_beats(make_client):
    """RNF05: a different source (other chunk size) needs no change in the server."""
    client, _ = make_client()
    x = synthetic_ecg(20)
    beats = []
    with client:
        for chunk_seconds in (1.0, 0.37):
            sid = _session(client)
            with client.websocket_connect(f"/ws/monitor/{sid}") as monitor:
                monitor.receive_json()
                with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
                    ingest.receive_json()
                    messages = _until(monitor, _send(ingest, x, chunk_seconds))
            beats.append([(m["sample"], m["cls"]) for m in messages if m["type"] == "beat"])
    assert beats[0] and beats[0] == beats[1]


class FakeSource:
    """Server side of /ws/ingest for the simulator: sends a control message after the n-th chunk.

    A pause is followed by a resume `pause_s` later on the fake clock, once the simulator waits.
    """

    def __init__(self, now, controls, pause_s):
        self.now = now
        self.times = []
        self.messages = []
        self.inbox = asyncio.Queue()
        self.inbox.put_nowait('{"type": "hello", "next_t0": 0}')
        self._controls = dict(controls)
        self._pause_s = pause_s

    async def send(self, message):
        self.messages.append(message)
        self.times.append(self.now[0])
        action = self._controls.get(len(self.messages))
        if action:
            self.inbox.put_nowait(f'{{"type": "control", "action": "{action}"}}')
        if action == "pause":
            asyncio.get_running_loop().call_later(0.05, self._resume)

    def _resume(self):
        self.now[0] += self._pause_s
        self.inbox.put_nowait('{"type": "control", "action": "resume"}')

    async def recv(self):
        return await self.inbox.get()


def _fake_run(controls, seconds=5, speed=5.0, pause_s=100.0):
    now = [0.0]

    async def sleep(s):
        now[0] += s
        await asyncio.sleep(0)

    async def main():
        source = FakeSource(now, controls, pause_s)
        x = np.zeros(seconds * FS)
        sent = await simulator.stream(source, x, FS, 1.0, speed, clock=lambda: now[0], sleep=sleep)
        return sent, source

    return asyncio.run(main())


def test_simulator_paces_chunks_by_speed():
    sent, source = _fake_run([])
    assert sent == 5 and source.messages[-1] == '{"type": "end"}'
    assert np.diff(source.times[:5]) == pytest.approx([0.2] * 4)


def test_simulator_pause_shifts_the_schedule_without_a_burst():
    sent, source = _fake_run([(2, "pause")])
    gaps = np.diff(source.times[:sent])
    assert sent == 5
    assert (gaps >= 100.0).sum() == 1
    assert gaps[gaps < 100.0] == pytest.approx([0.2] * (len(gaps) - 1))


def test_simulator_stops_on_control():
    sent, source = _fake_run([(3, "stop")])
    assert sent == 3 and '{"type": "end"}' not in source.messages


def test_real_simulator_against_real_server(tmp_path):
    """End to end through the network: uvicorn + the simulator process code (websockets client)."""
    db = Database(tmp_path / "db.sqlite")
    app = create_app(CFG, FakeClassifier(), db, raw_dir=tmp_path, model_run="fake")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        url = f"http://127.0.0.1:{port}"
        sid = simulator.create_session(url, RECORD, use_profile=True, feedback_enabled=False)
        x = synthetic_ecg(8)
        sent = asyncio.run(simulator.run(simulator.ingest_url(url, sid), x, FS, 1.0, speed=100.0))
        assert sent == 8
        for _ in range(200):
            if db.session(sid)["ended_at"]:
                break
            time.sleep(0.01)
        assert db.session(sid)["ended_at"] is not None
        assert db.beat_count(sid) > 0
    finally:
        server.should_exit = True
        thread.join(timeout=5)
