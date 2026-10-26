"""Server F5 with the final model and a DS2 record: same results as the offline replay."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cardiac_twin.acquisition import simulator
from cardiac_twin.config import ROOT, load_config
from cardiac_twin.data.mitbih import load_record
from cardiac_twin.evaluation.replay import replay_record
from cardiac_twin.server.app import create_app
from cardiac_twin.storage.db import Database

pytestmark = [pytest.mark.data, pytest.mark.slow]

CFG = load_config()
FS = CFG.data.fs
RECORD = "213"
MODEL_RUN = ROOT / CFG.server.model_run
RAW_DIR = ROOT / CFG.data.raw_dir


@pytest.fixture(scope="module")
def classifier():
    if not (MODEL_RUN / "model.keras").exists():
        pytest.skip("modelo final ausente")
    from cardiac_twin.model.inference import BeatClassifier

    return BeatClassifier.load(MODEL_RUN, CFG.data.classes, CFG.model.batch_size)


@pytest.fixture(scope="module")
def signal():
    return load_record(RECORD, RAW_DIR, CFG.data.lead).signal


@pytest.fixture
def client(classifier, tmp_path):
    db = Database(tmp_path / "db.sqlite")
    app = create_app(CFG, classifier, db, raw_dir=RAW_DIR, model_run=MODEL_RUN.name)
    with TestClient(app) as c:
        yield c, db


def _stream(client, body, signal, stop=lambda m: False):
    """Send one chunk at a time and read its dashboard messages; stop after the chunk where `stop` holds."""
    sid = client.post("/api/sessions", json={"record_id": RECORD, "source": "external", **body}).json()[
        "session_id"
    ]
    size = round(CFG.engine.chunk_seconds * FS)
    messages, round_trip_ms = [], []
    with client.websocket_connect(f"/ws/monitor/{sid}") as monitor:
        monitor.receive_json()
        with client.websocket_connect(f"/ws/ingest/{sid}") as ingest:
            ingest.receive_json()
            for chunk in simulator.chunk_messages(signal, FS, size):
                start = time.perf_counter()
                ingest.send_json(chunk)
                while (m := monitor.receive_json())["type"] != "signal":
                    messages.append(m)
                round_trip_ms.append((time.perf_counter() - start) * 1000)
                if any(stop(m) for m in messages):
                    break
            else:
                ingest.send_json({"type": "end"})
    return sid, messages, np.array(round_trip_ms)


def _of(messages, kind):
    return [m for m in messages if m["type"] == kind]


@pytest.mark.parametrize("use_profile", [True, False], ids=["B", "A"])
def test_server_gives_the_offline_replay_results(client, classifier, signal, use_profile):
    """Rule 2: the server and the evaluation of Section 4.6 run the same pipeline on the same record."""
    c, db = client
    sid, messages, round_trip_ms = _stream(c, {"use_profile": use_profile}, signal)
    offline = replay_record(
        CFG, classifier, RECORD, RAW_DIR, use_profile=use_profile, rpeak_source=CFG.engine.rpeak_source
    )

    beats = _of(messages, "beat")
    assert [b["sample"] for b in beats] == offline.beat_sample.tolist()
    assert [CFG.data.classes.index(b["cls"]) for b in beats] == offline.beat_cls.tolist()
    assert [b["level"] for b in beats] == offline.beat_level.tolist()
    np.testing.assert_array_equal([np.nan if b["kl"] is None else b["kl"] for b in beats], offline.beat_kl)

    windows = _of(messages, "kl")
    assert [w["t"] for w in windows] == offline.window_t.tolist()
    np.testing.assert_array_equal([w["value"] for w in windows], offline.window_kl)
    np.testing.assert_array_equal([w["L"] for w in windows], offline.window_threshold)

    alerts = _of(messages, "alert")
    assert alerts
    assert [a["t"] for a in alerts] == offline.alert_t.tolist()
    assert [a["episode_id"] for a in alerts] == offline.alert_episode.tolist()

    assert db.beat_count(sid) == len(beats)
    assert len(db.alerts(sid)) == len(alerts)
    assert db.session(sid)["ended_at"] is not None
    assert np.percentile(round_trip_ms, 95) < CFG.evaluation.latency_budget_ms


def test_alert_from_the_model_and_discard_with_feedback(client, signal):
    c, db = client
    calibrated = lambda m: m["type"] == "alert" and m["kl"] is not None  # noqa: E731
    sid, messages, _ = _stream(c, {"use_profile": True, "feedback_enabled": True}, signal, stop=calibrated)
    a = next(m for m in messages if calibrated(m))
    k = CFG.model.context_beats

    assert a["cls"] != "N" or a["level"] == "critical"
    assert sum(a["prob"]) == pytest.approx(1, abs=1e-5)
    assert a["prob"][CFG.data.classes.index(a["cls"])] == max(a["prob"])
    assert len(a["attention"]) == k and sum(a["attention"]) == pytest.approx(1, abs=1e-5)
    assert len(a["context_samples"]) == k and a["sample"] in a["context_samples"]
    assert a["segment_t0"] <= min(a["context_samples"])
    assert a["segment_t0"] + len(a["segment"]) > max(a["context_samples"])
    beat = next(b for b in _of(messages, "beat") if b["sample"] == a["sample"])
    assert beat["prob"] == a["prob"] and beat["attention"] == a["attention"]

    with c.websocket_connect(f"/ws/monitor/{sid}") as monitor:
        monitor.receive_json()
        r = c.post(f"/api/alerts/{a['id']}/decision", json={"decision": "discard"})
        assert monitor.receive_json() == {"type": "decision", "id": a["id"], "decision": "discard"}
        assert monitor.receive_json() == {"type": "profile_updated", "version": 2}
    assert r.json()["profile_version"] == 2
    profiles = c.get(f"/api/sessions/{sid}/history").json()["profiles"]
    assert [p["version"] for p in profiles] == [1, 2]
    assert profiles[0]["L"] == pytest.approx(a["L"])
    assert db.alert(a["id"])["status"] == "discarded"
