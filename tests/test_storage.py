from cardiac_twin.storage.db import Database


def _db(tmp_path):
    db = Database(tmp_path / "sub" / "t.sqlite")
    db.create_session("s1", "100", use_profile=True, feedback_enabled=False, model_run="m")
    return db


def test_schema_is_idempotent(tmp_path):
    path = tmp_path / "t.sqlite"
    Database(path).upsert_records([("100", "DS2", 1805.6, 360)])
    db = Database(path)
    assert db.records() == [{"record_id": "100", "split": "DS2", "duration_s": 1805.6, "fs": 360}]


def test_session_lifecycle(tmp_path):
    db = _db(tmp_path)
    assert db.session("s1")["ended_at"] is None
    db.end_session("s1")
    ended = db.session("s1")["ended_at"]
    db.end_session("s1")
    assert db.session("s1")["ended_at"] == ended is not None
    assert db.session("nope") is None


def test_beats_events_profiles(tmp_path):
    db = _db(tmp_path)
    db.save_beats("s1", [(100, "N", 0.9, [0.2] * 5), (400, "V", 0.8, [0.2] * 5)])
    assert db.beat_count("s1") == 2
    eid = db.save_event("s1", "beat", 400, "alert", "V", 0.3, 0.2, 1.0, 1.001)
    assert db.events("s1")[0]["id"] == eid
    profile = {"version": 1, "edges": {"f": [0.0, 1.0]}, "q": {"f": [1.0]}, "mu": 0.1, "sigma": 0.05}
    db.save_profile("s1", profile, k=1.5, threshold=0.175)
    db.save_profile("s1", {**profile, "version": 2}, k=1.5, threshold=0.175)
    assert [p["version"] for p in db.profiles("s1")] == [1, 2]


def test_alert_decision_only_once(tmp_path):
    db = _db(tmp_path)
    eid = db.save_event("s1", "beat", 400, "alert", "V", None, None, 1.0, 1.001)
    db.create_alert("s1-400", "s1", eid, 1, {"cls": "V", "segment": [0.1, 0.2]})
    assert db.alert("s1-400")["status"] == "pending"
    assert db.decide_alert("s1-400", "discard")
    assert not db.decide_alert("s1-400", "confirm")
    alert = db.alert("s1-400")
    assert alert["status"] == "discarded" and alert["decided_at"] is not None
    assert alert["payload"]["segment"] == [0.1, 0.2]
    assert db.alerts("s1") == [alert]
