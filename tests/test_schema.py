"""Decision record schema (FR25) and audit rules (append-only, reasons logged)."""
import sqlite3

import pytest

from src.decisions.run import run
from src.store import record

FR25 = {"advisory_time", "resolution", "estimates_json", "recommendation_json", "override_reason", "config_version",
        "crew_hours_entered", "sensor_value", "height_source", "prob_source", "model_version", "output_hash"}


def test_fr25_columns(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(records)")}
    assert FR25 <= cols


def test_decision_b_fields(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    b = next(o for o in run("ian", "T-12h", conn=conn) if o["decision"] == "B")
    row = record.get_record(conn, b["record_id"])
    assert row["height_source"] == "ESTIMATED" and row["prob_source"] == "PSURGE"
    assert row["sensor_value"] is None and "BLANK" in row["sensor_note"]
    assert record.load_config_text(conn, row["config_version"])  # config snapshot stored for replay


def test_records_are_append_only(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    a = run("milton", "T-72h", conn=conn)[0]
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE records SET estimates_json = '{}' WHERE record_id = ?", (a["record_id"],))
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM records WHERE record_id = ?", (a["record_id"],))


def test_override_requires_reason(tmp_path):
    conn = record.connect(tmp_path / "r.db")
    a = run("milton", "T-72h", conn=conn)[0]
    with pytest.raises(ValueError):
        record.add_action(conn, "OVERRIDE", "P1 Storm Director", a["record_id"], reason="  ")
    record.add_action(conn, "OVERRIDE", "P1 Storm Director", a["record_id"], reason="Staging at Sebring: road closure",
                      payload={"mutual_aid_crews": 1500})
    row = record.get_record(conn, a["record_id"])
    assert row["status"] == "OVERRIDDEN" and row["override_reason"].startswith("Staging")
    acts = record.list_actions(conn, a["record_id"])
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM actions WHERE action_id = ?", (int(acts["action_id"].iloc[0]),))
